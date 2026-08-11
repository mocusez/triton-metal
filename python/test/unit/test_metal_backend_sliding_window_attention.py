"""Sliding-window self-attention on the Metal backend.

The verbatim `leet-triton/hard-sliding_window_self_attention.py` kernel: the
same online-softmax loop shape as `hard-mult_head_attention.py`, but with a
band mask `|offset_m - offset_n| <= window_size` inside the softmax, no head
dimension (1D grid), separate M/N kernel args, a `range(0, cdiv(N, BLOCK_N))`
loop instead of `range(0, N, BLOCK_N)`, and `S / sqrt(d)` instead of
`S * (1/sqrt(d_head))`.

STATUS: supported since Phase C/D of metal-sliding-window-attention-plan.md.
`metal.flash_attention` grew an optional `window` operand (band predicate
`|row - key| <= window` on both the running max and the softmax numerator, plus
a `m_old == m_new` guard so a fully-out-of-band block does not turn
`exp(-inf - -inf)` into a NaN that poisons the row) and an optional `h`
(absent => no head split, `d_head == d_model`, column offset 0). The matcher
learned the cdiv loop form, the `S / sqrt(d)` scale spelling, the two-level
addptr addressing, the `select`-to-zero numerator chain, the mul+add
accumulation spelling, and separate M/N bounds.

The history is worth keeping in view. Before Phase A, `tryFlashAttentionLoop`
claimed this kernel on structural head-counts alone — 3 iter_args / 2 dots /
2 reduces / one trans / 1 store / 1 divsi, all satisfied by coincidence — and
emitted plain full attention with the window mask dropped and `N`/`d_model`/`h`
bound to the function's `tl.cdiv` divsi instead of kernel args. Those bindings
resolved to buffer 0 in the emitter, so the generated MSL read `Q[0][0]` as the
sequence length and the kernel silently wrote NOTHING (output buffer left
untouched; measured on M=32/64/33) or a single element (M=48). Hence
`test_sliding_window_attention_writes_nothing_or_errors` below, which pins the
failure MODE rather than the numbers.

Numerical note: the kernel takes its block max over the band-masked logits with
a `-100` fill and no bounds mask, while the emitter maxes over
`in-bounds AND in-band`. Both are correct — the online softmax divides by its
own running sum, so the result is invariant under any shift of the running max
— and they agree to ~2e-07 here. The emitter's choice is the more robust one.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import triton  # noqa: E402
import triton.language as tl  # noqa: E402

libmetal = pytest.importorskip(
    "triton._C.libtriton.metal",
    reason="Metal backend pybind module not built into libtriton",
)
if not torch.backends.mps.is_available():
    pytest.skip(
        "Metal backend requires an MPS-enabled PyTorch (Apple Silicon)",
        allow_module_level=True,
    )


# Verbatim copy of leet-triton/hard-sliding_window_self_attention.py.
@triton.jit
def attention(
    Q, K, V,
    output,
    M, N, d, window_size,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr
):
    pid = tl.program_id(0)
    offset_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offset_d = tl.arange(0, BLOCK_D)
    mask_m = offset_m < M
    mask_d = offset_d < d
    vals_q = tl.load(Q + offset_m[:, None] * d + offset_d[None, :], mask_m[:, None] & mask_d[None, :], other = 0.0)
    scale = tl.sqrt(d.to(tl.float32))

    out_vals = tl.zeros((BLOCK_M, BLOCK_D), dtype = tl.float32)
    ma = tl.full((BLOCK_M,), float("-inf"), dtype = tl.float32)
    sum = tl.full((BLOCK_M,), 0.0, dtype = tl.float32)

    for step in range(0, tl.cdiv(N, BLOCK_N)):
        offset_n = step * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offset_n < N

        vals_k = tl.load(K + offset_n[:, None] * d + offset_d[None, :], mask_n[:, None] & mask_d[None, :], other = 0.0)

        vals_qk = tl.dot(vals_q, tl.permute(vals_k, (1, 0)), allow_tf32 = False) / scale

        mask = tl.abs(offset_m[:, None] - offset_n[None, :]) <= window_size

        vals_qk_ma = tl.where(mask, vals_qk, float(-100))
        ma_now = tl.maximum(tl.max(vals_qk_ma, axis = 1), ma)
        vals_exp = tl.where(mask_m[:, None] & mask_n[None, :], tl.exp(vals_qk - ma_now[:, None]), 0.0)
        vals_exp = tl.where(mask, vals_exp, 0.0)
        sum_now = tl.sum(vals_exp, axis = 1)

        vals_v = tl.load(V + offset_n[:, None] * d + offset_d[None, :], mask_n[:, None] & mask_d[None, :], other = 0.0)

        out_vals = out_vals * tl.exp(ma - ma_now)[:, None] + tl.dot(vals_exp, vals_v, allow_tf32 = False)

        sum = sum * tl.exp(ma - ma_now) + sum_now
        ma = ma_now

    tl.store(output + offset_m[:, None] * d + offset_d[None, :], out_vals / sum[:, None], mask = mask_m[:, None] & mask_d[None, :])


def _solve(Q, K, V, output, M, d, window_size):
    """Verbatim driver from the leet file (BLOCK_M = BLOCK_N = 16)."""
    BLOCK_M = 16
    BLOCK_N = 16
    BLOCK_D = max(16, triton.next_power_of_2(d))
    grid = (triton.cdiv(M, BLOCK_M), )
    attention[grid](Q, K, V, output, M, M, d, window_size, BLOCK_M, BLOCK_N, BLOCK_D)


def _reference(Q, K, V, M, d, window_size):
    """Band-masked softmax attention.

    Not `F.scaled_dot_product_attention` — building its `attn_mask` for a band
    is more indirection than just writing the three lines. Verified against the
    kernel under TRITON_INTERPRET=1 (maxerr 2.4e-07), so this reference pins
    the kernel's semantics, not the backend's.
    """
    Qc, Kc, Vc = Q.cpu().float(), K.cpu().float(), V.cpu().float()
    scores = (Qc @ Kc.T) / (d ** 0.5)
    idx = torch.arange(M)
    keep = (idx[:, None] - idx[None, :]).abs() <= window_size
    scores = scores.masked_fill(~keep, float("-inf"))
    return torch.softmax(scores, dim=-1) @ Vc


@pytest.mark.parametrize(
    "M, d, window_size",
    [
        (64, 16, 8),    # block-aligned
        (33, 16, 7),    # ragged M -> row tail mask
        (48, 32, 16),   # d == BLOCK_D, no column padding
        (32, 12, 4),    # d < BLOCK_D (=16) -> padded-column masking
        (64, 16, 0),    # diagonal only; denom == 1 for every row
        (64, 16, 1),    # tridiagonal. Triton drops an argument equal to 1 from
                        # the kernel signature and folds it into a `dense<1>`
                        # constant, so this band width arrives as the op's
                        # `window_const` attribute, not as a buffer operand.
        (64, 16, 2),    # the smallest band that still comes through as an arg
        (32, 16, 64),   # window >= N -> degenerates to full attention
        (128, 16, 3),   # window < BLOCK_N -> whole key blocks fall outside the
                        # band, so the running max stays -inf across them. This
                        # is the case that needs the emitter's scaler guard:
                        # exp(-inf - -inf) is NaN and would poison the row.
        (256, 16, 8),   # multiple programs over tgid.x
        (64, 64, 16),   # BLOCK_D = 64 -> 5664 threadgroup floats (under budget)
    ],
)
def test_sliding_window_attention(M, d, window_size):
    torch.manual_seed(0x5A + M * 3 + d + window_size)
    Q = torch.randn(M, d, dtype=torch.float32, device="mps").contiguous()
    K = torch.randn(M, d, dtype=torch.float32, device="mps").contiguous()
    V = torch.randn(M, d, dtype=torch.float32, device="mps").contiguous()
    out = torch.zeros(M, d, dtype=torch.float32, device="mps").contiguous()

    _solve(Q, K, V, out, M, d, window_size)

    expected = _reference(Q, K, V, M, d, window_size)
    torch.testing.assert_close(out.cpu(), expected, atol=1e-3, rtol=1e-3)


def test_sliding_window_attention_over_budget_is_correct_not_rejected():
    """d = 128 -> BLOCK_D = 128 -> 10784 threadgroup floats > the 8192 budget.

    This used to be a HARD REJECT, and pinned as one: the simdgroup body needs
    the whole working set in threadgroup memory, so an over-budget tile had
    nowhere to go and the compile failed. `metal.fused_attention` carries a
    scalar per-key body as its correctness floor and picks between the two by
    threadgroup budget, so an over-budget shape now falls back and computes the
    right answer instead of failing.

    The assertion is NUMERIC on purpose. The thing that could go wrong with the
    fallback is a wrong answer, not a crash, so a bare "it compiles now" check
    would pass just as happily on a body that silently drops the band mask.
    """
    M, d, window_size = 64, 128, 16
    torch.manual_seed(0xB0D)
    Q = torch.randn(M, d, dtype=torch.float32, device="mps").contiguous()
    K = torch.randn(M, d, dtype=torch.float32, device="mps").contiguous()
    V = torch.randn(M, d, dtype=torch.float32, device="mps").contiguous()
    out = torch.zeros(M, d, dtype=torch.float32, device="mps").contiguous()

    _solve(Q, K, V, out, M, d, window_size)

    expected = _reference(Q, K, V, M, d, window_size)
    torch.testing.assert_close(out.cpu(), expected, atol=1e-3, rtol=1e-3)


def test_sliding_window_attention_single_row_rejects():
    """M == 1 is outside the envelope, and must fail loudly rather than quietly.

    Triton folds any kernel argument equal to 1 out of the signature. For the
    band width that is handled (`window_const`), but M is also the key count N
    here, so `tl.cdiv(N, BLOCK_N)` collapses to a constant loop bound and the
    row/key bounds lose their buffers — three separate reasons the template
    verifier cannot bind the emitter's operands. Supporting it would mean an
    attribute fallback for every scalar, which is not worth it for a one-row
    attention; what matters is that it is a compile error and not a silent
    wrong answer.
    """
    M, d, window_size = 1, 16, 4
    torch.manual_seed(0x1)
    Q = torch.randn(M, d, dtype=torch.float32, device="mps").contiguous()
    K = torch.randn(M, d, dtype=torch.float32, device="mps").contiguous()
    V = torch.randn(M, d, dtype=torch.float32, device="mps").contiguous()
    out = torch.zeros(M, d, dtype=torch.float32, device="mps").contiguous()

    with pytest.raises(RuntimeError):
        _solve(Q, K, V, out, M, d, window_size)


def test_sliding_window_attention_writes_nothing_or_errors():
    """Regression lock for the SILENT-miscompile mode specifically.

    The Phase-A bug was not "wrong numbers" — it was the kernel returning
    successfully having written zero elements, which every allclose-style
    assertion in the file above would report as a plain numeric mismatch. This
    test states the real invariant: the launch either raises, or it writes
    every output element. A kernel that returns success while leaving the
    output buffer untouched fails here and nowhere else.

    Survives Phase C/D unchanged (then it takes the second branch).
    """
    M, d, window_size = 64, 16, 8
    torch.manual_seed(0x51E)
    Q = torch.randn(M, d, dtype=torch.float32, device="mps").contiguous()
    K = torch.randn(M, d, dtype=torch.float32, device="mps").contiguous()
    V = torch.randn(M, d, dtype=torch.float32, device="mps").contiguous()
    sentinel = 1234.5
    out = torch.full((M, d), sentinel, dtype=torch.float32, device="mps").contiguous()

    try:
        _solve(Q, K, V, out, M, d, window_size)
    except RuntimeError:
        return  # rejected outright — acceptable, and what Phase A produces

    untouched = int((out.cpu() == sentinel).sum())
    assert untouched == 0, (
        f"kernel reported success but left {untouched}/{M * d} output elements "
        "at the sentinel value — this is the silent-miscompile mode "
        "(metal-sliding-window-attention-plan.md §0)"
    )
