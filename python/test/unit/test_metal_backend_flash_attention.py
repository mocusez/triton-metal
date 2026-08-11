"""Flash attention (online-softmax multi-head attention) on the Metal backend.

The verbatim leet-triton `hard-mult_head_attention.py` kernel: a `scf.for` over
key blocks carrying (2D accumulator, running sum, running max) that does two
`tl.dot`s (Q@K^T and P@V) with a masked online softmax in between. The matmul
track can't absorb the second dot (its A operand is a computed `exp`, not a
load), so `tryFlashAttentionLoop` (TritonGPUToMetal.cpp) recognizes the whole
loop + divide epilogue + masked store and lowers it to one `metal.flash_attention`
op, whose emitter renders the Phase-0-validated simdgroup body (both matmuls on
simdgroup hardware; S/P/O + running state in the threadgroup scalar domain).

Compared against `torch.scaled_dot_product_attention` per head. Envelope:
BLOCKSIZE_N == 32, num_warps == 4 (matches the leet driver), tile dims multiples
of 8, threadgroup working set <= 32 KiB. d_head varies: d_head < BLOCKSIZE_d (=
max(16, d_head)) exercises the padded-column masking (`d < d_head`); d_head ==
BD is the unpadded path. N covers block-aligned, masked-tail, and ragged cases.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import triton
import triton.language as tl

libmetal = pytest.importorskip(
    "triton._C.libtriton.metal",
    reason="Metal backend pybind module not built into libtriton",
)
if not torch.backends.mps.is_available():
    pytest.skip(
        "Metal backend requires an MPS-enabled PyTorch (Apple Silicon)",
        allow_module_level=True,
    )


@triton.jit
def mha_kernel(Q_ptr, K_ptr, V_ptr, output_ptr,
               N, d_model, h,
               BLOCKSIZE_N: tl.constexpr,
               BLOCKSIZE_d: tl.constexpr):
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    offset = tl.arange(0, BLOCKSIZE_N)
    d_head = d_model // h
    offset_N = pid0 * BLOCKSIZE_N + offset
    offset_d = pid1 * d_head + tl.arange(0, BLOCKSIZE_d)
    mask_d = tl.arange(0, BLOCKSIZE_d) < d_head
    offset_Q = offset_N[:, None] * d_model + offset_d[None, :]
    mask_Q = (offset_N[:, None] < N) & mask_d[None, :]
    data_Q = tl.load(Q_ptr + offset_Q, mask=mask_Q)
    attention_logit_scale = 1.0 / tl.sqrt(d_head + 0.0)
    accumulator = tl.zeros((BLOCKSIZE_N, BLOCKSIZE_d), dtype=tl.float32)
    softmax_running_sum = tl.zeros([BLOCKSIZE_N], dtype=tl.float32)
    softmax_current_max = tl.full([BLOCKSIZE_N], float("-inf"), dtype=tl.float32)
    for current_index in range(0, N, BLOCKSIZE_N):
        current_N_offset = current_index + offset
        current_N_mask = current_N_offset < N
        offset_K = current_N_offset[:, None] * d_model + offset_d[None, :]
        mask_K = current_N_mask[:, None] & mask_d[None, :]
        data_K = tl.load(K_ptr + offset_K, mask=mask_K, other=0.0)
        offset_V = current_N_offset[:, None] * d_model + offset_d[None, :]
        mask_V = current_N_mask[:, None] & mask_d[None, :]
        data_V = tl.load(V_ptr + offset_V, mask=mask_V, other=0.0)
        attention_logits = tl.dot(data_Q, tl.trans(data_K)) * attention_logit_scale
        attention_logits_mask = (offset_N[:, None] < N) & (current_N_offset[None, :] < N)
        attention_logits = tl.where(attention_logits_mask, attention_logits, float("-inf"))
        current_block_max = tl.max(attention_logits, axis=1)
        max_value = tl.maximum(current_block_max, softmax_current_max)
        softmax_scaler = tl.exp(softmax_current_max - max_value)
        softmax_current_max = max_value
        attention_logits_shift = attention_logits - max_value[:, None]
        softmax_nom = tl.exp(attention_logits_shift)
        softmax_denom = tl.sum(softmax_nom, axis=1)
        softmax_running_sum = tl.fma(softmax_running_sum, softmax_scaler, softmax_denom)
        accumulator = tl.fma(accumulator, softmax_scaler[:, None], tl.dot(softmax_nom, data_V))
    accumulator = accumulator / softmax_running_sum[:, None]
    tl.store(output_ptr + offset_Q, accumulator, mask=mask_Q)


def _reference(Q, K, V, N, d_model, h):
    d_head = d_model // h
    ref = torch.zeros(N, d_model, dtype=torch.float32)
    Qc, Kc, Vc = Q.cpu(), K.cpu(), V.cpu()
    for hi in range(h):
        sl = slice(hi * d_head, (hi + 1) * d_head)
        ref[:, sl] = torch.nn.functional.scaled_dot_product_attention(
            Qc[:, sl].unsqueeze(0), Kc[:, sl].unsqueeze(0), Vc[:, sl].unsqueeze(0)
        ).squeeze(0)
    return ref


@pytest.mark.parametrize(
    "N, d_model, h",
    [
        (64, 64, 4),    # block-aligned, d_head=16 (== BD)
        (48, 64, 4),    # masked tail (48 = 32 + 16)
        (33, 64, 4),    # ragged
        (128, 128, 8),  # more heads, d_head=16
        (96, 32, 2),    # fewer heads, d_head=16
        (256, 64, 4),   # many key blocks
        (64, 64, 8),    # d_head=8 < BD=16 -> padded-column masking
        (64, 64, 2),    # d_head=32 == BD=32 (no padding)
        (96, 96, 3),    # d_head=32, ragged N
    ],
)
def test_flash_attention_online_softmax(N, d_model, h):
    torch.manual_seed(0xF1A5 + N * 7 + d_model + h)
    Q = torch.randn(N, d_model, dtype=torch.float32, device="mps").contiguous()
    K = torch.randn(N, d_model, dtype=torch.float32, device="mps").contiguous()
    V = torch.randn(N, d_model, dtype=torch.float32, device="mps").contiguous()
    out = torch.zeros(N, d_model, dtype=torch.float32, device="mps").contiguous()

    BLOCKSIZE_N = 32
    BLOCKSIZE_d = max(16, d_model // h)
    grid = (triton.cdiv(N, BLOCKSIZE_N), h)
    mha_kernel[grid](Q, K, V, out, N, d_model, h, BLOCKSIZE_N, BLOCKSIZE_d, num_warps=4)

    expected = _reference(Q, K, V, N, d_model, h)
    torch.testing.assert_close(out.cpu(), expected, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize(
    "N, d_model, h",
    [
        (64, 64, 4),   # block-aligned
        (48, 64, 4),   # masked tail
        (33, 64, 4),   # ragged
        (96, 96, 3),   # d_head=32 > BLOCKSIZE_N
    ],
)
def test_flash_attention_block16_lane_guard(N, d_model, h):
    """BLOCKSIZE_N == 16, i.e. bm < 32 — the FA emitter's per-row buffers.

    The emitter runs the online-softmax stage on one warp with `q = _fa_lane`
    ranging over all 32 lanes, but `_fa_rmax`/`_fa_rsum` hold bm floats,
    `_fa_pbuf` bm*bn and `_fa_obuf` bm*bd. At bm == 32 that is exactly in
    bounds, which is why the bm == 32 cases above never caught it; at bm == 16
    lanes 16..31 write past the end of every one of them (both the `row < N`
    arm and its pbuf-zeroing else arm). Measured before the `q < bm` guard:
    maxerr 1.7e+04 on (64, 64, 4). See metal-sliding-window-attention-plan.md
    §1c — the sliding-window driver uses BLOCK_M = 16, so this had to be fixed
    before that kernel could work regardless of the matcher gates.
    """
    torch.manual_seed(0xB16 + N * 7 + d_model + h)
    Q = torch.randn(N, d_model, dtype=torch.float32, device="mps").contiguous()
    K = torch.randn(N, d_model, dtype=torch.float32, device="mps").contiguous()
    V = torch.randn(N, d_model, dtype=torch.float32, device="mps").contiguous()
    out = torch.zeros(N, d_model, dtype=torch.float32, device="mps").contiguous()

    BLOCKSIZE_N = 16
    BLOCKSIZE_d = max(16, d_model // h)
    grid = (triton.cdiv(N, BLOCKSIZE_N), h)
    mha_kernel[grid](Q, K, V, out, N, d_model, h, BLOCKSIZE_N, BLOCKSIZE_d, num_warps=4)

    expected = _reference(Q, K, V, N, d_model, h)
    torch.testing.assert_close(out.cpu(), expected, atol=1e-3, rtol=1e-3)


# ---------------------------------------------------------------------------
# Decoys: kernels the FA matcher must NOT claim (Phase B template verifier).
# ---------------------------------------------------------------------------
#
# Each of these is structurally IDENTICAL to mha_kernel — 3 iter_args (one
# rank-2 accumulator + two rank-1), 2 tt.dot, 2 tt.reduce, one transposed dot
# operand, 1 tt.store, 1 arith.divsi with kernel-arg operands feeding
# `pid1 * d_head` — so every gate the matcher had before Phase B passes. Only
# the SEMANTICS differ, in one op each.
#
# Measured with the pre-Phase-B matcher, all three compiled and returned
# silently wrong answers (maxerr 2.4 / 1.4 / 0.95 against their own
# references); `tryFlashAttentionLoop` dropped the difference on the floor and
# emitted plain full attention. The Phase-B role walk rejects each with a
# distinct reason ("softmax mask is not exactly (row < N) & (key < N)",
# "logits are not <dot> * scale", "logit scale is not 1/sqrt(d_head)"; set
# TRITON_METAL_FA_DEBUG=1 to print them).
#
# The assertion is deliberately "raises OR numerically correct", not "raises":
# it states the invariant that actually matters (no silent wrong answer) and
# stays valid if a later phase teaches the emitter one of these variants.


@triton.jit
def _decoy_kernel(Q_ptr, K_ptr, V_ptr, output_ptr, N, d_model, h,
                  BLOCKSIZE_N: tl.constexpr, BLOCKSIZE_d: tl.constexpr,
                  VARIANT: tl.constexpr):
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    offset = tl.arange(0, BLOCKSIZE_N)
    d_head = d_model // h
    offset_N = pid0 * BLOCKSIZE_N + offset
    offset_d = pid1 * d_head + tl.arange(0, BLOCKSIZE_d)
    mask_d = tl.arange(0, BLOCKSIZE_d) < d_head
    offset_Q = offset_N[:, None] * d_model + offset_d[None, :]
    mask_Q = (offset_N[:, None] < N) & mask_d[None, :]
    data_Q = tl.load(Q_ptr + offset_Q, mask=mask_Q)
    scale = 1.0 / tl.sqrt(d_head + 0.0)
    if VARIANT == 2:  # forgot the sqrt — a plausible typo, silently wrong
        scale = 1.0 / (d_head + 0.0)
    accumulator = tl.zeros((BLOCKSIZE_N, BLOCKSIZE_d), dtype=tl.float32)
    running_sum = tl.zeros([BLOCKSIZE_N], dtype=tl.float32)
    running_max = tl.full([BLOCKSIZE_N], float("-inf"), dtype=tl.float32)
    for current_index in range(0, N, BLOCKSIZE_N):
        current_N_offset = current_index + offset
        current_N_mask = current_N_offset < N
        offset_K = current_N_offset[:, None] * d_model + offset_d[None, :]
        mask_K = current_N_mask[:, None] & mask_d[None, :]
        data_K = tl.load(K_ptr + offset_K, mask=mask_K, other=0.0)
        data_V = tl.load(V_ptr + offset_K, mask=mask_K, other=0.0)
        logits = tl.dot(data_Q, tl.trans(data_K)) * scale
        if VARIANT == 1:  # ALiBi-style linear positional bias
            logits = logits + (offset_N[:, None] - current_N_offset[None, :]) * 0.125
        logits_mask = (offset_N[:, None] < N) & (current_N_offset[None, :] < N)
        if VARIANT == 0:  # causal
            logits_mask = logits_mask & (current_N_offset[None, :] <= offset_N[:, None])
        logits = tl.where(logits_mask, logits, float("-inf"))
        max_value = tl.maximum(tl.max(logits, axis=1), running_max)
        scaler = tl.exp(running_max - max_value)
        running_max = max_value
        nom = tl.exp(logits - max_value[:, None])
        running_sum = tl.fma(running_sum, scaler, tl.sum(nom, axis=1))
        accumulator = tl.fma(accumulator, scaler[:, None], tl.dot(nom, data_V))
    tl.store(output_ptr + offset_Q, accumulator / running_sum[:, None], mask=mask_Q)


def _decoy_reference(Q, K, V, N, d_model, h, variant):
    d_head = d_model // h
    Qc, Kc, Vc = Q.cpu(), K.cpu(), V.cpu()
    out = torch.zeros(N, d_model, dtype=torch.float32)
    idx = torch.arange(N)
    for hi in range(h):
        sl = slice(hi * d_head, (hi + 1) * d_head)
        denom = d_head if variant == 2 else d_head ** 0.5
        scores = (Qc[:, sl] @ Kc[:, sl].T) / denom
        if variant == 1:
            scores = scores + (idx[:, None] - idx[None, :]) * 0.125
        if variant == 0:
            scores = scores.masked_fill(idx[None, :] > idx[:, None], float("-inf"))
        out[:, sl] = torch.softmax(scores, dim=-1) @ Vc[:, sl]
    return out


@pytest.mark.parametrize(
    "variant, what",
    [
        (0, "causal mask"),
        (1, "ALiBi-style linear bias"),
        (2, "scale missing the sqrt"),
    ],
)
def test_flash_attention_rejects_near_miss_kernels(variant, what):
    N, d_model, h = 64, 64, 4
    torch.manual_seed(0xDEC0 + variant)
    Q = torch.randn(N, d_model, dtype=torch.float32, device="mps").contiguous()
    K = torch.randn(N, d_model, dtype=torch.float32, device="mps").contiguous()
    V = torch.randn(N, d_model, dtype=torch.float32, device="mps").contiguous()
    out = torch.zeros(N, d_model, dtype=torch.float32, device="mps").contiguous()

    grid = (triton.cdiv(N, 32), h)
    try:
        _decoy_kernel[grid](Q, K, V, out, N, d_model, h, 32, max(16, d_model // h),
                            variant, num_warps=4)
    except RuntimeError:
        return  # rejected — the expected outcome today

    expected = _decoy_reference(Q, K, V, N, d_model, h, variant)
    err = (out.cpu() - expected).abs().max().item()
    assert err < 1e-3, (
        f"the FA matcher claimed a kernel with a {what} and dropped it: "
        f"maxerr {err:.3e}. It must either reject the kernel or implement the "
        "variant — see metal-sliding-window-attention-plan.md §4 Phase B."
    )


# `N == BLOCKSIZE_N` (and `h == 1`, which Triton folds out of the signature) is
# the shape the loop-anchored FA matcher cannot claim, so before
# `metal.fused_attention` existed this was a hard
# `convert-tritongpu-to-metal failed` — on the SAME kernel that compiles fine at
# N=64. A single-block sweep is the degenerate case of every attention kernel,
# so it is worth pinning separately from the parametrized sweep above.
@pytest.mark.parametrize("N, d_model, h", [(32, 32, 1), (16, 16, 1), (32, 64, 2)])
def test_flash_attention_single_key_block(N, d_model, h):
    torch.manual_seed(0x5106 + N)
    Q = torch.randn(N, d_model, dtype=torch.float32, device="mps").contiguous()
    K = torch.randn(N, d_model, dtype=torch.float32, device="mps").contiguous()
    V = torch.randn(N, d_model, dtype=torch.float32, device="mps").contiguous()
    out = torch.zeros(N, d_model, dtype=torch.float32, device="mps").contiguous()

    BLOCKSIZE_N = 32
    BLOCKSIZE_d = max(16, d_model // h)
    grid = (triton.cdiv(N, BLOCKSIZE_N), h)
    mha_kernel[grid](Q, K, V, out, N, d_model, h, BLOCKSIZE_N, BLOCKSIZE_d, num_warps=4)

    expected = _reference(Q, K, V, N, d_model, h)
    torch.testing.assert_close(out.cpu(), expected, atol=1e-3, rtol=1e-3)
