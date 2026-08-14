"""Flash attention (online-softmax multi-head attention) on the Metal backend.

The verbatim leet-triton `hard-mult_head_attention.py` kernel: a `scf.for` over
key blocks carrying (2D accumulator, running sum, running max) that does two
`tl.dot`s (Q@K^T and P@V) with a masked online softmax in between. The matmul
track can't absorb the second dot (its A operand is a computed `exp`, not a
load), so `tryFusedAttention` (TritonGPUToMetal.cpp) recognizes the whole loop +
divide epilogue + masked store and lowers it to one `metal.fused_attention` op,
whose emitter renders the simdgroup body (both matmuls on simdgroup hardware;
S/P/O + running state in the threadgroup scalar domain).

This suite used to back a dedicated `metal.flash_attention` op. That op is gone:
this suite and the sliding-window one both reached parity on the generalized op,
which was the deletion precondition -- an op is deletable when every suite it
BACKS is at parity, not just the one named after it.

Compared against `torch.scaled_dot_product_attention` per head. Envelope:
BLOCKSIZE_N == 32, num_warps == 4 (matches the leet driver), tile dims multiples
of 8, threadgroup working set <= 32 KiB. d_head varies: d_head < BLOCKSIZE_d (=
max(16, d_head)) exercises the padded-column masking (`d < d_head`); d_head ==
BD is the unpadded path. N covers block-aligned, masked-tail, and ragged cases.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

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


@triton.jit
def _grouped_query_attention_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    stride_qh, stride_qs, stride_qd,
    stride_kh, stride_ks, stride_kd,
    stride_vh, stride_vs, stride_vd,
    stride_oh, stride_os, stride_od,
    seq_len, head_dim, scale, groups,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    start_m = tl.program_id(0)
    q_head_idx = tl.program_id(1)
    kv_head_idx = q_head_idx // groups

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    mask_m = offs_m < seq_len
    mask_d = offs_d < head_dim

    q_ptrs = (q_ptr + q_head_idx * stride_qh +
              offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd)
    q = tl.load(q_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0)

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    for start_n in range(0, seq_len, BLOCK_N):
        offs_n_curr = start_n + offs_n
        mask_n = offs_n_curr < seq_len

        k_ptrs = (k_ptr + kv_head_idx * stride_kh +
                  offs_n_curr[:, None] * stride_ks +
                  offs_d[None, :] * stride_kd)
        v_ptrs = (v_ptr + kv_head_idx * stride_vh +
                  offs_n_curr[:, None] * stride_vs +
                  offs_d[None, :] * stride_vd)
        kv_mask = mask_n[:, None] & mask_d[None, :]
        k = tl.load(k_ptrs, mask=kv_mask, other=0.0)
        v = tl.load(v_ptrs, mask=kv_mask, other=0.0)

        qk = tl.dot(q, tl.trans(k)) * scale
        score_mask = mask_m[:, None] & mask_n[None, :]
        qk = tl.where(score_mask, qk, float("-inf"))
        m_i_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(qk - m_i_new[:, None])
        p = tl.where(score_mask, p, 0.0)
        l_i = alpha * l_i + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p, v)
        m_i = m_i_new

    acc = acc / l_i[:, None]
    o_ptrs = (o_ptr + q_head_idx * stride_oh +
              offs_m[:, None] * stride_os + offs_d[None, :] * stride_od)
    tl.store(o_ptrs, acc, mask=mask_m[:, None] & mask_d[None, :])


def _launch_grouped_query_attention(
    Q, K, V, output, num_q_heads, num_kv_heads, seq_len, head_dim,
):
    block_d = triton.next_power_of_2(max(16, head_dim))
    if block_d >= 256:
        block_m, block_n = 16, 16
    elif block_d >= 128:
        block_m, block_n = 32, 32
    else:
        block_m, block_n = 64, 64
    grid = (triton.cdiv(seq_len, block_m), num_q_heads)
    return _grouped_query_attention_kernel[grid](
        Q, K, V, output,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        output.stride(0), output.stride(1), output.stride(2),
        seq_len, head_dim, head_dim ** -0.5,
        num_q_heads // num_kv_heads,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_D=block_d,
    )


def _gqa_reference(Q, K, V, num_q_heads, num_kv_heads):
    groups = num_q_heads // num_kv_heads
    return torch.nn.functional.scaled_dot_product_attention(
        Q.cpu(),
        K.cpu().repeat_interleave(groups, dim=0),
        V.cpu().repeat_interleave(groups, dim=0),
    )


@pytest.mark.parametrize(
    "num_q_heads, num_kv_heads, seq_len, head_dim",
    [
        pytest.param(4, 2, 64, 16, id="grouped-heads-block-aligned"),
        pytest.param(4, 2, 33, 16, id="ragged-sequence"),
        pytest.param(4, 2, 64, 12, id="padded-head-dim"),
        pytest.param(8, 2, 129, 32, id="multiple-key-blocks"),
    ],
)
def test_leet_grouped_query_attention_matches_reference(
    num_q_heads, num_kv_heads, seq_len, head_dim
):
    torch.manual_seed(
        0x60A + num_q_heads * 31 + num_kv_heads * 7 + seq_len + head_dim)
    Q = torch.randn(
        num_q_heads, seq_len, head_dim, dtype=torch.float32,
        device="mps").contiguous()
    K = torch.randn(
        num_kv_heads, seq_len, head_dim, dtype=torch.float32,
        device="mps").contiguous()
    V = torch.randn(
        num_kv_heads, seq_len, head_dim, dtype=torch.float32,
        device="mps").contiguous()
    sentinel = 1234.5
    out = torch.full_like(Q, sentinel)

    compiled = _launch_grouped_query_attention(
        Q, K, V, out, num_q_heads, num_kv_heads, seq_len, head_dim)
    torch.mps.synchronize()

    if (num_q_heads, num_kv_heads, seq_len, head_dim) == (4, 2, 64, 16):
        msl = compiled.asm["metal"]
        if isinstance(msl, bytes):
            msl = msl.decode()
        assert "metal.fused_attention" in msl
        for head_offset in (
            "_fa_qhoff", "_fa_khoff", "_fa_vhoff", "_fa_ohoff"
        ):
            assert head_offset in msl

    actual = out.cpu()
    assert actual.isfinite().all()
    assert not torch.any(actual == sentinel)
    torch.testing.assert_close(
        actual,
        _gqa_reference(Q, K, V, num_q_heads, num_kv_heads),
        atol=1e-3,
        rtol=1e-3,
    )


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


def _load_alibi_attention():
    path = (Path(__file__).resolve().parents[3] / "leet-triton" /
            "medium-attention_with_linear_biases.py")
    if not path.is_file():
        pytest.skip(f"leet-triton fixture not present: {path}")
    spec = importlib.util.spec_from_file_location(
        "leet_triton_attention_with_linear_biases", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _alibi_reference(Q, K, V, M, N, d, alpha):
    Qc, Kc, Vc = Q.cpu(), K.cpu(), V.cpu()
    off_m = torch.arange(M, dtype=torch.float32)[:, None]
    off_n = torch.arange(N, dtype=torch.float32)[None, :]
    scores = (Qc @ Kc.T) * (d ** -0.5)
    scores = scores + alpha * (off_m - off_n)
    return torch.softmax(scores, dim=1) @ Vc


def _launch_alibi_attention(alibi, Q, K, V, out, M, N, d, alpha):
    K_t = K.T.contiguous()
    block_m = 16
    block_n = 16
    block_d = 64
    grid = (triton.cdiv(M, block_m), triton.cdiv(d, block_d))
    return alibi.alibi_attention_fwd[grid](
        Q, K_t, V, out,
        M, N, d, alpha, d ** -0.5,
        Q.stride(0), Q.stride(1),
        K_t.stride(0), K_t.stride(1),
        V.stride(0), V.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_N=block_n,
        BLOCK_SIZE_D=block_d,
    )


@pytest.mark.parametrize(
    "M, N, d, alpha",
    [
        (16, 16, 32, 0.0),
        (17, 19, 32, 0.125),
        (16, 16, 64, 0.125),
        (31, 33, 96, -0.0625),
        (18, 47, 96, 0.25),
    ],
)
def test_leet_alibi_attention_matches_reference(M, N, d, alpha):
    """The leet-triton ALiBi attention kernel compiles on Metal and is numeric."""
    torch.manual_seed(0xA11B1 + M * 17 + N * 3 + d)
    Q = torch.randn(M, d, dtype=torch.float32, device="mps").contiguous()
    K = torch.randn(N, d, dtype=torch.float32, device="mps").contiguous()
    V = torch.randn(N, d, dtype=torch.float32, device="mps").contiguous()
    out = torch.zeros(M, d, dtype=torch.float32, device="mps").contiguous()

    alibi = _load_alibi_attention()
    inspect_msl = (M, N, d, alpha) == (31, 33, 96, -0.0625)
    if inspect_msl:
        compiled = _launch_alibi_attention(alibi, Q, K, V, out, M, N, d, alpha)
    else:
        # Exercise the public leet-triton entry point, including its
        # `K.T.contiguous()` storage conversion and 2-D grid calculation.
        alibi.solve(Q, K, V, out, M, N, d, alpha)
    torch.mps.synchronize()

    if inspect_msl:
        msl = compiled.asm["metal"]
        if isinstance(msl, bytes):
            msl = msl.decode()
        assert "metal.fused_attention" in msl
        assert "feature-tiled QK full-dhead sweep" in msl
        assert msl.count("simdgroup_multiply_accumulate") >= 2

    expected = _alibi_reference(Q, K, V, M, N, d, alpha)
    torch.testing.assert_close(out.cpu(), expected, atol=1e-3, rtol=1e-3)


def test_flash_attention_launches_one_warp():
    """A fused-attention kernel must launch 32 threads, whatever num_warps says.

    The op's body runs entirely under `ltid.x < 32`, so a source kernel written
    with the customary `num_warps=4` would launch 128 threads and have 96 of
    them exit immediately — measured at ~11% on this backend, for nothing.
    The compiler therefore reports `threads_per_group` and the driver prefers
    it over `num_warps * 32`.

    Asserted on the MECHANISM rather than on a timing: a perf regression here
    is silent and a wall-clock assertion in a test suite is a flake generator.
    The numbers are in the commit that added it.
    """
    N, d_model, h = 128, 64, 4
    torch.manual_seed(0xF1)
    Q = torch.randn(N, d_model, dtype=torch.float32, device="mps").contiguous()
    K = torch.randn(N, d_model, dtype=torch.float32, device="mps").contiguous()
    V = torch.randn(N, d_model, dtype=torch.float32, device="mps").contiguous()
    out = torch.zeros(N, d_model, dtype=torch.float32, device="mps").contiguous()

    for num_warps in (1, 2, 4, 8):
        compiled = mha_kernel[(triton.cdiv(N, 32), h)](
            Q, K, V, out, N, d_model, h, 32, max(16, d_model // h),
            num_warps=num_warps)
        assert getattr(compiled.metadata, "threads_per_group", None) == 32, (
            f"num_warps={num_warps} did not report a single-warp launch")
        # The override must not change what the kernel computes.
        torch.mps.synchronize()
        torch.testing.assert_close(out.cpu(), _reference(Q, K, V, N, d_model, h),
                                   atol=1e-3, rtol=1e-3)


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
        compiled = _decoy_kernel[grid](Q, K, V, out, N, d_model, h, 32,
                                       max(16, d_model // h), variant,
                                       num_warps=4)
    except RuntimeError:
        # Rejection is still acceptable for a variant nothing implements, but
        # causal and ALiBi must be claimed by `metal.fused_attention` now.
        # Letting those variants take this branch would turn the tests into
        # compile-only guards and miss regressions in the emitted score region.
        assert variant == 2, (
            f"a {what} must be CLAIMED by metal.fused_attention now, not "
            "rejected — see flash_attention_reject_causal.mlir and the "
            "leet-triton ALiBi regression"
        )
        return

    if variant in (0, 1):
        msl = compiled.asm["metal"]
        if isinstance(msl, bytes):
            msl = msl.decode()
        assert "metal.fused_attention" in msl

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


def _load_softmax_attention_backward():
    path = (Path(__file__).resolve().parents[3] / "leet-triton" /
            "medium-softmax_attention_backward.py")
    if not path.is_file():
        pytest.skip(f"leet-triton fixture not present: {path}")
    spec = importlib.util.spec_from_file_location(
        "leet_triton_softmax_attention_backward", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "M, N, d, force_chunk_rows",
    [
        (16, 16, 16, None),
        (17, 19, 20, None),
        (32, 48, 32, 16),
        (64, 256, 128, None),
    ],
)
def test_softmax_attention_backward(M, N, d, force_chunk_rows):
    """The leet-triton backward kernel compiles and matches CPU autograd."""
    torch.manual_seed(0xBADC0DE + M + N + d)
    q_cpu = torch.randn(M, d, dtype=torch.float32, requires_grad=True)
    k_cpu = torch.randn(N, d, dtype=torch.float32, requires_grad=True)
    v_cpu = torch.randn(N, d, dtype=torch.float32, requires_grad=True)
    do_cpu = torch.randn(M, d, dtype=torch.float32)

    scale = d ** -0.5
    expected = (torch.softmax(q_cpu @ k_cpu.T * scale, dim=1) @ v_cpu)
    expected.backward(do_cpu)

    Q = q_cpu.detach().to("mps").contiguous()
    K = k_cpu.detach().to("mps").contiguous()
    V = v_cpu.detach().to("mps").contiguous()
    dO = do_cpu.to("mps").contiguous()
    dQ = torch.empty_like(Q)
    dK = torch.empty_like(K)
    dV = torch.empty_like(V)

    backward = _load_softmax_attention_backward()
    if force_chunk_rows is not None:
        backward._SCRATCH_ELEMS = force_chunk_rows * backward._pad16(N)
    backward.solve(Q, K, V, dO, dQ, dK, dV, M, N, d)
    torch.mps.synchronize()

    torch.testing.assert_close(dQ.cpu(), q_cpu.grad, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(dK.cpu(), k_cpu.grad, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(dV.cpu(), v_cpu.grad, atol=1e-5, rtol=1e-5)
