"""Universal tt.dot on the Metal backend.

Covers Phase 1 of `.omc/plans/metal-universal-matmul.md`:

- **AC2** (non-square multiples-of-8 f32): grid in M and/or N, K-tiled or single.
  Runtime M/N not a multiple of 8 also pass via the masked `tt.store` epilogue
  (`test_dot_f32_mn_partial_tile`).
- **AC3** (fp16 inputs, f32 accumulator): canonical-3-iter-arg path uses
  threadgroup-staged loads which implicit-cast `half`/`bfloat` -> `float`.
  bf16 single-dot passes via `rewriteSingleDot` SimdgroupLoadDeviceStaged staging
  (`test_dot_bf16_singletile`).
- **AC5** (K-loop tiled, K_TILES in {1,2,4,8}): the canonical-3-iter-arg
  matcher already covers this once the launcher arg-mask fix is in place.

Shape conventions:
- `BLOCK_M = BLOCK_N = BLOCK_K = 8` matches Apple's native simdgroup-matrix
  tile granularity (Principle 4 of the plan: tile granularity is fixed at
  hardware, kernel shape variability is encoded via the tile-grid).
- The kernel is a single canonical 3-iter-arg matmul; correctness of each
  axis (M-tile, N-tile, K-tile) is verified against `torch.matmul`.
"""
from __future__ import annotations

import os
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
def dot_universal_kernel(
    a_ptr, b_ptr, c_ptr,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    K_TILES: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, K_TILES):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc)


def _run(M, N, K, dtype_in=torch.float32, *, seed=0xC0FFEE):
    torch.manual_seed(seed)
    a = torch.randn((M, K), dtype=dtype_in).contiguous()
    b = torch.randn((K, N), dtype=dtype_in).contiguous()
    c = torch.zeros((M, N), dtype=torch.float32).contiguous()
    K_TILES = K // 8
    dot_universal_kernel[(triton.cdiv(M, 8), triton.cdiv(N, 8))](
        a, b, c,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=8, BLOCK_N=8, BLOCK_K=8, K_TILES=K_TILES,
    )
    ref = a.float() @ b.float()
    return c, ref


# ----------------------------------------------------------------------------
# AC2: non-square multiples-of-8 f32
# ----------------------------------------------------------------------------
@pytest.mark.parametrize(
    "M,N,K",
    [
        pytest.param(24, 8, 8, id="grid_m"),
        pytest.param(8, 24, 8, id="grid_n"),
        pytest.param(8, 8, 24, id="kloop_3"),
        pytest.param(16, 32, 48, id="grid_mn_kloop"),
    ],
)
def test_dot_f32_nonsquare(M, N, K):
    os.environ.setdefault(
        "TRITON_REPRODUCER_PATH", f"/tmp/dot-universal-{M}x{N}x{K}.mlir"
    )
    c, ref = _run(M, N, K, torch.float32)
    torch.testing.assert_close(c, ref, atol=1e-4, rtol=1e-4)


# AC2 masked-tail path: kernels whose static dot shape is still 8x8 but
# whose runtime (M, N) are not multiples of 8. Uses a `tl.store(c, acc,
# mask=(offs_m < M) & (offs_n < N))` so the compiler can extract M, N from
# the mask and route through the threadgroup-scratch + coop-loop predicated
# store. A/B sides are pre-padded to multiples of 8 so device loads stay
# in-bounds (the masked store discards the out-of-bounds rows/cols of acc).
@triton.jit
def dot_universal_masked_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    K_TILES: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, K_TILES):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask)


def _run_padded(M, N, K, *, seed=0xC0FFEE):
    """Allocate padded A/B/C so device loads of the tail tile stay in-bounds,
    then compare the unpadded slice against torch.matmul."""
    Mp = (M + 7) // 8 * 8
    Np = (N + 7) // 8 * 8
    torch.manual_seed(seed)
    a_pad = torch.zeros((Mp, K), dtype=torch.float32)
    b_pad = torch.zeros((K, Np), dtype=torch.float32)
    a_pad[:M, :] = torch.randn((M, K), dtype=torch.float32)
    b_pad[:, :N] = torch.randn((K, N), dtype=torch.float32)
    c_pad = torch.zeros((Mp, Np), dtype=torch.float32)
    K_TILES = K // 8
    dot_universal_masked_kernel[(Mp // 8, Np // 8)](
        a_pad, b_pad, c_pad,
        M, N,
        a_pad.stride(0), a_pad.stride(1),
        b_pad.stride(0), b_pad.stride(1),
        c_pad.stride(0), c_pad.stride(1),
        BLOCK_M=8, BLOCK_N=8, BLOCK_K=8, K_TILES=K_TILES,
    )
    ref = a_pad[:M, :].float() @ b_pad[:, :N].float()
    return c_pad[:M, :N].contiguous(), ref


@pytest.mark.parametrize(
    "M,N,K",
    [
        pytest.param(33,  8,  8, id="m_tail_only"),
        pytest.param( 8, 17,  8, id="n_tail_only"),
        pytest.param(33, 17, 16, id="both_tails_kloop_2"),
    ],
)
def test_dot_f32_mn_partial_tile(M, N, K):
    """AC2: runtime M/N not a multiple of 8. Masked tt.store triggers the
    threadgroup-scratch + coop-loop predicated emitter path."""
    os.environ.setdefault(
        "TRITON_REPRODUCER_PATH",
        f"/tmp/dot-partial-{M}x{N}x{K}.mlir",
    )
    c, ref = _run_padded(M, N, K)
    torch.testing.assert_close(c, ref, atol=1e-4, rtol=1e-4)


# ----------------------------------------------------------------------------
# AC3: fp16 / bf16 inputs with f32 accumulator
# ----------------------------------------------------------------------------
def test_dot_fp16():
    """fp16 inputs via the canonical-3-iter-arg path; threadgroup staging
    does an implicit `half -> float` cast inside the cooperative copy, so
    the simdgroup_float8x8 load from threadgroup is well-typed."""
    M = N = K = 32
    c, ref = _run(M, N, K, torch.float16)
    # K * 2^-10 = K * 9.77e-4
    tol = K * (2.0 ** -10)
    torch.testing.assert_close(c, ref, atol=tol, rtol=tol)


def test_dot_bf16_singletile():
    M = N = K = 8
    c, ref = _run(M, N, K, torch.bfloat16)
    tol = K * (2.0 ** -7)
    torch.testing.assert_close(c, ref, atol=tol, rtol=tol)


# Verbatim compute shape from
# leet-triton/medium-fp16_batched_matrix_multiplication.py.  Unlike the
# canonical universal kernel above, each dot operand is explicitly extended
# from fp16 to fp32 before Triton inserts the blocked -> dot-operand relayout.
@triton.jit
def batched_fp16_dot_kernel(
    a_ptr, b_ptr, c_ptr, BATCH, M, N, K, BLOCK_SIZE: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    a_ptr += pid_b * M * K
    b_ptr += pid_b * K * N
    c_ptr += pid_b * M * N

    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)

    for ks in range(0, K, BLOCK_SIZE):
        offs_k = ks + tl.arange(0, BLOCK_SIZE)
        offs_a = offs_m[:, None] * K + offs_k[None, :]
        mask_a = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a = tl.load(a_ptr + offs_a, mask=mask_a, other=0.0).to(tl.float32)
        offs_b = offs_k[:, None] * N + offs_n[None, :]
        mask_b = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        b = tl.load(b_ptr + offs_b, mask=mask_b, other=0.0).to(tl.float32)
        acc += tl.dot(a, b)

    offs_c = offs_m[:, None] * N + offs_n[None, :]
    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptr + offs_c, acc.to(tl.float16), mask=mask_c)


@pytest.mark.parametrize("batch,M,N,K", [(2, 64, 64, 64), (3, 70, 66, 50)])
def test_batched_fp16_dot_explicit_fp32_operands(batch, M, N, K):
    torch.manual_seed(0xC0FFEE)
    a = torch.randn((batch, M, K), dtype=torch.float16).contiguous()
    b = torch.randn((batch, K, N), dtype=torch.float16).contiguous()
    c = torch.zeros((batch, M, N), dtype=torch.float16).contiguous()
    grid = (batch, triton.cdiv(M, 64), triton.cdiv(N, 64))
    batched_fp16_dot_kernel[grid](a, b, c, batch, M, N, K, BLOCK_SIZE=64)
    ref = torch.bmm(a.float(), b.float()).half()
    tol = K * (2.0 ** -9)
    torch.testing.assert_close(c.float(), ref.float(), atol=tol, rtol=tol)


# Compute shape from leet-triton/medium-batched_matrix_multiplication.py.  The
# batch dimension remains in every tile, so tl.dot is rank-3 for both unit and
# multi-element batch tiles.
@triton.jit
def batched_f32_rank3_dot_kernel(
    a, b, c, BATCH, M, N, K,
    BLOCK_BATCH: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    COMBINE_OFFSETS: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_b = pid_b * BLOCK_BATCH + tl.arange(0, BLOCK_BATCH)

    mask_m = offs_m < M
    mask_n = offs_n < N
    mask_b = offs_b < BATCH

    offs_m64 = offs_m.to(tl.int64)
    offs_n64 = offs_n.to(tl.int64)
    offs_b64 = offs_b.to(tl.int64)
    M64 = tl.full((), M, tl.int64)
    N64 = tl.full((), N, tl.int64)
    K64 = tl.full((), K, tl.int64)

    acc = tl.zeros((BLOCK_BATCH, BLOCK_M, BLOCK_N), dtype=tl.float32)

    for i in range(0, K, BLOCK_K):
        offs_k = i + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        offs_k64 = offs_k.to(tl.int64)

        a_batch_offset = offs_b64[:, None, None] * (M64 * K64)
        a_row_offset = offs_m64[None, :, None] * K64
        a_k_offset = offs_k64[None, None, :]
        if COMBINE_OFFSETS:
            a_ptrs = a + (a_batch_offset + a_row_offset + a_k_offset)
        else:
            a_ptrs = a + a_batch_offset + a_row_offset + a_k_offset
        a_mask = (
            mask_b[:, None, None]
            & mask_m[None, :, None]
            & mask_k[None, None, :]
        )
        tile_a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        b_batch_offset = offs_b64[:, None, None] * (K64 * N64)
        b_k_offset = offs_k64[None, :, None] * N64
        b_col_offset = offs_n64[None, None, :]
        if COMBINE_OFFSETS:
            b_ptrs = b + (b_batch_offset + b_k_offset + b_col_offset)
        else:
            b_ptrs = b + b_batch_offset + b_k_offset + b_col_offset
        b_mask = (
            mask_b[:, None, None]
            & mask_k[None, :, None]
            & mask_n[None, None, :]
        )
        tile_b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc = tl.dot(tile_a, tile_b, acc=acc, input_precision="ieee")

    c_batch_offset = offs_b64[:, None, None] * (M64 * N64)
    c_row_offset = offs_m64[None, :, None] * N64
    c_col_offset = offs_n64[None, None, :]
    if COMBINE_OFFSETS:
        c_ptrs = c + (c_batch_offset + c_row_offset + c_col_offset)
    else:
        c_ptrs = c + c_batch_offset + c_row_offset + c_col_offset
    c_mask = (
        mask_b[:, None, None]
        & mask_m[None, :, None]
        & mask_n[None, None, :]
    )
    tl.store(c_ptrs, acc, mask=c_mask)


@pytest.mark.parametrize(
    "batch,M,N,K,block_batch,combine_offsets",
    [
        (2, 64, 64, 64, 1, False),
        (3, 70, 66, 50, 1, False),
        (2, 33, 17, 96, 1, False),
        (4, 64, 64, 64, 2, False),
        (3, 70, 66, 50, 2, False),
        (5, 33, 17, 96, 4, False),
        (3, 33, 17, 96, 2, False),
        (3, 33, 17, 96, 2, True),
    ],
)
def test_batched_f32_rank3_dot_batch_tile(
    batch, M, N, K, block_batch, combine_offsets,
):
    torch.manual_seed(0xC0FFEE)
    a = torch.randn((batch, M, K), dtype=torch.float32).contiguous()
    b = torch.randn((batch, K, N), dtype=torch.float32).contiguous()
    c = torch.zeros((batch, M, N), dtype=torch.float32).contiguous()
    grid = (
        triton.cdiv(M, 64),
        triton.cdiv(N, 64),
        triton.cdiv(batch, block_batch),
    )
    batched_f32_rank3_dot_kernel[grid](
        a, b, c, batch, M, N, K,
        BLOCK_BATCH=block_batch, BLOCK_M=64, BLOCK_N=64, BLOCK_K=64,
        COMBINE_OFFSETS=combine_offsets,
    )
    ref = torch.bmm(a, b)
    torch.testing.assert_close(c, ref, atol=1e-4, rtol=1e-4)


@triton.jit
def int4_weight_only_dot_kernel(
    x,
    wq,
    scales,
    y,
    M,
    N,
    K,
    group_size: tl.constexpr,
    stride_xm,
    stride_xk,
    stride_wqn,
    stride_wqk,
    stride_sn,
    stride_sk,
    stride_ym,
    stride_yn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        offs_x1k = k + tl.arange(0, BLOCK_SIZE_K // 2) * 2
        mask_x1 = (offs_m[:, None] < M) & (offs_x1k[None, :] < K)
        tile_x1 = tl.load(
            x + offs_m[:, None] * stride_xm + offs_x1k[None, :] * stride_xk,
            mask=mask_x1,
            other=0.0,
        ).to(tl.float32)

        offs_x2k = k + tl.arange(0, BLOCK_SIZE_K // 2) * 2 + 1
        mask_x2 = (offs_m[:, None] < M) & (offs_x2k[None, :] < K)
        tile_x2 = tl.load(
            x + offs_m[:, None] * stride_xm + offs_x2k[None, :] * stride_xk,
            mask=mask_x2,
            other=0.0,
        ).to(tl.float32)

        offs_sk = (k // group_size) + tl.arange(0, BLOCK_SIZE_K // group_size)
        mask_s = (offs_n[:, None] < N) & (offs_sk[None, :] < (K // group_size))
        tile_s = tl.load(
            scales + offs_n[:, None] * stride_sn + offs_sk[None, :] * stride_sk,
            mask=mask_s,
            other=0.0,
        ).to(tl.float32)
        tile_s = tl.broadcast_to(
            tile_s[:, :, None],
            (BLOCK_SIZE_N, BLOCK_SIZE_K // group_size, group_size // 2),
        )
        tile_s = tl.reshape(tile_s, (BLOCK_SIZE_N, BLOCK_SIZE_K // 2))

        offs_wk = (k // 2) + tl.arange(0, BLOCK_SIZE_K // 2)
        mask_w = (offs_n[:, None] < N) & (offs_wk[None, :] < (K // 2))
        tile_wq = tl.load(
            wq + offs_n[:, None] * stride_wqn + offs_wk[None, :] * stride_wqk,
            mask=mask_w,
            other=0x88,
        )
        tile_w1 = (((tile_wq & 0xF0) >> 4).to(tl.float32) - 8.0) * tile_s
        tile_w2 = ((tile_wq & 0x0F).to(tl.float32) - 8.0) * tile_s

        acc = tl.dot(tile_x1, tl.trans(tile_w1), acc=acc, input_precision="ieee")
        acc = tl.dot(tile_x2, tl.trans(tile_w2), acc=acc, input_precision="ieee")

    mask_y = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(
        y + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn,
        acc.to(tl.float16),
        mask=mask_y,
    )


@pytest.mark.parametrize(
    "M,N,K,group_size",
    [
        pytest.param(64, 64, 64, 16, id="single_tile_group16"),
        pytest.param(64, 64, 64, 32, id="single_tile_group32"),
        pytest.param(70, 66, 128, 64, id="mn_tail_kloop_group64"),
    ],
)
def test_int4_weight_only_dot_runs_computed_dequant_operand(
    M, N, K, group_size
):
    torch.manual_seed(0x1A4)
    x = torch.randn((M, K), dtype=torch.float16).contiguous()
    w_int = torch.randint(-8, 8, (N, K), dtype=torch.int16)
    hi = ((w_int[:, 0::2] + 8).to(torch.uint8) << 4)
    lo = (w_int[:, 1::2] + 8).to(torch.uint8)
    wq = (hi | lo).contiguous()
    scales = torch.rand((N, K // group_size), dtype=torch.float32).contiguous()
    y = torch.empty((M, N), dtype=torch.float16).contiguous()

    int4_weight_only_dot_kernel[
        (triton.cdiv(M, 64), triton.cdiv(N, 64))
    ](
        x,
        wq,
        scales,
        y,
        M,
        N,
        K,
        group_size,
        x.stride(0),
        x.stride(1),
        wq.stride(0),
        wq.stride(1),
        scales.stride(0),
        scales.stride(1),
        y.stride(0),
        y.stride(1),
        BLOCK_SIZE_M=64,
        BLOCK_SIZE_N=64,
        BLOCK_SIZE_K=max(32, group_size),
    )

    scale_expanded = scales.float().repeat_interleave(group_size, dim=1)
    w_dequant = w_int.float() * scale_expanded
    ref = (x.float() @ w_dequant.t()).half()
    tol = K * (2.0 ** -9)
    torch.testing.assert_close(y.float(), ref.float(), atol=tol, rtol=tol)


# ----------------------------------------------------------------------------
# AC5: K-loop tiled, K_TILES in {1, 2, 4, 8}
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("K_TILES", [1, 2, 4, 8])
def test_dot_kloop_tiled(K_TILES):
    """K-tiled accumulation with BLOCK_K=8, so K in {8, 16, 32, 64}. Verified
    by the existing 3-iter-arg matcher (plan-step path)."""
    M = N = 8
    K = 8 * K_TILES
    c, ref = _run(M, N, K, torch.float32, seed=0xC0FFEE + K_TILES)
    torch.testing.assert_close(c, ref, atol=1e-4, rtol=1e-4)


# ----------------------------------------------------------------------------
# AC4: multi-warp M/N tile-grid partition (Phase 2)
# ----------------------------------------------------------------------------
@triton.jit
def dot_multiwarp_kernel(
    a_ptr, b_ptr, c_ptr,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    K_TILES: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, K_TILES):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc)


def _run_multiwarp(num_warps, M=64, N=64, K=32, seed=0xAC4):
    torch.manual_seed(seed)
    a = torch.randn((M, K), dtype=torch.float32).contiguous()
    b = torch.randn((K, N), dtype=torch.float32).contiguous()
    c = torch.zeros((M, N), dtype=torch.float32).contiguous()
    dot_multiwarp_kernel[(1, 1)](
        a, b, c,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=M, BLOCK_N=N, BLOCK_K=8, K_TILES=K // 8,
        num_warps=num_warps,
    )
    ref = a.float() @ b.float()
    return c, ref


@pytest.mark.parametrize("num_warps", [1, 2, 4])
def test_dot_multiwarp(num_warps):
    """AC4: 64×64 dot with K_TILES=4, partitioned across `num_warps`
    SIMD-groups via `simdgroup_index_in_threadgroup`. factorWarps picks
    (warpsM, warpsN) maximizing min(warpsM, warpsN) under divisibility
    constraints: 1→(1,1), 2→(1,2)/(2,1), 4→(2,2)."""
    os.environ.setdefault(
        "TRITON_REPRODUCER_PATH",
        f"/tmp/dot-multiwarp-nw{num_warps}.mlir",
    )
    c, ref = _run_multiwarp(num_warps)
    torch.testing.assert_close(c, ref, atol=1e-5, rtol=1e-5)


def _capture_msl(num_warps):
    """Compile dot_multiwarp_kernel and return the emitted MSL source. The
    cache directory whose `dot_multiwarp_kernel.json` reports the matching
    `num_warps` is the only candidate, so this works even when multiple
    parametrize iterations co-exist in `~/.triton/cache`."""
    import json
    M, N, K = 64, 64, 32
    a = torch.zeros((M, K), dtype=torch.float32).contiguous()
    b = torch.zeros((K, N), dtype=torch.float32).contiguous()
    c = torch.zeros((M, N), dtype=torch.float32).contiguous()
    dot_multiwarp_kernel[(1, 1)](
        a, b, c,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=M, BLOCK_N=N, BLOCK_K=8, K_TILES=K // 8,
        num_warps=num_warps,
    )
    for root, _dirs, files in os.walk(os.path.expanduser("~/.triton/cache")):
        if "dot_multiwarp_kernel.json" not in files:
            continue
        if "dot_multiwarp_kernel.metal" not in files:
            continue
        with open(os.path.join(root, "dot_multiwarp_kernel.json")) as fh:
            if json.load(fh).get("num_warps") != num_warps:
                continue
        with open(os.path.join(root, "dot_multiwarp_kernel.metal")) as fh:
            return fh.read()
    return ""


@pytest.mark.parametrize("num_warps", [1, 2, 4])
def test_dot_multiwarp_msl_predicates(num_warps):
    """AC4-S5: the emitted MSL contains exactly
    `(8/warpsM) * (8/warpsN) * K_TILES = 256/num_warps` occurrences of
    `simdgroup_multiply_accumulate(`. For num_warps>1 the MSL also has
    `simdgroup_index_in_threadgroup` (the kernel parameter attribute) and
    per-warp slice indexing via `_stage_shared[sgid]`."""
    msl = _capture_msl(num_warps)
    assert msl, "no MSL captured (cache miss)"
    expected_mma = 256 // num_warps
    mma_count = msl.count("simdgroup_multiply_accumulate(")
    assert mma_count == expected_mma, (
        f"num_warps={num_warps}: mma count {mma_count} != expected {expected_mma}"
    )
    if num_warps > 1:
        assert "simdgroup_index_in_threadgroup" in msl, (
            "multi-warp MSL missing simdgroup_index_in_threadgroup parameter")
        assert "_stage_shared[sgid]" in msl, (
            "multi-warp MSL missing per-warp stage buffer slice _stage_shared[sgid]")
    else:
        assert "_stage_shared[" in msl
        # Single-warp Branch A: shared buffer is [elems], not [num_warps][elems].
        assert "_stage_shared[sgid]" not in msl, (
            "single-warp MSL should not slice _stage_shared by sgid")
