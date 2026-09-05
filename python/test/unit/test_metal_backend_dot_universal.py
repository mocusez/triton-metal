"""Universal tt.dot on the Metal backend.

Covers Phase 1 of the implementation notes:

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
    a_ptr,
    b_ptr,
    c_ptr,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
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


@triton.jit
def dot_scaled_e4m3_u8_kernel(
    a_ptr,
    b_ptr,
    a_scale_ptr,
    b_scale_ptr,
    c_ptr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a = tl.load(a_ptr + offs_m[:, None] * BLOCK_K + offs_k[None, :])
    b = tl.load(b_ptr + offs_k[:, None] * BLOCK_N + offs_n[None, :])
    a_scale = tl.load(a_scale_ptr + offs_m[:, None])
    b_scale = tl.load(b_scale_ptr + offs_n[:, None])
    result = tl.dot_scaled(
        a,
        a_scale,
        "e4m3",
        b,
        b_scale,
        "e4m3",
        fast_math=False,
    )
    tl.store(c_ptr + offs_m[:, None] * BLOCK_N + offs_n[None, :], result)


@triton.jit
def dot_scaled_e4m3_masked_u8_kernel(
    a_ptr,
    b_ptr,
    a_scale_ptr,
    b_scale_ptr,
    c_ptr,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    OTHER: tl.constexpr,
):
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
    b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
    a = tl.load(
        a_ptr + offs_m[:, None] * K + offs_k[None, :],
        mask=a_mask,
        other=OTHER,
    )
    b = tl.load(
        b_ptr + offs_k[:, None] * N + offs_n[None, :],
        mask=b_mask,
        other=OTHER,
    )
    a_scale = tl.load(a_scale_ptr + offs_m[:, None])
    b_scale = tl.load(b_scale_ptr + offs_n[:, None])
    result = tl.dot_scaled(
        a,
        a_scale,
        "e4m3",
        b,
        b_scale,
        "e4m3",
        fast_math=False,
    )
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(
        c_ptr + offs_m[:, None] * BLOCK_N + offs_n[None, :],
        result,
        mask=c_mask,
    )


@triton.jit
def dot_scaled_e4m3_batched_u8_kernel(
    a_ptr,
    b_ptr,
    a_scale_ptr,
    b_scale_ptr,
    c_ptr,
    BATCH: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    PER_BATCH_SCALES: tl.constexpr,
    A_SCALE_BATCH_STRIDE: tl.constexpr,
    B_SCALE_BATCH_STRIDE: tl.constexpr,
):
    offs_b = tl.arange(0, BATCH)
    offs_m = tl.arange(0, M)
    offs_n = tl.arange(0, N)
    offs_k = tl.arange(0, K)
    offs_scale_k = tl.arange(0, K // 32)
    a = tl.load(a_ptr + offs_b[:, None, None] * M * K + offs_m[None, :, None] * K + offs_k[None, None, :])
    b = tl.load(b_ptr + offs_b[:, None, None] * K * N + offs_k[None, :, None] * N + offs_n[None, None, :])
    if PER_BATCH_SCALES:
        a_scale = tl.load(
            a_scale_ptr
            + offs_b[:, None, None] * A_SCALE_BATCH_STRIDE
            + offs_m[None, :, None] * (K // 32)
            + offs_scale_k[None, None, :]
        )
        b_scale = tl.load(
            b_scale_ptr
            + offs_b[:, None, None] * B_SCALE_BATCH_STRIDE
            + offs_n[None, :, None] * (K // 32)
            + offs_scale_k[None, None, :]
        )
    else:
        a_scale = tl.load(
            a_scale_ptr + offs_b[:, None, None] * 0 + offs_m[None, :, None] * (K // 32) + offs_scale_k[None, None, :]
        )
        b_scale = tl.load(
            b_scale_ptr + offs_b[:, None, None] * 0 + offs_n[None, :, None] * (K // 32) + offs_scale_k[None, None, :]
        )
    result = tl.dot_scaled(
        a,
        a_scale,
        "e4m3",
        b,
        b_scale,
        "e4m3",
        fast_math=False,
    )
    tl.store(
        c_ptr + offs_b[:, None, None] * M * N + offs_m[None, :, None] * N + offs_n[None, None, :],
        result,
    )


@triton.jit
def dot_scaled_e4m3_batched_dynamic_scale_stride_u8_kernel(
    a_ptr,
    b_ptr,
    a_scale_ptr,
    b_scale_ptr,
    c_ptr,
    a_scale_batch_stride,
    b_scale_batch_stride,
    BATCH: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    SCALE_BATCH_ORIGIN: tl.constexpr,
):
    offs_b = tl.arange(0, BATCH)
    offs_m = tl.arange(0, M)
    offs_n = tl.arange(0, N)
    offs_k = tl.arange(0, K)
    offs_scale_k = tl.arange(0, K // 32)
    a = tl.load(a_ptr + offs_b[:, None, None] * M * K + offs_m[None, :, None] * K + offs_k[None, None, :])
    b = tl.load(b_ptr + offs_b[:, None, None] * K * N + offs_k[None, :, None] * N + offs_n[None, None, :])
    scale_batch = offs_b + SCALE_BATCH_ORIGIN
    a_scale = tl.load(
        a_scale_ptr
        + scale_batch[:, None, None] * a_scale_batch_stride
        + offs_m[None, :, None] * (K // 32)
        + offs_scale_k[None, None, :]
    )
    b_scale = tl.load(
        b_scale_ptr
        + scale_batch[:, None, None] * b_scale_batch_stride
        + offs_n[None, :, None] * (K // 32)
        + offs_scale_k[None, None, :]
    )
    result = tl.dot_scaled(
        a,
        a_scale,
        "e4m3",
        b,
        b_scale,
        "e4m3",
        fast_math=False,
    )
    tl.store(
        c_ptr + offs_b[:, None, None] * M * N + offs_m[None, :, None] * N + offs_n[None, None, :],
        result,
    )


@triton.jit
def dot_scaled_e4m3_batched_masked_u8_kernel(
    a_ptr,
    b_ptr,
    a_scale_ptr,
    b_scale_ptr,
    c_ptr,
    actual_batch,
    M,
    N,
    K,
    BLOCK_BATCH: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    MASK_BATCH: tl.constexpr,
    BATCH_BOUND_BIAS: tl.constexpr,
):
    offs_b = tl.arange(0, BLOCK_BATCH)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_mask = (offs_m[None, :, None] < M) & (offs_k[None, None, :] < K)
    b_mask = (offs_k[None, :, None] < K) & (offs_n[None, None, :] < N)
    c_mask = (offs_m[None, :, None] < M) & (offs_n[None, None, :] < N)
    if MASK_BATCH:
        batch_mask = offs_b[:, None, None] < actual_batch
        b_batch_mask = batch_mask
        if BATCH_BOUND_BIAS:
            b_batch_mask = offs_b[:, None, None] < actual_batch + BATCH_BOUND_BIAS
        a_mask &= batch_mask
        b_mask &= b_batch_mask
        c_mask &= batch_mask
    a = tl.load(
        a_ptr + offs_b[:, None, None] * BLOCK_M * BLOCK_K + offs_m[None, :, None] * BLOCK_K + offs_k[None, None, :],
        mask=a_mask,
        other=0,
    )
    b = tl.load(
        b_ptr + offs_b[:, None, None] * BLOCK_K * BLOCK_N + offs_k[None, :, None] * BLOCK_N + offs_n[None, None, :],
        mask=b_mask,
        other=0,
    )
    a_scale = tl.load(a_scale_ptr + offs_b[:, None, None] * 0 + offs_m[None, :, None] * (BLOCK_K // 32))
    b_scale = tl.load(b_scale_ptr + offs_b[:, None, None] * 0 + offs_n[None, :, None] * (BLOCK_K // 32))
    result = tl.dot_scaled(
        a,
        a_scale,
        "e4m3",
        b,
        b_scale,
        "e4m3",
        fast_math=False,
    )
    tl.store(
        c_ptr + offs_b[:, None, None] * BLOCK_M * BLOCK_N + offs_m[None, :, None] * BLOCK_N + offs_n[None, None, :],
        result,
        mask=c_mask,
    )


@triton.jit
def dot_scaled_loop_kernel(
    a_ptr,
    b_ptr,
    a_scale_ptr,
    b_scale_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SCALE_FACTOR: tl.constexpr,
    LOOP_START: tl.constexpr,
    A_ELEM_TYPE: tl.constexpr,
    B_ELEM_TYPE: tl.constexpr,
):
    offs_m = tl.arange(0, M)
    offs_n = tl.arange(0, N)
    offs_k = tl.arange(0, BLOCK_K)
    offs_scale_k = tl.arange(0, BLOCK_K // SCALE_FACTOR)
    acc = tl.zeros((M, N), dtype=tl.float32)
    for k_start in range(LOOP_START, K, BLOCK_K):
        a = tl.load(a_ptr + offs_m[:, None] * K + (k_start + offs_k[None, :]))
        b = tl.load(b_ptr + (k_start + offs_k[:, None]) * N + offs_n[None, :])
        a_scale = tl.load(
            a_scale_ptr + offs_m[:, None] * (K // SCALE_FACTOR) + k_start // SCALE_FACTOR + offs_scale_k[None, :]
        )
        b_scale = tl.load(
            b_scale_ptr + offs_n[:, None] * (K // SCALE_FACTOR) + k_start // SCALE_FACTOR + offs_scale_k[None, :]
        )
        acc = tl.dot_scaled(
            a,
            a_scale,
            A_ELEM_TYPE,
            b,
            b_scale,
            B_ELEM_TYPE,
            acc,
            fast_math=False,
        )
    tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc)


@triton.jit
def dot_scaled_masked_loop_kernel(
    a_ptr,
    b_ptr,
    a_scale_ptr,
    b_scale_ptr,
    c_ptr,
    actual_m,
    actual_n,
    actual_k,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SCALE_FACTOR: tl.constexpr,
    LOOP_START: tl.constexpr,
    A_ELEM_TYPE: tl.constexpr,
    B_ELEM_TYPE: tl.constexpr,
    LOAD_OTHER: tl.constexpr,
    A_K_BOUND_BIAS: tl.constexpr,
):
    offs_m = tl.arange(0, M)
    offs_n = tl.arange(0, N)
    offs_k = tl.arange(0, BLOCK_K)
    offs_scale_k = tl.arange(0, BLOCK_K // SCALE_FACTOR)
    acc = tl.zeros((M, N), dtype=tl.float32)
    for k_start in range(LOOP_START, K, BLOCK_K):
        logical_k = k_start + offs_k
        a_mask = (offs_m[:, None] < actual_m) & (logical_k[None, :] < actual_k + A_K_BOUND_BIAS)
        b_mask = (logical_k[:, None] < actual_k) & (offs_n[None, :] < actual_n)
        a = tl.load(
            a_ptr + offs_m[:, None] * K + logical_k[None, :],
            mask=a_mask,
            other=LOAD_OTHER,
        )
        b = tl.load(
            b_ptr + logical_k[:, None] * N + offs_n[None, :],
            mask=b_mask,
            other=LOAD_OTHER,
        )
        a_scale = tl.load(
            a_scale_ptr + offs_m[:, None] * (K // SCALE_FACTOR) + k_start // SCALE_FACTOR + offs_scale_k[None, :]
        )
        b_scale = tl.load(
            b_scale_ptr + offs_n[:, None] * (K // SCALE_FACTOR) + k_start // SCALE_FACTOR + offs_scale_k[None, :]
        )
        acc = tl.dot_scaled(
            a,
            a_scale,
            A_ELEM_TYPE,
            b,
            b_scale,
            B_ELEM_TYPE,
            acc,
            fast_math=False,
        )
    output_mask = (offs_m[:, None] < actual_m) & (offs_n[None, :] < actual_n)
    tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc, mask=output_mask)


@triton.jit
def dot_scaled_dynamic_upper_masked_loop_kernel(
    a_ptr,
    b_ptr,
    a_scale_ptr,
    b_scale_ptr,
    c_ptr,
    actual_m,
    actual_n,
    actual_k,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SCALE_FACTOR: tl.constexpr,
    A_ELEM_TYPE: tl.constexpr,
    B_ELEM_TYPE: tl.constexpr,
    LOOP_UPPER_BIAS: tl.constexpr,
    A_SCALE_ROW_STRIDE_BIAS: tl.constexpr,
):
    offs_m = tl.arange(0, M)
    offs_n = tl.arange(0, N)
    offs_k = tl.arange(0, BLOCK_K)
    offs_scale_k = tl.arange(0, BLOCK_K // SCALE_FACTOR)
    loop_k = tl.minimum(tl.maximum(actual_k + LOOP_UPPER_BIAS, 0), K)
    acc = tl.zeros((M, N), dtype=tl.float32)
    for k_start in range(0, loop_k, BLOCK_K):
        logical_k = k_start + offs_k
        a_mask = (offs_m[:, None] < actual_m) & (logical_k[None, :] < actual_k)
        b_mask = (logical_k[:, None] < actual_k) & (offs_n[None, :] < actual_n)
        a = tl.load(
            a_ptr + offs_m[:, None] * K + logical_k[None, :],
            mask=a_mask,
            other=0,
        )
        b = tl.load(
            b_ptr + logical_k[:, None] * N + offs_n[None, :],
            mask=b_mask,
            other=0,
        )
        a_scale = tl.load(
            a_scale_ptr
            + offs_m[:, None] * (K // SCALE_FACTOR + A_SCALE_ROW_STRIDE_BIAS)
            + k_start // SCALE_FACTOR
            + offs_scale_k[None, :]
        )
        b_scale = tl.load(
            b_scale_ptr + offs_n[:, None] * (K // SCALE_FACTOR) + k_start // SCALE_FACTOR + offs_scale_k[None, :]
        )
        acc = tl.dot_scaled(
            a,
            a_scale,
            A_ELEM_TYPE,
            b,
            b_scale,
            B_ELEM_TYPE,
            acc,
            fast_math=False,
        )
    output_mask = (offs_m[:, None] < actual_m) & (offs_n[None, :] < actual_n)
    tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc, mask=output_mask)


@triton.jit
def dot_scaled_dynamic_upper_unmasked_loop_kernel(
    a_ptr,
    b_ptr,
    a_scale_ptr,
    b_scale_ptr,
    c_ptr,
    actual_k,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SCALE_FACTOR: tl.constexpr,
):
    offs_m = tl.arange(0, M)
    offs_n = tl.arange(0, N)
    offs_k = tl.arange(0, BLOCK_K)
    offs_scale_k = tl.arange(0, BLOCK_K // SCALE_FACTOR)
    loop_k = tl.minimum(tl.maximum(actual_k, 0), K)
    acc = tl.zeros((M, N), dtype=tl.float32)
    for k_start in range(0, loop_k, BLOCK_K):
        a = tl.load(a_ptr + offs_m[:, None] * K + (k_start + offs_k[None, :]))
        b = tl.load(b_ptr + (k_start + offs_k[:, None]) * N + offs_n[None, :])
        a_scale = tl.load(
            a_scale_ptr + offs_m[:, None] * (K // SCALE_FACTOR) + k_start // SCALE_FACTOR + offs_scale_k[None, :]
        )
        b_scale = tl.load(
            b_scale_ptr + offs_n[:, None] * (K // SCALE_FACTOR) + k_start // SCALE_FACTOR + offs_scale_k[None, :]
        )
        acc = tl.dot_scaled(
            a,
            a_scale,
            "fp16",
            b,
            b_scale,
            "fp16",
            acc,
            fast_math=False,
        )
    tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc)


@triton.jit
def dot_scaled_loop_e2m1_u8_kernel(
    a_ptr,
    b_ptr,
    a_scale_ptr,
    b_scale_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    LOOP_START: tl.constexpr,
    LHS_K_PACK: tl.constexpr,
    RHS_K_PACK: tl.constexpr,
    PACKED_K_DIVISOR: tl.constexpr,
):
    offs_m = tl.arange(0, M)
    offs_n = tl.arange(0, N)
    offs_k = tl.arange(0, BLOCK_K)
    offs_scale_k = tl.arange(0, BLOCK_K // 32)
    acc = tl.zeros((M, N), dtype=tl.float32)
    for k_start in range(LOOP_START, K, BLOCK_K):
        if LHS_K_PACK:
            lhs_packed_k: tl.constexpr = BLOCK_K // 2
            offs_lhs_packed_k = tl.arange(0, lhs_packed_k)
            a = tl.load(a_ptr + offs_m[:, None] * (K // 2) + (k_start // PACKED_K_DIVISOR + offs_lhs_packed_k[None, :]))
        else:
            packed_m: tl.constexpr = M // 2
            offs_packed_m = tl.arange(0, packed_m)
            a = tl.load(a_ptr + offs_packed_m[:, None] * K + (k_start + offs_k[None, :]))
        if RHS_K_PACK:
            rhs_packed_k: tl.constexpr = BLOCK_K // 2
            offs_rhs_packed_k = tl.arange(0, rhs_packed_k)
            b = tl.load(b_ptr + (k_start // PACKED_K_DIVISOR + offs_rhs_packed_k[:, None]) * N + offs_n[None, :])
        else:
            packed_n: tl.constexpr = N // 2
            offs_packed_n = tl.arange(0, packed_n)
            b = tl.load(b_ptr + (k_start + offs_k[:, None]) * packed_n + offs_packed_n[None, :])
        a_scale = tl.load(a_scale_ptr + offs_m[:, None] * (K // 32) + k_start // 32 + offs_scale_k[None, :])
        b_scale = tl.load(b_scale_ptr + offs_n[:, None] * (K // 32) + k_start // 32 + offs_scale_k[None, :])
        acc = tl.dot_scaled(
            a,
            a_scale,
            "e2m1",
            b,
            b_scale,
            "e2m1",
            acc,
            fast_math=False,
            lhs_k_pack=LHS_K_PACK,
            rhs_k_pack=RHS_K_PACK,
        )
    tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc)


@triton.jit
def dot_scaled_e2m1_u8_kernel(
    a_ptr,
    b_ptr,
    a_scale_ptr,
    b_scale_ptr,
    c_ptr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    LHS_K_PACK: tl.constexpr,
    RHS_K_PACK: tl.constexpr,
):
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    if LHS_K_PACK:
        LHS_PACKED_K: tl.constexpr = BLOCK_K // 2
        offs_lhs_packed_k = tl.arange(0, LHS_PACKED_K)
        a = tl.load(a_ptr + offs_m[:, None] * LHS_PACKED_K + offs_lhs_packed_k[None, :])
    else:
        PACKED_M: tl.constexpr = BLOCK_M // 2
        offs_packed_m = tl.arange(0, PACKED_M)
        a = tl.load(a_ptr + offs_packed_m[:, None] * BLOCK_K + offs_k[None, :])
    if RHS_K_PACK:
        RHS_PACKED_K: tl.constexpr = BLOCK_K // 2
        offs_rhs_packed_k = tl.arange(0, RHS_PACKED_K)
        b = tl.load(b_ptr + offs_rhs_packed_k[:, None] * BLOCK_N + offs_n[None, :])
    else:
        PACKED_N: tl.constexpr = BLOCK_N // 2
        offs_packed_n = tl.arange(0, PACKED_N)
        b = tl.load(b_ptr + offs_k[:, None] * PACKED_N + offs_packed_n[None, :])
    SCALE_K: tl.constexpr = BLOCK_K // 32
    offs_scale_k = tl.arange(0, SCALE_K)
    a_scale = tl.load(a_scale_ptr + offs_m[:, None] * SCALE_K + offs_scale_k[None, :])
    b_scale = tl.load(b_scale_ptr + offs_n[:, None] * SCALE_K + offs_scale_k[None, :])
    result = tl.dot_scaled(
        a,
        a_scale,
        "e2m1",
        b,
        b_scale,
        "e2m1",
        fast_math=False,
        lhs_k_pack=LHS_K_PACK,
        rhs_k_pack=RHS_K_PACK,
    )
    tl.store(c_ptr + offs_m[:, None] * BLOCK_N + offs_n[None, :], result)


@triton.jit
def dot_scaled_e5m2_u8_kernel(
    a_ptr,
    b_ptr,
    a_scale_ptr,
    b_scale_ptr,
    c_ptr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a = tl.load(a_ptr + offs_m[:, None] * BLOCK_K + offs_k[None, :])
    b = tl.load(b_ptr + offs_k[:, None] * BLOCK_N + offs_n[None, :])
    a_scale = tl.load(a_scale_ptr + offs_m[:, None])
    b_scale = tl.load(b_scale_ptr + offs_n[:, None])
    result = tl.dot_scaled(
        a,
        a_scale,
        "e5m2",
        b,
        b_scale,
        "e5m2",
        fast_math=False,
    )
    tl.store(c_ptr + offs_m[:, None] * BLOCK_N + offs_n[None, :], result)


def _run_dot_scaled_e4m3_u8(a, *, a_scale_raw=127):
    b = torch.zeros((32, 16), dtype=torch.uint8)
    diagonal = torch.arange(16)
    b[diagonal, diagonal] = 0x38
    a_scale = torch.full((16, 1), a_scale_raw, dtype=torch.uint8)
    b_scale = torch.full((16, 1), 127, dtype=torch.uint8)
    output = torch.empty((16, 16), dtype=torch.float32, device="mps")

    dot_scaled_e4m3_u8_kernel[(1,)](
        a.to("mps"),
        b.to("mps"),
        a_scale.to("mps"),
        b_scale.to("mps"),
        output,
        BLOCK_M=16,
        BLOCK_N=16,
        BLOCK_K=32,
        num_warps=4,
    )
    torch.mps.synchronize()
    return output.cpu()


def test_dot_scaled_e4m3_masked_ragged_mnk():
    M, N, K = 13, 11, 27
    block_m, block_n, block_k = 16, 16, 32
    one = 0x38
    a = torch.full((M, K), one, dtype=torch.uint8, device="mps")
    b = torch.full((K, N), one, dtype=torch.uint8, device="mps")
    a_scale = torch.full((block_m, 1), 127, dtype=torch.uint8, device="mps")
    b_scale = torch.full((block_n, 1), 127, dtype=torch.uint8, device="mps")
    sentinel = -4321.0
    output = torch.full((block_m, block_n), sentinel, dtype=torch.float32, device="mps")

    dot_scaled_e4m3_masked_u8_kernel[(1,)](
        a,
        b,
        a_scale,
        b_scale,
        output,
        M,
        N,
        K,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        OTHER=0,
        num_warps=4,
    )
    torch.mps.synchronize()
    actual = output.cpu()
    assert torch.equal(actual[:M, :N], torch.full((M, N), float(K)))
    assert torch.all(actual[M:, :] == sentinel)
    assert torch.all(actual[:, N:] == sentinel)


def test_dot_scaled_masked_nonzero_other_is_named(capfd):
    M = N = 8
    K = 16
    a = torch.zeros((M, K), dtype=torch.uint8, device="mps")
    b = torch.zeros((K, N), dtype=torch.uint8, device="mps")
    a_scale = torch.full((16, 1), 127, dtype=torch.uint8, device="mps")
    b_scale = torch.full((16, 1), 127, dtype=torch.uint8, device="mps")
    output = torch.empty((16, 16), dtype=torch.float32, device="mps")

    with pytest.raises(Exception):
        dot_scaled_e4m3_masked_u8_kernel[(1,)](
            a,
            b,
            a_scale,
            b_scale,
            output,
            M,
            N,
            K,
            BLOCK_M=16,
            BLOCK_N=16,
            BLOCK_K=32,
            OTHER=0x38,
            num_warps=4,
        )
    assert "zero-filled rectangular A/B loads" in capfd.readouterr().err


def test_dot_scaled_e4m3_batched_shared_scales():
    batch, M, N, K = 2, 16, 16, 64
    one = 0x38
    two = 0x40
    a = torch.stack(
        [
            torch.full((M, K), one, dtype=torch.uint8),
            torch.full((M, K), two, dtype=torch.uint8),
        ]
    ).to("mps")
    b = torch.stack(
        [
            torch.full((K, N), one, dtype=torch.uint8),
            torch.full((K, N), two, dtype=torch.uint8),
        ]
    ).to("mps")
    a_scale = torch.full((M, K // 32), 127, dtype=torch.uint8, device="mps")
    b_scale = torch.full((N, K // 32), 127, dtype=torch.uint8, device="mps")
    output = torch.empty((batch, M, N), dtype=torch.float32, device="mps")

    dot_scaled_e4m3_batched_u8_kernel[(1,)](
        a,
        b,
        a_scale,
        b_scale,
        output,
        batch,
        M,
        N,
        K,
        PER_BATCH_SCALES=False,
        A_SCALE_BATCH_STRIDE=0,
        B_SCALE_BATCH_STRIDE=0,
        num_warps=4,
    )
    torch.mps.synchronize()
    expected = torch.stack([torch.full((M, N), float(K)), torch.full((M, N), float(4 * K))])
    assert torch.equal(output.cpu(), expected)


def test_dot_scaled_e4m3_batched_per_batch_scales():
    batch, M, N, K = 2, 16, 16, 64
    one = 0x38
    a = torch.full((batch, M, K), one, dtype=torch.uint8, device="mps")
    b = torch.full((batch, K, N), one, dtype=torch.uint8, device="mps")
    a_scale = torch.full((batch, M, K // 32), 127, dtype=torch.uint8, device="mps")
    b_scale = torch.full((batch, N, K // 32), 127, dtype=torch.uint8, device="mps")
    a_scale[1] = 128
    b_scale[1] = 129
    output = torch.empty((batch, M, N), dtype=torch.float32, device="mps")

    dot_scaled_e4m3_batched_u8_kernel[(1,)](
        a,
        b,
        a_scale,
        b_scale,
        output,
        batch,
        M,
        N,
        K,
        PER_BATCH_SCALES=True,
        A_SCALE_BATCH_STRIDE=M * (K // 32),
        B_SCALE_BATCH_STRIDE=N * (K // 32),
        num_warps=4,
    )
    torch.mps.synchronize()
    expected = torch.stack([torch.full((M, N), float(K)), torch.full((M, N), float(8 * K))])
    assert torch.equal(output.cpu(), expected)


def test_dot_scaled_batched_noncontiguous_scale_batch_stride_reads_padded_scales():
    batch, M, N, K = 2, 16, 16, 64
    one = 0x38
    a = torch.full((batch, M, K), one, dtype=torch.uint8, device="mps")
    b = torch.full((batch, K, N), one, dtype=torch.uint8, device="mps")
    a_scale = torch.full((batch * (M + 1), K // 32), 255, dtype=torch.uint8, device="mps")
    b_scale = torch.full((batch * (N + 1), K // 32), 255, dtype=torch.uint8, device="mps")
    a_scale[:M] = 127
    a_scale[M + 1 : M + 1 + M] = 128
    b_scale[:N] = 127
    b_scale[N + 1 : N + 1 + N] = 129
    output = torch.empty((batch, M, N), dtype=torch.float32, device="mps")

    dot_scaled_e4m3_batched_u8_kernel[(1,)](
        a,
        b,
        a_scale,
        b_scale,
        output,
        batch,
        M,
        N,
        K,
        PER_BATCH_SCALES=True,
        A_SCALE_BATCH_STRIDE=(M + 1) * (K // 32),
        B_SCALE_BATCH_STRIDE=(N + 1) * (K // 32),
        num_warps=4,
    )
    torch.mps.synchronize()
    expected = torch.stack([torch.full((M, N), float(K)), torch.full((M, N), float(8 * K))])
    assert torch.equal(output.cpu(), expected)


def test_dot_scaled_batched_dynamic_scale_batch_stride_reads_padded_scales():
    batch, M, N, K = 2, 16, 16, 64
    one = 0x38
    a = torch.full((batch, M, K), one, dtype=torch.uint8, device="mps")
    b = torch.full((batch, K, N), one, dtype=torch.uint8, device="mps")
    a_scale = torch.full((batch * (M + 1), K // 32), 255, dtype=torch.uint8, device="mps")
    b_scale = torch.full((batch * (N + 1), K // 32), 255, dtype=torch.uint8, device="mps")
    a_scale[:M] = 127
    a_scale[M + 1 : M + 1 + M] = 128
    b_scale[:N] = 127
    b_scale[N + 1 : N + 1 + N] = 129
    output = torch.empty((batch, M, N), dtype=torch.float32, device="mps")

    dot_scaled_e4m3_batched_dynamic_scale_stride_u8_kernel[(1,)](
        a,
        b,
        a_scale,
        b_scale,
        output,
        (M + 1) * (K // 32),
        (N + 1) * (K // 32),
        batch,
        M,
        N,
        K,
        SCALE_BATCH_ORIGIN=0,
        num_warps=4,
    )
    torch.mps.synchronize()
    expected = torch.stack([torch.full((M, N), float(K)), torch.full((M, N), float(8 * K))])
    assert torch.equal(output.cpu(), expected)


def test_dot_scaled_batched_dynamic_scale_batch_origin_is_named(capfd):
    batch, M, N, K = 2, 16, 16, 64
    a = torch.full((batch, M, K), 0x38, dtype=torch.uint8, device="mps")
    b = torch.full((batch, K, N), 0x38, dtype=torch.uint8, device="mps")
    a_scale = torch.full(((batch + 1) * (M + 1), K // 32), 127, dtype=torch.uint8, device="mps")
    b_scale = torch.full(((batch + 1) * (N + 1), K // 32), 127, dtype=torch.uint8, device="mps")
    output = torch.empty((batch, M, N), dtype=torch.float32, device="mps")

    with pytest.raises(Exception):
        dot_scaled_e4m3_batched_dynamic_scale_stride_u8_kernel[(1,)](
            a,
            b,
            a_scale,
            b_scale,
            output,
            (M + 1) * (K // 32),
            (N + 1) * (K // 32),
            batch,
            M,
            N,
            K,
            SCALE_BATCH_ORIGIN=1,
            num_warps=4,
        )
    assert "canonical zero-based per-batch scale stride" in capfd.readouterr().err


def test_dot_scaled_e4m3_batched_masked_ragged_mnk():
    block_batch, block_m, block_n, block_k = 2, 16, 16, 32
    M, N, K = 13, 11, 27
    one = 0x38
    a = torch.full((block_batch, block_m, block_k), one, dtype=torch.uint8, device="mps")
    b = torch.full((block_batch, block_k, block_n), one, dtype=torch.uint8, device="mps")
    a_scale = torch.full((block_m, 1), 127, dtype=torch.uint8, device="mps")
    b_scale = torch.full((block_n, 1), 127, dtype=torch.uint8, device="mps")
    sentinel = -4321.0
    output = torch.full(
        (block_batch, block_m, block_n),
        sentinel,
        dtype=torch.float32,
        device="mps",
    )

    dot_scaled_e4m3_batched_masked_u8_kernel[(1,)](
        a,
        b,
        a_scale,
        b_scale,
        output,
        block_batch,
        M,
        N,
        K,
        BLOCK_BATCH=block_batch,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        MASK_BATCH=False,
        BATCH_BOUND_BIAS=0,
        num_warps=4,
    )
    torch.mps.synchronize()
    actual = output.cpu()
    assert torch.equal(actual[:, :M, :N], torch.full((block_batch, M, N), float(K)))
    assert torch.all(actual[:, M:, :] == sentinel)
    assert torch.all(actual[:, :, N:] == sentinel)


def test_dot_scaled_e4m3_batched_masked_batch_tail():
    block_batch, block_m, block_n, block_k = 2, 16, 16, 32
    actual_batch, M, N, K = 1, 13, 11, 27
    one = 0x38
    a = torch.full((block_batch, block_m, block_k), one, dtype=torch.uint8, device="mps")
    b = torch.full((block_batch, block_k, block_n), one, dtype=torch.uint8, device="mps")
    a_scale = torch.full((block_m, 1), 127, dtype=torch.uint8, device="mps")
    b_scale = torch.full((block_n, 1), 127, dtype=torch.uint8, device="mps")
    sentinel = -4321.0
    output = torch.full(
        (block_batch, block_m, block_n),
        sentinel,
        dtype=torch.float32,
        device="mps",
    )

    dot_scaled_e4m3_batched_masked_u8_kernel[(1,)](
        a,
        b,
        a_scale,
        b_scale,
        output,
        actual_batch,
        M,
        N,
        K,
        BLOCK_BATCH=block_batch,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        MASK_BATCH=True,
        BATCH_BOUND_BIAS=0,
        num_warps=4,
    )
    torch.mps.synchronize()
    actual = output.cpu()
    assert torch.equal(actual[0, :M, :N], torch.full((M, N), float(K)))
    assert torch.all(actual[0, M:, :] == sentinel)
    assert torch.all(actual[0, :, N:] == sentinel)
    assert torch.all(actual[1] == sentinel)


def test_dot_scaled_batched_mismatched_batch_tail_is_named(capfd):
    block_batch, block_m, block_n, block_k = 2, 16, 16, 32
    one = 0x38
    a = torch.full((block_batch, block_m, block_k), one, dtype=torch.uint8, device="mps")
    b = torch.full((block_batch, block_k, block_n), one, dtype=torch.uint8, device="mps")
    a_scale = torch.full((block_m, 1), 127, dtype=torch.uint8, device="mps")
    b_scale = torch.full((block_n, 1), 127, dtype=torch.uint8, device="mps")
    output = torch.empty((block_batch, block_m, block_n), dtype=torch.float32, device="mps")

    with pytest.raises(Exception):
        dot_scaled_e4m3_batched_masked_u8_kernel[(1,)](
            a,
            b,
            a_scale,
            b_scale,
            output,
            1,
            block_m,
            block_n,
            block_k,
            BLOCK_BATCH=block_batch,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            MASK_BATCH=True,
            BATCH_BOUND_BIAS=1,
            num_warps=4,
        )
    assert "matching batch-tail bounds" in capfd.readouterr().err


def _loop_scaled_inputs(*, M, N, K, one, two, dtype, scale_factor):
    a = torch.full((M, K), one, dtype=dtype)
    a[:, K // 2 :] = two
    b = torch.full((K, N), one, dtype=dtype)
    a_scale = torch.full((M, K // scale_factor), 127, dtype=torch.uint8)
    a_scale[:, K // (2 * scale_factor) :] = 128
    b_scale = torch.full((N, K // scale_factor), 127, dtype=torch.uint8)
    return tuple(tensor.to("mps") for tensor in (a, b, a_scale, b_scale))


@pytest.mark.parametrize(
    ("elem_type", "one", "two", "dtype", "scale_factor"),
    [
        pytest.param("e4m3", 0x38, 0x40, torch.uint8, 32, id="e4m3-sf32"),
        pytest.param("e5m2", 0x3C, 0x40, torch.uint8, 32, id="e5m2-sf32"),
        pytest.param("fp16", 1.0, 2.0, torch.float16, 16, id="fp16-sf16"),
        pytest.param("fp16", 1.0, 2.0, torch.float16, 32, id="fp16-sf32"),
        pytest.param("bf16", 1.0, 2.0, torch.bfloat16, 16, id="bf16-sf16"),
        pytest.param("bf16", 1.0, 2.0, torch.bfloat16, 32, id="bf16-sf32"),
    ],
)
def test_dot_scaled_loop_accumulates_payload_and_scale_groups(elem_type, one, two, dtype, scale_factor):
    M = N = 16
    K = 64
    a, b, a_scale, b_scale = _loop_scaled_inputs(
        M=M, N=N, K=K, one=one, two=two, dtype=dtype, scale_factor=scale_factor
    )
    output = torch.empty((M, N), dtype=torch.float32, device="mps")

    dot_scaled_loop_kernel[(1,)](
        a,
        b,
        a_scale,
        b_scale,
        output,
        M,
        N,
        K,
        BLOCK_K=scale_factor,
        SCALE_FACTOR=scale_factor,
        LOOP_START=0,
        A_ELEM_TYPE=elem_type,
        B_ELEM_TYPE=elem_type,
        num_warps=4,
    )
    torch.mps.synchronize()
    assert torch.equal(output.cpu(), torch.full((M, N), 160.0))


@pytest.mark.parametrize(
    ("a_elem_type", "b_elem_type", "a_one", "a_two", "b_one", "b_two"),
    [
        pytest.param("e4m3", "e5m2", 0x38, 0x40, 0x40, 0x44, id="a-e4m3-b-e5m2"),
        pytest.param("e5m2", "e4m3", 0x3C, 0x40, 0x40, 0x48, id="a-e5m2-b-e4m3"),
    ],
)
@pytest.mark.parametrize(
    ("K", "loop_start", "expected"),
    [
        pytest.param(64, 0, 320.0, id="zero-start"),
        pytest.param(96, 32, 416.0, id="nonzero-start"),
    ],
)
def test_dot_scaled_loop_mixed_e4m3_e5m2_accumulates_payload_and_scale_groups(
    a_elem_type,
    b_elem_type,
    a_one,
    a_two,
    b_one,
    b_two,
    K,
    loop_start,
    expected,
):
    M = N = 16
    a = torch.full((M, K), a_one, dtype=torch.uint8)
    a[:, K // 2 :] = a_two
    b = torch.full((K, N), b_one, dtype=torch.uint8)
    b[K // 2 :, :] = b_two
    a_scale = torch.full((M, K // 32), 127, dtype=torch.uint8)
    a_scale[:, 1:] = 128
    b_scale = torch.full((N, K // 32), 127, dtype=torch.uint8)
    b_scale[:, 1:] = 126
    output = torch.empty((M, N), dtype=torch.float32, device="mps")

    dot_scaled_loop_kernel[(1,)](
        a.to("mps"),
        b.to("mps"),
        a_scale.to("mps"),
        b_scale.to("mps"),
        output,
        M,
        N,
        K,
        BLOCK_K=32,
        SCALE_FACTOR=32,
        LOOP_START=loop_start,
        A_ELEM_TYPE=a_elem_type,
        B_ELEM_TYPE=b_elem_type,
        num_warps=4,
    )
    torch.mps.synchronize()
    assert torch.equal(output.cpu(), torch.full((M, N), expected))


@pytest.mark.parametrize(
    ("elem_type", "one", "two", "dtype", "scale_factor"),
    [
        pytest.param("e4m3", 0x38, 0x40, torch.uint8, 32, id="e4m3-sf32"),
        pytest.param("e5m2", 0x3C, 0x40, torch.uint8, 32, id="e5m2-sf32"),
        pytest.param("fp16", 1.0, 2.0, torch.float16, 16, id="fp16-sf16"),
        pytest.param("fp16", 1.0, 2.0, torch.float16, 32, id="fp16-sf32"),
        pytest.param("bf16", 1.0, 2.0, torch.bfloat16, 16, id="bf16-sf16"),
        pytest.param("bf16", 1.0, 2.0, torch.bfloat16, 32, id="bf16-sf32"),
    ],
)
def test_dot_scaled_loop_accumulates_nonzero_start_payload_and_scale_groups(elem_type, one, two, dtype, scale_factor):
    M = N = 16
    K = 4 * scale_factor
    a, b, a_scale, b_scale = _loop_scaled_inputs(
        M=M, N=N, K=K, one=one, two=two, dtype=dtype, scale_factor=scale_factor
    )
    output = torch.empty((M, N), dtype=torch.float32, device="mps")

    dot_scaled_loop_kernel[(1,)](
        a,
        b,
        a_scale,
        b_scale,
        output,
        M,
        N,
        K,
        BLOCK_K=scale_factor,
        SCALE_FACTOR=scale_factor,
        LOOP_START=scale_factor,
        A_ELEM_TYPE=elem_type,
        B_ELEM_TYPE=elem_type,
        num_warps=4,
    )
    torch.mps.synchronize()
    assert torch.equal(output.cpu(), torch.full((M, N), float(9 * scale_factor)))


def test_dot_scaled_loop_misaligned_nonzero_start_is_named(capfd):
    M = N = 16
    scale_factor = 32
    K = 4 * scale_factor
    a, b, a_scale, b_scale = _loop_scaled_inputs(
        M=M, N=N, K=K, one=0x38, two=0x40, dtype=torch.uint8, scale_factor=scale_factor
    )
    output = torch.empty((M, N), dtype=torch.float32, device="mps")

    with pytest.raises(Exception):
        dot_scaled_loop_kernel[(1,)](
            a,
            b,
            a_scale,
            b_scale,
            output,
            M,
            N,
            K,
            BLOCK_K=scale_factor,
            SCALE_FACTOR=scale_factor,
            LOOP_START=scale_factor // 2,
            A_ELEM_TYPE="e4m3",
            B_ELEM_TYPE="e4m3",
            num_warps=4,
        )
    assert "nonnegative aligned divisible bounds" in capfd.readouterr().err


def test_dot_scaled_loop_negative_start_is_named(capfd):
    M = N = 16
    scale_factor = 32
    K = 2 * scale_factor
    a, b, a_scale, b_scale = _loop_scaled_inputs(
        M=M, N=N, K=K, one=0x38, two=0x40, dtype=torch.uint8, scale_factor=scale_factor
    )
    output = torch.empty((M, N), dtype=torch.float32, device="mps")

    with pytest.raises(Exception):
        dot_scaled_loop_kernel[(1,)](
            a,
            b,
            a_scale,
            b_scale,
            output,
            M,
            N,
            K,
            BLOCK_K=scale_factor,
            SCALE_FACTOR=scale_factor,
            LOOP_START=-scale_factor,
            A_ELEM_TYPE="e4m3",
            B_ELEM_TYPE="e4m3",
            num_warps=4,
        )
    assert "nonnegative aligned divisible bounds" in capfd.readouterr().err


@pytest.mark.parametrize(
    ("elem_type", "one", "two", "dtype", "scale_factor"),
    [
        pytest.param("e4m3", 0x38, 0x40, torch.uint8, 32, id="e4m3-sf32"),
        pytest.param("fp16", 1.0, 2.0, torch.float16, 16, id="fp16-sf16"),
    ],
)
@pytest.mark.parametrize(
    ("actual_k", "expected_value"),
    [
        pytest.param(-1, 0.0, id="negative-k-clamps-to-zero"),
        pytest.param(48, 96.0, id="partial-k"),
        pytest.param(80, 160.0, id="oversized-k-clamps-to-static-upper"),
    ],
)
def test_dot_scaled_masked_loop_clamps_k_and_preserves_mn_tails(
    elem_type,
    one,
    two,
    dtype,
    scale_factor,
    actual_k,
    expected_value,
):
    M = N = 16
    K = 64
    actual_m, actual_n = 13, 11
    a, b, a_scale, b_scale = _loop_scaled_inputs(
        M=M, N=N, K=K, one=one, two=two, dtype=dtype, scale_factor=scale_factor
    )
    sentinel = -7.0
    output = torch.full((M, N), sentinel, dtype=torch.float32, device="mps")

    dot_scaled_masked_loop_kernel[(1,)](
        a,
        b,
        a_scale,
        b_scale,
        output,
        actual_m,
        actual_n,
        actual_k,
        M,
        N,
        K,
        BLOCK_K=scale_factor,
        SCALE_FACTOR=scale_factor,
        LOOP_START=0,
        A_ELEM_TYPE=elem_type,
        B_ELEM_TYPE=elem_type,
        LOAD_OTHER=0,
        A_K_BOUND_BIAS=0,
        num_warps=4,
    )
    torch.mps.synchronize()
    expected = torch.full((M, N), sentinel)
    expected[:actual_m, :actual_n] = expected_value
    assert torch.equal(output.cpu(), expected)


@pytest.mark.parametrize(
    ("elem_type", "one", "two", "dtype", "scale_factor"),
    [
        pytest.param("e4m3", 0x38, 0x40, torch.uint8, 32, id="e4m3-sf32"),
        pytest.param("fp16", 1.0, 2.0, torch.float16, 16, id="fp16-sf16"),
    ],
)
@pytest.mark.parametrize(
    ("actual_k", "expected_value"),
    [
        pytest.param(-1, 0.0, id="negative-k-clamps-to-zero"),
        pytest.param(0, 0.0, id="zero-k"),
        pytest.param(27, 27.0, id="partial-first-group"),
        pytest.param(48, 96.0, id="partial-k"),
        pytest.param(64, 160.0, id="capacity-k"),
        pytest.param(80, 160.0, id="oversized-k-clamps-to-capacity"),
    ],
)
def test_dot_scaled_dynamic_upper_masked_loop_clamps_k_and_preserves_mn_tails(
    elem_type,
    one,
    two,
    dtype,
    scale_factor,
    actual_k,
    expected_value,
):
    M = N = 16
    K = 64
    actual_m, actual_n = 13, 11
    a, b, a_scale, b_scale = _loop_scaled_inputs(
        M=M, N=N, K=K, one=one, two=two, dtype=dtype, scale_factor=scale_factor
    )
    sentinel = -7.0
    output = torch.full((M, N), sentinel, dtype=torch.float32, device="mps")

    dot_scaled_dynamic_upper_masked_loop_kernel[(1,)](
        a,
        b,
        a_scale,
        b_scale,
        output,
        actual_m,
        actual_n,
        actual_k,
        M,
        N,
        K,
        BLOCK_K=scale_factor,
        SCALE_FACTOR=scale_factor,
        A_ELEM_TYPE=elem_type,
        B_ELEM_TYPE=elem_type,
        LOOP_UPPER_BIAS=0,
        A_SCALE_ROW_STRIDE_BIAS=0,
        num_warps=4,
    )
    torch.mps.synchronize()
    expected = torch.full((M, N), sentinel)
    expected[:actual_m, :actual_n] = expected_value
    assert torch.equal(output.cpu(), expected)


def test_dot_scaled_dynamic_upper_unmasked_loop_is_named(capfd):
    M = N = 16
    K = 64
    scale_factor = 16
    a, b, a_scale, b_scale = _loop_scaled_inputs(
        M=M, N=N, K=K, one=1.0, two=2.0, dtype=torch.float16, scale_factor=scale_factor
    )
    output = torch.empty((M, N), dtype=torch.float32, device="mps")

    with pytest.raises(Exception):
        dot_scaled_dynamic_upper_unmasked_loop_kernel[(1,)](
            a,
            b,
            a_scale,
            b_scale,
            output,
            48,
            M,
            N,
            K,
            BLOCK_K=scale_factor,
            SCALE_FACTOR=scale_factor,
            num_warps=4,
        )
    assert "dynamic-upper form requires matched zero-filled rectangular A/B loads" in capfd.readouterr().err


def test_dot_scaled_dynamic_upper_masked_loop_rejects_noncanonical_upper(capfd):
    M = N = 16
    K = 64
    actual_m, actual_n, actual_k = 13, 11, 48
    a, b, a_scale, b_scale = _loop_scaled_inputs(M=M, N=N, K=K, one=1.0, two=2.0, dtype=torch.float16, scale_factor=16)
    output = torch.empty((M, N), dtype=torch.float32, device="mps")

    with pytest.raises(Exception):
        dot_scaled_dynamic_upper_masked_loop_kernel[(1,)](
            a,
            b,
            a_scale,
            b_scale,
            output,
            actual_m,
            actual_n,
            actual_k,
            M,
            N,
            K,
            BLOCK_K=16,
            SCALE_FACTOR=16,
            A_ELEM_TYPE="fp16",
            B_ELEM_TYPE="fp16",
            LOOP_UPPER_BIAS=1,
            A_SCALE_ROW_STRIDE_BIAS=0,
            num_warps=4,
        )
    assert "min(max(mask_k, 0), full_k) upper" in capfd.readouterr().err


def test_dot_scaled_dynamic_upper_masked_loop_rejects_padded_scale_row(capfd):
    M = N = 16
    K = 64
    actual_m, actual_n, actual_k = 13, 11, 48
    a, b, a_scale, b_scale = _loop_scaled_inputs(M=M, N=N, K=K, one=1.0, two=2.0, dtype=torch.float16, scale_factor=16)
    output = torch.empty((M, N), dtype=torch.float32, device="mps")

    with pytest.raises(Exception):
        dot_scaled_dynamic_upper_masked_loop_kernel[(1,)](
            a,
            b,
            a_scale,
            b_scale,
            output,
            actual_m,
            actual_n,
            actual_k,
            M,
            N,
            K,
            BLOCK_K=16,
            SCALE_FACTOR=16,
            A_ELEM_TYPE="fp16",
            B_ELEM_TYPE="fp16",
            LOOP_UPPER_BIAS=0,
            A_SCALE_ROW_STRIDE_BIAS=1,
            num_warps=4,
        )
    assert "requires direct unmasked row-major i8 scale loads" in capfd.readouterr().err


def test_dot_scaled_masked_loop_nonzero_other_is_named(capfd):
    M = N = 16
    K = 64
    actual_m, actual_n, actual_k = 13, 11, 48
    a, b, a_scale, b_scale = _loop_scaled_inputs(M=M, N=N, K=K, one=1.0, two=2.0, dtype=torch.float16, scale_factor=16)
    output = torch.empty((M, N), dtype=torch.float32, device="mps")

    with pytest.raises(Exception):
        dot_scaled_masked_loop_kernel[(1,)](
            a,
            b,
            a_scale,
            b_scale,
            output,
            actual_m,
            actual_n,
            actual_k,
            M,
            N,
            K,
            BLOCK_K=16,
            SCALE_FACTOR=16,
            LOOP_START=0,
            A_ELEM_TYPE="fp16",
            B_ELEM_TYPE="fp16",
            LOAD_OTHER=1,
            A_K_BOUND_BIAS=0,
            num_warps=4,
        )
    assert "matched zero-filled rectangular A/B loads" in capfd.readouterr().err


def test_dot_scaled_masked_loop_mismatched_k_bound_is_named(capfd):
    M = N = 16
    K = 64
    actual_m, actual_n, actual_k = 13, 11, 48
    a, b, a_scale, b_scale = _loop_scaled_inputs(M=M, N=N, K=K, one=1.0, two=2.0, dtype=torch.float16, scale_factor=16)
    output = torch.empty((M, N), dtype=torch.float32, device="mps")

    with pytest.raises(Exception):
        dot_scaled_masked_loop_kernel[(1,)](
            a,
            b,
            a_scale,
            b_scale,
            output,
            actual_m,
            actual_n,
            actual_k,
            M,
            N,
            K,
            BLOCK_K=16,
            SCALE_FACTOR=16,
            LOOP_START=0,
            A_ELEM_TYPE="fp16",
            B_ELEM_TYPE="fp16",
            LOAD_OTHER=0,
            A_K_BOUND_BIAS=1,
            num_warps=4,
        )
    assert "matched zero-filled rectangular A/B loads" in capfd.readouterr().err


def test_dot_scaled_masked_loop_nonzero_start_is_named(capfd):
    M = N = 16
    K = 64
    actual_m, actual_n, actual_k = 13, 11, 48
    a, b, a_scale, b_scale = _loop_scaled_inputs(M=M, N=N, K=K, one=1.0, two=2.0, dtype=torch.float16, scale_factor=16)
    output = torch.empty((M, N), dtype=torch.float32, device="mps")

    with pytest.raises(Exception):
        dot_scaled_masked_loop_kernel[(1,)](
            a,
            b,
            a_scale,
            b_scale,
            output,
            actual_m,
            actual_n,
            actual_k,
            M,
            N,
            K,
            BLOCK_K=16,
            SCALE_FACTOR=16,
            LOOP_START=16,
            A_ELEM_TYPE="fp16",
            B_ELEM_TYPE="fp16",
            LOAD_OTHER=0,
            A_K_BOUND_BIAS=0,
            num_warps=4,
        )
    assert "masked form currently requires a zero-based K loop" in capfd.readouterr().err


def _loop_e2m1_inputs(*, M, N, K, lhs_k_pack, rhs_k_pack):
    a_shape = (M, K // 2) if lhs_k_pack else (M // 2, K)
    b_shape = (K // 2, N) if rhs_k_pack else (K, N // 2)
    a = torch.full(a_shape, 0x22, dtype=torch.uint8)
    a[:, a.shape[1] // 2 :] = 0x44
    b = torch.full(b_shape, 0x22, dtype=torch.uint8)
    a_scale = torch.full((M, K // 32), 127, dtype=torch.uint8)
    a_scale[:, K // 64 :] = 128
    b_scale = torch.full((N, K // 32), 127, dtype=torch.uint8)
    return tuple(tensor.to("mps") for tensor in (a, b, a_scale, b_scale))


@pytest.mark.parametrize(
    ("lhs_k_pack", "rhs_k_pack"),
    [
        pytest.param(True, True, id="kk"),
        pytest.param(True, False, id="ko"),
        pytest.param(False, True, id="ok"),
        pytest.param(False, False, id="oo"),
    ],
)
def test_dot_scaled_loop_e2m1_accumulates_packed_payload_and_scale_groups(lhs_k_pack, rhs_k_pack):
    M = N = 16
    K = 64
    a, b, a_scale, b_scale = _loop_e2m1_inputs(
        M=M,
        N=N,
        K=K,
        lhs_k_pack=lhs_k_pack,
        rhs_k_pack=rhs_k_pack,
    )
    output = torch.empty((M, N), dtype=torch.float32, device="mps")

    dot_scaled_loop_e2m1_u8_kernel[(1,)](
        a,
        b,
        a_scale,
        b_scale,
        output,
        M,
        N,
        K,
        BLOCK_K=32,
        LOOP_START=0,
        LHS_K_PACK=lhs_k_pack,
        RHS_K_PACK=rhs_k_pack,
        PACKED_K_DIVISOR=2,
        num_warps=4,
    )
    torch.mps.synchronize()
    assert torch.equal(output.cpu(), torch.full((M, N), 160.0))


@pytest.mark.parametrize(
    ("lhs_k_pack", "rhs_k_pack"),
    [
        pytest.param(True, True, id="kk"),
        pytest.param(True, False, id="ko"),
        pytest.param(False, True, id="ok"),
        pytest.param(False, False, id="oo"),
    ],
)
def test_dot_scaled_loop_e2m1_accumulates_nonzero_start_packed_payload_and_scale_groups(lhs_k_pack, rhs_k_pack):
    M = N = 16
    K = 96
    a, b, a_scale, b_scale = _loop_e2m1_inputs(
        M=M,
        N=N,
        K=K,
        lhs_k_pack=lhs_k_pack,
        rhs_k_pack=rhs_k_pack,
    )
    output = torch.empty((M, N), dtype=torch.float32, device="mps")

    dot_scaled_loop_e2m1_u8_kernel[(1,)](
        a,
        b,
        a_scale,
        b_scale,
        output,
        M,
        N,
        K,
        BLOCK_K=32,
        LOOP_START=32,
        LHS_K_PACK=lhs_k_pack,
        RHS_K_PACK=rhs_k_pack,
        PACKED_K_DIVISOR=2,
        num_warps=4,
    )
    torch.mps.synchronize()
    assert torch.equal(output.cpu(), torch.full((M, N), 224.0))


@pytest.mark.parametrize(
    ("lhs_k_pack", "rhs_k_pack"),
    [
        pytest.param(True, False, id="lhs"),
        pytest.param(False, True, id="rhs"),
    ],
)
def test_dot_scaled_loop_e2m1_wrong_packed_k_divisor_is_named(capfd, lhs_k_pack, rhs_k_pack):
    M = N = 16
    K = 64
    a, b, a_scale, b_scale = _loop_e2m1_inputs(
        M=M,
        N=N,
        K=K,
        lhs_k_pack=lhs_k_pack,
        rhs_k_pack=rhs_k_pack,
    )
    output = torch.empty((M, N), dtype=torch.float32, device="mps")

    with pytest.raises(Exception):
        dot_scaled_loop_e2m1_u8_kernel[(1,)](
            a,
            b,
            a_scale,
            b_scale,
            output,
            M,
            N,
            K,
            BLOCK_K=32,
            LOOP_START=0,
            LHS_K_PACK=lhs_k_pack,
            RHS_K_PACK=rhs_k_pack,
            PACKED_K_DIVISOR=4,
            num_warps=4,
        )
    assert "canonical contiguous full-K A and B pointer arithmetic" in capfd.readouterr().err


def _run_dot_scaled_e5m2_u8(a):
    b = torch.zeros((32, 16), dtype=torch.uint8)
    diagonal = torch.arange(16)
    b[diagonal, diagonal] = 0x3C
    a_scale = torch.full((16, 1), 127, dtype=torch.uint8)
    b_scale = torch.full((16, 1), 127, dtype=torch.uint8)
    output = torch.empty((16, 16), dtype=torch.float32, device="mps")

    dot_scaled_e5m2_u8_kernel[(1,)](
        a.to("mps"),
        b.to("mps"),
        a_scale.to("mps"),
        b_scale.to("mps"),
        output,
        BLOCK_M=16,
        BLOCK_N=16,
        BLOCK_K=32,
        num_warps=4,
    )
    torch.mps.synchronize()
    return output.cpu()


def _run_dot_scaled_e2m1_u8(a, b, *, lhs_k_pack=True, rhs_k_pack=True, a_scale_raw=None, b_scale_raw=None):
    M = a.shape[0] if lhs_k_pack else a.shape[0] * 2
    logical_k_a = a.shape[1] * 2 if lhs_k_pack else a.shape[1]
    logical_k_b = b.shape[0] * 2 if rhs_k_pack else b.shape[0]
    N = b.shape[1] if rhs_k_pack else b.shape[1] * 2
    assert logical_k_a == logical_k_b
    logical_k = logical_k_a
    scale_groups = logical_k // 32
    if a_scale_raw is None:
        a_scale_raw = torch.full((M, scale_groups), 127, dtype=torch.uint8)
    if b_scale_raw is None:
        b_scale_raw = torch.full((N, scale_groups), 127, dtype=torch.uint8)
    assert a_scale_raw.shape == (M, scale_groups)
    assert b_scale_raw.shape == (N, scale_groups)
    output = torch.empty((M, N), dtype=torch.float32, device="mps")

    dot_scaled_e2m1_u8_kernel[(1,)](
        a.to("mps"),
        b.to("mps"),
        a_scale_raw.to("mps"),
        b_scale_raw.to("mps"),
        output,
        BLOCK_M=M,
        BLOCK_N=N,
        BLOCK_K=logical_k,
        LHS_K_PACK=lhs_k_pack,
        RHS_K_PACK=rhs_k_pack,
        num_warps=4,
    )
    torch.mps.synchronize()
    return output.cpu()


def _decode_e2m1_packed(packed, dim):
    values = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])
    low = values[(packed & 0xF).long()]
    high = values[(packed >> 4).long()]
    return torch.stack((low, high), dim=dim + 1).flatten(dim, dim + 1)


def test_dot_scaled_e2m1_a_payload_bytes_exhaustive():
    # Every physical A byte contributes its low then high nibble to adjacent
    # logical-K columns.  An encoded +1 identity on B exposes all 512 decoded
    # values independently in the 16x32 output.
    a = torch.arange(256, dtype=torch.uint8).reshape(16, 16)
    b = torch.zeros((16, 32), dtype=torch.uint8)
    for diagonal in range(32):
        b[diagonal // 2, diagonal] = 0x2 << (4 * (diagonal & 1))

    actual = _run_dot_scaled_e2m1_u8(a, b)
    reference = _decode_e2m1_packed(a, dim=1)
    assert torch.equal(actual, reference)


def test_dot_scaled_e2m1_b_payload_bytes_exhaustive():
    # Mirror the exhaustive byte/address check for B.  Packed A now supplies
    # the logical identity, so both nibbles of all 256 B bytes are observable.
    a = torch.zeros((32, 16), dtype=torch.uint8)
    for diagonal in range(32):
        a[diagonal, diagonal // 2] = 0x2 << (4 * (diagonal & 1))
    b = torch.arange(256, dtype=torch.uint8).reshape(16, 16)

    actual = _run_dot_scaled_e2m1_u8(a, b)
    reference = _decode_e2m1_packed(b, dim=0)
    assert torch.equal(actual, reference)


def test_dot_scaled_e2m1_lhs_outer_packed_payload_bytes_exhaustive():
    a = torch.arange(256, dtype=torch.uint8).reshape(8, 32)
    b = torch.zeros((16, 32), dtype=torch.uint8)
    for diagonal in range(32):
        b[diagonal // 2, diagonal] = 0x2 << (4 * (diagonal & 1))

    actual = _run_dot_scaled_e2m1_u8(a, b, lhs_k_pack=False)
    reference = _decode_e2m1_packed(a, dim=0)
    assert torch.equal(actual, reference)


def test_dot_scaled_e2m1_rhs_outer_packed_payload_bytes_exhaustive():
    a = torch.zeros((32, 16), dtype=torch.uint8)
    for diagonal in range(32):
        a[diagonal, diagonal // 2] = 0x2 << (4 * (diagonal & 1))
    b = torch.arange(256, dtype=torch.uint8).reshape(32, 8)

    actual = _run_dot_scaled_e2m1_u8(a, b, rhs_k_pack=False)
    reference = _decode_e2m1_packed(b, dim=1)
    assert torch.equal(actual, reference)


def test_dot_scaled_e2m1_both_outer_packed_with_distinct_scales():
    a = torch.arange(256, dtype=torch.uint8).reshape(8, 32)
    b = torch.arange(255, -1, -1, dtype=torch.int32).to(torch.uint8).reshape(32, 8)
    a_scale_raw = torch.tensor(([126, 127, 128, 127] * 4), dtype=torch.uint8).reshape(16, 1)
    b_scale_raw = torch.tensor(([128, 127, 126, 127] * 4), dtype=torch.uint8).reshape(16, 1)

    actual = _run_dot_scaled_e2m1_u8(
        a,
        b,
        lhs_k_pack=False,
        rhs_k_pack=False,
        a_scale_raw=a_scale_raw,
        b_scale_raw=b_scale_raw,
    )
    decoded_a = _decode_e2m1_packed(a, dim=0)
    decoded_b = _decode_e2m1_packed(b, dim=1)
    a_scales = torch.pow(2.0, a_scale_raw.float() - 127.0)
    b_scales = torch.pow(2.0, b_scale_raw.float() - 127.0).T
    reference = (decoded_a @ decoded_b) * a_scales * b_scales
    assert torch.equal(actual, reference)


@pytest.mark.parametrize("lhs_k_pack,rhs_k_pack", [(False, True), (True, False)])
def test_dot_scaled_e2m1_mixed_packing_uses_logical_k_scale_groups(lhs_k_pack, rhs_k_pack):
    M, N, K = 16, 16, 64
    a_shape = (M, K // 2) if lhs_k_pack else (M // 2, K)
    b_shape = (K // 2, N) if rhs_k_pack else (K, N // 2)
    a = torch.full(a_shape, 0x22, dtype=torch.uint8)
    b = torch.full(b_shape, 0x22, dtype=torch.uint8)
    a_scale_raw = torch.tensor(([127, 129] * M), dtype=torch.uint8).reshape(M, 2)
    b_scale_raw = torch.tensor(([127, 128] * N), dtype=torch.uint8).reshape(N, 2)

    actual = _run_dot_scaled_e2m1_u8(
        a,
        b,
        lhs_k_pack=lhs_k_pack,
        rhs_k_pack=rhs_k_pack,
        a_scale_raw=a_scale_raw,
        b_scale_raw=b_scale_raw,
    )
    # Each logical payload is +1. Group 0 contributes 32 * 1 * 1 and group 1
    # contributes 32 * 4 * 2. Reusing physical packed-K for scale lookup would
    # incorrectly select group 0 for the entire reduction.
    assert torch.equal(actual, torch.full((M, N), 288.0))


def test_dot_scaled_e5m2_exhaustive_finite_payloads():
    finite_raw = [value for value in range(256) if value & 0x7F < 0x7C]
    raw = torch.tensor(finite_raw + [0] * 8, dtype=torch.uint8).reshape(16, 16)
    a = torch.zeros((16, 32), dtype=torch.uint8)
    a[:, :16] = raw

    actual = _run_dot_scaled_e5m2_u8(a)
    reference = raw.reshape(-1).view(torch.float8_e5m2).float().reshape(16, 16)
    assert torch.isfinite(reference).all()
    assert torch.equal(actual, reference)


def test_dot_scaled_e5m2_special_value_classes():
    special = torch.tensor(
        [0x7C, 0xFC, 0x7D, 0x7E, 0x7F, 0xFD, 0xFE, 0xFF],
        dtype=torch.uint8,
    )
    a = torch.zeros((16, 32), dtype=torch.uint8)
    diagonal = torch.arange(special.numel())
    a[diagonal, diagonal] = special

    actual = _run_dot_scaled_e5m2_u8(a).diagonal()
    assert actual[0].item() == float("inf")
    assert actual[1].item() == float("-inf")
    assert torch.isnan(actual[2:8]).all()
    assert torch.equal(actual[8:], torch.zeros_like(actual[8:]))


def test_dot_scaled_e4m3_exhaustive_finite_payloads():
    # Each output selects one A element through an E4M3 +1 diagonal in B.  Keep
    # NaNs out of this matrix because NaN*0 would intentionally contaminate the
    # other outputs in its row; NaN class is covered independently below.
    finite_raw = [value for value in range(256) if value & 0x7F != 0x7F]
    raw = torch.tensor(finite_raw + [0, 0], dtype=torch.uint8).reshape(16, 16)
    a = torch.zeros((16, 32), dtype=torch.uint8)
    a[:, :16] = raw

    actual = _run_dot_scaled_e4m3_u8(a)
    reference = raw.reshape(-1).view(torch.float8_e4m3fn).float().reshape(16, 16)
    assert torch.isfinite(reference).all()
    assert torch.equal(actual, reference)


def test_dot_scaled_e4m3_nan_class():
    a = torch.zeros((16, 32), dtype=torch.uint8)
    a[0, 0] = 0x7F
    a[1, 1] = 0xFF

    actual = _run_dot_scaled_e4m3_u8(a)
    assert torch.isnan(actual[:2]).all()
    assert torch.equal(actual[2:], torch.zeros_like(actual[2:]))


# Raw 10 is the smallest boundary here whose selected products stay out of the
# target's native BF16/F32 denormal-flush region; 127/254/255 cover unity,
# overflow, and the required E8M0 NaN sentinel.
@pytest.mark.parametrize("scale_raw", [10, 127, 254, 255])
def test_dot_scaled_e4m3_bf16_scale_boundaries(scale_raw):
    payloads = torch.tensor(
        [
            0x01,
            0x03,
            0x07,
            0x08,
            0x38,
            0x3F,
            0x78,
            0x7E,
            0x81,
            0x83,
            0x87,
            0x88,
            0xB8,
            0xBF,
            0xF8,
            0xFE,
        ],
        dtype=torch.uint8,
    )
    a = torch.zeros((16, 32), dtype=torch.uint8)
    diagonal = torch.arange(16)
    a[diagonal, diagonal] = payloads
    actual = _run_dot_scaled_e4m3_u8(a, a_scale_raw=scale_raw).diagonal()

    if scale_raw == 255:
        assert torch.isnan(actual).all()
        return
    scale_bits = torch.tensor([scale_raw << 7], dtype=torch.uint16)
    scale = scale_bits.view(torch.bfloat16)
    reference = (payloads.view(torch.float8_e4m3fn).to(torch.bfloat16) * scale).float()
    torch.testing.assert_close(actual, reference, rtol=0, atol=0, equal_nan=True)


def _run(M, N, K, dtype_in=torch.float32, *, seed=0xC0FFEE):
    torch.manual_seed(seed)
    a = torch.randn((M, K), dtype=dtype_in).contiguous()
    b = torch.randn((K, N), dtype=dtype_in).contiguous()
    c = torch.zeros((M, N), dtype=torch.float32).contiguous()
    K_TILES = K // 8
    dot_universal_kernel[(triton.cdiv(M, 8), triton.cdiv(N, 8))](
        a,
        b,
        c,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_M=8,
        BLOCK_N=8,
        BLOCK_K=8,
        K_TILES=K_TILES,
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
    os.environ.setdefault("TRITON_REPRODUCER_PATH", f"/tmp/dot-universal-{M}x{N}x{K}.mlir")
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
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
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
        a_pad,
        b_pad,
        c_pad,
        M,
        N,
        a_pad.stride(0),
        a_pad.stride(1),
        b_pad.stride(0),
        b_pad.stride(1),
        c_pad.stride(0),
        c_pad.stride(1),
        BLOCK_M=8,
        BLOCK_N=8,
        BLOCK_K=8,
        K_TILES=K_TILES,
    )
    ref = a_pad[:M, :].float() @ b_pad[:, :N].float()
    return c_pad[:M, :N].contiguous(), ref


@pytest.mark.parametrize(
    "M,N,K",
    [
        pytest.param(33, 8, 8, id="m_tail_only"),
        pytest.param(8, 17, 8, id="n_tail_only"),
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
    tol = K * (2.0**-10)
    torch.testing.assert_close(c, ref, atol=tol, rtol=tol)


def test_dot_bf16_singletile():
    M = N = K = 8
    c, ref = _run(M, N, K, torch.bfloat16)
    tol = K * (2.0**-7)
    torch.testing.assert_close(c, ref, atol=tol, rtol=tol)


# Verbatim compute shape from
# python/test/unit/fixtures/metal_leet/medium-fp16_batched_matrix_multiplication.py.  Unlike the
# canonical universal kernel above, each dot operand is explicitly extended
# from fp16 to fp32 before Triton inserts the blocked -> dot-operand relayout.
@triton.jit
def batched_fp16_dot_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    BATCH,
    M,
    N,
    K,
    BLOCK_SIZE: tl.constexpr,
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
    tol = K * (2.0**-9)
    torch.testing.assert_close(c.float(), ref.float(), atol=tol, rtol=tol)


# Compute shape from python/test/unit/fixtures/metal_leet/medium-batched_matrix_multiplication.py.  The
# batch dimension remains in every tile, so tl.dot is rank-3 for both unit and
# multi-element batch tiles.
@triton.jit
def batched_f32_rank3_dot_kernel(
    a,
    b,
    c,
    BATCH,
    M,
    N,
    K,
    BLOCK_BATCH: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
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
        a_mask = mask_b[:, None, None] & mask_m[None, :, None] & mask_k[None, None, :]
        tile_a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        b_batch_offset = offs_b64[:, None, None] * (K64 * N64)
        b_k_offset = offs_k64[None, :, None] * N64
        b_col_offset = offs_n64[None, None, :]
        if COMBINE_OFFSETS:
            b_ptrs = b + (b_batch_offset + b_k_offset + b_col_offset)
        else:
            b_ptrs = b + b_batch_offset + b_k_offset + b_col_offset
        b_mask = mask_b[:, None, None] & mask_k[None, :, None] & mask_n[None, None, :]
        tile_b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc = tl.dot(tile_a, tile_b, acc=acc, input_precision="ieee")

    c_batch_offset = offs_b64[:, None, None] * (M64 * N64)
    c_row_offset = offs_m64[None, :, None] * N64
    c_col_offset = offs_n64[None, None, :]
    if COMBINE_OFFSETS:
        c_ptrs = c + (c_batch_offset + c_row_offset + c_col_offset)
    else:
        c_ptrs = c + c_batch_offset + c_row_offset + c_col_offset
    c_mask = mask_b[:, None, None] & mask_m[None, :, None] & mask_n[None, None, :]
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
    batch,
    M,
    N,
    K,
    block_batch,
    combine_offsets,
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
        a,
        b,
        c,
        batch,
        M,
        N,
        K,
        BLOCK_BATCH=block_batch,
        BLOCK_M=64,
        BLOCK_N=64,
        BLOCK_K=64,
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
def test_int4_weight_only_dot_runs_computed_dequant_operand(M, N, K, group_size):
    torch.manual_seed(0x1A4)
    x = torch.randn((M, K), dtype=torch.float16).contiguous()
    w_int = torch.randint(-8, 8, (N, K), dtype=torch.int16)
    hi = (w_int[:, 0::2] + 8).to(torch.uint8) << 4
    lo = (w_int[:, 1::2] + 8).to(torch.uint8)
    wq = (hi | lo).contiguous()
    scales = torch.rand((N, K // group_size), dtype=torch.float32).contiguous()
    y = torch.empty((M, N), dtype=torch.float16).contiguous()

    int4_weight_only_dot_kernel[(triton.cdiv(M, 64), triton.cdiv(N, 64))](
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
    tol = K * (2.0**-9)
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
    a_ptr,
    b_ptr,
    c_ptr,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
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
        a,
        b,
        c,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_M=M,
        BLOCK_N=N,
        BLOCK_K=8,
        K_TILES=K // 8,
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
        a,
        b,
        c,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_M=M,
        BLOCK_N=N,
        BLOCK_K=8,
        K_TILES=K // 8,
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
    assert mma_count == expected_mma, f"num_warps={num_warps}: mma count {mma_count} != expected {expected_mma}"
    if num_warps > 1:
        assert "simdgroup_index_in_threadgroup" in msl, (
            "multi-warp MSL missing simdgroup_index_in_threadgroup parameter"
        )
        assert "_stage_shared[sgid]" in msl, "multi-warp MSL missing per-warp stage buffer slice _stage_shared[sgid]"
    else:
        assert "_stage_shared[" in msl
        # Single-warp Branch A: shared buffer is [elems], not [num_warps][elems].
        assert "_stage_shared[sgid]" not in msl, "single-warp MSL should not slice _stage_shared by sgid"


# --- transposed B ---------------------------------------------------------
#
# `tl.dot(a, tl.trans(b))` with b stored [N, K] is how any nn.Linear-style
# matmul is written, and it was SILENTLY WRONG at 64x64 tiles: the multi-tile
# simdgroup matcher declines there on its register guard, and the scalar-dot
# tier that picks the dot up peeled through the `tt.trans` to reach the root
# load — discarding the transpose and indexing B as if it were [K, N].
#
# The same is true of the address-transposed form `b + n*K + k`, which carries
# no `tt.trans` at all: every simdgroup matcher detects trans-B by looking for
# that op, so it staged an ordinary [K, N] tile with `stride_b` read off the
# unit row axis.
#
# Both are checked against torch here, and the tile sizes span the tiers: 16/32
# take a simdgroup path, 64 falls to scalar_dot.


@triton.jit
def _transb_trans_kernel(a_ptr, b_ptr, c_ptr, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """b is [N, K]; the transpose is a real tt.trans."""
    om = tl.arange(0, BM)
    on = tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        ok = k + tl.arange(0, BK)
        av = tl.load(a_ptr + om[:, None] * K + ok[None, :], mask=(om[:, None] < M) & (ok[None, :] < K), other=0.0)
        bnk = tl.load(b_ptr + on[:, None] * K + ok[None, :], mask=(on[:, None] < N) & (ok[None, :] < K), other=0.0)
        acc = tl.dot(av, tl.trans(bnk), acc)
    tl.store(c_ptr + om[:, None] * N + on[None, :], acc, mask=(om[:, None] < M) & (on[None, :] < N))


@triton.jit
def _transb_address_kernel(a_ptr, b_ptr, c_ptr, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """b is [N, K] and the transpose lives only in the address arithmetic."""
    om = tl.arange(0, BM)
    on = tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        ok = k + tl.arange(0, BK)
        av = tl.load(a_ptr + om[:, None] * K + ok[None, :], mask=(om[:, None] < M) & (ok[None, :] < K), other=0.0)
        bv = tl.load(b_ptr + on[None, :] * K + ok[:, None], mask=(ok[:, None] < K) & (on[None, :] < N), other=0.0)
        acc += tl.dot(av, bv)
    tl.store(c_ptr + om[:, None] * N + on[None, :], acc, mask=(om[:, None] < M) & (on[None, :] < N))


@pytest.mark.parametrize("kernel", [_transb_trans_kernel, _transb_address_kernel])
@pytest.mark.parametrize("size,bk", [(16, 16), (32, 32), (64, 64), (64, 32)])
def test_dot_transposed_b(kernel, size, bk):
    torch.manual_seed(size + bk)
    m = n = k = size
    a = torch.rand(m, k, dtype=torch.float32)
    b = torch.rand(k, n, dtype=torch.float32)
    c = torch.zeros(m, n, dtype=torch.float32, device="mps")
    kernel[(1, 1)](a.to("mps"), b.t().contiguous().to("mps"), c, m, n, k, BM=m, BN=n, BK=bk, num_warps=1)
    torch.mps.synchronize()
    torch.testing.assert_close(c.cpu(), a @ b, atol=2e-4, rtol=2e-4)


@triton.jit
def _dot_runtime_extent_one_kernel(a_ptr, b_ptr, c_ptr, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    om = tl.arange(0, BM)
    on = tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        ok = k + tl.arange(0, BK)
        av = tl.load(a_ptr + om[:, None] * K + ok[None, :], mask=(om[:, None] < M) & (ok[None, :] < K), other=0.0)
        bv = tl.load(b_ptr + ok[:, None] * N + on[None, :], mask=(ok[:, None] < K) & (on[None, :] < N), other=0.0)
        acc += tl.dot(av, bv)
    tl.store(c_ptr + om[:, None] * N + on[None, :], acc, mask=(om[:, None] < M) & (on[None, :] < N))


def test_unclaimed_dot_is_rejected_not_crashed(capfd):
    """A matmul with a runtime extent of 1 matches no tier: Triton specializes
    the argument to a constant, which changes the tile order and puts the shape
    outside every matcher's envelope.

    Before, that unclaimed dot reached applyFullConversion, where a decline does
    not raise — it segfaults the process during the failed conversion's
    rollback. The assertion is therefore that the caller SURVIVES with a named
    error, not merely that compilation failed."""
    a = torch.rand(16, 16, dtype=torch.float32, device="mps")
    b = torch.rand(16, 1, dtype=torch.float32, device="mps")
    c = torch.zeros(16, 1, dtype=torch.float32, device="mps")
    with pytest.raises(Exception):
        _dot_runtime_extent_one_kernel[(1, 1)](a, b, c, 16, 1, 16, BM=16, BN=16, BK=16, num_warps=1)
    assert "no matmul lowering matched this tl.dot" in capfd.readouterr().err

    # The process must still be able to compile: a rejection that poisons the
    # context is no better than the crash it replaced.
    a2 = torch.rand(16, 16, dtype=torch.float32, device="mps")
    b2 = torch.rand(16, 16, dtype=torch.float32, device="mps")
    c2 = torch.zeros(16, 16, dtype=torch.float32, device="mps")
    _dot_runtime_extent_one_kernel[(1, 1)](a2, b2, c2, 16, 16, 16, BM=16, BN=16, BK=16, num_warps=1)
    torch.mps.synchronize()
    torch.testing.assert_close(c2.cpu(), a2.cpu() @ b2.cpu(), atol=2e-4, rtol=2e-4)


# --- BK > BN must not let the A tile define the output's geometry -----------
#
# `findLargestRank2Tile` returns the (thread, iv) -> element bijection that
# every op in the kernel shares, including the STORE's address decomposition.
# It picked the largest rank-2 tile in the module — and a matmul's A tile is
# `BM x BK`, which beats the `BM x BN` output whenever BK > BN.
#
# The store then walked its result with BK columns: an 8x8 output with BK=16 ran
# 4 tile-loop iterations decomposed `% 16` and read its 64-element result
# scratch at indices up to 127. Wrong numbers, no diagnostic. BN=8 makes
# `BK > BN` true for every BK above 8, so this reproduced at EXACT extents, not
# only ragged ones — which is why the parametrization below sweeps both.


@triton.jit
def _bk_gt_bn_matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    om = pm * BM + tl.arange(0, BM)
    on = pn * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        ok = k + tl.arange(0, BK)
        av = tl.load(a_ptr + om[:, None] * K + ok[None, :], mask=(om[:, None] < M) & (ok[None, :] < K), other=0.0)
        bv = tl.load(b_ptr + ok[:, None] * N + on[None, :], mask=(ok[:, None] < K) & (on[None, :] < N), other=0.0)
        acc += tl.dot(av, bv)
    tl.store(c_ptr + om[:, None] * N + on[None, :], acc, mask=(om[:, None] < M) & (on[None, :] < N))


@pytest.mark.parametrize("bm,bn,bk", [(8, 8, 16), (8, 8, 64), (16, 8, 32), (32, 8, 64), (16, 16, 32), (32, 16, 64)])
@pytest.mark.parametrize("ragged", [False, True])
def test_matmul_block_k_larger_than_block_n(bm, bn, bk, ragged):
    m, n, k = (bm - 1, bn - 1, bk - 1) if ragged else (bm, bn, bk)
    torch.manual_seed(bm * 100 + bn * 10 + bk)
    a = torch.randn(m, k, dtype=torch.float32)
    b = torch.randn(k, n, dtype=torch.float32)
    c = torch.zeros(m, n, dtype=torch.float32, device="mps")
    _bk_gt_bn_matmul_kernel[(triton.cdiv(m, bm), triton.cdiv(n, bn))](
        a.to("mps"), b.to("mps"), c, m, n, k, BM=bm, BN=bn, BK=bk, num_warps=1
    )
    torch.mps.synchronize()
    torch.testing.assert_close(c.cpu(), a @ b, atol=2e-4, rtol=2e-4)


@triton.jit
def _transb_single_column_kernel(a_ptr, b_ptr, c_ptr, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    om = tl.arange(0, BM)
    on = tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        ok = k + tl.arange(0, BK)
        av = tl.load(a_ptr + om[:, None] * K + ok[None, :], mask=(om[:, None] < M) & (ok[None, :] < K), other=0.0)
        bnk = tl.load(b_ptr + on[:, None] * K + ok[None, :], mask=(on[:, None] < N) & (ok[None, :] < K), other=0.0)
        acc = tl.dot(av, tl.trans(bnk), acc)
    tl.store(c_ptr + om[:, None] * N + on[None, :], acc, mask=(om[:, None] < M) & (on[None, :] < N))


@pytest.mark.parametrize("bm,bn,bk", [(32, 16, 16), (32, 16, 64), (64, 32, 32), (16, 16, 16), (8, 8, 8)])
def test_transb_single_runtime_column(bm, bn, bk):
    """N == 1 flips the tile to column-major (Triton specializes the argument
    to a constant), and ScalarDotLowering decomposed the flat position
    row-major regardless: the dot indexed A and B with `/BN, %BN` while the
    store addressed with `%BM, /BM`, so every lane computed a different element
    than the one it stored."""
    m, n, k = bm, 1, bk
    torch.manual_seed(bm * 31 + bn)
    a = torch.randn(m, k, dtype=torch.float32)
    b = torch.randn(k, n, dtype=torch.float32)
    c = torch.zeros(m, n, dtype=torch.float32, device="mps")
    _transb_single_column_kernel[(1, 1)](
        a.to("mps"), b.t().contiguous().to("mps"), c, m, n, k, BM=bm, BN=bn, BK=bk, num_warps=1
    )
    torch.mps.synchronize()
    torch.testing.assert_close(c.cpu(), a @ b, atol=2e-4, rtol=2e-4)


@triton.jit
def _multiwarp_multitile_matmul_kernel(
    a_ptr, b_ptr, c_ptr, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr
):
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    om = pm * BM + tl.arange(0, BM)
    on = pn * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        ok = k + tl.arange(0, BK)
        av = tl.load(a_ptr + om[:, None] * K + ok[None, :], mask=(om[:, None] < M) & (ok[None, :] < K), other=0.0)
        bv = tl.load(b_ptr + ok[:, None] * N + on[None, :], mask=(ok[:, None] < K) & (on[None, :] < N), other=0.0)
        acc += tl.dot(av, bv)
    tl.store(c_ptr + om[:, None] * N + on[None, :], acc, mask=(om[:, None] < M) & (on[None, :] < N))


@pytest.mark.parametrize("num_warps", [1, 2, 4])
def test_multiwarp_multitile_matmul_is_deterministic(num_warps):
    """The tile loop's TRIP COUNT and its INDEX DECOMPOSITION have to describe
    the same tile. They came from two different walkers, and once one preferred
    the output tile and the other still took the largest, a num_warps>=2 matmul
    took E=32 from the A tile and `flat = tid*8 + iv` from the 32x16 output — so
    the loop ran four times too many and read its 512-element result scratch
    past the end.

    That failed on roughly three launches in four, which reads as a race and is
    not one, so this repeats the launch: a single run passes too often to catch
    it."""
    bm, bn, bk = 32, 16, 64
    m, n, k = 64, 32, 192
    torch.manual_seed(0)
    a = torch.rand(m, k, dtype=torch.float32)
    b = torch.rand(k, n, dtype=torch.float32)
    ref = a @ b
    a_mps, b_mps = a.to("mps"), b.to("mps")
    for _ in range(8):
        c = torch.zeros(m, n, dtype=torch.float32, device="mps")
        _multiwarp_multitile_matmul_kernel[(triton.cdiv(m, bm), triton.cdiv(n, bn))](
            a_mps, b_mps, c, m, n, k, BM=bm, BN=bn, BK=bk, num_warps=num_warps
        )
        torch.mps.synchronize()
        torch.testing.assert_close(c.cpu(), ref, atol=2e-4, rtol=2e-4)


# --- int8 x int8 -> i32 ------------------------------------------------------
#
# An integer `tl.dot` was rejected: every dot path checked for an f32 result, so
# the operands' `#ttg.dot_op` conversions survived to dialect conversion and
# failed there. It rides the scalar_dot K-loop, which is shape-generic — the
# simdgroup fast path stays float-only because Apple's matrix units are.
#
# The comparison is exact: int8 operands in [-8, 8) over K <= 64 cannot overflow
# i32, so any difference from torch is a real error and not rounding.


@triton.jit
def _int8_dot_kernel(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    am = tl.arange(0, M)[:, None] * K + tl.arange(0, K)[None, :]
    bm = tl.arange(0, K)[:, None] * N + tl.arange(0, N)[None, :]
    cm = tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :]
    tl.store(C + cm, tl.dot(tl.load(A + am), tl.load(B + bm), out_dtype=tl.int32))


@pytest.mark.parametrize("num_warps", [1, 4])
@pytest.mark.parametrize("M, N, K", [(16, 16, 16), (32, 32, 32), (16, 32, 64), (64, 16, 16), (32, 16, 8)])
def test_int8_dot_matches_torch_exactly(M, N, K, num_warps):
    torch.manual_seed(M * N + K)
    a = torch.randint(-8, 8, (M, K), dtype=torch.int8)
    b = torch.randint(-8, 8, (K, N), dtype=torch.int8)
    c = torch.zeros(M, N, dtype=torch.int32, device="mps")
    _int8_dot_kernel[(1,)](a.to("mps"), b.to("mps"), c, M, N, K, num_warps=num_warps)
    torch.mps.synchronize()
    assert torch.equal(c.cpu(), a.to(torch.int32) @ b.to(torch.int32))


# --- 3-D batched dot ---------------------------------------------------------
#
# A batched `tl.dot` uses the BxMxN result tile as both the function-loop and
# scalar-dot geometry. This remains true when K is the largest dimension and
# either BxMxK or BxKxN would otherwise win a largest-tensor heuristic.


@triton.jit
def _batched_dot_kernel(A, B, C, BATCH: tl.constexpr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    b = tl.arange(0, BATCH)[:, None, None]
    am = b * (M * K) + tl.arange(0, M)[None, :, None] * K + tl.arange(0, K)[None, None, :]
    bm = b * (K * N) + tl.arange(0, K)[None, :, None] * N + tl.arange(0, N)[None, None, :]
    cm = b * (M * N) + tl.arange(0, M)[None, :, None] * N + tl.arange(0, N)[None, None, :]
    tl.store(C + cm, tl.dot(tl.load(A + am), tl.load(B + bm)))


@pytest.mark.parametrize(
    "BATCH, M, N, K",
    [
        (2, 16, 16, 16),
        (2, 32, 32, 32),
        (4, 32, 32, 32),
        (2, 64, 16, 16),
        (8, 16, 16, 16),
        (2, 32, 16, 64),
        (2, 16, 32, 64),
    ],
)
def test_batched_dot_matches_torch(BATCH, M, N, K):
    torch.manual_seed(BATCH * 100 + M + K)
    a = torch.rand(BATCH, M, K, dtype=torch.float32)
    b = torch.rand(BATCH, K, N, dtype=torch.float32)
    c = torch.zeros(BATCH, M, N, dtype=torch.float32, device="mps")
    _batched_dot_kernel[(1,)](a.to("mps"), b.to("mps"), c, BATCH, M, N, K)
    torch.mps.synchronize()
    torch.testing.assert_close(c.cpu(), torch.bmm(a, b), atol=1e-4, rtol=1e-4)
