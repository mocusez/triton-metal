"""End-to-end f32 canonical-dot regression tests for 8x8 and 16x16 tiles.

Both cases pass. Together they lock the modern SIMD-group matrix emitter, the
canonical three-iter-arg loop unroller, staged device loads, accumulator
initialization, and launcher argument filtering. The resolved failure history
is summarized next to the tests instead of being presented as current status.
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
def dot_8x8_f32_kernel(
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
    K_TILES: tl.constexpr,  # static trip count for v4 canonical unroller
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    # Loop bound is constexpr (K_TILES) — required by v4 canonical-3-iter_arg unroller
    # which gates on static trip count at TritonGPUToMetal.cpp:3119.
    for _ in range(0, K_TILES):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc)


# iter-8 ship state (2026-05-20):
# - [8,8,8]: PASS
# - [16,16,16]: PASS (xfail removed)
#
# Root cause of the residual iter-7 failure (resolved):
# `third_party/metal/backend/driver.py`'s `MetalLauncher` built
# `_arg_types` by filtering both constexpr AND specialized args (Triton
# inlines `stride=1` as a constant and drops it from the kernel signature),
# but then iterated `zip(args, self._arg_types)` over the FULL positional
# args list — Python's zip silently truncated and mis-aligned the args
# with MSL buffer slots, so `v4` (semantically stride_bk) was being bound
# to the runtime value of stride_ak=1 instead. The simdgroup_load for B
# then ran with stride=1 instead of the real stride, reading wrong data
# and dropping iter-1's contribution at specific output columns. Fixed by
# introducing `MetalLauncher._arg_mask` and filtering `args` by it before
# the zip so positional alignment with `_arg_types` is preserved.
#
# Supporting iter-8 improvements that landed alongside the launcher fix
# (correct and necessary for multi-threadgroup pid-driven origin handling):
# - `findOriginScalarInPtrChain` / `findStrideSplatSourceInPtrChain` walk
#   Triton's canonical 2-level matmul addptr chain
#   `addptr(broadcast(addptr(splat(arg), inner_off)), outer_off)`
#   (TritonGPUToMetal.cpp:3028-3084).
# - `metal.simdgroup_matrix_zero` op emits `simdgroup_<elem><M>x<N>(0.0f)`
#   constructor init for the accumulator (Apple's matmul/FA pattern).
# - `metal.simdgroup_load_device_staged` op stages A/B tiles through
#   threadgroup memory before simdgroup_load (Apple's FA pattern).
# - SimdgroupMultiplyAccumulateOp emits `sgmma(acc, A, B, acc)` in-place
#   accumulator pattern.


@pytest.mark.parametrize(
    "M, N, K",
    [
        pytest.param(8, 8, 8),
        pytest.param(16, 16, 16),
    ],
)
def test_dot_f32_8x8(M, N, K):
    """Canonical 8x8 tile with one or two K tiles, f32 throughout.

    On failure, TRITON_REPRODUCER_PATH=/tmp/dot-8x8-f32-fail.mlir captures the
    MLIR reproducer.
    """
    os.environ.setdefault(
        "TRITON_REPRODUCER_PATH", "/tmp/dot-8x8-f32-fail.mlir"
    )
    torch.manual_seed(0xC0FFEE)
    a = torch.randn((M, K), dtype=torch.float32).contiguous()
    b = torch.randn((K, N), dtype=torch.float32).contiguous()
    c = torch.zeros((M, N), dtype=torch.float32).contiguous()
    grid = (triton.cdiv(M, 8), triton.cdiv(N, 8))
    K_TILES = (K + 8 - 1) // 8  # 8→1, 16→2; both ∈ [1, 8] (v4 trip-count bound)
    dot_8x8_f32_kernel[grid](
        a, b, c,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=8, BLOCK_N=8, BLOCK_K=8,
        K_TILES=K_TILES,
    )
    expected = torch.matmul(a, b)
    torch.testing.assert_close(c, expected, atol=1e-5, rtol=1e-5)
