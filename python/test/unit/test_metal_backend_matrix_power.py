"""Matrix power (repeated squaring) on the Metal backend.

Runs the verbatim `leet-triton/medium-matrix_power.py` kernel: a tiled `tt.dot`
(BLOCK_W/H=64, BLOCK_K=32) accumulating over a *static-bound, tile-indexed*
K-loop, stored directly with a masked `tt.store` (NO alpha/beta epilogue).

This bare `dot -> tt.store` loop matches none of the fast SIMD-group matchers
(canonical-3-iter_arg / runtime-K recompute / 8x8 unroll all reject it: static
`cdiv(n, 32)` bound, `k*BK + arange` tile-index K addressing, 64x64 masked
tiles). It is the total correctness fallback: after Tier 1 declines, the late
`finalizeScalarDots` sweep collapses the whole reduction to a single
`metal.scalar_dot` over `[0, stride_a)`. See `tryScalarDotLoopFallback`
(`allowStoreConsumer=true`) / `finalizeScalarDots` in TritonGPUToMetal.cpp.
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


# Verbatim copy of leet-triton/medium-matrix_power.py.
@triton.jit
def matmul_kernel(A, B, C, n: tl.constexpr, BLOCK_SIZE_W: tl.constexpr,
                  BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_K: tl.constexpr):
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)
    offsets_i = BLOCK_SIZE_W * pid_i + tl.arange(0, BLOCK_SIZE_W)[:, None]
    offsets_j = BLOCK_SIZE_H * pid_j + tl.arange(0, BLOCK_SIZE_H)[None, :]
    C_tile = tl.zeros((BLOCK_SIZE_W, BLOCK_SIZE_H), dtype=tl.float32)
    for k in range(tl.cdiv(n, BLOCK_SIZE_K)):
        offsets_k = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        A_rows = tl.load(A + offsets_i * n + offsets_k[None, :],
                         mask=(offsets_i < n) & (offsets_k[None, :] < n), other=0.0)
        B_cols = tl.load(B + offsets_k[:, None] * n + offsets_j,
                         mask=(offsets_j < n) & (offsets_k[:, None] < n), other=0.0)
        C_tile = tl.dot(A_rows, B_cols, acc=C_tile)
    tl.store(C + offsets_i * n + offsets_j, C_tile,
             mask=(offsets_i < n) & (offsets_j < n))


def solve(input: torch.Tensor, output: torch.Tensor, N: int, P: int):
    if P == 1:
        output.copy_(input)
        return
    tmp1 = torch.zeros_like(input)
    solve(input, tmp1, N, P // 2)
    BLOCK_SIZE_W = 64
    BLOCK_SIZE_H = 64
    BLOCK_SIZE_K = 32
    grid = (triton.cdiv(N, BLOCK_SIZE_W), triton.cdiv(N, BLOCK_SIZE_H))
    if P % 2 == 0:
        matmul_kernel[grid](tmp1, tmp1, output, N, BLOCK_SIZE_W, BLOCK_SIZE_H, BLOCK_SIZE_K)
    else:
        tmp2 = torch.zeros_like(input)
        matmul_kernel[grid](tmp1, tmp1, tmp2, N, BLOCK_SIZE_W, BLOCK_SIZE_H, BLOCK_SIZE_K)
        matmul_kernel[grid](tmp2, input, output, N, BLOCK_SIZE_W, BLOCK_SIZE_H, BLOCK_SIZE_K)


def _run(N, P, *, seed=0):
    torch.manual_seed(seed)
    # Scale by 1/sqrt(N) so repeated squaring stays well within fp32 range.
    inp = (torch.randn((N, N), dtype=torch.float32) / (N ** 0.5)).contiguous()
    out = torch.zeros_like(inp)
    solve(inp, out, N, P)
    ref = torch.matrix_power(inp, P)
    return out, ref


@pytest.mark.parametrize(
    "N,P",
    [
        pytest.param(64, 1, id="single_tile_p1"),
        pytest.param(64, 2, id="single_tile_p2"),
        pytest.param(64, 3, id="single_tile_p3_odd"),
        pytest.param(64, 5, id="single_tile_p5"),
        pytest.param(64, 8, id="single_tile_p8"),
        pytest.param(64, 16, id="single_tile_p16"),
        pytest.param(128, 3, id="multi_tile_p3"),
        pytest.param(128, 5, id="multi_tile_p5"),
        pytest.param(256, 4, id="multi_tile_p4_large"),
        pytest.param(100, 3, id="ragged_p3"),
        pytest.param(70, 3, id="ragged_small_p3"),
        pytest.param(96, 4, id="ragged_96_p4"),
    ],
)
def test_metal_matrix_power(N, P):
    out, ref = _run(N, P)
    # f32 accumulate; error grows with P (chained matmuls) and N (reduction len).
    tol = max(1e-3, P * N * (2.0 ** -20))
    torch.testing.assert_close(out, ref, atol=tol, rtol=tol)
