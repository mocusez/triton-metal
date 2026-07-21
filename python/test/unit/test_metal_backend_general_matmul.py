"""General GEMM with fused epilogue on the Metal backend.

Runs the verbatim `leet-triton/medium-general_matrix_multiplication.py` kernel:
a single tiled `tt.dot` (TILE=64) feeding an `alpha*(A@B) + beta*C` epilogue,
fp16 inputs / f32 accumulate / fp16 output, with masked partial tiles.

This shape matches no SIMD-group matmul matcher (they all fuse `dot -> tt.store`
with no epilogue); it is lowered by the per-thread scalar `metal.scalar_dot`
correctness fallback and bridged into the ordinary tile-loop epilogue. See
`tryScalarDotFallback` / `ScalarDotLowering` in TritonGPUToMetal.cpp.
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


# Verbatim copy of leet-triton/medium-general_matrix_multiplication.py.
@triton.jit
def kernel(a, b, c, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
           alpha: tl.constexpr, beta: tl.constexpr, TILE_SIZE: tl.constexpr):
    bx = tl.program_id(0)
    by = tl.program_id(1)

    ar = tl.arange(0, TILE_SIZE)

    row = by * TILE_SIZE
    col = bx * TILE_SIZE

    iters = tl.cdiv(K, TILE_SIZE)
    output = (ar[:, None] * ar[None, :]) * 0.0
    output = tl.cast(output, tl.float32)
    ay_off = ar[:, None] + row
    bx_off = ar[None, :] + col

    ay_off_mask = ay_off < M
    bx_off_mask = bx_off < N

    for i in range(iters):
        ax_off = ar[None, :] + (i * TILE_SIZE)
        by_off = ar[:, None] + (i * TILE_SIZE)
        adata = tl.load(a + ay_off * K + ax_off, mask=(ax_off < K) & ay_off_mask, other=0.0)
        bdata = tl.load(b + by_off * N + bx_off, mask=bx_off_mask & (by_off < K), other=0.0)
        output = tl.dot(tl.cast(adata, tl.float32), tl.cast(bdata, tl.float32), acc=output)

    c_offset = c + ay_off * N + bx_off
    c_mask = ay_off_mask & bx_off_mask
    cdata = tl.load(c_offset, mask=c_mask, other=0.0)
    output = output * alpha + tl.cast(cdata, tl.float32) * beta
    output = tl.cast(output, tl.float16)
    tl.store(c_offset, output, mask=c_mask)


def _solve(a, b, c, M, N, K, alpha, beta):
    TILE_SIZE = 64
    grid = (triton.cdiv(N, TILE_SIZE), triton.cdiv(M, TILE_SIZE))
    kernel[grid](a, b, c, M=M, N=N, K=K, alpha=alpha, beta=beta, TILE_SIZE=TILE_SIZE)


def _run(M, N, K, alpha, beta, *, seed=0):
    torch.manual_seed(seed)
    a = torch.randn((M, K), dtype=torch.float16).contiguous()
    b = torch.randn((K, N), dtype=torch.float16).contiguous()
    c = torch.randn((M, N), dtype=torch.float16).contiguous()
    c_orig = c.clone()
    _solve(a, b, c, M, N, K, alpha, beta)
    ref = (alpha * (a.float() @ b.float()) + beta * c_orig.float()).to(torch.float16)
    return c, ref


@pytest.mark.parametrize(
    "M,N,K,alpha,beta",
    [
        pytest.param(64, 64, 64, 1.0, 0.0, id="square_ab"),
        pytest.param(64, 64, 64, 1.0, 1.0, id="square_gemm"),
        pytest.param(64, 64, 64, 2.0, 0.5, id="square_scaled"),
        pytest.param(64, 64, 32, 1.0, 1.0, id="k_lt_tile"),
        pytest.param(128, 128, 64, 1.0, 1.0, id="multi_tile"),
        pytest.param(128, 96, 64, 1.5, 0.5, id="nonsquare_n_tail"),
        pytest.param(100, 70, 50, 1.0, 1.0, id="ragged_all"),
    ],
)
def test_metal_general_matmul(M, N, K, alpha, beta):
    c, ref = _run(M, N, K, alpha, beta)
    # fp16 output; tolerance scales with K (accumulation error) + fp16 rounding.
    tol = max(1e-2, K * (2.0 ** -9))
    torch.testing.assert_close(c.float(), ref.float(), atol=tol, rtol=tol)
