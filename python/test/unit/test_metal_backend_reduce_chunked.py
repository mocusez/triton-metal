"""Rank-2 axis=1 reduce — `(M, N)` sweep including M*N >> tpb shapes.

Exercises `ReduceLowering`'s rank-2 axis=1 body across a sweep of `(M, N)`
shapes. The kernel sums a 2D `(M, N)` tile along axis=1 and writes the
resulting `(M,)` vector.

The L3a-tileloop-2 redesign makes the body self-contained: it walks back to
the producing `tt.load` and reduces each row directly from device memory into
a per-row threadgroup buffer (`rowBuf[M]`), hoisted above the FuncOpLowering
tile loop so it runs once. This removed the old `M*N == tpb` assumption (and
the old chunked body's threadgroup over-allocation), so every shape below —
including the large `M*N >> tpb` ones that were previously xfail — now produces
correct results.
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
def reduce_sum_axis1_kernel(
    x_ptr,
    out_ptr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    addr = offs_m[:, None] * BLOCK_N + offs_n[None, :]
    x = tl.load(x_ptr + addr)
    s = tl.sum(x, axis=1)
    tl.store(out_ptr + offs_m, s)


@pytest.mark.parametrize(
    "M, N",
    [
        # In-budget shape: M*N == tpb (the single-(row,col)-per-thread case).
        (8, 16),     # 128 B, M*N == 128 == tpb (4 warps × 32).
        # M*N >> tpb shapes. Previously xfail (the old chunked body assumed
        # M*N == tpb and produced wrong results / over-allocated threadgroup
        # memory). The L3a-tileloop-2 redesign reduces each row directly from
        # device memory into a per-row threadgroup buffer, so these now pass.
        (1024, 8),
        (1024, 16),
        (1024, 64),    # matches conv1d's reduce shape.
        (2048, 64),
        (512, 128),
    ],
)
def test_reduce_chunked_axis1_f32(M, N):
    torch.manual_seed(0xC0FFEE)
    x = torch.randn((M, N), dtype=torch.float32).contiguous()
    out = torch.zeros((M,), dtype=torch.float32).contiguous()
    reduce_sum_axis1_kernel[(1, 1, 1)](
        x, out, BLOCK_M=M, BLOCK_N=N
    )
    expected = torch.sum(x.cpu(), dim=1).to(torch.float32)
    torch.testing.assert_close(out, expected, atol=1e-4, rtol=1e-4)
