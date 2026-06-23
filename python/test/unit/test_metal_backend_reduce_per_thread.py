"""Per-thread-owned axis=1 sum reduce on the Metal backend (M*N >> tpb).

Exercises the rank-2 axis=1 `ReduceLowering` body on the conv1d-style layout
where the reduce axis is fully serial within each thread
(`threadsPerCTA[axis_dim] == 1`) and `M*N >> tpb`.

The L3a-tileloop-2 redesign makes the body self-contained: it walks back to
the producing `tt.load` and reduces each row directly from device memory into
a per-row threadgroup buffer (`rowBuf[M]`) hoisted above the tile loop. This
closed the former carry-forward gap (the old body gave one scalar per
(thread, tile-iv) pair and could not express the per-row gather), so the
multi-element-per-thread cases below now produce correct results.
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
def reduce_per_thread_kernel(
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
        (1024, 64),
        (512, 32),
        (256, 128),
    ],
)
def test_reduce_per_thread_owned_f32(M, N):
    torch.manual_seed(0xC0FFEE)
    x = torch.randn((M, N), dtype=torch.float32).contiguous()
    out = torch.zeros((M,), dtype=torch.float32).contiguous()
    reduce_per_thread_kernel[(1, 1, 1)](x, out, BLOCK_M=M, BLOCK_N=N)
    expected = torch.sum(x.cpu(), dim=1)
    torch.testing.assert_close(out, expected, atol=1e-3, rtol=1e-3)
