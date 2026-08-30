"""Session L3a: axis=1 sum reduce on the Metal backend (f32 + i32).

Exercises the new `ReduceLowering` row-scan body produced by the
`convert-tritongpu-to-metal` pass. The kernel loads a 2D `(M, N)` block,
calls `tl.sum(x, axis=1)`, and stores the resulting `(M,)` vector. We
compare against `torch.sum(input, dim=1)` (bit-exact for i32, with
small tolerance for f32 ordering effects).

See the implementation notes.
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
    # axis=1 reduce: tensor<BLOCK_M x BLOCK_N> -> tensor<BLOCK_M>
    s = tl.sum(x, axis=1)
    tl.store(out_ptr + offs_m, s)


# L2b (2026-06-03): end-to-end i32 reduce now works. `tt.load` / `tt.store`
# on i32 pointers are supported by routing the i32 data buffer through ui32
# storage (memref element + threadgroup scratch) and bridging back to signless
# i32 for arith via builtin.unrealized_conversion_cast — see
# the implementation notes. The reduce body already
# handled i32 (lit `reduce_sum_axis.mlir` Section 2); the load/store path was
# the only gap, so the i32 cases below are now plain (non-xfail) passes.


@pytest.mark.parametrize(
    "M, N, dtype",
    [
        (8, 16, torch.float32),
        (4, 32, torch.float32),
        (8, 16, torch.int32),
        (4, 32, torch.int32),
    ],
)
def test_reduce_sum_axis1(M, N, dtype):
    torch.manual_seed(0xC0FFEE)
    if dtype is torch.float32:
        x = torch.randn((M, N), dtype=dtype).contiguous()
    else:
        x = torch.randint(-100, 100, (M, N), dtype=dtype).contiguous()
    out = torch.zeros((M,), dtype=dtype).contiguous()
    reduce_sum_axis1_kernel[(1, 1, 1)](
        x, out, BLOCK_M=M, BLOCK_N=N
    )
    expected = torch.sum(x.cpu(), dim=1).to(dtype)
    if dtype is torch.float32:
        torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)
    else:
        assert torch.equal(out, expected), (
            f"i32 reduce mismatch: out={out} expected={expected}"
        )
