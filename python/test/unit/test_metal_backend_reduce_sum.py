"""Session L3a: axis=1 sum reduce on the Metal backend (f32 + i32).

Exercises the new `ReduceLowering` row-scan body produced by the
`convert-tritongpu-to-metal` pass. The kernel loads a 2D `(M, N)` block,
calls `tl.sum(x, axis=1)`, and stores the resulting `(M,)` vector. We
compare against `torch.sum(input, dim=1)` (bit-exact for i32, with
small tolerance for f32 ordering effects).

See `.omc/specs/deep-interview-leet-triton-l3a-reduce-body-axis1.md`.
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
if not hasattr(libmetal, "launch_kernel_with_pipeline"):
    pytest.skip(
        "Metal runtime not compiled (non-Darwin build or Xcode CLT absent)",
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


# Honest-divergence note (per L3a spec §"Honest divergence policy"): the
# spec calls for i32 reduce coverage alongside f32. The reduce body
# itself supports i32 (see lit `reduce_sum_axis.mlir` Section 2). End-to-
# end i32 kernels however require `tl.load` / `tl.store` on i32 pointers,
# which the backend explicitly defers to Session L2b (`test_metal_backend
# _int_arith.py` documents: "uint8/i32 data path is deferred"). We mark
# the i32 cases as xfail so the parametrize matrix matches the spec
# without falsely claiming i32 i/o works.
_I32_XFAIL_REASON = (
    "i32 tt.load/tt.store deferred to L2b; reduce body itself handles i32 "
    "per lit reduce_sum_axis.mlir Section 2 (Metal_Type's signless-i32 "
    "exclusion bridged via ui32 storage in ReduceLowering)"
)


@pytest.mark.parametrize(
    "M, N, dtype",
    [
        (8, 16, torch.float32),
        (4, 32, torch.float32),
        pytest.param(
            8, 16, torch.int32,
            marks=pytest.mark.xfail(reason=_I32_XFAIL_REASON, strict=True),
        ),
        pytest.param(
            4, 32, torch.int32,
            marks=pytest.mark.xfail(reason=_I32_XFAIL_REASON, strict=True),
        ),
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
