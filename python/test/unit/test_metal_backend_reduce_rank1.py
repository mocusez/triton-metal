"""Rank-1 tt.reduce on the Metal backend (f32/i32, sum/max).

All f32 rank-1 reduce cases (sum+max, BLOCK ∈ {32, 64, 128, 256, 512, 1024})
pass end-to-end via the spt-fold in `lowerRank1Reduce`.

Kernel shape: load a row of length N from a 1D `!tt.ptr`, reduce via
`tl.sum(row, axis=0)` or `tl.max(row, axis=0)`, store the resulting scalar.

Coverage:
  BLOCK_SIZE ∈ {32, 64, 128, 256, 512, 1024}
  dtype      ∈ {f32, i32}
  op         ∈ {sum, max}

Notes:
  - f32 sum+max: all 12 parametrized cases pass.
  - i32 cases: still xfail — `tt.load`/`tt.store` end-to-end for i32
    deferred to L2b. Mirrors `test_metal_backend_reduce_sum.py` i32 xfail.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import triton  # noqa: E402
import triton.language as tl  # noqa: E402

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
def reduce_sum_rank1_kernel(
    x_ptr,
    out_ptr,
    BLOCK: tl.constexpr,
):
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    s = tl.sum(x, axis=0)
    tl.store(out_ptr + 0, s)


@triton.jit
def reduce_max_rank1_kernel(
    x_ptr,
    out_ptr,
    BLOCK: tl.constexpr,
):
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    s = tl.max(x, axis=0)
    tl.store(out_ptr + 0, s)


_I32_XFAIL_REASON = (
    "i32 tt.load/tt.store end-to-end deferred to L2b — mirrors "
    "test_metal_backend_reduce_sum.py i32 xfail. The rank-1 reduce body "
    "itself supports i32 (ui32-bridged storage like the rank-2 path); "
    "only the surrounding load/store path is gated."
)


_NUM_WARPS = 8   # threads_per_block = num_warps * 32 = 256


def _block_params():
    # f32 sum+max pass for all BLOCK sizes; i32 xfails preserved (L2b).
    return [pytest.param(BLOCK) for BLOCK in [32, 64, 128, 256, 512, 1024]]


@pytest.mark.parametrize("BLOCK", _block_params())
def test_reduce_sum_rank1_f32(BLOCK):
    torch.manual_seed(0xC0FFEE)
    x = torch.randn((BLOCK,), dtype=torch.float32).contiguous()
    out = torch.zeros((1,), dtype=torch.float32).contiguous()
    reduce_sum_rank1_kernel[(1, 1, 1)](x, out, BLOCK=BLOCK, num_warps=_NUM_WARPS)
    expected = torch.sum(x.cpu())
    torch.testing.assert_close(out[0].item(), expected.item(),
                               atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("BLOCK", _block_params())
def test_reduce_max_rank1_f32(BLOCK):
    torch.manual_seed(0xC0FFEE)
    x = torch.randn((BLOCK,), dtype=torch.float32).contiguous()
    out = torch.zeros((1,), dtype=torch.float32).contiguous()
    reduce_max_rank1_kernel[(1, 1, 1)](x, out, BLOCK=BLOCK, num_warps=_NUM_WARPS)
    expected = torch.max(x.cpu())
    torch.testing.assert_close(out[0].item(), expected.item(),
                               atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("BLOCK", _block_params())
@pytest.mark.xfail(reason=_I32_XFAIL_REASON, strict=False)
def test_reduce_sum_rank1_i32(BLOCK):
    torch.manual_seed(0xC0FFEE)
    x = torch.randint(-100, 100, (BLOCK,), dtype=torch.int32).contiguous()
    out = torch.zeros((1,), dtype=torch.int32).contiguous()
    reduce_sum_rank1_kernel[(1, 1, 1)](x, out, BLOCK=BLOCK, num_warps=_NUM_WARPS)
    expected = torch.sum(x.cpu()).to(torch.int32)
    assert out[0].item() == expected.item()


@pytest.mark.parametrize("BLOCK", _block_params())
@pytest.mark.xfail(reason=_I32_XFAIL_REASON, strict=False)
def test_reduce_max_rank1_i32(BLOCK):
    torch.manual_seed(0xC0FFEE)
    x = torch.randint(-100, 100, (BLOCK,), dtype=torch.int32).contiguous()
    out = torch.zeros((1,), dtype=torch.int32).contiguous()
    reduce_max_rank1_kernel[(1, 1, 1)](x, out, BLOCK=BLOCK, num_warps=_NUM_WARPS)
    expected = torch.max(x.cpu()).to(torch.int32)
    assert out[0].item() == expected.item()
