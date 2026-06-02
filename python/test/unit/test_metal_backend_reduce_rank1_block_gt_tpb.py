"""Rank-1 tt.reduce on the Metal backend with BLOCK > tpb (Wall 14).

Tutorial02 autotune sweep produces BLOCK_SIZE = next_power_of_2(n_cols) up
to 16384. With num_warps=8 and threads_per_warp=32, tpb=256, so the new
envelope covers E = BLOCK/tpb up to 64. This file exercises the bounded-
unroll path at three representative shapes:

  (N=2048,  BLOCK=2048)    — E=8     simplest BLOCK > tpb
  (N=4097,  BLOCK=8192)    — E=32    masked (n_cols < BLOCK)
  (N=12672, BLOCK=16384)   — E=64    autotune endpoint

Both `tl.sum` and `tl.max` combiners are exercised ⇒ 6 sub-tests.

See .omc/plans/tutorial02-wall14-block-gt-tpb-consensus.md AC8.
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
def reduce_sum_rank1_masked_kernel(
    x_ptr,
    out_ptr,
    N,
    BLOCK: tl.constexpr,
):
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    s = tl.sum(x, axis=0)
    tl.store(out_ptr + 0, s)


@triton.jit
def reduce_max_rank1_masked_kernel(
    x_ptr,
    out_ptr,
    N,
    BLOCK: tl.constexpr,
):
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    NEG_INF = float("-inf")
    x = tl.load(x_ptr + offs, mask=mask, other=NEG_INF)
    s = tl.max(x, axis=0)
    tl.store(out_ptr + 0, s)


_NUM_WARPS = 8  # threads_per_block = num_warps * warp_size = 8 * 32 = 256


_SHAPES = [
    # (N, BLOCK) — N is the actual length; BLOCK is next_power_of_2(N).
    (2048, 2048),    # E = 2048 / 256 = 8
    (4097, 8192),    # E = 8192 / 256 = 32, masked (n_cols < BLOCK)
    (12672, 16384),  # E = 16384 / 256 = 64, autotune endpoint
]


@pytest.mark.parametrize("N,BLOCK", _SHAPES)
def test_reduce_sum_rank1_f32_block_gt_tpb(N, BLOCK):
    torch.manual_seed(0xC0FFEE)
    x = torch.randn((N,), dtype=torch.float32).contiguous()
    out = torch.zeros((1,), dtype=torch.float32).contiguous()
    reduce_sum_rank1_masked_kernel[(1, 1, 1)](
        x, out, N, BLOCK=BLOCK, num_warps=_NUM_WARPS
    )
    expected = torch.sum(x.cpu())
    torch.testing.assert_close(
        out[0].item(), expected.item(), atol=1e-3, rtol=1e-3
    )


@pytest.mark.parametrize("N,BLOCK", _SHAPES)
def test_reduce_max_rank1_f32_block_gt_tpb(N, BLOCK):
    torch.manual_seed(0xC0FFEE)
    x = torch.randn((N,), dtype=torch.float32).contiguous()
    out = torch.zeros((1,), dtype=torch.float32).contiguous()
    reduce_max_rank1_masked_kernel[(1, 1, 1)](
        x, out, N, BLOCK=BLOCK, num_warps=_NUM_WARPS
    )
    expected = torch.max(x.cpu())
    torch.testing.assert_close(
        out[0].item(), expected.item(), atol=1e-5, rtol=1e-5
    )
