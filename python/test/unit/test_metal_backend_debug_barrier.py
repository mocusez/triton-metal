"""`tl.debug_barrier()` on the Metal backend.

`tl.debug_barrier()` lowers to `ttg.barrier`, which had no conversion pattern
at all. The decline therefore happened INSIDE `applyFullConversion`, which on
this backend takes the whole process down (SIGSEGV/SIGABRT, intermittently)
rather than raising a catchable error — so every kernel containing one was
unusable and told you nothing about why.

`metal.barrier` already emits exactly the right thing,
`threadgroup_barrier(mem_flags::mem_threadgroup)`, so `BarrierLowering` is a
direct substitution. The `addrSpace` attribute is dropped: Metal's threadgroup
barrier orders threadgroup and device memory together, which is at least as
strong as any subset `ttg.barrier` can name.

The multi-warp case is the one that matters — with a single warp a barrier is
a no-op in practice, so it would pass even if the op were silently dropped.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import triton  # noqa: E402
import triton.language as tl  # noqa: E402

pytest.importorskip(
    "triton._C.libtriton.metal",
    reason="Metal backend pybind module not built into libtriton",
)
if not torch.backends.mps.is_available():
    pytest.skip(
        "Metal backend requires an MPS-enabled PyTorch (Apple Silicon)",
        allow_module_level=True,
    )


@triton.jit
def _barrier_copy_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    v = tl.load(x_ptr + i)
    tl.debug_barrier()
    tl.store(out_ptr + i, v)


@pytest.mark.parametrize("num_warps", [1, 2, 4])
@pytest.mark.parametrize("block", [64, 256])
def test_debug_barrier_compiles_and_copies(block, num_warps):
    x = torch.rand(block, device="mps")
    out = torch.zeros(block, device="mps")
    _barrier_copy_kernel[(1,)](x, out, block, num_warps=num_warps)
    torch.mps.synchronize()
    assert torch.equal(out.cpu(), x.cpu())


@triton.jit
def _barrier_only_kernel(out_ptr):
    tl.debug_barrier()
    tl.store(out_ptr, 1.0)


def test_debug_barrier_alone():
    """A barrier with no tile around it: nothing else can carry the lowering."""
    out = torch.zeros(1, device="mps")
    _barrier_only_kernel[(1,)](out)
    torch.mps.synchronize()
    assert out.cpu().item() == 1.0
