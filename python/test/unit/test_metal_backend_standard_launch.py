"""End-to-end: standard `add_kernel[grid](x, y, out, ...)` on Metal with torch.

Acceptance test for `.omc/specs/deep-interview-metal-standard-launch.md`
(AC.L4 + AC.L5). Requires torch + Darwin libmetal runtime.
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
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)


def test_standard_launch_vector_add():
    N = 100
    BLOCK_SIZE = 128  # >= N so grid=(1,1,1) is sufficient
    torch.manual_seed(0xC0FFEE)
    x = torch.rand(BLOCK_SIZE, dtype=torch.float32)
    y = torch.rand(BLOCK_SIZE, dtype=torch.float32)
    out = torch.zeros(BLOCK_SIZE, dtype=torch.float32)

    add_kernel[(1, 1, 1)](x, y, out, N, BLOCK_SIZE=BLOCK_SIZE)

    expected = x + y
    # Active threads (first N): bit-exact.
    torch.testing.assert_close(out[:N], expected[:N], atol=0, rtol=0)


def test_standard_launch_hooks_fire():
    """AC.L5: launch_enter_hook and launch_exit_hook each invoked exactly once."""
    import triton.knobs as knobs

    N = 64
    BLOCK_SIZE = 128
    x = torch.zeros(BLOCK_SIZE, dtype=torch.float32)
    y = torch.zeros(BLOCK_SIZE, dtype=torch.float32)
    out = torch.zeros(BLOCK_SIZE, dtype=torch.float32)

    enter_count = [0]
    exit_count = [0]

    def enter_hook(metadata):
        enter_count[0] += 1

    def exit_hook(metadata):
        exit_count[0] += 1

    prior_enter = knobs.runtime.launch_enter_hook
    prior_exit = knobs.runtime.launch_exit_hook
    knobs.runtime.launch_enter_hook = enter_hook
    knobs.runtime.launch_exit_hook = exit_hook
    try:
        add_kernel[(1, 1, 1)](x, y, out, N, BLOCK_SIZE=BLOCK_SIZE)
    finally:
        knobs.runtime.launch_enter_hook = prior_enter
        knobs.runtime.launch_exit_hook = prior_exit

    assert enter_count[0] == 1, f"enter hook invoked {enter_count[0]}x, expected 1"
    assert exit_count[0] == 1, f"exit hook invoked {exit_count[0]}x, expected 1"
