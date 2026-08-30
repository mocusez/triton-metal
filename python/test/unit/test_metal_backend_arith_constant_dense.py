"""arith.constant dense<C> (splat) via tl.full.

Acceptance test for AC.OPS4 in
the implementation notes.
Compiles + launches a Triton kernel that uses `tl.full([BLOCK_SIZE],
7.0, dtype=tl.float32)`. The Triton frontend emits this as
`arith.constant dense<7.000000e+00> : tensor<128xf32, #blocked>`. The
new `ArithConstantDenseLowering` pattern collapses the dense splat to a
scalar `arith.constant 7.0 : f32`. Result: `out == x + 7.0` bit-exact.
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
def add_kernel_with_bias(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    bias = tl.full([BLOCK_SIZE], 7.0, dtype=tl.float32)
    tl.store(output_ptr + offsets, x + bias, mask=mask)


def test_arith_constant_dense_via_tl_full():
    N = 100
    BLOCK_SIZE = 128  # N < BLOCK_SIZE exercises the masked tail too
    torch.manual_seed(0xC0FFEE)
    x = torch.rand(BLOCK_SIZE, dtype=torch.float32)
    out = torch.zeros(BLOCK_SIZE, dtype=torch.float32)

    add_kernel_with_bias[(1, 1, 1)](x, out, N, BLOCK_SIZE=BLOCK_SIZE)

    expected = x + 7.0
    # Active region (first N): bit-exact equivalence.
    torch.testing.assert_close(out[:N], expected[:N], atol=0, rtol=0)
