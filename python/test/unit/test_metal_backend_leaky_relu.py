"""End-to-end: leaky_relu via tl.where on the Metal backend.

Verifies that `tl.where(x > 0, x, 0.01 * x)` — which lowers to
`arith.cmpf` + `arith.select` on tensors after the TritonGPU pipeline —
compiles to MSL and dispatches on the Metal device. Sourced from
`python/test/unit/fixtures/metal_leet/easy-leaky_Relu.py`.

Acceptance bar (per the implementation notes):
- Kernel compiles to MSL (compile-only path; always runs when pybind
  module is built).
- Kernel dispatches without error and produces a finite fp32 output of
  the correct shape (runtime path; gated by MPS availability).
- Qualitative leaky-relu behavior: positive inputs pass through, negative
  inputs are scaled by ~0.01.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

libmetal = pytest.importorskip(
    "triton._C.libtriton.metal",
    reason="Metal backend pybind module not built into libtriton",
)


@triton.jit
def leaky_relu_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    output = tl.where(x > 0, x, 0.01 * x)
    tl.store(output_ptr + offsets, output, mask=mask)


def test_leaky_relu_compiles_to_msl():
    """Compile-only: tl.where lowers through the Metal conversion pipeline to MSL."""
    signature = {
        "input_ptr": "*fp32",
        "output_ptr": "*fp32",
        "n_elements": "i32",
        "BLOCK_SIZE": "constexpr",
    }
    src = ASTSource(
        fn=leaky_relu_kernel,
        signature=signature,
        constexprs={"BLOCK_SIZE": 1024},
    )
    target = GPUTarget(backend="metal", arch=80, warp_size=32)
    compiled = triton.compile(src, target=target, options={"num_warps": 4})

    raw = compiled.asm["metal"]
    msl = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    assert msl, "MSL output is empty"
    assert "kernel void leaky_relu_kernel" in msl, (
        f"MSL missing kernel entry point.\n--- MSL ---\n{msl}\n"
    )
    # tl.where lowers to a ternary; MSL emits `(cond ? a : b)`.
    assert "?" in msl and ":" in msl, (
        f"MSL missing ternary operator for tl.where.\n--- MSL ---\n{msl}\n"
    )


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="Metal backend requires an MPS-enabled PyTorch (Apple Silicon)",
)
def test_leaky_relu_dispatches_and_output_is_sane():
    """Runtime: dispatch leaky_relu and check finiteness + qualitative behavior."""
    N = 4096
    BLOCK_SIZE = 1024
    torch.manual_seed(0xC0FFEE)
    # Mix of positive and negative inputs to exercise both branches.
    x = (torch.rand((N,), dtype=torch.float32) - 0.5) * 2.0
    x = x.contiguous()
    out = torch.zeros((N,), dtype=torch.float32).contiguous()

    grid = (triton.cdiv(N, BLOCK_SIZE),)
    leaky_relu_kernel[grid](x, out, N, BLOCK_SIZE=BLOCK_SIZE)

    assert out.shape == (N,), f"shape mismatch: {out.shape}"
    assert torch.isfinite(out).all(), "non-finite values in output"

    # Qualitative leaky-relu: positives pass through, negatives scaled.
    expected = torch.where(x > 0, x, 0.01 * x)
    # Loose tolerance — the spec asks for sanity, not bit-exactness.
    torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)
