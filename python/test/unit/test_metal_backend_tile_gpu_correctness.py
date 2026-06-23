"""GPU correctness for BLOCK_SIZE > threads_per_block (tile loop).

The block-size-loop session shipped IR + MSL emission for the outer
scf.for; the standard-launch session shipped the @triton.jit -> GPU
path. This pytest closes the loop by asserting that the canonical
vector_add produces bit-exact correct output on real Apple GPU
hardware across multiple tile sizes (elem_per_thread = 1, 8, 16).

See `.omc/specs/deep-interview-metal-tile-gpu-correctness.md`.
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


# BLOCK_SIZE → elem_per_thread mapping under 128 threads per block:
#   128  -> E=1  (degenerate; no tile loop emitted)
#   1024 -> E=8  (matches the canonical Triton sizePerThread=[1] strided layout)
#   2048 -> E=16 (further stress of the outer scf.for trip count)
@pytest.mark.parametrize("BLOCK_SIZE", [128, 1024, 2048])
def test_tile_gpu_correctness(BLOCK_SIZE):
    N = BLOCK_SIZE - 24  # leave a non-empty masked tail at every shape
    torch.manual_seed(0xC0FFEE)
    x = torch.rand(BLOCK_SIZE, dtype=torch.float32)
    y = torch.rand(BLOCK_SIZE, dtype=torch.float32)
    out = torch.zeros(BLOCK_SIZE, dtype=torch.float32)

    add_kernel[(1, 1, 1)](x, y, out, N, BLOCK_SIZE=BLOCK_SIZE)

    expected = x + y
    # Active region (first N): bit-exact equivalence to CPU float-add.
    torch.testing.assert_close(out[:N], expected[:N], atol=0, rtol=0)
