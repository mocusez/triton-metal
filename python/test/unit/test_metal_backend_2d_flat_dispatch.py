"""2D-shaped torch tensors via the existing 1D kernel + launcher.

Confirmation test for `.omc/specs/deep-interview-metal-2d-flat-dispatch.md`.
The MetalLauncher's `tensor.numpy().tobytes()` flattens any-rank tensors;
the d2h side reshapes via `np.frombuffer(...).reshape(tensor.shape)`. So
2D-shaped inputs round-trip transparently through the 1D kernel without
any new lowerings. Full canonical 2D Triton support (tt.expand_dims,
tt.broadcast, 2D #blocked layout in TTGIR) is the next slice.
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


def test_2d_shaped_tensors_via_flat_dispatch():
    M, N = 8, 16  # M*N = 128 fits in a single BLOCK_SIZE=128 threadgroup
    n_elements = M * N
    torch.manual_seed(0xC0FFEE)
    x = torch.rand(M, N, dtype=torch.float32)
    y = torch.rand(M, N, dtype=torch.float32)
    out = torch.zeros(M, N, dtype=torch.float32)

    # 1D grid + 1D kernel; the 2D tensor shape is preserved by the
    # launcher's flatten/reshape round-trip.
    add_kernel[(1, 1, 1)](x, y, out, n_elements, BLOCK_SIZE=128)

    assert out.shape == (M, N), f"shape changed: {out.shape}"
    assert out.dim() == 2
    torch.testing.assert_close(out, x + y, atol=0, rtol=0)
