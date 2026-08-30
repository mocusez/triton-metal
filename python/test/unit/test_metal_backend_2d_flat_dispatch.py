"""2D-shaped torch tensors via the existing 1D kernel + launcher.

Confirmation test for the implementation notes.
The MetalLauncher's `tensor.numpy().tobytes()` flattens any-rank tensors;
the d2h side reshapes via `np.frombuffer(...).reshape(tensor.shape)`. So
2D-shaped inputs round-trip transparently through the 1D kernel without
any new lowerings. Full canonical 2D Triton support (tt.expand_dims,
tt.broadcast, 2D #blocked layout in TTGIR) is the next slice.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

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


def _load_leet_2d_convolution():
    path = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "metal_leet"
        / "medium-2d_convolution.py"
    )
    if not path.is_file():
        pytest.skip(f"Metal Leet fixture not present: {path}")
    spec = importlib.util.spec_from_file_location("leet_triton_medium_2d_convolution", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_leet_2d_convolution_actual_file_matches_cpu_conv2d():
    torch.manual_seed(0x2D0C)
    input_rows, input_cols = 35, 41
    kernel_rows, kernel_cols = 3, 3
    output_rows = input_rows - kernel_rows + 1
    output_cols = input_cols - kernel_cols + 1

    inp = torch.randn(input_rows, input_cols, dtype=torch.float32, device="mps").contiguous()
    ker = torch.randn(kernel_rows, kernel_cols, dtype=torch.float32, device="mps").contiguous()
    out = torch.empty(output_rows, output_cols, dtype=torch.float32, device="mps").contiguous()

    module = _load_leet_2d_convolution()
    module.solve(inp, ker, out, input_rows, input_cols, kernel_rows, kernel_cols)
    torch.mps.synchronize()

    expected = torch.nn.functional.conv2d(
        inp.cpu().view(1, 1, input_rows, input_cols),
        ker.cpu().view(1, 1, kernel_rows, kernel_cols),
    ).view(output_rows, output_cols)
    torch.testing.assert_close(out.cpu(), expected, atol=1e-5, rtol=1e-5)
