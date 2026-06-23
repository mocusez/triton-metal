"""L2b.6: tutorial-04 low-memory dropout (the int32-mask `_dropout` kernel)
end-to-end on the Metal backend.

This is the headline value-leverage case that drove Session L2b (i32
`tt.load`/`tt.store`). The kernel masked-loads an int32 `x_keep` tensor, a
float32 `x` tensor, computes `tl.where(x_keep, x / (1 - p), 0.0)`, and
masked-stores the float32 result. It exercises:
  * masked i32 `tt.load` (routed through ui32 storage — Session L2b), and
  * scalar float arith (`1 - p`) emitted in MSL (`arith.subf` translateValue
    case added alongside L2b).

The seeded-dropout kernel from the tutorial (`tl.rand` / Philox PRNG) is a
separate feature out of L2b scope and is intentionally not covered here.
See `.omc/specs/l2b-i32-tt-load-store-extension.md`.
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

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def _dropout(x_ptr, x_keep_ptr, output_ptr, n_elements, p, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    x_keep = tl.load(x_keep_ptr + offsets, mask=mask)
    output = tl.where(x_keep, x / (1 - p), 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)


def _dropout_ref(x, x_keep, p):
    return torch.where(x_keep.to(torch.bool), x / (1 - p), torch.zeros_like(x))


@pytest.mark.parametrize("n_elements", [10, 128, 1024, 4096])
@pytest.mark.parametrize("p", [0.3, 0.5])  # exercise the scalar (1 - p) float path
def test_dropout_i32_mask(n_elements, p):
    torch.manual_seed(0xC0FFEE + n_elements)
    x = torch.randn(size=(n_elements, ), device=DEVICE)
    x_keep = (torch.rand(size=(n_elements, ), device=DEVICE) > p).to(torch.int32)
    output = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    _dropout[grid](x, x_keep, output, n_elements, p, BLOCK_SIZE=1024)
    expected = _dropout_ref(x, x_keep, p)
    torch.testing.assert_close(output.cpu(), expected.cpu(), atol=1e-5, rtol=1e-5)
