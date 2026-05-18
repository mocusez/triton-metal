"""L2b: uint8/i8 masked tt.load + tt.store on the Metal backend.

Drives the color_inversion leet kernel end-to-end (TTIR -> TTGIR ->
convert-tritongpu-to-metal -> MSL -> MPS launch) and asserts bit-exact
agreement with the torch reference across 5 deterministic runs.

The L2b session relaxed three sites in the metal backend:

  * `MaskedLoadLowering` dtype gate
    (`third_party/metal/lib/Conversion/TritonGPUToMetal/TritonGPUToMetal.cpp`)
    now accepts signless `i8` in addition to `FloatType`.
  * The `other`-default constant emission branches on `IntegerType` and
    emits an `IntegerAttr` zero (no `getFloatAttr` path).
  * `Metal_Type` in `MetalOps.td` adds signless `I8` so `metal.get_element`
    and friends verify on `i8` element types.

See `.omc/specs/deep-interview-leet-triton-l2b-uint8-load-store.md`.
"""

from __future__ import annotations

import sys

import pytest

torch = pytest.importorskip("torch")

import triton
import triton.language as tl


pytestmark = pytest.mark.skipif(
    sys.platform != "darwin" or not torch.backends.mps.is_available(),
    reason="Metal backend requires Darwin + MPS",
)


@triton.jit
def _invert_kernel(image_ptr, width, height, BLOCK_SIZE: tl.constexpr):
    image_ptr = image_ptr.to(tl.pointer_type(tl.uint8))
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE * 4 + tl.arange(0, BLOCK_SIZE * 4)
    mask = (offsets < width * height * 4) & (offsets % 4 != 3)
    image = tl.load(image_ptr + offsets, mask=mask)
    image = 255 - image
    tl.store(image_ptr + offsets, image, mask=mask)


def _run_one(width: int, height: int) -> None:
    torch.manual_seed(0)
    image = torch.randint(
        0, 256, (height * width * 4,), dtype=torch.uint8, device="mps"
    )
    expected = image.clone()
    rgb_mask = (torch.arange(image.numel(), device="mps") % 4) != 3
    expected[rgb_mask] = 255 - expected[rgb_mask]

    BLOCK_SIZE = 1024
    n_pixels = width * height
    grid = (triton.cdiv(n_pixels, BLOCK_SIZE),)
    _invert_kernel[grid](image, width, height, BLOCK_SIZE)

    assert torch.equal(image, expected), (
        f"mismatch at {(image != expected).nonzero().flatten()[:8].tolist()}"
    )


@pytest.mark.parametrize("run_idx", range(5))
def test_color_inversion_64x32_bit_exact(run_idx: int) -> None:
    """Determinism: 5 independent runs must each pass bit-exact."""
    _run_one(64, 32)
