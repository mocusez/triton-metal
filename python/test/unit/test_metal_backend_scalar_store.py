"""Option β carry-forward AC2 / plan Step 5.5 — scalar-ptr `tt.store` path.

Exercises the inline scalar-ptr fallback inside `StoreLowering`
(`third_party/metal/lib/Conversion/TritonGPUToMetal/TritonGPUToMetal.cpp:2826-2841`)
that was added in US-002 of the option-beta-spt-load-lowering ralph. The path
fires when `tt.store` consumes a bare `!tt.ptr<T>` (no `tt.addptr`, no tensor
layout) — the same IR shape that rank-1 reduce result stores produce when
Triton folds `tl.store(out_ptr + 0, s)` to `tt.store %out_ptr, %s : !tt.ptr<f32>`.

Scope: f32 only. The scalar-ptr branch emits `metal.store` with the raw
adaptor value; it does NOT bridge a signless i32 through ui32 (`metal.store`
expects Metal_Type — the bridging the rank-1 reduce path performs at
`:1825` is absent from the scalar-ptr branch). i32 coverage requires that
bridging or a dialect tweak and is deferred.

See `.omc/plans/option-beta-carry-forward.md` Step 5 / AC2.
"""

from __future__ import annotations

import sys

import pytest

torch = pytest.importorskip("torch")

import triton  # noqa: E402
import triton.language as tl  # noqa: E402


pytestmark = pytest.mark.skipif(
    sys.platform != "darwin" or not torch.backends.mps.is_available(),
    reason="Metal backend requires Darwin + MPS",
)


@triton.jit
def _scalar_store_sum_kernel(
    in_ptr,
    out_ptr,
    BLOCK: tl.constexpr,
):
    """Sum a BLOCK-sized tile, then write the scalar result via a bare
    `tl.store(out_ptr, s)` — Triton folds the +0 offset, producing
    `tt.store %out_ptr, %s : !tt.ptr<f32>` with no `tt.addptr`."""
    offs = tl.arange(0, BLOCK)
    x = tl.load(in_ptr + offs)
    s = tl.sum(x, axis=0)
    tl.store(out_ptr, s)


@pytest.mark.parametrize("BLOCK", [32, 64, 128, 256])
def test_scalar_store_f32_after_reduce(BLOCK: int) -> None:
    torch.manual_seed(BLOCK)
    inp = torch.randn(BLOCK, dtype=torch.float32, device="mps")
    out = torch.zeros(1, dtype=torch.float32, device="mps")

    _scalar_store_sum_kernel[(1,)](inp, out, BLOCK)

    expected = inp.sum()
    err = (out[0] - expected).abs().item()
    # f32 partial-sum across BLOCK elements; tolerance scales with BLOCK.
    assert err < 1e-3 * BLOCK, (
        f"BLOCK={BLOCK}: got {out[0].item()}, expected {expected.item()}, err={err}"
    )
