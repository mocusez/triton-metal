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

See the implementation notes Step 5 / AC2.
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


@triton.jit
def _scalar_i32_copy_guarded_kernel(in_ptr, out_ptr, B, S):
    """Per-program early-return guard + scalar i32 load/store over a MULTI-LEVEL
    scalar `tt.addptr` chain (`ptr + pid*S + j` is two nested addptr ops).

    Exercises three lowerings together: (1) `cf.cond_br` early-return
    structured into `scf.if` (structureEarlyReturns), (2) scalar i32 `tt.load`
    (ScalarLoadLowering widened to i32 via ui32 storage), (3) scalar i32
    `tt.store` over an accumulated scalar-addptr offset (StoreLowering)."""
    pid = tl.program_id(0)
    if pid >= B:
        return
    for j in range(S):
        v = tl.load(in_ptr + pid * S + j)
        tl.store(out_ptr + pid * S + j, v + 1)


@pytest.mark.parametrize("B, S", [(1, 4), (3, 5), (8, 1)])
def test_scalar_i32_load_store_guarded(B: int, S: int) -> None:
    torch.manual_seed(B * 100 + S)
    inp = torch.randint(-1000, 1000, (B, S), dtype=torch.int32, device="mps")
    # Launch two EXTRA programs; the `pid >= B` guard must make them return
    # without touching the sentinel rows.
    SENT = torch.iinfo(torch.int32).min
    out = torch.full((B + 2, S), SENT, dtype=torch.int32, device="mps")

    _scalar_i32_copy_guarded_kernel[(B + 2,)](inp, out, B, S)

    # Valid rows: out == in + 1 (scalar i32 load/store, multi-level addptr).
    torch.testing.assert_close(out[:B].cpu(), (inp + 1).cpu(), atol=0, rtol=0)
    # Guarded rows: untouched sentinel (cf.cond_br early-return fired).
    assert (out[B:].cpu() == SENT).all(), "early-return guard did not fire"
