"""End-to-end correctness tests for L1d2 staged-transpose
(`.omc/specs/deep-interview-leet-triton-l1d2-staged-transpose-body.md`).

L1d2 ships `ConvertLayoutLowering`'s staged-transpose body for in-envelope
rank-2 blocked->blocked `ttg.convert_layout` ops with `sizePerThread=[1,1]`
on both sides. The body emits the canonical 5-op sequence
`threadgroup_alloca -> tg_store_indexed[srcIdx] -> barrier ->
tg_load_indexed[dstIdx] -> barrier -> replaceOp`.

Honest divergences from the spec's AC.T2 / AC.T4 envelope (surface, do
not silently work around):

* The leet `easy-matrix_transpose.py` kernel as shipped (BLOCK_N=16,
  default num_warps=4) lowers to a TTGIR whose `ttg.convert_layout`
  endpoints have `sizePerThread=[1,2]` / `[2,1]` (the post-coalesce
  vectorized shape). L1d2 restricts to `sizePerThread=[1,1]` only; the
  vectorized path is L1d3 territory. To exercise the [1,1] envelope at
  the leet's 16x16 shape we MUST force `num_warps=8` (256 threads ==
  16*16 elements per tile, one element per thread).

* AC.T2's 32x32 and 48x80 cases land outside the [1,1] envelope under
  any practical `num_warps`: 32x32=1024 elements would need 1024
  threads/threadgroup (32 warps with `warp_size=32`) which exceeds Apple
  Silicon's per-threadgroup thread budget, and 48x80=3840 is
  unreachable. They are left for L1d3 (vectorized staging).

* The masked transpose path (the canonical leet pattern with `mask=...`
  on both load and store) currently exhibits a downstream miscompile
  when the cvt result flows into a masked-store's `scf.if` block — even
  when mask is trivially true. The miscompile shows up as a deterministic
  drop of writes from threads in non-zero warps and occasionally a race
  on warp 0. This is INDEPENDENT of L1d2's staging-body correctness:
  the unmasked transpose kernel (no `tt.load.mask` / `tt.store.mask`)
  passes bit-exact. Until the masked-store + cvt-body interaction is
  fixed, the masked path is marked xfail here. The fix likely belongs
  in `MaskedStoreLowering` (materializing the cvt's threadgroup-load as
  a top-level value before the `scf.if` body) and is out of L1d2's
  stated scope.
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
def transpose_kernel_unmasked(
    input_ptr, output_ptr, BLOCK_N: tl.constexpr
):
    """Square-tile transpose without masking. Exercises the cvt body in
    isolation, free of the masked-store interaction documented above."""
    x = tl.arange(0, BLOCK_N)
    y = tl.arange(0, BLOCK_N)
    xi = x[None, :]
    yi = y[:, None]
    tile = tl.load(input_ptr + yi * BLOCK_N + xi)
    tl.store(output_ptr + xi * BLOCK_N + yi, tile)


@pytest.mark.parametrize(
    "BLOCK_N,num_warps",
    [
        # 16x16 = 256 elements; num_warps=8 * warp_size=32 = 256 threads ->
        # sizePerThread=[1,1] on both src and dst encodings (in-envelope for
        # L1d2). Matches the leet matrix_transpose shape after launch-time
        # `num_warps` override.
        pytest.param(16, 8, id="16x16_nw8"),
        # 8x8 = 64 elements with num_warps=2 -> 64 threads, sizePerThread=
        # [1,1]. Exercises a smaller transpose tile through the same body.
        pytest.param(8, 2, id="8x8_nw2"),
    ],
)
def test_staged_transpose_unmasked(BLOCK_N, num_warps):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    if device == "cpu":
        pytest.skip("MPS device required to launch metal kernels")
    torch.manual_seed(0xC0FFEE)
    inp = torch.arange(
        BLOCK_N * BLOCK_N, dtype=torch.float32, device=device
    ).reshape(BLOCK_N, BLOCK_N).contiguous()
    out = torch.zeros(
        BLOCK_N, BLOCK_N, dtype=torch.float32, device=device
    ).contiguous()
    transpose_kernel_unmasked[(1,)](
        inp, out, BLOCK_N, num_warps=num_warps
    )
    expected = inp.t().contiguous()
    assert torch.equal(out, expected), (
        f"staged-transpose BLOCK_N={BLOCK_N} num_warps={num_warps} "
        f"max-abs-err={(out - expected).abs().max().item()}\n"
        f"out=\n{out.cpu()}\nexpected=\n{expected.cpu()}"
    )


@triton.jit
def transpose_kernel_masked(
    input_ptr,
    output_ptr,
    rows,
    cols,
    sir,
    sic,
    sor,
    soc,
    BLOCK_N: tl.constexpr,
):
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    x = pid_x * BLOCK_N + tl.arange(0, BLOCK_N)
    y = pid_y * BLOCK_N + tl.arange(0, BLOCK_N)
    xi = x[None, :]
    yi = y[:, None]
    mask = (xi < cols) & (yi < rows)
    tile = tl.load(input_ptr + yi * sir + xi * sic, mask=mask, other=0.0)
    tl.store(output_ptr + xi * sor + yi * soc, tile, mask=mask)


# SUPERSEDED 2026-06-03 — see the RESOLVED note just above the @parametrize
# decorator below. The historical hypothesis (a per-warp lane-aliasing
# miscompile in Apple's `tg_load_indexed` codegen) is kept for context, but
# the current lowering no longer emits that shape and the cases now pass.
#
# L1d2b honest divergence (`.omc/specs/deep-interview-leet-triton-l1d2b-...md`):
#
# L1d2b implements the MSL emitter's *inline-barrier contract* —
# `metal.tg_load_indexed` results are force-materialised as named MSL
# let-bindings AT their IR position, so subsequent uses inside any
# `scf.if(mask){…}` body emitted by `MaskedStoreLowering` render as the
# let-binding name and the load expression is never re-inlined across
# the barrier. AC.M1–M4 are met: see the canary lit fixture at
# `test/Dialect/Metal/metal-translate/tg_load_in_scf_if.mlir` and the
# dumped MSL for this kernel, which now shows
#   `float vN = v<tgbuf>[<dstIdx>];`  *before* the `if (mask) { … }`
# block, with the body referencing `vN` rather than re-evaluating the
# threadgroup load.
#
# However, the *runtime* miscompile this masked variant exhibits is NOT
# fully eliminated by the inline-barrier contract. Empirically, even
# with the spec-compliant MSL (let-binding hoisted out of the if-body;
# additional probes attempted: trailing-barrier drop, `volatile`
# qualifier on the let-binding, scf.if-condition let-binding) the
# masked transpose still drops higher-warp lane stores and produces
# the same warp-0-style result skew as the pre-L1d2b emit. The
# unmasked staged-transpose, which differs only by the absence of the
# `if (mask) { … }` wrapper around the trailing `metal.store`, passes
# bit-exact with the identical cvt body and let-binding hoisted.
#
# Per-lane diagnosis (8x8 nw=2 single threadgroup, mask uniformly true):
#   * out[0..7]  correct (lanes 0..7 wrote their transposed values).
#   * out[8..14] equal lane (i+8)'s SHIFTED tg_load — as if those lanes'
#     `id.x % 8` and `id.x / 8` mapped to lid+7 rather than lid. The
#     pattern is a per-warp lane-aliasing miscompile when the cvt
#     output flows into `if(mask){devstore;}` even though `mask` is
#     uniformly true and the if-body touches no threadgroup memory.
#
# Conclusion: the Apple Metal compiler bug class that L1d2 surfaced is
# broader than the inline-barrier theory hypothesised. Triggering the
# miscompile does not require `tg_load_indexed` to be inlined into the
# `scf.if` body — it occurs whenever a `threadgroup_barrier;
# if (cond) { devstore; }` shape is present, even when the if-body
# touches no threadgroup memory and `cond` is uniformly true at
# runtime. Surfacing this divergence (per spec § "Reporting
# expectations" item 6) rather than silently working around it.
# Candidate next-step fix loci, all out of L1d2b non-goals:
#   1. Materialise the masked-store's condition+address into thread-
#      local let-bindings *before* the trailing barrier so the
#      `scf.if` body becomes a pure device-memory store of pre-computed
#      operands (will require touching `MaskedStoreLowering` or a new
#      pre-emit MLIR pass).
#   2. Drop the cvt body's trailing `metal.barrier` for single-cvt
#      kernels (touches `ConvertLayoutLowering`).
#   3. Lift the entire masked-store out of `scf.if` form via a select
#      on the address (clamped-to-zero on masked-off lanes), avoiding
#      the divergent control flow shape entirely (touches
#      `MaskedStoreLowering`).
#
# Marking the parametrized cases `xfail(strict=False)` until the
# correct fix locus is approved. AC.T2's parametrization over
# (16×16 nw=8, 8×8 nw=2) is preserved per the spec so the regression
# coverage stays in place once the eventual fix lands. The new canary
# lit fixture is an AC.M2-level regression test for the let-binding
# hoisting that the runtime case sits on top of.
# RESOLVED 2026-06-03 (XPASS disposition): the L1d2c lane-aliasing
# miscompile described above no longer reproduces. The single-threadgroup
# (sizePerThread=[1,1], E=1) masked staged-transpose now lowers to a direct
# address-arithmetic gather/scatter — the dumped MSL contains no cross-lane
# `metal.tg_load_indexed` + `threadgroup_barrier` reload (the threadgroup
# scratch write is dead; the device store reuses the loaded value), so the
# Apple codegen bug cannot fire. Verified 30/30 deterministic bit-exact
# passes for both params. Flipped from xfail to plain pass; the test is
# retained as a regression guard — if a future codegen change reintroduces
# the cross-lane reload shape and the Apple bug returns, these cases fail.
@pytest.mark.parametrize(
    "rows,cols,BLOCK_N,num_warps",
    [
        pytest.param(16, 16, 16, 8, id="16x16_nw8"),
        pytest.param(8, 8, 8, 2, id="8x8_nw2"),
    ],
)
def test_staged_transpose_masked(rows, cols, BLOCK_N, num_warps):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    if device == "cpu":
        pytest.skip("MPS device required to launch metal kernels")
    torch.manual_seed(0)
    inp = torch.arange(
        rows * cols, dtype=torch.float32, device=device
    ).reshape(rows, cols).contiguous()
    out = torch.zeros(cols, rows, dtype=torch.float32, device=device).contiguous()
    grid = (triton.cdiv(cols, BLOCK_N), triton.cdiv(rows, BLOCK_N))
    transpose_kernel_masked[grid](
        inp,
        out,
        rows,
        cols,
        cols,
        1,
        rows,
        1,
        BLOCK_N,
        num_warps=num_warps,
    )
    expected = inp.t().contiguous()
    assert torch.equal(out, expected), (
        f"masked staged-transpose {rows}x{cols} BLOCK_N={BLOCK_N} "
        f"nw={num_warps} max-abs-err={(out - expected).abs().max().item()}\n"
        f"out=\n{out.cpu()}\nexpected=\n{expected.cpu()}"
    )
