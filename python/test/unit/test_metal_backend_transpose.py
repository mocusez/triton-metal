"""End-to-end correctness tests for L1d2 staged-transpose
(`.omc/specs/deep-interview-leet-triton-l1d2-staged-transpose-body.md`).

L1d2 ships `ConvertLayoutLowering`'s staged-transpose body for in-envelope
rank-2 blocked->blocked `ttg.convert_layout` ops with `sizePerThread=[1,1]`
on both sides. The body emits the canonical 5-op sequence
`threadgroup_alloca -> tg_store_indexed[srcIdx] -> barrier ->
tg_load_indexed[dstIdx] -> barrier -> replaceOp`.

Honest divergences from the spec's AC.T2 / AC.T4 envelope (surface, do
not silently work around):

* The tests below force `num_warps=8` at the leet's 16x16 shape (256
  threads == 16*16 elements, one element per thread) because the STAGED
  body is what they exist to cover, and it only ever handles one element
  per thread. The shipped kernel's own default (BLOCK_N=16, num_warps=4)
  lowers to `sizePerThread=[1,2]` / `[2,1]` instead.

* SUPERSEDED by L1d3 (`normalizeConsumerSideBlockedDivergentCvt`): the
  `sizePerThread > 1` shapes this note used to call unreachable — the
  leet default, and 32x32 under any practical `num_warps` — no longer
  need the staged body at all. The relabel is carried by re-encoding the
  consumer cone, so no data crosses lanes and the element count per
  thread stops mattering. Covered by
  `test_trans_combined_with_second_tile` at the bottom of this file and
  by `convert_layout_consumer_reencode.mlir`.

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


# --------------------------------------------------------------------------
# `tl.trans` and rank-2 `tl.reshape` — the one-rank-2-tile premise.
#
# The tests above transpose through the ADDRESS (`store(o + xi*N + yi, tile)`),
# which never puts a second rank-2 tile shape in the kernel. `tl.trans` and a
# shape-changing `tl.reshape` do, and until 2026-08-17 the backend answered
# every index range from one module-wide tile (`findLargestRank2Tile`) and
# forwarded `tt.trans` unconditionally. The measured result:
#
#   * a standalone `tl.trans` stored the UNtransposed tile — silently, at every
#     non-square shape and at every square shape of at least threadgroup size
#     (4x4 and sub-tpb 8x8 happened to come out right, which is why nothing in
#     this suite noticed);
#   * `tl.reshape(v, (8, 16))` of a 16x8 tile wrote with the SOURCE row stride,
#     leaving 65 of 128 output slots at zero;
#   * `tl.reshape` of a rank-1 tile into a rank-2 one took the process down
#     (SIGSEGV/SIGABRT on 9 of 30 runs of one shape) because the rank-2 range
#     found no rank-2 tile and `MakeRangeLowering` declined inside the
#     conversion.
#
# Every `tl.trans` in `leet-triton/` sits inside a `tl.dot`, where the matmul
# matchers walk through it on their own terms — which is exactly why these
# shapes need their own coverage.
# --------------------------------------------------------------------------


@triton.jit
def trans_store_kernel(x_ptr, o_ptr, M: tl.constexpr, N: tl.constexpr):
    i = tl.arange(0, M)
    j = tl.arange(0, N)
    v = tl.load(x_ptr + i[:, None] * N + j[None, :])
    tl.store(o_ptr + j[:, None] * M + i[None, :], tl.trans(v))


@pytest.mark.parametrize(
    "M,N,num_warps",
    [
        # Non-square: the load tile is [M,N] and the store tile [N,M], so the
        # two ranges must decompose against different row lengths.
        pytest.param(16, 8, 4, id="16x8_nw4"),
        pytest.param(8, 16, 4, id="8x16_nw4"),
        pytest.param(64, 16, 4, id="64x16_nw4"),
        pytest.param(4, 8, 1, id="4x8_nw1"),
        pytest.param(8, 4, 2, id="8x4_nw2"),
        # Square: one shape, but the two sides' `order` differ, so forwarding
        # the trans is only right if nothing re-encoded it on the way.
        pytest.param(16, 16, 4, id="16x16_nw4"),
        pytest.param(32, 32, 4, id="32x32_nw4"),
        # Sub-threadgroup squares — the two that were accidentally correct
        # before the fix, kept as controls.
        pytest.param(8, 8, 4, id="8x8_nw4"),
        pytest.param(4, 4, 1, id="4x4_nw1"),
    ],
)
def test_trans_rank2_stored(M, N, num_warps):
    device = "mps"
    x = torch.arange(M * N, dtype=torch.float32, device=device).reshape(M, N)
    out = torch.zeros(N, M, dtype=torch.float32, device=device)
    trans_store_kernel[(1,)](x, out, M, N, num_warps=num_warps)
    expected = x.cpu().t().contiguous()
    assert torch.equal(out.cpu(), expected), (
        f"tl.trans {M}x{N} nw={num_warps}\nout=\n{out.cpu()}\n"
        f"expected=\n{expected}"
    )


@triton.jit
def trans_reduce_kernel(x_ptr, o_ptr, M: tl.constexpr, N: tl.constexpr):
    i = tl.arange(0, M)
    j = tl.arange(0, N)
    v = tl.load(x_ptr + i[:, None] * N + j[None, :])
    tl.store(o_ptr + tl.arange(0, N), tl.sum(tl.trans(v), 1))


def test_trans_feeding_reduce():
    """A reduce re-derives its own coordinates, so this path was already right
    before the fix — pinned so the fix cannot regress it."""
    M, N = 16, 8
    x = torch.arange(M * N, dtype=torch.float32, device="mps").reshape(M, N)
    out = torch.zeros(N, dtype=torch.float32, device="mps")
    trans_reduce_kernel[(1,)](x, out, M, N)
    assert torch.allclose(out.cpu(), x.cpu().t().sum(1), atol=1e-5)


@triton.jit
def reshape_rank2_kernel(
    x_ptr, o_ptr, M: tl.constexpr, N: tl.constexpr,
    P: tl.constexpr, Q: tl.constexpr
):
    i = tl.arange(0, M)
    j = tl.arange(0, N)
    v = tl.load(x_ptr + i[:, None] * N + j[None, :])
    r = tl.reshape(v, (P, Q))
    p = tl.arange(0, P)
    q = tl.arange(0, Q)
    tl.store(o_ptr + p[:, None] * Q + q[None, :], r)


@pytest.mark.parametrize(
    "M,N,P,Q",
    [
        pytest.param(16, 8, 8, 16, id="16x8_to_8x16"),
        pytest.param(16, 16, 8, 32, id="16x16_to_8x32"),
        pytest.param(32, 32, 16, 64, id="32x32_to_16x64"),
        pytest.param(8, 8, 4, 16, id="8x8_to_4x16"),
    ],
)
def test_reshape_rank2_shape_change(M, N, P, Q):
    x = torch.arange(M * N, dtype=torch.float32, device="mps").reshape(M, N)
    out = torch.zeros(P, Q, dtype=torch.float32, device="mps")
    reshape_rank2_kernel[(1,)](x, out, M, N, P, Q)
    expected = x.cpu().reshape(P, Q)
    assert torch.equal(out.cpu(), expected), (
        f"tl.reshape {M}x{N}->{P}x{Q}: "
        f"{int((out.cpu() == 0).sum())} of {P * Q} slots left at zero"
    )


@triton.jit
def reshape_rank1_to_rank2_kernel(
    x_ptr, o_ptr, B: tl.constexpr, P: tl.constexpr, Q: tl.constexpr
):
    v = tl.reshape(tl.load(x_ptr + tl.arange(0, B)), (P, Q))
    p = tl.arange(0, P)
    q = tl.arange(0, Q)
    tl.store(o_ptr + p[:, None] * Q + q[None, :], v)


@pytest.mark.parametrize(
    "P,Q",
    [
        pytest.param(8, 16, id="8x16"),
        pytest.param(16, 8, id="16x8"),
        pytest.param(4, 4, id="4x4"),
        pytest.param(32, 32, id="32x32"),
        pytest.param(2, 64, id="2x64"),
    ],
)
def test_reshape_rank1_to_rank2(P, Q):
    """Was a process kill, not a wrong answer: the rank-2 range found only the
    rank-1 source tile and `MakeRangeLowering` declined mid-conversion."""
    B = P * Q
    x = torch.arange(B, dtype=torch.float32, device="mps")
    out = torch.zeros(P, Q, dtype=torch.float32, device="mps")
    reshape_rank1_to_rank2_kernel[(1,)](x, out, B, P, Q)
    assert torch.equal(out.cpu().reshape(-1), x.cpu())


@triton.jit
def reshape_rank2_to_rank1_kernel(
    x_ptr, o_ptr, B: tl.constexpr, P: tl.constexpr, Q: tl.constexpr
):
    p = tl.arange(0, P)
    q = tl.arange(0, Q)
    v = tl.load(x_ptr + p[:, None] * Q + q[None, :])
    tl.store(o_ptr + tl.arange(0, B), tl.reshape(v, (B,)))


@pytest.mark.parametrize(
    "P,Q", [pytest.param(8, 16, id="8x16"), pytest.param(16, 8, id="16x8")]
)
def test_reshape_rank2_to_rank1(P, Q):
    """The direction that always worked — the control for the pair."""
    B = P * Q
    x = torch.arange(B, dtype=torch.float32, device="mps").reshape(P, Q)
    out = torch.zeros(B, dtype=torch.float32, device="mps")
    reshape_rank2_to_rank1_kernel[(1,)](x, out, B, P, Q)
    assert torch.equal(out.cpu(), x.cpu().reshape(-1))


# --------------------------------------------------------------------------
# L1d3: a transposed tile COMBINED with a second tile.
#
# `tl.trans(x) * tl.load(y)` leaves a rank-2 blocked->blocked relabel between
# two genuinely different lane mappings, and the staged transpose above can only
# exchange one element per thread — with more, the publish and the read both sit
# inside the scalarized tile loop and the slot a lane needs at iteration `iv` is
# written by another lane at some other iteration. So this compiled only while
# `M*N <= 32*num_warps`; 12 of the 18 cases below were refused outright.
#
# The fix moves the LAYOUT instead of the data: the forward cone (the multiply,
# the second load, the store's address cone) is re-encoded into the transposed
# layout and the relabel is erased. No threadgroup memory, no barrier, and no
# dependence on sizePerThread at all.
# --------------------------------------------------------------------------


@triton.jit
def trans_times_tile_kernel(
    x_ptr, y_ptr, o_ptr, M: tl.constexpr, N: tl.constexpr
):
    i = tl.arange(0, M)
    j = tl.arange(0, N)
    v = tl.trans(tl.load(x_ptr + i[:, None] * N + j[None, :]))
    w = tl.load(y_ptr + j[:, None] * M + i[None, :])
    tl.store(o_ptr + j[:, None] * M + i[None, :], v * w)


@pytest.mark.parametrize("num_warps", [1, 4, 8])
@pytest.mark.parametrize(
    "M,N",
    [
        pytest.param(16, 16, id="16x16"),
        pytest.param(16, 8, id="16x8"),
        pytest.param(8, 16, id="8x16"),
        pytest.param(32, 32, id="32x32"),
        pytest.param(64, 16, id="64x16"),
        pytest.param(8, 8, id="8x8"),
    ],
)
def test_trans_combined_with_second_tile(M, N, num_warps):
    x = torch.arange(M * N, dtype=torch.float32, device="mps").reshape(M, N)
    y = torch.arange(N * M, dtype=torch.float32, device="mps").reshape(N, M) * 0.5
    out = torch.zeros(N, M, dtype=torch.float32, device="mps")
    trans_times_tile_kernel[(1,)](x, y, out, M, N, num_warps=num_warps)
    expected = x.cpu().t().contiguous() * y.cpu()
    assert torch.equal(out.cpu(), expected), (
        f"trans*tile {M}x{N} nw={num_warps}\nout=\n{out.cpu()}\n"
        f"expected=\n{expected}"
    )


@triton.jit
def trans_plus_scalar_masked_kernel(
    x_ptr, o_ptr, rows, M: tl.constexpr, N: tl.constexpr
):
    i = tl.arange(0, M)
    j = tl.arange(0, N)
    v = tl.trans(tl.load(x_ptr + i[:, None] * N + j[None, :]))
    tl.store(
        o_ptr + j[:, None] * M + i[None, :],
        v + 1.0,
        mask=j[:, None] < rows,
    )


@pytest.mark.parametrize("num_warps", [1, 4])
@pytest.mark.parametrize(
    "M,N", [pytest.param(16, 16, id="16x16"), pytest.param(32, 16, id="32x16")]
)
def test_trans_elementwise_masked_store(M, N, num_warps):
    """The store's MASK cone has to be re-encoded alongside its address cone."""
    rows = N // 2
    x = torch.arange(M * N, dtype=torch.float32, device="mps").reshape(M, N)
    out = torch.zeros(N, M, dtype=torch.float32, device="mps")
    trans_plus_scalar_masked_kernel[(1,)](x, out, rows, M, N, num_warps=num_warps)
    expected = torch.zeros(N, M)
    expected[:rows] = x.cpu().t()[:rows] + 1.0
    assert torch.equal(out.cpu(), expected), (
        f"masked trans {M}x{N} nw={num_warps}\nout=\n{out.cpu()}\n"
        f"expected=\n{expected}"
    )


# --------------------------------------------------------------------------
# Rank-3 `tl.permute` — the same premise one rank up.
#
# Triton does not lower a rank-3 permute into index arithmetic the way it does a
# rank-2 transpose. It folds the WHOLE permutation into the load index's
# `#ttg.linear` layout and leaves the reshape and the transpose as flat
# identities, so the permutation exists nowhere else in the IR. This backend
# imposes its own `element == localTid` mapping on every range, which erased the
# op outright — `tl.permute` came back a plain copy — and the shape was
# therefore refused with "tl.arange has no per-element index under this layout".
#
# `planLinearRange` evaluates the layout's basis vectors instead: element index
# = XOR of one basis per set bit of (register, lane, warp). A zero basis is a
# broadcast (two threads holding one element), which XOR handles for free.
# --------------------------------------------------------------------------


@triton.jit
def permute_rank3_kernel(
    x_ptr, o_ptr, A: tl.constexpr, B: tl.constexpr, C: tl.constexpr,
    P0: tl.constexpr, P1: tl.constexpr, P2: tl.constexpr
):
    i = tl.arange(0, A * B * C)
    v = tl.reshape(tl.load(x_ptr + i), (A, B, C))
    t = tl.permute(v, (P0, P1, P2))
    tl.store(o_ptr + i, tl.reshape(t, (A * B * C,)))


@pytest.mark.parametrize("num_warps", [4, 8])
@pytest.mark.parametrize(
    "perm",
    [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)],
    ids=lambda p: "".join(str(d) for d in p),
)
@pytest.mark.parametrize(
    "A,B,C",
    [
        pytest.param(2, 4, 8, id="2x4x8"),
        pytest.param(4, 4, 4, id="4x4x4"),
        pytest.param(2, 2, 16, id="2x2x16"),
        pytest.param(8, 4, 2, id="8x4x2"),
    ],
)
def test_permute_rank3(A, B, C, perm, num_warps):
    x = torch.arange(A * B * C, dtype=torch.float32, device="mps")
    out = torch.zeros(A * B * C, dtype=torch.float32, device="mps")
    permute_rank3_kernel[(1,)](x, out, A, B, C, *perm, num_warps=num_warps)
    expected = x.cpu().reshape(A, B, C).permute(*perm).reshape(-1)
    assert torch.equal(out.cpu(), expected), (
        f"tl.permute {A}x{B}x{C} perm={perm} nw={num_warps}\n"
        f"out={out.cpu()[:16].tolist()}\nexpected={expected[:16].tolist()}"
    )
