"""End-to-end correctness tests for L1d2 staged-transpose
(the implementation notes).

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

* The formerly failing masked transpose path now passes deterministically.
  The direct address-arithmetic path avoids the cross-lane reload shape that
  triggered the Apple codegen issue; the parametrized cases remain as plain
  regression tests.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import triton  # noqa: E402
import triton.language as tl  # noqa: E402

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
# L1d2b honest divergence (the implementation notes):
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
# Every `tl.trans` in `python/test/unit/fixtures/metal_leet/` sits inside a `tl.dot`, where the matmul
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


# --------------------------------------------------------------------------
# The full-tile exchange: a permute whose tile holds MORE than one element per
# thread.
#
# `ConvertLayoutLowering`'s staged transpose can only ever move one element per
# thread — above that its publish and its read both sit inside the scalarized
# tile loop, and the slot a lane needs at iteration `iv` is written by another
# lane at some other iteration. So `tl.permute` compiled only while the tile fit
# one element per thread, which at num_warps=1 means 32 elements.
#
# The way through is to split the tile loop: publish the whole tile in one loop,
# barrier, read it in the next. That split happens where the loop is BUILT
# (`FuncOpLowering` splices the body into the `scf.for`), because a conversion
# pattern cannot fission a loop it is running inside. The body is partitioned by
# DEPENDENCE — everything the exchanged value is built from goes in the first
# loop, everything else in the second, and values needed on both sides are
# recomputed rather than carried across.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("num_warps", [1, 2])
@pytest.mark.parametrize(
    "perm",
    [(0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)],
    ids=lambda p: "".join(str(d) for d in p),
)
@pytest.mark.parametrize(
    "A,B,C",
    [
        pytest.param(2, 4, 8, id="2x4x8"),
        pytest.param(4, 4, 4, id="4x4x4"),
        pytest.param(8, 4, 2, id="8x4x2"),
    ],
)
def test_permute_rank3_exchanged(A, B, C, perm, num_warps):
    """Both forms of the exchange. Above one element per thread the tile loop
    runs more than once and the publish/read pair has to span a SPLIT loop;
    at exactly one element there is no loop and the pair is straight-line code
    with a barrier between. Neither shape compiled before."""
    x = torch.arange(A * B * C, dtype=torch.float32, device="mps")
    out = torch.zeros(A * B * C, dtype=torch.float32, device="mps")
    permute_rank3_kernel[(1,)](x, out, A, B, C, *perm, num_warps=num_warps)
    expected = x.cpu().reshape(A, B, C).permute(*perm).reshape(-1)
    assert torch.equal(out.cpu(), expected), (
        f"exchanged tl.permute {A}x{B}x{C} perm={perm} nw={num_warps}\n"
        f"out={out.cpu()[:16].tolist()}\nexpected={expected[:16].tolist()}"
    )


# --------------------------------------------------------------------------
# The rank-3 permute spelled the way a rank-3 kernel is actually written: a
# `[:, None, None]` / `[None, :, None]` / `[None, None, :]` address cone, not a
# rank-1 load reshaped into a cube.
#
# The two spellings take different routes. Reshaped, the whole permutation ends
# up inside the load index's `#ttg.linear` layout and `planLinearRange` reads it
# off the basis vectors (the section above). Written as an address cone, the
# permutation is a real `tt.trans` plus the `ttg.convert_layout` Triton inserts
# before the store — and THAT cvt is the op that moves the data, which took six
# attempts to establish: the trans is a relabel, the exchange carries it
# correctly, and the fault was in what the two halves index.
#
# The cvt needs the full-tile phase, granted only to a function that already has
# one, so every shape above a single element per thread was refused by name. It
# is granted here when the DESTINATION layout's flat index is the thread-major
# expression `localTid * E + iv` that the consuming store's own address
# arithmetic emits. That is the boundary, and it is not the register band it
# resembles: `8x4x4` at one warp is exact under the full reversal at E = 4 while
# `2x4x16` at one warp, also E = 4, is refused.
# --------------------------------------------------------------------------


@triton.jit
def permute_rank3_addr_kernel(
    x_ptr, o_ptr, A: tl.constexpr, B: tl.constexpr, C: tl.constexpr,
    S0: tl.constexpr, S1: tl.constexpr, S2: tl.constexpr,
    P0: tl.constexpr, P1: tl.constexpr, P2: tl.constexpr
):
    a = tl.arange(0, A)[:, None, None]
    b = tl.arange(0, B)[None, :, None]
    c = tl.arange(0, C)[None, None, :]
    v = tl.load(x_ptr + a * (B * C) + b * C + c)
    p = tl.permute(v, (P0, P1, P2))
    i0 = tl.arange(0, S0)[:, None, None]
    i1 = tl.arange(0, S1)[None, :, None]
    i2 = tl.arange(0, S2)[None, None, :]
    tl.store(o_ptr + i0 * (S1 * S2) + i1 * S2 + i2, p)


def _run_permute_rank3_addr(A, B, C, perm, num_warps):
    torch.manual_seed(0)
    x = torch.randn(A * B * C, dtype=torch.float32)
    out = torch.zeros(A * B * C, dtype=torch.float32, device="mps")
    s = tuple((A, B, C)[i] for i in perm)
    compiled = permute_rank3_addr_kernel[(1,)](
        x.to("mps"), out, A, B, C, s[0], s[1], s[2], *perm,
        num_warps=num_warps)
    torch.mps.synchronize()
    want = x.reshape(A, B, C).permute(*perm)
    return out.cpu().reshape(want.shape), want, compiled


@pytest.mark.parametrize("num_warps,A,B,C", [
    pytest.param(1, 2, 4, 8, id="2x4x8w1e2"),
    pytest.param(2, 2, 4, 8, id="2x4x8w2e1"),
    pytest.param(1, 4, 4, 4, id="4x4x4w1e2"),
    pytest.param(1, 8, 4, 4, id="8x4x4w1e4"),
    pytest.param(2, 4, 8, 8, id="4x8x8w2e4"),
    pytest.param(4, 8, 8, 8, id="8x8x8w4e4"),
])
@pytest.mark.parametrize(
    "perm",
    [(0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)],
    ids=lambda p: "".join(str(d) for d in p),
)
def test_permute_rank3_address_cone_matches_torch(A, B, C, perm, num_warps):
    """Every one of these was a named refusal before the cvt got the phase."""
    got, want, _ = _run_permute_rank3_addr(A, B, C, perm, num_warps)
    torch.testing.assert_close(got, want)


@pytest.mark.parametrize("num_warps,A,B,C", [
    pytest.param(1, 2, 4, 8, id="2x4x8w1e2"),
    pytest.param(1, 4, 8, 8, id="4x8x8w1e8"),
    pytest.param(4, 8, 8, 8, id="8x8x8w4e4"),
])
def test_permute_rank3_address_cone_identity_is_unchanged(A, B, C, num_warps):
    """The control: the same kernel, two rank-3 tile shapes in it, and no
    permutation. It needs no exchange and was already exact, including at the
    E = 8 shape the permuting arm still refuses — so the refusal is about the
    permutation and not about the rank or the band."""
    got, want, _ = _run_permute_rank3_addr(A, B, C, (0, 1, 2), num_warps)
    torch.testing.assert_close(got, want)


def test_permute_rank3_address_cone_publishes_through_threadgroup_memory():
    """Census the emitted MSL, not just the numbers.

    Exact would also be the answer if the permute had quietly become a copy at
    a shape where the permutation happens to be the identity of the data. The
    mechanism this commit installs is the two-loop exchange, so the barrier and
    the threadgroup buffer have to be there.
    """
    _got, _want, compiled = _run_permute_rank3_addr(2, 4, 8, (0, 2, 1), 1)
    msl = compiled.asm["metal"]
    if isinstance(msl, bytes):
        msl = msl.decode()
    assert "threadgroup_barrier" in msl, msl[:600]
    assert "threadgroup " in msl, msl[:600]


@pytest.mark.parametrize("num_warps,A,B,C,perm", [
    pytest.param(1, 4, 8, 8, (0, 2, 1), id="4x8x8w1e8_021"),
    pytest.param(1, 2, 8, 16, (1, 0, 2), id="2x8x16w1e8_102"),
    pytest.param(2, 2, 16, 16, (2, 1, 0), id="2x16x16w2e8_210"),
    pytest.param(1, 2, 4, 16, (2, 1, 0), id="2x4x16w1e4_210"),
    pytest.param(2, 2, 8, 16, (1, 2, 0), id="2x8x16w2e4_120"),
])
def test_permute_rank3_address_cone_off_thread_major_matches_torch(
        A, B, C, perm, num_warps):
    """These were refused by name: their destination LAYOUT is not the
    thread-major expression the store addresses by, and the exchange used the
    layouts' own plans. The exchange now publishes and reads under the
    thread-major labels the values actually carry -- the operand's, with its
    axes permuted through the `tt.trans`, and the store's -- which is exact
    whatever either layout is. `s_rank3_permute.py`: 19 named -> 0, 0 wrong.
    """
    got, want, _ = _run_permute_rank3_addr(A, B, C, perm, num_warps)
    torch.testing.assert_close(got, want)


# --------------------------------------------------------------------------
# The same permutation with NO layout op at all: a rank-3 load whose index cone
# carries the permuted strides.
#
# This arm went in as the control for the permute above — the same function
# written the other way — and it came back RED: silently wrong whenever the
# INNERMOST destination axis does not carry the source's unit stride, at every
# shape and warp count from one element per thread upward. `(1, 0, 2)` and the
# identity, which leave axis 2 innermost, were exact. Every rank-3 address probe
# before it held the inner stride at 1 (`.omc/fuzz/rank3_address.py`'s `strided`
# mode is `i * PS + j * RS + k`), which is why it had never been seen.
#
# Triton's coalescer reads the blocked `order` off the address's contiguity, so
# such a kernel holds two rank-3 encodings of the SAME shape differing only in
# `order`, with a `ttg.convert_layout` between them.
# `normalizeBlockedDivergentCvts` re-encodes the load's producer cone to the
# store's layout so the cvt collapses — and `remapDivergentConeEncoding` handled
# a tensor under ONE `#ttg.slice` but returned nullptr for nested ones. A rank-3
# index cone reaches its tile through TWO (`offs[:, None, None]`), so the whole
# rank-1 range layer fell outside the cone: the rank-3 tensors moved to the
# destination encoding while the ranges that index them kept the source's
# `order`, and the emitted load decomposed `localTid` as (2,4,8) while the store
# decomposed it as (2,8,4).
# --------------------------------------------------------------------------


@triton.jit
def rank3_strided_load_kernel(
    x_ptr, o_ptr, S0: tl.constexpr, S1: tl.constexpr, S2: tl.constexpr,
    T0: tl.constexpr, T1: tl.constexpr, T2: tl.constexpr
):
    i0 = tl.arange(0, S0)[:, None, None]
    i1 = tl.arange(0, S1)[None, :, None]
    i2 = tl.arange(0, S2)[None, None, :]
    v = tl.load(x_ptr + i0 * T0 + i1 * T1 + i2 * T2)
    tl.store(o_ptr + i0 * (S1 * S2) + i1 * S2 + i2, v)


def _run_rank3_strided_load(A, B, C, perm, num_warps):
    dims = (A, B, C)
    src = (B * C, C, 1)
    s = tuple(dims[i] for i in perm)
    t = tuple(src[i] for i in perm)
    x = torch.arange(A * B * C, dtype=torch.float32)
    out = torch.zeros(A * B * C, dtype=torch.float32, device="mps")
    compiled = rank3_strided_load_kernel[(1,)](
        x.to("mps"), out, s[0], s[1], s[2], t[0], t[1], t[2],
        num_warps=num_warps)
    torch.mps.synchronize()
    want = x.reshape(A, B, C).permute(*perm)
    return out.cpu().reshape(want.shape), want, compiled


@pytest.mark.parametrize("num_warps,A,B,C", [
    pytest.param(2, 2, 4, 8, id="2x4x8w2e1"),
    pytest.param(1, 2, 4, 8, id="2x4x8w1e2"),
    pytest.param(1, 8, 4, 4, id="8x4x4w1e4"),
    pytest.param(1, 4, 8, 8, id="4x8x8w1e8"),
    pytest.param(2, 2, 16, 16, id="2x16x16w2e8"),
])
@pytest.mark.parametrize(
    "perm",
    [(0, 2, 1), (1, 2, 0), (2, 0, 1), (2, 1, 0)],
    ids=lambda p: "".join(str(d) for d in p),
)
def test_rank3_load_with_non_unit_inner_stride_matches_torch(
        A, B, C, perm, num_warps):
    """The four permutations that move a strided source axis innermost."""
    got, want, _ = _run_rank3_strided_load(A, B, C, perm, num_warps)
    assert torch.equal(got, want), (
        f"{A}x{B}x{C} perm={perm} nw={num_warps}\n"
        f"got={got.flatten()[:12].tolist()}\n"
        f"want={want.flatten()[:12].tolist()}"
    )


@pytest.mark.parametrize("num_warps", [1, 2, 4])
@pytest.mark.parametrize(
    "perm", [(0, 1, 2), (1, 0, 2)],
    ids=lambda p: "".join(str(d) for d in p),
)
def test_rank3_load_with_unit_inner_stride_is_unchanged(perm, num_warps):
    """The control: the two permutations that leave axis 2 innermost were
    exact on the old binary too, which is what located the defect on the
    INNER axis rather than on rank 3 or on strided addressing as such."""
    got, want, _ = _run_rank3_strided_load(4, 8, 8, perm, num_warps)
    assert torch.equal(got, want)


def test_rank3_strided_load_is_normalized_not_staged():
    """Census the mechanism: this is the cone re-encode, not the exchange.

    A correct answer through a threadgroup publish/barrier/read would pass the
    numeric tests above while paying for staging a tile that needs no cross-lane
    move at all — the load can simply be gathered under the store's layout.
    """
    _got, _want, compiled = _run_rank3_strided_load(2, 4, 8, (0, 2, 1), 2)
    msl = compiled.asm["metal"]
    if isinstance(msl, bytes):
        msl = msl.decode()
    assert "threadgroup_barrier" not in msl, msl[:600]
    assert "threadgroup " not in msl, msl[:600]


@triton.jit
def _trans_add_reduce_kernel(X, W, Y, M: tl.constexpr, N: tl.constexpr,
                             AXIS: tl.constexpr):
    m = tl.arange(0, M)
    n = tl.arange(0, N)
    x = tl.load(X + m[:, None] * N + n[None, :])
    w = tl.load(W + n[:, None] * M + m[None, :])
    s = tl.sum(tl.trans(x) + w, AXIS)
    if AXIS == 0:
        tl.store(Y + m, s)
    else:
        tl.store(Y + n, s)


# `tl.sum(tl.trans(x) + w, axis)`: the reduce replays its source cone from
# memory, stepping through the `ttg.convert_layout` Triton puts after the
# trans as a shape-only op. Once that cvt had been split into an exchange's
# publish/read, the READ's operand was a `dense<0>` placeholder, and the replay
# folded `0 + w` -- every element wrong, silently. The replay now reads through
# a read to its publish's source. 📏 `s_trans_consumers.py` 16 wrong -> 0; the
# shapes whose cvt cannot be exchanged stay a named refusal.
@pytest.mark.parametrize("M,N,num_warps", [
    (8, 8, 1), (16, 16, 2), (16, 16, 4), (16, 32, 4), (32, 16, 4), (8, 64, 4),
    (64, 8, 4), (4, 128, 4),
])
@pytest.mark.parametrize("axis", [0, 1])
def test_trans_add_reduce_matches_torch(M, N, num_warps, axis):
    torch.manual_seed(M * 31 + N + num_warps + axis)
    x = torch.randn(M, N)
    w = torch.randn(N, M)
    y = torch.zeros(M if axis == 0 else N, device="mps")
    _trans_add_reduce_kernel[(1,)](x.to("mps"), w.to("mps"), y, M, N, axis,
                                   num_warps=num_warps)
    torch.mps.synchronize()
    torch.testing.assert_close(y.cpu(), (x.T + w).sum(axis), atol=1e-4, rtol=1e-4)


# Shapes whose trans cvt used to be refused ("broader staged-transpose
# deferred") because neither layout was thread-major. The exchange now runs
# between the two thread-major labels. `s_trans_consumers.py`: 34 named -> 0.
@pytest.mark.parametrize("M,N,num_warps", [
    (16, 32, 1), (32, 32, 1), (64, 64, 4), (8, 64, 1), (64, 8, 2), (4, 128, 1),
    (32, 16, 2),
])
@pytest.mark.parametrize("axis", [0, 1])
def test_trans_add_reduce_beyond_thread_major_layouts(M, N, num_warps, axis):
    torch.manual_seed(M * 17 + N + num_warps + axis)
    x = torch.randn(M, N)
    w = torch.randn(N, M)
    y = torch.zeros(M if axis == 0 else N, device="mps")
    _trans_add_reduce_kernel[(1,)](x.to("mps"), w.to("mps"), y, M, N, axis,
                                   num_warps=num_warps)
    torch.mps.synchronize()
    torch.testing.assert_close(y.cpu(), (x.T + w).sum(axis), atol=1e-4, rtol=1e-4)


@triton.jit
def _rank3_reduce_with_independent_broadcast_kernel(
        X, Y, S, O, A: tl.constexpr, B: tl.constexpr, C: tl.constexpr,
        M: tl.constexpr, N: tl.constexpr, T: tl.constexpr):
    f = tl.arange(0, A * B * C)
    v = tl.reshape(tl.load(X + f), (A, B, C))
    tl.store(S + tl.arange(0, A * C), tl.reshape(tl.sum(v, 1), (A * C,)))
    i = tl.arange(0, M)
    j = tl.arange(0, N)
    for t in range(0, T):
        r = tl.load(Y + t * N + j)
        back = r[None, :] + tl.zeros((M, N), tl.float32)
        tl.store(O + t * M * N + i[:, None] * N + j[None, :], back)


@pytest.mark.parametrize("trips", [1, 2])
def test_rank3_reduce_with_independent_wider_broadcast_is_named_then_recovers(
        trips, capfd):
    a, b, c, m, n = 4, 8, 4, 16, 64
    x = torch.arange(a * b * c, dtype=torch.float32)
    y = torch.arange(trips * n, dtype=torch.float32)
    sums = torch.zeros(a * c, device="mps")
    out = torch.zeros(trips * m * n, device="mps")
    with pytest.raises(Exception, match="convert-tritongpu-to-metal failed"):
        _rank3_reduce_with_independent_broadcast_kernel[(1,)](
            x.to("mps"), y.to("mps"), sums, out, a, b, c, m, n, trips,
            num_warps=1)
    diagnostic = capfd.readouterr()
    assert "this non-local tt.broadcast cannot be scheduled" in (
        diagnostic.err + diagnostic.out)
    assert torch.all(sums == 0) and torch.all(out == 0)

    survivor_x = torch.randn(m, n)
    w = torch.randn(n, m)
    survivor = torch.zeros(m, device="mps")
    _trans_add_reduce_kernel[(1,)](survivor_x.to("mps"), w.to("mps"), survivor,
                                   m, n, 0, num_warps=1)
    torch.mps.synchronize()
    expected = (survivor_x.T + w).sum(0)
    torch.testing.assert_close(survivor.cpu(), expected, atol=1e-4, rtol=1e-4)
