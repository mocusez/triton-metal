"""Rank-2 axis=0 (per-column) reduce on the Metal backend (Session L3a2).

`tt.reduce(tile[BM,BN], axis=0)` sums each column over its BM rows. Unlike the
axis=1 (per-row) reduce, axis=0's output columns are INDEPENDENT — each is owned
by one output thread, which sums `device[offs(m, myCol)]` over m locally (no
threadgroup buffer, no barrier). `lowerRank2Axis0Reduce` re-derives the per-(row,
col) device address and mask from the load's own offset/mask cones via
`evalRank2ConeAt`, so the runtime row stride N and the per-program column base
(pid*BN) are recovered from the cone.

The tutorial-05 backward's `_layer_norm_bwd_dwdb` reduces a LOOP-CARRIED 2D
accumulator (`dw += load_tile`, `db += load_tile`) over axis 0;
`reassociateLoopCarriedAxis0Reduce` rewrites that to `scf.for(s1d += reduce(
load_tile, axis=0))` (per accumulator) so the reduce sees a per-iteration direct
masked load.

Combines: f32 sum/max and i32 sum/product/max/min. Product is direct-unmasked
load only;
the other combines support both direct and loop-carried forms (the
reassociation reassociates any same-kind loop-carried accumulator whose tensor
update op has an elementwise scalarizing lowering — arith.add{f,i},
metal.binary_exp for f32 max, and scalar arith.maxsi/minsi for i32 max/min). Sum
uses scalar arith; f32 max uses metal.binary_exp; i32 max/min use cmpi+select in
the reduce body (binary_exp rejects signless i32).

MORE COLUMNS THAN THREADS (output E>1, BN > tpb). Each thread then owns several
output columns, one per tile-loop iteration, and which one it owns on a given
iteration is a property of the CONSUMING value's layout — `tid + iv*tpb` when
the bands interleave, `tid*E + iv` when a thread holds a contiguous band. The
lowering reads that layout off the rank-1 value the result becomes (the same
tile, and the same index function, the consuming store uses) rather than
assuming one. A result re-expanded into a 2-D tile (`mean[None, :]`) needs one more thing:
no lane holds the column its tile position names, so the reduce leaves ALL of
its columns in a per-column threadgroup buffer and the re-expansion reads that
buffer at whichever column it needs, replaying the elementwise cone between the
two. That path is f32-only and wants a rank-1 consumer to take the output
geometry from; without one the shape is still refused before conversion.

ADDRESS SPELLING (`test_*_two_level_addptr` below). Triton emits one of two
shapes for the same tile address, chosen purely by how the Python is
parenthesised:

    offs = rows[:, None] * N + cols[None, :]   one level:
    tl.load(In + offs)                           addptr(splat(In), offs)

    tl.load(In + rows[:, None] * N             two levels:
                + cols[None, :])                 addptr(broadcast(addptr(
                                                   splat(In), rows*N)), cols)

Every case above uses the first. The lowering used to read only the OUTERMOST
`ap.getOffset()`, so under the second spelling the inner `rows*N` term was
dropped: every row aliased row 0 and the reduce returned `BM * tile[0, col]` —
plausible numbers, no crash, no diagnostic. `evalAddPtrChainAt` now sums the
whole chain (the row half sits below a `tt.broadcast`, so the walk has to peel
shape ops), which also picks up SCALAR chain terms such as a per-program
`In + b * stride` base that the old outer-offset-only read discarded too.
Both spellings are pinned here — the second is the one leet-triton kernels
(e.g. `medium-batch_normalization.py`) actually write.
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
def _colsum(In, Out, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # Loop-carried 2D accumulator reduced over axis 0 → per-column sum.
    pid = tl.program_id(0)
    cols = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for i in range(0, M, BLOCK_M):
        rows = i + tl.arange(0, BLOCK_M)
        mask = (rows[:, None] < M) & (cols[None, :] < N)
        offs = rows[:, None] * N + cols[None, :]
        acc += tl.load(In + offs, mask=mask, other=0.)
    s = tl.sum(acc, axis=0)
    tl.store(Out + cols, s, mask=cols < N)


# BLOCK_N == 128 (tpb at num_warps=4) so output E==1; N spans one program (<=128)
# and several (256/700/1024). Non-divisible M (100/40) exercises the row mask;
# non-pow2 N (700/333) the column mask.
@pytest.mark.parametrize("M, N", [(32, 128), (64, 256), (96, 700), (128, 1024),
                                  (256, 333), (100, 128), (40, 200)])
def test_reduce_axis0_colsum(M, N):
    torch.manual_seed(M * 13 + N)
    inp = torch.randn(M, N, device="mps")
    out = torch.empty(N, device="mps")
    _colsum[(triton.cdiv(N, 128),)](inp, out, M, N, BLOCK_M=32, BLOCK_N=128)
    torch.mps.synchronize()
    ref = inp.cpu().sum(0)
    torch.testing.assert_close(out.cpu(), ref, atol=1e-3, rtol=1e-3)


@triton.jit
def _layer_norm_bwd_dwdb(DW, DB, FINAL_DW, FINAL_DB, M, N,
                         BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    # Verbatim tutorial-05 Stage 2: two loop-carried 2D accumulators, each
    # reduced over axis 0.
    pid = tl.program_id(0)
    cols = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    dw = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    db = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(0, M, BLOCK_SIZE_M):
        rows = i + tl.arange(0, BLOCK_SIZE_M)
        mask = (rows[:, None] < M) & (cols[None, :] < N)
        offs = rows[:, None] * N + cols[None, :]
        dw += tl.load(DW + offs, mask=mask, other=0.)
        db += tl.load(DB + offs, mask=mask, other=0.)
    sum_dw = tl.sum(dw, axis=0)
    sum_db = tl.sum(db, axis=0)
    tl.store(FINAL_DW + cols, sum_dw, mask=cols < N)
    tl.store(FINAL_DB + cols, sum_db, mask=cols < N)


# Two fused accumulators → a 2-result scf.for; the reassociation reassociates
# each result independently.
@pytest.mark.parametrize("M, N", [(32, 128), (64, 256), (96, 700), (256, 1024)])
def test_bwd_dwdb_verbatim(M, N):
    torch.manual_seed(M * 31 + N)
    dw = torch.randn(M, N, device="mps")
    db = torch.randn(M, N, device="mps")
    fdw = torch.empty(N, device="mps")
    fdb = torch.empty(N, device="mps")
    grid = (triton.cdiv(N, 128),)
    _layer_norm_bwd_dwdb[grid](dw, db, fdw, fdb, M, N, BLOCK_SIZE_M=32,
                               BLOCK_SIZE_N=128)
    torch.mps.synchronize()
    torch.testing.assert_close(fdw.cpu(), dw.cpu().sum(0), atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(fdb.cpu(), db.cpu().sum(0), atol=1e-3, rtol=1e-3)


# --- Combines beyond f32 sum (Session L3a2 generalization) -----------------
# f32 max and i32 sum/product/max/min. Sum uses scalar arith; product uses
# ui32 metal.binary_exp for modulo-2^32 behavior; f32 max uses
# metal.binary_exp maxOp; i32 max/min use cmpi+select (binary_exp rejects
# signless i32). Both direct and loop-carried (reassociation + elementwise
# maxnumf / maxsi / minsi lowerings).


@triton.jit
def _colmax(In, Out, M, N, BM: tl.constexpr, BN: tl.constexpr):
    # loop-carried column max (handles M > BM).
    pid = tl.program_id(0)
    cols = pid * BN + tl.arange(0, BN)
    acc = tl.full((BM, BN), -1e30, tl.float32)
    for i in range(0, M, BM):
        rows = i + tl.arange(0, BM)
        mask = (rows[:, None] < M) & (cols[None, :] < N)
        offs = rows[:, None] * N + cols[None, :]
        acc = tl.maximum(acc, tl.load(In + offs, mask=mask, other=-1e30))
    tl.store(Out + cols, tl.max(acc, axis=0), mask=cols < N)


@pytest.mark.parametrize("M, N", [(32, 128), (64, 256), (96, 700), (100, 1024)])
def test_reduce_axis0_colmax_f32(M, N):
    torch.manual_seed(M * 7 + N)
    inp = torch.randn(M, N, device="mps")
    out = torch.empty(N, device="mps")
    _colmax[(triton.cdiv(N, 128),)](inp, out, M, N, BM=32, BN=128)
    torch.mps.synchronize()
    torch.testing.assert_close(out.cpu(), inp.cpu().amax(0), atol=1e-4, rtol=1e-4)


@triton.jit
def _colsum_i32(In, Out, M, N, BM: tl.constexpr, BN: tl.constexpr):
    pid = tl.program_id(0)
    cols = pid * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), tl.int32)
    for i in range(0, M, BM):
        rows = i + tl.arange(0, BM)
        mask = (rows[:, None] < M) & (cols[None, :] < N)
        offs = rows[:, None] * N + cols[None, :]
        acc += tl.load(In + offs, mask=mask, other=0)
    tl.store(Out + cols, tl.sum(acc, axis=0), mask=cols < N)


@pytest.mark.parametrize("M, N", [(32, 128), (64, 256), (96, 700)])
def test_reduce_axis0_colsum_i32(M, N):
    torch.manual_seed(M * 11 + N)
    inp = torch.randint(-500, 500, (M, N), dtype=torch.int32, device="mps")
    out = torch.empty(N, dtype=torch.int32, device="mps")
    _colsum_i32[(triton.cdiv(N, 128),)](inp, out, M, N, BM=32, BN=128)
    torch.mps.synchronize()
    assert torch.equal(out.cpu(), inp.cpu().sum(0, dtype=torch.int32))


@triton.jit
def _product_combine(a, b):
    return a * b


@triton.jit
def _colproduct_i32(In, Out, BM: tl.constexpr, BN: tl.constexpr):
    cols = tl.arange(0, BN)
    rows = tl.arange(0, BM)
    offs = rows[:, None] * BN + cols[None, :]
    x = tl.load(In + offs)
    product = tl.reduce(x, axis=0, combine_fn=_product_combine)
    tl.store(Out + cols, product)


@triton.jit
def _colproduct_f32(In, Out, BM: tl.constexpr, BN: tl.constexpr):
    cols = tl.arange(0, BN)
    rows = tl.arange(0, BM)
    offs = rows[:, None] * BN + cols[None, :]
    x = tl.load(In + offs)
    product = tl.reduce(x, axis=0, combine_fn=_product_combine)
    tl.store(Out + cols, product)


@triton.jit
def _colmax_f32_direct(In, Out, BM: tl.constexpr, BN: tl.constexpr):
    cols = tl.arange(0, BN)
    rows = tl.arange(0, BM)
    offs = rows[:, None] * BN + cols[None, :]
    x = tl.load(In + offs)
    tl.store(Out + cols, tl.max(x, axis=0))


@triton.jit
def _colmin_f32_direct(In, Out, BM: tl.constexpr, BN: tl.constexpr):
    cols = tl.arange(0, BN)
    rows = tl.arange(0, BM)
    offs = rows[:, None] * BN + cols[None, :]
    x = tl.load(In + offs)
    tl.store(Out + cols, tl.min(x, axis=0))


@pytest.mark.parametrize("M,N", [(4, 16), (8, 32)])
def test_reduce_axis0_colproduct_i32_direct(M, N):
    inp = torch.ones((M, N), dtype=torch.int32)
    inp[0] = 2
    inp[1] = -3
    if M >= 4:
        inp[2, 0] = 65536
        inp[3, 0] = 65536
    out = torch.empty(N, dtype=torch.int32, device="mps")
    _colproduct_i32[(1,)](inp.to("mps"), out, BM=M, BN=N)
    torch.mps.synchronize()
    expected_values = []
    for column in inp.T.tolist():
        bits = 1
        for value in column:
            bits = (bits * (value & 0xFFFFFFFF)) & 0xFFFFFFFF
        expected_values.append(bits if bits < 0x80000000 else bits - 0x100000000)
    expected = torch.tensor(expected_values, dtype=torch.int32)
    assert torch.equal(out.cpu(), expected)


@pytest.mark.parametrize("M,N", [(4, 16), (8, 32)])
def test_reduce_axis0_colproduct_f32_direct(M, N):
    inp = torch.ones((M, N), dtype=torch.float32)
    inp[0] = 1.25
    inp[1] = -0.5
    inp[2, 0] = 0.0
    inp[2, 1] = -0.0
    inp[2, 2] = float("inf")
    inp[2, 3] = float("nan")
    out = torch.empty(N, dtype=torch.float32, device="mps")
    _colproduct_f32[(1,)](inp.to("mps"), out, BM=M, BN=N)
    torch.mps.synchronize()
    expected = torch.prod(inp, dim=0)
    actual = out.cpu()
    finite = torch.isfinite(expected)
    torch.testing.assert_close(actual[finite], expected[finite], atol=1e-6, rtol=1e-6)
    assert torch.isneginf(actual[2])
    assert torch.isnan(actual[3])
    assert torch.equal(actual.view(torch.int32)[[0, 1]],
                       expected.view(torch.int32)[[0, 1]])


@pytest.mark.parametrize(
    "kernel, fill, check",
    [
        (_colmax_f32_direct, float("-inf"), torch.isneginf),
        (_colmin_f32_direct, float("inf"), torch.isposinf),
    ],
)
@pytest.mark.parametrize("M,N", [(4, 16), (8, 32)])
def test_reduce_axis0_f32_extrema_exact_infinity_identity(kernel, fill, check,
                                                          M, N):
    inp = torch.full((M, N), fill, dtype=torch.float32, device="mps")
    out = torch.empty(N, dtype=torch.float32, device="mps")
    kernel[(1,)](inp, out, BM=M, BN=N)
    torch.mps.synchronize()
    assert check(out).all()


@triton.jit
def _colmax_i32(In, Out, M, N, BM: tl.constexpr, BN: tl.constexpr):
    # direct i32 column max (M <= BM); cmpi+select combine.
    pid = tl.program_id(0)
    cols = pid * BN + tl.arange(0, BN)
    rows = tl.arange(0, BM)
    mask = (rows[:, None] < M) & (cols[None, :] < N)
    offs = rows[:, None] * N + cols[None, :]
    x = tl.load(In + offs, mask=mask, other=-2147483648)
    tl.store(Out + cols, tl.max(x, axis=0), mask=cols < N)


@triton.jit
def _colmin_i32(In, Out, M, N, BM: tl.constexpr, BN: tl.constexpr):
    pid = tl.program_id(0)
    cols = pid * BN + tl.arange(0, BN)
    rows = tl.arange(0, BM)
    mask = (rows[:, None] < M) & (cols[None, :] < N)
    offs = rows[:, None] * N + cols[None, :]
    x = tl.load(In + offs, mask=mask, other=2147483647)
    tl.store(Out + cols, tl.min(x, axis=0), mask=cols < N)


@pytest.mark.parametrize("kernel, red",
                         [(_colmax_i32, lambda t: t.amax(0)),
                          (_colmin_i32, lambda t: t.amin(0))])
@pytest.mark.parametrize("M, N", [(32, 128), (16, 256), (32, 700)])
def test_reduce_axis0_minmax_i32_direct(kernel, red, M, N):
    torch.manual_seed(M * 17 + N)
    inp = torch.randint(-500, 500, (M, N), dtype=torch.int32, device="mps")
    out = torch.empty(N, dtype=torch.int32, device="mps")
    kernel[(triton.cdiv(N, 128),)](inp, out, M, N, BM=32, BN=128)
    torch.mps.synchronize()
    assert torch.equal(out.cpu(), red(inp.cpu()))


@triton.jit
def _colmax_i32_lc(In, Out, M, N, BM: tl.constexpr, BN: tl.constexpr):
    # loop-carried i32 column max (handles M > BM); ArithIntMinMax elementwise
    # lowering + cmpi+select reduce body.
    pid = tl.program_id(0)
    cols = pid * BN + tl.arange(0, BN)
    acc = tl.full((BM, BN), -2147483648, tl.int32)
    for i in range(0, M, BM):
        rows = i + tl.arange(0, BM)
        mask = (rows[:, None] < M) & (cols[None, :] < N)
        offs = rows[:, None] * N + cols[None, :]
        acc = tl.maximum(acc, tl.load(In + offs, mask=mask, other=-2147483648))
    tl.store(Out + cols, tl.max(acc, axis=0), mask=cols < N)


@triton.jit
def _colmin_i32_lc(In, Out, M, N, BM: tl.constexpr, BN: tl.constexpr):
    pid = tl.program_id(0)
    cols = pid * BN + tl.arange(0, BN)
    acc = tl.full((BM, BN), 2147483647, tl.int32)
    for i in range(0, M, BM):
        rows = i + tl.arange(0, BM)
        mask = (rows[:, None] < M) & (cols[None, :] < N)
        offs = rows[:, None] * N + cols[None, :]
        acc = tl.minimum(acc, tl.load(In + offs, mask=mask, other=2147483647))
    tl.store(Out + cols, tl.min(acc, axis=0), mask=cols < N)


@pytest.mark.parametrize("kernel, red",
                         [(_colmax_i32_lc, lambda t: t.amax(0)),
                          (_colmin_i32_lc, lambda t: t.amin(0))])
@pytest.mark.parametrize("M, N", [(32, 128), (64, 256), (96, 700), (100, 1024)])
def test_reduce_axis0_minmax_i32_loopcarried(kernel, red, M, N):
    torch.manual_seed(M * 23 + N)
    inp = torch.randint(-500, 500, (M, N), dtype=torch.int32, device="mps")
    out = torch.empty(N, dtype=torch.int32, device="mps")
    kernel[(triton.cdiv(N, 128),)](inp, out, M, N, BM=32, BN=128)
    torch.mps.synchronize()
    assert torch.equal(out.cpu(), red(inp.cpu()))


# --- Two-level addptr addressing (see the module docstring) ----------------
# `In + rows[:, None] * N + cols[None, :]` instead of `In + offs`. Same tile,
# same results; before `evalAddPtrChainAt` these all returned BM * tile[0, col].


@triton.jit
def _colsum_two_level(In, Out, M, N, BM: tl.constexpr, BN: tl.constexpr):
    pid = tl.program_id(0)
    cols = pid * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for i in range(0, M, BM):
        rows = i + tl.arange(0, BM)
        mask = (rows[:, None] < M) & (cols[None, :] < N)
        acc += tl.load(In + rows[:, None] * N + cols[None, :], mask=mask, other=0.)
    tl.store(Out + cols, tl.sum(acc, axis=0), mask=cols < N)


@triton.jit
def _colsum_two_level_direct(In, Out, M, N, BM: tl.constexpr, BN: tl.constexpr):
    # No loop: the reduce sees the tt.load directly, not the reassociated form.
    pid = tl.program_id(0)
    cols = pid * BN + tl.arange(0, BN)
    rows = tl.arange(0, BM)
    mask = (rows[:, None] < M) & (cols[None, :] < N)
    v = tl.load(In + rows[:, None] * N + cols[None, :], mask=mask, other=0.)
    tl.store(Out + cols, tl.sum(v, axis=0), mask=cols < N)


# M == BM (single trip) through M >> BM; ragged M/N; BN of 16 and 128 so the
# tile's blocked layout differs (BN=16 gives sizePerThread=[1,2]).
@pytest.mark.parametrize("M, N, BM, BN", [(32, 128, 32, 128), (100, 128, 32, 128),
                                          (96, 700, 32, 128), (256, 333, 32, 128),
                                          (64, 32, 16, 16), (17, 48, 8, 16),
                                          (16, 16, 16, 16)])
def test_reduce_axis0_colsum_two_level_addptr(M, N, BM, BN):
    torch.manual_seed(M * 3 + N)
    inp = torch.randn(M, N, device="mps")
    out = torch.empty(N, device="mps")
    _colsum_two_level[(triton.cdiv(N, BN),)](inp, out, M, N, BM=BM, BN=BN)
    torch.mps.synchronize()
    ref = inp.cpu().sum(0)
    torch.testing.assert_close(out.cpu(), ref, atol=1e-3, rtol=1e-3)
    # The old bug was exactly BM * row 0 — assert we are not back on it even if
    # some future rewrite makes the numbers merely "close".
    assert not torch.allclose(out.cpu(), BM * inp.cpu()[0], atol=1e-3)


@pytest.mark.parametrize("M, N, BM, BN", [(32, 128, 32, 128), (16, 256, 16, 128),
                                          (16, 16, 16, 16)])
def test_reduce_axis0_colsum_two_level_addptr_direct(M, N, BM, BN):
    torch.manual_seed(M * 5 + N)
    inp = torch.randn(M, N, device="mps")
    out = torch.empty(N, device="mps")
    _colsum_two_level_direct[(triton.cdiv(N, BN),)](inp, out, M, N, BM=BM, BN=BN)
    torch.mps.synchronize()
    torch.testing.assert_close(out.cpu(), inp.cpu().sum(0), atol=1e-3, rtol=1e-3)


@triton.jit
def _colmax_two_level(In, Out, M, N, BM: tl.constexpr, BN: tl.constexpr):
    pid = tl.program_id(0)
    cols = pid * BN + tl.arange(0, BN)
    acc = tl.full((BM, BN), -1e30, tl.float32)
    for i in range(0, M, BM):
        rows = i + tl.arange(0, BM)
        mask = (rows[:, None] < M) & (cols[None, :] < N)
        acc = tl.maximum(acc, tl.load(In + rows[:, None] * N + cols[None, :],
                                      mask=mask, other=-1e30))
    tl.store(Out + cols, tl.max(acc, axis=0), mask=cols < N)


@triton.jit
def _colsum_i32_two_level(In, Out, M, N, BM: tl.constexpr, BN: tl.constexpr):
    pid = tl.program_id(0)
    cols = pid * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), tl.int32)
    for i in range(0, M, BM):
        rows = i + tl.arange(0, BM)
        mask = (rows[:, None] < M) & (cols[None, :] < N)
        acc += tl.load(In + rows[:, None] * N + cols[None, :], mask=mask, other=0)
    tl.store(Out + cols, tl.sum(acc, axis=0), mask=cols < N)


@pytest.mark.parametrize("M, N", [(96, 700), (100, 1024)])
def test_reduce_axis0_colmax_f32_two_level_addptr(M, N):
    torch.manual_seed(M * 7 + N)
    inp = torch.randn(M, N, device="mps")
    out = torch.empty(N, device="mps")
    _colmax_two_level[(triton.cdiv(N, 128),)](inp, out, M, N, BM=32, BN=128)
    torch.mps.synchronize()
    torch.testing.assert_close(out.cpu(), inp.cpu().amax(0), atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("M, N", [(96, 700), (64, 256)])
def test_reduce_axis0_colsum_i32_two_level_addptr(M, N):
    torch.manual_seed(M * 11 + N)
    inp = torch.randint(-500, 500, (M, N), dtype=torch.int32, device="mps")
    out = torch.empty(N, dtype=torch.int32, device="mps")
    _colsum_i32_two_level[(triton.cdiv(N, 128),)](inp, out, M, N, BM=32, BN=128)
    torch.mps.synchronize()
    assert torch.equal(out.cpu(), inp.cpu().sum(0, dtype=torch.int32))


@triton.jit
def _colsum_batched(In, Out, M, N, S, BM: tl.constexpr, BN: tl.constexpr):
    # A SCALAR per-program base (`In + b * S`) sits below the two tensor offsets;
    # the outer-offset-only read dropped it along with the row term.
    b = tl.program_id(1)
    pid = tl.program_id(0)
    cols = pid * BN + tl.arange(0, BN)
    base = In + b * S
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for i in range(0, M, BM):
        rows = i + tl.arange(0, BM)
        mask = (rows[:, None] < M) & (cols[None, :] < N)
        acc += tl.load(base + rows[:, None] * N + cols[None, :], mask=mask, other=0.)
    tl.store(Out + b * N + cols, tl.sum(acc, axis=0), mask=cols < N)


@pytest.mark.parametrize("B, M, N, BM, BN", [(3, 64, 128, 32, 128),
                                             (2, 40, 60, 16, 64)])
def test_reduce_axis0_scalar_base_two_level_addptr(B, M, N, BM, BN):
    torch.manual_seed(B * 100 + M)
    inp = torch.randn(B, M, N, device="mps")
    out = torch.empty(B, N, device="mps")
    _colsum_batched[(triton.cdiv(N, BN), B)](inp, out, M, N, M * N, BM=BM, BN=BN)
    torch.mps.synchronize()
    torch.testing.assert_close(out.cpu(), inp.cpu().sum(1), atol=1e-3, rtol=1e-3)


# --- COMPUTED cone sources ------------------------------------------------
# Everything above reduces a device tile directly (or the loop-carried
# accumulator the reassociation produces). A computed tile —
# `tl.where(mask, x - mean[None, :], 0.) ** 2` in batch norm's variance pass —
# has no single tensor to re-read, so it is re-derived per (row, col) by
# `evalRank2ConeAt`, the evaluator the axis=1 reduce uses for softmax cones.
# The row loop always runs the full compile-time BM, so the representative
# load's mask has to gate the cone's own addresses (`g_coneAddrGuard`) and not
# just its value — hence the ragged-M cases below.


@triton.jit
def _colsum_computed_cone(In, Out, M, N, BM: tl.constexpr, BN: tl.constexpr):
    # The mask is the ordinary `andi` of two comparisons — which is what caught
    # `rank2ConeSupported` missing arith.and/or while `evalRank2ConeAt` had them
    # all along, i.e. every masked cone was rejected before the evaluator ran.
    pid = tl.program_id(0)
    cols = pid * BN + tl.arange(0, BN)
    acc = tl.zeros((BN,), dtype=tl.float32)
    for i in range(0, M, BM):
        rows = i + tl.arange(0, BM)
        mask = (rows[:, None] < M) & (cols[None, :] < N)
        v = tl.load(In + rows[:, None] * N + cols[None, :], mask=mask, other=0.)
        v = tl.where(mask, v, 0.0)
        acc += tl.sum(v * v, axis=0)
    tl.store(Out + cols, acc, mask=cols < N)


# Ragged M (100/17 vs BM) drives the in-bounds guard on the cone's own loads:
# the row loop always runs the full compile-time BM.
@pytest.mark.parametrize("M, N, BM, BN", [(64, 128, 32, 128), (100, 128, 32, 128),
                                          (17, 48, 8, 16), (64, 32, 16, 16)])
def test_reduce_axis0_computed_cone(M, N, BM, BN):
    torch.manual_seed(M * 3 + N)
    x = torch.randn(M, N, device="mps")
    out = torch.zeros(N, device="mps")
    _colsum_computed_cone[(triton.cdiv(N, BN),)](x, out, M, N, BM=BM, BN=BN)
    torch.mps.synchronize()
    ref = (x.cpu().double() ** 2).sum(0).float()
    torch.testing.assert_close(out.cpu(), ref, atol=1e-3, rtol=1e-4)


@triton.jit
def _colvar_chained(In, Out, M, N, BM: tl.constexpr, BN: tl.constexpr):
    # The staged leaf: the second reduce's cone subtracts the FIRST reduce's
    # result, an scf.for result the cone evaluator cannot re-emit. It resolves
    # to the thread's own converted scalar, sound because this reduce's column
    # IS that thread's column.
    pid = tl.program_id(0)
    cols = pid * BN + tl.arange(0, BN)
    mean = tl.zeros((BN,), dtype=tl.float32)
    for i in range(0, M, BM):
        rows = i + tl.arange(0, BM)
        mask = (rows[:, None] < M) & (cols[None, :] < N)
        mean += tl.sum(tl.load(In + rows[:, None] * N + cols[None, :], mask=mask, other=0.), axis=0)
    mean /= M
    var = tl.zeros((BN,), dtype=tl.float32)
    for i in range(0, M, BM):
        rows = i + tl.arange(0, BM)
        mask = (rows[:, None] < M) & (cols[None, :] < N)
        v = tl.load(In + rows[:, None] * N + cols[None, :], mask=mask, other=0.)
        v = tl.where(mask, v - mean[None, :], 0.0)
        var += tl.sum(v * v, axis=0)
    tl.store(Out + cols, var / M, mask=cols < N)


@pytest.mark.parametrize("M, N, BM, BN", [(64, 128, 32, 128), (100, 40, 16, 16),
                                          (33, 64, 16, 32)])
def test_reduce_axis0_chained_cone(M, N, BM, BN):
    torch.manual_seed(M * 5 + N)
    x = torch.randn(M, N, device="mps")
    out = torch.zeros(N, device="mps")
    _colvar_chained[(triton.cdiv(N, BN),)](x, out, M, N, BM=BM, BN=BN)
    torch.mps.synchronize()
    xc = x.cpu().double()
    ref = ((xc - xc.mean(0)) ** 2).mean(0).float()
    torch.testing.assert_close(out.cpu(), ref, atol=1e-5, rtol=1e-4)


@triton.jit
def _kmeans_assign_axis0_argmin(Points, Centers, IndexOut, N,
                                K: tl.constexpr, BK: tl.constexpr,
                                BN: tl.constexpr):
    pid = tl.program_id(0)
    cols = pid * BN + tl.arange(0, BN)
    ks = tl.arange(0, BK)
    mask = cols < N
    px = tl.load(Points + cols * 2, mask=mask, other=0.0)
    py = tl.load(Points + cols * 2 + 1, mask=mask, other=0.0)
    cx = tl.load(Centers + ks * 2, mask=ks < K, other=0.0)
    cy = tl.load(Centers + ks * 2 + 1, mask=ks < K, other=0.0)
    dx = px[None, :] - cx[:, None]
    dy = py[None, :] - cy[:, None]
    dist = tl.where(ks[:, None] < K, dx * dx + dy * dy, float("inf"))
    tl.store(IndexOut + cols, tl.argmin(dist, axis=0), mask=mask)


@pytest.mark.parametrize("N, K", [(1, 1), (128, 4), (257, 3)])
def test_reduce_axis0_argmin_computed_kmeans_tile(N, K):
    torch.manual_seed(N * 29 + K)
    points = torch.randn(N, 2, dtype=torch.float32, device="mps")
    centers = torch.randn(K, 2, dtype=torch.float32, device="mps")
    if K >= 3:
        centers[2] = centers[0]
        points[:7] = centers[0]
    index_out = torch.empty(N, dtype=torch.int32, device="mps")

    _kmeans_assign_axis0_argmin[(triton.cdiv(N, 128),)](
        points, centers, index_out, N, K=K,
        BK=triton.next_power_of_2(K), BN=128, num_warps=4
    )
    torch.mps.synchronize()

    distances = ((points.cpu()[:, None, :] - centers.cpu()[None, :, :]) ** 2).sum(2)
    expected_index = distances.argmin(dim=1)
    torch.testing.assert_close(
        index_out.cpu(), expected_index.to(torch.int32), atol=0, rtol=0
    )


# ---------------------------------------------------------------------------
# More output columns than the threadgroup has threads (output E > 1).
#
# This used to be refused outright, on the assumption that the output layout
# would be one the tile-loop model mis-addresses. The layout is no longer
# assumed: the column a thread owns on a given iteration comes from the rank-1
# tile the result becomes, which is the same tile the consuming store indexes
# with. The same kernel at four warps (tpb=128, E=1) and at one warp (tpb=32,
# E=4) must give the same answer.
# ---------------------------------------------------------------------------


@triton.jit
def _colsum_direct(In, Out, BM: tl.constexpr, BN: tl.constexpr):
    rows = tl.arange(0, BM)[:, None]
    cols = tl.arange(0, BN)[None, :]
    tile = tl.load(In + rows * BN + cols)
    tl.store(Out + tl.arange(0, BN), tl.sum(tile, axis=0))


@triton.jit
def _colmax_direct(In, Out, BM: tl.constexpr, BN: tl.constexpr):
    rows = tl.arange(0, BM)[:, None]
    cols = tl.arange(0, BN)[None, :]
    tile = tl.load(In + rows * BN + cols)
    tl.store(Out + tl.arange(0, BN), tl.max(tile, axis=0))


@pytest.mark.parametrize("BM, BN, num_warps", [
    (8, 64, 1),     # E_out = 2
    (8, 128, 1),    # E_out = 4
    (4, 256, 1),    # E_out = 8
    (8, 256, 4),    # E_out = 2, four warps
    (16, 512, 4),   # E_out = 4
    (8, 1024, 8),   # E_out = 4, eight warps
    (32, 128, 1),   # more rows than threads as well
])
def test_reduce_axis0_more_columns_than_threads(BM, BN, num_warps):
    torch.manual_seed(BM * 101 + BN + num_warps)
    inp = torch.randn(BM, BN)
    out = torch.empty(BN, device="mps")
    _colsum_direct[(1,)](inp.to("mps").contiguous(), out, BM=BM, BN=BN,
                         num_warps=num_warps)
    torch.mps.synchronize()
    torch.testing.assert_close(out.cpu().double(), inp.double().sum(0),
                               atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("BM, BN, num_warps", [(8, 128, 1), (8, 256, 4)])
def test_reduce_axis0_more_columns_than_threads_max(BM, BN, num_warps):
    """A maximum reads the same address relation as the sum but folds with a
    different operator, so it catches a column index that is off by a band
    rather than merely mis-scaled."""
    torch.manual_seed(BM + BN)
    inp = torch.randn(BM, BN)
    out = torch.empty(BN, device="mps")
    _colmax_direct[(1,)](inp.to("mps").contiguous(), out, BM=BM, BN=BN,
                         num_warps=num_warps)
    torch.mps.synchronize()
    torch.testing.assert_close(out.cpu(), inp.max(dim=0).values, atol=1e-5,
                               rtol=1e-5)


@pytest.mark.parametrize("M, N, num_warps", [(64, 256, 1), (96, 700, 2),
                                             (100, 512, 4)])
def test_reduce_axis0_colsum_masked_more_columns_than_threads(M, N, num_warps):
    """The loop-carried, masked, multi-program form -- tutorial-05's shape --
    with the column block wider than the threadgroup."""
    torch.manual_seed(M * 7 + N + num_warps)
    inp = torch.randn(M, N, device="mps")
    out = torch.empty(N, device="mps")
    block_n = 256
    _colsum[(triton.cdiv(N, block_n),)](inp, out, M, N, BLOCK_M=32,
                                        BLOCK_N=block_n, num_warps=num_warps)
    torch.mps.synchronize()
    torch.testing.assert_close(out.cpu(), inp.cpu().sum(0), atol=1e-3,
                               rtol=1e-3)


@triton.jit
def _colsum_i32_direct(In, Out, BM: tl.constexpr, BN: tl.constexpr):
    rows = tl.arange(0, BM)[:, None]
    cols = tl.arange(0, BN)[None, :]
    tile = tl.load(In + rows * BN + cols)
    tl.store(Out + tl.arange(0, BN), tl.sum(tile, axis=0))


def test_reduce_axis0_more_columns_than_threads_i32():
    torch.manual_seed(5)
    BM, BN = 8, 128
    inp = torch.randint(-500, 500, (BM, BN), dtype=torch.int32)
    out = torch.empty(BN, dtype=torch.int32, device="mps")
    _colsum_i32_direct[(1,)](inp.to("mps").contiguous(), out, BM=BM, BN=BN,
                             num_warps=1)
    torch.mps.synchronize()
    assert torch.equal(out.cpu(), inp.sum(0, dtype=torch.int32))


@triton.jit
def _colsum_rows_per_program(In, Out, BM: tl.constexpr, BN: tl.constexpr):
    pid = tl.program_id(0)
    rows = tl.arange(0, BM)[:, None]
    cols = tl.arange(0, BN)[None, :]
    tile = tl.load(In + (pid * BM + rows) * BN + cols)
    tl.store(Out + pid * BN + tl.arange(0, BN), tl.sum(tile, axis=0))


def test_reduce_axis0_more_columns_than_threads_multiprogram():
    """Each threadgroup addresses its own columns by a LOCAL thread id; a
    global one would shift every group's window by tgid*tpb."""
    torch.manual_seed(6)
    P, BM, BN = 3, 8, 128
    inp = torch.randn(P * BM, BN)
    out = torch.zeros(P, BN, device="mps")
    _colsum_rows_per_program[(P,)](inp.to("mps").contiguous(), out, BM=BM,
                                   BN=BN, num_warps=1)
    torch.mps.synchronize()
    torch.testing.assert_close(out.cpu().double(),
                               inp.view(P, BM, BN).double().sum(1), atol=1e-4,
                               rtol=1e-4)


@triton.jit
def _colmean_reexpanded(In, OutMean, OutDev, BM: tl.constexpr,
                        BN: tl.constexpr):
    rows = tl.arange(0, BM)[:, None]
    cols = tl.arange(0, BN)[None, :]
    tile = tl.load(In + rows * BN + cols)
    mean = tl.sum(tile, axis=0) / BM
    tl.store(OutMean + tl.arange(0, BN), mean)
    tl.store(OutDev + rows * BN + cols, tile - mean[None, :])


def test_reduce_axis0_reexpanded_result_is_lowered():
    """`mean[None, :]` at more columns than threads used to be refused, and had
    to be: the re-expansion republishes one column per thread, and no lane held
    the column its tile position named. The reduce now leaves ALL of its columns
    in a per-column buffer and the re-expansion reads that buffer at whichever
    column it needs, so this is the ordinary centring kernel."""
    BM, BN = 8, 128
    torch.manual_seed(19)
    inp = torch.randn(BM, BN)
    mean = torch.zeros(BN, device="mps")
    dev = torch.zeros(BM, BN, device="mps")
    _colmean_reexpanded[(1,)](inp.to("mps").contiguous(), mean, dev, BM=BM,
                              BN=BN, num_warps=1)
    torch.mps.synchronize()
    want_mean = inp.double().mean(0)
    torch.testing.assert_close(mean.cpu().double(), want_mean, atol=1e-4,
                               rtol=1e-4)
    torch.testing.assert_close(dev.cpu().double(), inp.double() - want_mean,
                               atol=1e-4, rtol=1e-4)


@triton.jit
def _colsum_reexpanded_only(In, OutDev, BM: tl.constexpr, BN: tl.constexpr):
    rows = tl.arange(0, BM)[:, None]
    cols = tl.arange(0, BN)[None, :]
    tile = tl.load(In + rows * BN + cols)
    tl.store(OutDev + rows * BN + cols, tile - tl.sum(tile, axis=0) / BM)


def test_reduce_axis0_reexpanded_without_a_rank1_consumer_is_named(capfd):
    """What is still refused is a result that never becomes a rank-1 value at
    all: the per-thread output geometry then has nothing to come from. The
    refusal is made before conversion, because a pattern that declines inside
    it takes the process down."""
    BM, BN = 8, 128
    inp = torch.randn(BM, BN, device="mps")
    dev = torch.zeros(BM, BN, device="mps")
    with pytest.raises(Exception) as error:
        _colsum_reexpanded_only[(1,)](inp, dev, BM=BM, BN=BN, num_warps=1)
    assert "convert-tritongpu-to-metal failed" in str(error.value)
    reason = capfd.readouterr()
    assert "rank-1 consumer carrying exactly those columns" in (reason.err
                                                                + reason.out)

    survivor = torch.randn(4, 32, device="mps")
    out = torch.empty(32, device="mps")
    _colsum_direct[(1,)](survivor, out, BM=4, BN=32, num_warps=1)
    torch.mps.synchronize()
    torch.testing.assert_close(out.cpu(), survivor.cpu().sum(0), atol=1e-4,
                               rtol=1e-4)


@triton.jit
def _colmean_reexpanded_var(In, OutMean, OutVar, OutDev, BM: tl.constexpr,
                            BN: tl.constexpr):
    rows = tl.arange(0, BM)[:, None]
    cols = tl.arange(0, BN)[None, :]
    tile = tl.load(In + rows * BN + cols)
    mean = tl.sum(tile, axis=0) / BM
    dev = tile - mean[None, :]
    var = tl.sum(dev * dev, axis=0) / BM
    tl.store(OutMean + tl.arange(0, BN), mean)
    tl.store(OutVar + tl.arange(0, BN), var)
    tl.store(OutDev + rows * BN + cols, dev)


@pytest.mark.parametrize("BM,BN,num_warps", [(8, 128, 1), (8, 256, 4),
                                             (4, 512, 4), (8, 1024, 8),
                                             (32, 128, 1), (8, 64, 1)])
def test_reduce_axis0_batch_norm_wide_columns(BM, BN, num_warps):
    """Two column reduces where the second consumes the first's re-expansion --
    batch norm's shape, with more columns than the threadgroup has threads."""
    torch.manual_seed(BM * 53 + BN + num_warps)
    inp = torch.randn(BM, BN)
    mean = torch.zeros(BN, device="mps")
    var = torch.zeros(BN, device="mps")
    dev = torch.zeros(BM, BN, device="mps")
    _colmean_reexpanded_var[(1,)](inp.to("mps").contiguous(), mean, var, dev,
                                  BM=BM, BN=BN, num_warps=num_warps)
    torch.mps.synchronize()
    want_mean = inp.double().mean(0)
    want_dev = inp.double() - want_mean
    torch.testing.assert_close(mean.cpu().double(), want_mean, atol=1e-4,
                               rtol=1e-4)
    torch.testing.assert_close(var.cpu().double(), (want_dev * want_dev).mean(0),
                               atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(dev.cpu().double(), want_dev, atol=1e-4,
                               rtol=1e-4)
