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

Combines: f32 sum/max and i32 sum/product/signed/unsigned
extrema/and/or/xor. Product is direct-unmasked load only; the other combines
support both direct and loop-carried forms (the reassociation reassociates any
same-kind loop-carried accumulator whose tensor update op has an elementwise
scalarizing lowering). Sum and bitwise combines use scalar arith; f32 max uses
metal.binary_exp; integer extrema use cmpi+select with the appropriate
signedness in the reduce body.

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


@triton.jit
def _and_combine(a, b):
    return a & b


@triton.jit
def _or_combine(a, b):
    return a | b


@triton.jit
def _xor_combine(a, b):
    return a ^ b


@triton.jit
def _colbitwise_i32(In, Out, M, N, BM: tl.constexpr, BN: tl.constexpr,
                    OP: tl.constexpr):
    pid = tl.program_id(0)
    cols = pid * BN + tl.arange(0, BN)
    rows = tl.arange(0, BM)
    mask = (rows[:, None] < M) & (cols[None, :] < N)
    offs = rows[:, None] * N + cols[None, :]
    other = -1 if OP == 0 else 0
    x = tl.load(In + offs, mask=mask, other=other)
    if OP == 0:
        result = tl.reduce(x, axis=0, combine_fn=_and_combine)
    elif OP == 1:
        result = tl.reduce(x, axis=0, combine_fn=_or_combine)
    else:
        result = tl.reduce(x, axis=0, combine_fn=_xor_combine)
    tl.store(Out + cols, result, mask=cols < N)


@triton.jit
def _colbitwise_i32_lc(In, Out, M, N, BM: tl.constexpr, BN: tl.constexpr,
                       OP: tl.constexpr):
    pid = tl.program_id(0)
    cols = pid * BN + tl.arange(0, BN)
    identity = -1 if OP == 0 else 0
    acc = tl.full((BM, BN), identity, tl.int32)
    for i in range(0, M, BM):
        rows = i + tl.arange(0, BM)
        mask = (rows[:, None] < M) & (cols[None, :] < N)
        offs = rows[:, None] * N + cols[None, :]
        x = tl.load(In + offs, mask=mask, other=identity)
        if OP == 0:
            acc = acc & x
        elif OP == 1:
            acc = acc | x
        else:
            acc = acc ^ x
    if OP == 0:
        result = tl.reduce(acc, axis=0, combine_fn=_and_combine)
    elif OP == 1:
        result = tl.reduce(acc, axis=0, combine_fn=_or_combine)
    else:
        result = tl.reduce(acc, axis=0, combine_fn=_xor_combine)
    tl.store(Out + cols, result, mask=cols < N)


def _bitwise_reference(inp, op):
    result = inp[0].clone()
    for row in inp[1:]:
        if op == 0:
            result = result & row
        elif op == 1:
            result = result | row
        else:
            result = result ^ row
    return result


@pytest.mark.parametrize("op", [0, 1, 2], ids=["and", "or", "xor"])
@pytest.mark.parametrize("M,N,BM,BN", [(8, 16, 8, 16), (8, 256, 8, 256),
                                       (5, 37, 8, 64)])
def test_reduce_axis0_bitwise_i32_direct(M, N, BM, BN, op):
    torch.manual_seed(M * 29 + N + op)
    inp = torch.randint(-2147483648, 2147483647, (M, N), dtype=torch.int32)
    out = torch.empty(N, dtype=torch.int32, device="mps")
    _colbitwise_i32[(triton.cdiv(N, BN),)](inp.to("mps"), out, M, N, BM=BM,
                                          BN=BN, OP=op, num_warps=1)
    torch.mps.synchronize()
    assert torch.equal(out.cpu(), _bitwise_reference(inp, op))


@pytest.mark.parametrize("op", [0, 1, 2], ids=["and", "or", "xor"])
def test_reduce_axis0_bitwise_i32_loopcarried(op):
    M, N, BM, BN = 20, 64, 8, 64
    torch.manual_seed(37 + op)
    inp = torch.randint(-2147483648, 2147483647, (M, N), dtype=torch.int32)
    out = torch.empty(N, dtype=torch.int32, device="mps")
    _colbitwise_i32_lc[(1,)](inp.to("mps"), out, M, N, BM=BM, BN=BN, OP=op,
                            num_warps=1)
    torch.mps.synchronize()
    assert torch.equal(out.cpu(), _bitwise_reference(inp, op))


@triton.jit
def _colbitwise_i32_reexpanded(In, Out, OutTile, BM: tl.constexpr,
                               BN: tl.constexpr):
    rows = tl.arange(0, BM)[:, None]
    cols = tl.arange(0, BN)[None, :]
    tile = tl.load(In + rows * BN + cols)
    reduced = tl.reduce(tile, axis=0, combine_fn=_xor_combine)
    tl.store(Out + tl.arange(0, BN), reduced)
    tl.store(OutTile + rows * BN + cols, tile ^ reduced[None, :])


def test_reduce_axis0_bitwise_i32_wide_reexpansion_is_named(capfd):
    BM, BN = 8, 128
    inp = torch.randint(-1000, 1000, (BM, BN), dtype=torch.int32, device="mps")
    out = torch.empty(BN, dtype=torch.int32, device="mps")
    out_tile = torch.empty(BM, BN, dtype=torch.int32, device="mps")
    with pytest.raises(Exception, match="convert-tritongpu-to-metal failed"):
        _colbitwise_i32_reexpanded[(1,)](inp, out, out_tile, BM=BM, BN=BN,
                                        num_warps=1)
    diagnostic = capfd.readouterr()
    assert "re-expansion back to a 2-D tile needs an f32 result" in (
        diagnostic.err + diagnostic.out)

    survivor = torch.empty(BN, dtype=torch.int32, device="mps")
    _colbitwise_i32[(1,)](inp, survivor, BM, BN, BM=BM, BN=BN, OP=2,
                         num_warps=1)
    torch.mps.synchronize()
    assert torch.equal(survivor.cpu(), _bitwise_reference(inp.cpu(), 2))


@triton.jit
def _colextremum_u32(In, Out, M, N, BM: tl.constexpr, BN: tl.constexpr,
                     IS_MIN: tl.constexpr):
    pid = tl.program_id(0)
    cols = pid * BN + tl.arange(0, BN)
    rows = tl.arange(0, BM)
    mask = (rows[:, None] < M) & (cols[None, :] < N)
    offs = rows[:, None] * N + cols[None, :]
    other = -1 if IS_MIN else 0
    x = tl.load(In + offs, mask=mask, other=other)
    x = x.to(tl.uint32, bitcast=True)
    if IS_MIN:
        result = tl.min(x, axis=0)
    else:
        result = tl.max(x, axis=0)
    tl.store(Out + cols, result.to(tl.int32, bitcast=True), mask=cols < N)


@triton.jit
def _colextremum_u32_lc(In, Out, M, N, BM: tl.constexpr, BN: tl.constexpr,
                        IS_MIN: tl.constexpr):
    pid = tl.program_id(0)
    cols = pid * BN + tl.arange(0, BN)
    identity = 0xffffffff if IS_MIN else 0
    acc = tl.full((BM, BN), identity, tl.uint32)
    for i in range(0, M, BM):
        rows = i + tl.arange(0, BM)
        mask = (rows[:, None] < M) & (cols[None, :] < N)
        offs = rows[:, None] * N + cols[None, :]
        other = -1 if IS_MIN else 0
        x = tl.load(In + offs, mask=mask, other=other)
        x = x.to(tl.uint32, bitcast=True)
        if IS_MIN:
            acc = tl.minimum(acc, x)
        else:
            acc = tl.maximum(acc, x)
    if IS_MIN:
        result = tl.min(acc, axis=0)
    else:
        result = tl.max(acc, axis=0)
    tl.store(Out + cols, result.to(tl.int32, bitcast=True), mask=cols < N)


def _unsigned_extremum_reference(inp, is_min):
    unsigned = inp.to(torch.int64) & 0xffffffff
    result = unsigned.amin(0) if is_min else unsigned.amax(0)
    signed = torch.where(result >= 0x80000000, result - 0x100000000, result)
    return signed.to(torch.int32)


@pytest.mark.parametrize("is_min", [False, True], ids=["umax", "umin"])
@pytest.mark.parametrize("M,N,BM,BN", [(8, 16, 8, 16), (8, 256, 8, 256),
                                       (5, 37, 8, 64)])
def test_reduce_axis0_unsigned_extrema_direct(M, N, BM, BN, is_min):
    torch.manual_seed(M * 31 + N + is_min)
    inp = torch.randint(-2147483648, 2147483647, (M, N), dtype=torch.int32)
    out = torch.empty(N, dtype=torch.int32, device="mps")
    _colextremum_u32[(triton.cdiv(N, BN),)](inp.to("mps"), out, M, N, BM=BM,
                                           BN=BN, IS_MIN=is_min, num_warps=1)
    torch.mps.synchronize()
    assert torch.equal(out.cpu(), _unsigned_extremum_reference(inp, is_min))


@pytest.mark.parametrize("is_min", [False, True], ids=["umax", "umin"])
def test_reduce_axis0_unsigned_extrema_loopcarried(is_min):
    M, N, BM, BN = 20, 64, 8, 64
    torch.manual_seed(43 + is_min)
    inp = torch.randint(-2147483648, 2147483647, (M, N), dtype=torch.int32)
    out = torch.empty(N, dtype=torch.int32, device="mps")
    _colextremum_u32_lc[(1,)](inp.to("mps"), out, M, N, BM=BM, BN=BN,
                             IS_MIN=is_min, num_warps=1)
    torch.mps.synchronize()
    assert torch.equal(out.cpu(), _unsigned_extremum_reference(inp, is_min))


@triton.jit
def _colextremum_u32_reexpanded(In, Out, OutTile, BM: tl.constexpr,
                                BN: tl.constexpr):
    rows = tl.arange(0, BM)[:, None]
    cols = tl.arange(0, BN)[None, :]
    tile = tl.load(In + rows * BN + cols)
    reduced = tl.max(tile.to(tl.uint32, bitcast=True), axis=0)
    reduced = reduced.to(tl.int32, bitcast=True)
    tl.store(Out + tl.arange(0, BN), reduced)
    tl.store(OutTile + rows * BN + cols, tile ^ reduced[None, :])


def test_reduce_axis0_unsigned_wide_reexpansion_is_named(capfd):
    BM, BN = 8, 128
    inp = torch.randint(-1000, 1000, (BM, BN), dtype=torch.int32, device="mps")
    out = torch.empty(BN, dtype=torch.int32, device="mps")
    out_tile = torch.empty(BM, BN, dtype=torch.int32, device="mps")
    with pytest.raises(Exception, match="convert-tritongpu-to-metal failed"):
        _colextremum_u32_reexpanded[(1,)](inp, out, out_tile, BM=BM, BN=BN,
                                         num_warps=1)
    diagnostic = capfd.readouterr()
    assert "re-expansion back to a 2-D tile needs an f32 result" in (
        diagnostic.err + diagnostic.out)


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


@triton.jit(do_not_specialize=["runtime_offset", "input_stride", "output_stride"])
def _colsum_scalar_offset_replay(
    x_ptr, out_ptr, runtime_offset, input_stride, output_stride,
    BM: tl.constexpr, BN: tl.constexpr, RUNTIME: tl.constexpr,
    COMPUTED: tl.constexpr,
):
    if RUNTIME:
        offset = runtime_offset + 3
    else:
        offset = 7
    rows = tl.arange(0, BM)
    cols = tl.arange(0, BN)
    base = x_ptr + offset + tl.program_id(0) * input_stride
    x = tl.load(base + rows[:, None] * BN + cols[None, :])
    if COMPUTED:
        x = x * x + 0.5
    result = tl.sum(x, axis=0)
    tl.store(out_ptr + offset + tl.program_id(0) * output_stride + cols, result)


@pytest.mark.parametrize("computed", [False, True], ids=["direct", "computed"])
@pytest.mark.parametrize("runtime", [False, True], ids=["constexpr", "runtime"])
def test_reduce_axis0_scalar_offset_replay(computed, runtime):
    """Scalar pointer terms survive per-column replay with multiple output bands."""
    m, n, programs, guard = 16, 256, 3, 7
    input_stride = m * n + 11
    output_stride = n + 13
    # The runtime path reuses one unspecialized signature for two physical
    # displacements, including an arithmetic dependency on a scalar argument.
    for offset in ((7, 13) if runtime else (7,)):
        x_cpu = torch.full((offset + programs * input_stride + guard,), 257.0)
        expected = torch.full((offset + programs * output_stride + guard,), 257.0)
        for program in range(programs):
            values = (torch.arange(m)[:, None] + torch.arange(n)[None, :]) % 7 - 3
            values = values.float() + program * 4
            input_start = offset + program * input_stride
            x_cpu[input_start:input_start + m * n] = values.flatten()
            source = values * values + 0.5 if computed else values
            output_start = offset + program * output_stride
            expected[output_start:output_start + n] = source.sum(dim=0)
        x = x_cpu.to("mps")
        out = torch.full_like(expected, 257.0, device="mps")
        _colsum_scalar_offset_replay[(programs,)](
            x, out, offset - 3, input_stride, output_stride,
            BM=m, BN=n, RUNTIME=runtime, COMPUTED=computed, num_warps=4,
        )
        torch.mps.synchronize()
        # Whole-allocation checks include the prefix, program gaps and suffix.
        torch.testing.assert_close(out.cpu(), expected, atol=0, rtol=0)
        torch.testing.assert_close(x.cpu(), x_cpu, atol=0, rtol=0)


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


@pytest.mark.parametrize("N, K", [(1, 1), (257, 1), (128, 4), (257, 3)])
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


@triton.jit
def _axis0_two_pass_arg_extrema(In, IndexOut, ValueOut, M, N, Iterations, OutStride,
                               BM: tl.constexpr, BN: tl.constexpr,
                               IS_MIN: tl.constexpr, COMPUTED: tl.constexpr):
    rows = tl.arange(0, BM)
    cols = tl.program_id(0) * BN + tl.arange(0, BN)
    mask = (rows[:, None] < M) & (cols[None, :] < N)
    for iteration in range(Iterations):
        tile = tl.load(In + iteration * M * N + rows[:, None] * N + cols[None, :],
                       mask=mask, other=float("nan"))
        if COMPUTED:
            tile = tile * 2.0
        if IS_MIN:
            index = tl.argmin(tile, axis=0)
            value = tl.min(tile, axis=0)
        else:
            index = tl.argmax(tile, axis=0)
            value = tl.max(tile, axis=0)
        tl.store(IndexOut + iteration * OutStride + cols, index, mask=cols < N)
        tl.store(ValueOut + iteration * OutStride + cols, value, mask=cols < N)


@pytest.mark.parametrize("is_min", [True, False], ids=["argmin", "argmax"])
@pytest.mark.parametrize("computed", [False, True], ids=["direct", "computed"])
@pytest.mark.parametrize("BN, num_warps, iterations", [
    (32, 1, 1), (128, 4, 1), (128, 1, 3), (32, 4, 2),
])
def test_reduce_axis0_two_pass_arg_extrema(BN, num_warps, iterations, computed, is_min):
    # The integer second pass must replay f32 leaves and read the first pass's
    # value for its own column, including when one thread owns several columns.
    M, N, guard = 5, BN + 7, 11
    patterns = torch.tensor([
        [3., -4., 3., -4., 1.],             # tied finite extrema
        [float("nan"), 2., -3., float("nan"), 4.],
        [float("nan")] * M,
        [float("inf")] * M,
        [-float("inf")] * M,
        [-0., 0., -0., 0., -0.],
        [0., 1., 2., 3., -9.],
        [0., -1., -2., -3., 9.],
    ], dtype=torch.float32).T
    tile = patterns[:, torch.arange(N) % patterns.shape[1]]
    # Shift winners between runtime loop iterations to expose stale publication.
    source = torch.stack([tile.roll(i, dims=0) for i in range(iterations)])
    stride = N + 2 * guard
    indices = torch.full((iterations * stride,), -12345, dtype=torch.int32, device="mps")
    values = torch.full((iterations * stride,), 12345., dtype=torch.float32, device="mps")
    _axis0_two_pass_arg_extrema[(triton.cdiv(N, BN),)](
        source.to("mps"), indices[guard:], values[guard:], M, N, iterations, stride,
        BM=8, BN=BN, IS_MIN=is_min, COMPUTED=computed, num_warps=num_warps,
    )
    torch.mps.synchronize()

    reference = source * 2.0 if computed else source
    is_nan = reference.isnan()
    identity = float("inf") if is_min else -float("inf")
    finite = reference.masked_fill(is_nan, identity)
    extrema = finite.amin(dim=1) if is_min else finite.amax(dim=1)
    all_nan = is_nan.all(dim=1)
    extrema = extrema.masked_fill(all_nan, float("nan"))
    matches = torch.where(all_nan[:, None, :], is_nan, reference == extrema[:, None, :])
    expected_indices = torch.where(matches, torch.arange(M)[None, :, None], 0x7fffffff).amin(dim=1)
    actual_indices = indices.cpu().reshape(iterations, stride)
    actual_values = values.cpu().reshape(iterations, stride)
    torch.testing.assert_close(actual_indices[:, guard:guard + N], expected_indices.to(torch.int32),
                               atol=0, rtol=0)
    torch.testing.assert_close(actual_values[:, guard:guard + N], extrema, atol=0, rtol=0, equal_nan=True)
    for output, canary in ((actual_indices, -12345), (actual_values, 12345.)):
        assert torch.all(output[:, :guard] == canary)
        assert torch.all(output[:, guard + N:] == canary)


@triton.jit
def _axis0_arg_transposed_column_leaf(In, Out, B: tl.constexpr, RESHAPE: tl.constexpr):
    rows = tl.arange(0, B)
    cols = tl.arange(0, B)
    tile = tl.load(In + rows[:, None] * B + cols[None, :])
    minimum = tl.min(tile, axis=0)
    matches = tile == minimum[None, :]
    if RESHAPE:
        # Two transposes have even parity, but the intervening reshape changes
        # which original column each final coordinate reads.
        matches = tl.trans(tl.reshape(matches, (B // 2, B * 2)))
        matches = tl.trans(tl.reshape(matches, (B, B)))
        # Preserve a blocked consumer so the mixed-leaf guard diagnoses this,
        # rather than the earlier rejection of linear-encoded reductions.
        matches = (tile > 0.0) & matches
    else:
        matches = tl.trans(matches)
    candidates = tl.where(matches, rows[:, None], 0x7fffffff)
    tl.store(Out + cols, tl.min(candidates, axis=0))


@pytest.mark.parametrize("reshape", [False, True], ids=["transpose", "reshape_transposes"])
def test_reduce_axis0_transposed_mixed_column_leaf_is_named(capfd, reshape):
    # Square dimensions cannot prove that a staged column scalar still belongs
    # to the requested coordinate: transposition turns it into a row value.
    block = 8 if reshape else 32
    inp = torch.randn(block, block, device="mps")
    out = torch.full((block,), -12345, dtype=torch.int32, device="mps")
    with pytest.raises(Exception, match="convert-tritongpu-to-metal failed"):
        _axis0_arg_transposed_column_leaf[(1,)](inp, out, B=block, RESHAPE=reshape, num_warps=1)
    diagnostic = capfd.readouterr()
    assert "axis=0 i32 reduce requires mixed f32 leaves" in diagnostic.err + diagnostic.out
    assert "untransposed per-column reduction results" in diagnostic.err + diagnostic.out
    assert torch.all(out.cpu() == -12345)

    survivor = torch.empty(block, device="mps")
    _colsum_direct[(1,)](inp, survivor, BM=block, BN=block, num_warps=1)
    torch.mps.synchronize()
    torch.testing.assert_close(survivor.cpu(), inp.cpu().sum(0), atol=1e-4, rtol=1e-4)


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


@triton.jit
def _colreduce_masked_other_f32(In, Out, BOUND, OTHER,
                                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                                MASK_AXIS: tl.constexpr, OP: tl.constexpr,
                                HAS_OTHER: tl.constexpr):
    rows = tl.arange(0, BLOCK_M)[:, None]
    cols = tl.arange(0, BLOCK_N)[None, :]
    if MASK_AXIS == 0:
        m = rows < BOUND
    else:
        m = cols < BOUND
    if HAS_OTHER:
        v = tl.load(In + rows * BLOCK_N + cols, mask=m, other=OTHER)
    else:
        v = tl.load(In + rows * BLOCK_N + cols, mask=m)
    if OP == "max":
        r = tl.max(v, axis=0)
    elif OP == "min":
        r = tl.min(v, axis=0)
    else:
        r = tl.sum(v, axis=0)
    tl.store(Out + tl.arange(0, BLOCK_N), r)


@triton.jit
def _colreduce_masked_other_i32(In, Out, BOUND,
                                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                                OTHER: tl.constexpr, OP: tl.constexpr):
    rows = tl.arange(0, BLOCK_M)[:, None]
    cols = tl.arange(0, BLOCK_N)[None, :]
    v = tl.load(In + rows * BLOCK_N + cols, mask=rows < BOUND, other=OTHER)
    if OP == "max":
        r = tl.max(v, axis=0)
    elif OP == "min":
        r = tl.min(v, axis=0)
    else:
        r = tl.sum(v, axis=0)
    tl.store(Out + tl.arange(0, BLOCK_N), r)


# A masked-off row contributes the load's `other`, not the combine identity.
# `other` is deliberately away from every identity here: with 0.0 and a sum the
# two coincide, which is exactly why the existing `_colsum` above (mask +
# `other=0.`) cannot see the difference.
@pytest.mark.parametrize(
    "M, N, bound, mask_axis, op, other",
    [
        (4, 8, 2, 0, "max", 5.0),
        (4, 8, 2, 0, "min", -5.0),
        (4, 8, 2, 0, "sum", 3.0),
        (4, 8, 5, 1, "max", 5.0),      # whole columns masked
        (4, 8, 5, 1, "min", -5.0),
        (4, 8, 5, 1, "sum", 3.0),
        (32, 128, 7, 0, "max", 9.5),   # E_out == 1, four warps
        (32, 128, 7, 0, "sum", -2.5),
        (8, 256, 3, 0, "max", 4.25),   # more columns than threads
        (8, 256, 3, 0, "sum", 1.75),
    ],
    ids=["rowmask_max", "rowmask_min", "rowmask_sum", "colmask_max",
         "colmask_min", "colmask_sum", "wide_rowmask_max", "wide_rowmask_sum",
         "eout_rowmask_max", "eout_rowmask_sum"],
)
def test_reduce_axis0_masked_other_matches_torch(M, N, bound, mask_axis, op,
                                                 other):
    torch.manual_seed(M * 31 + N + bound)
    inp = torch.randn(M, N)
    rows = torch.arange(M)[:, None]
    cols = torch.arange(N)[None, :]
    keep = (rows < bound) if mask_axis == 0 else (cols < bound)
    tile = torch.where(keep, inp, torch.full((M, N), other))
    ref = (tile.max(dim=0).values if op == "max" else
           tile.min(dim=0).values if op == "min" else tile.sum(dim=0))
    out = torch.empty(N, device="mps")
    _colreduce_masked_other_f32[(1,)](
        inp.to("mps"), out, bound, other, BLOCK_M=M, BLOCK_N=N,
        MASK_AXIS=mask_axis, OP=op, HAS_OTHER=True, num_warps=4)
    torch.mps.synchronize()
    torch.testing.assert_close(out.cpu(), ref, atol=1e-4, rtol=1e-4)


# Without `other` the masked value is unspecified, so the backend drives it to
# the combine identity to keep the result deterministic. Pin that rule too:
# it is the only case the identity substitution is correct for.
@pytest.mark.parametrize(
    "op, identity",
    [("max", float("-inf")), ("min", float("inf")), ("sum", 0.0)],
    ids=["max_identity", "min_identity", "sum_identity"],
)
def test_reduce_axis0_masked_without_other_uses_identity(op, identity):
    M, N, bound = 4, 8, 2
    torch.manual_seed(7)
    inp = torch.randn(M, N)
    rows = torch.arange(M)[:, None]
    tile = torch.where(rows < bound, inp, torch.full((M, N), identity))
    ref = (tile.max(dim=0).values if op == "max" else
           tile.min(dim=0).values if op == "min" else tile.sum(dim=0))
    out = torch.empty(N, device="mps")
    _colreduce_masked_other_f32[(1,)](
        inp.to("mps"), out, bound, 0.0, BLOCK_M=M, BLOCK_N=N, MASK_AXIS=0,
        OP=op, HAS_OTHER=False, num_warps=4)
    torch.mps.synchronize()
    torch.testing.assert_close(out.cpu(), ref, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize(
    "op, other",
    [("max", 7), ("min", -7), ("sum", 3)],
    ids=["i32_max", "i32_min", "i32_sum"],
)
def test_reduce_axis0_masked_other_i32(op, other):
    M, N, bound = 4, 16, 2
    torch.manual_seed(11)
    inp = torch.randint(-50, 50, (M, N), dtype=torch.int32)
    rows = torch.arange(M)[:, None]
    tile = torch.where(rows < bound, inp, torch.full((M, N), other,
                                                     dtype=torch.int32))
    ref = (tile.max(dim=0).values if op == "max" else
           tile.min(dim=0).values if op == "min" else
           tile.sum(dim=0).to(torch.int32))
    out = torch.empty(N, dtype=torch.int32, device="mps")
    _colreduce_masked_other_i32[(1,)](
        inp.to("mps"), out, bound, BLOCK_M=M, BLOCK_N=N, OTHER=other, OP=op,
        num_warps=4)
    torch.mps.synchronize()
    torch.testing.assert_close(out.cpu(), ref)


@triton.jit
def _colsum_broadcast_by_row(In, Out, BM: tl.constexpr, BN: tl.constexpr):
    rows = tl.arange(0, BM)
    cols = tl.arange(0, BN)
    tile = tl.load(In + rows[:, None] * BN + cols[None, :])
    sums = tl.sum(tile, axis=0)
    tl.store(Out + rows[:, None] * BN + cols[None, :], sums[:, None])


def test_reduce_axis0_result_broadcast_by_row_matches_torch():
    block = 16
    inp = torch.arange(block * block, dtype=torch.float32).reshape(block, block)
    out = torch.zeros_like(inp, device="mps")
    _colsum_broadcast_by_row[(1,)](inp.to("mps"), out, BM=block, BN=block,
                                   num_warps=1)
    torch.mps.synchronize()
    expected = inp.sum(0)[:, None].expand(block, block)
    torch.testing.assert_close(out.cpu(), expected, atol=0, rtol=0)
