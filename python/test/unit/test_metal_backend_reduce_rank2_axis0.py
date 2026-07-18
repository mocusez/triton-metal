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

Combines: f32 sum/max and i32 sum/max/min. Sum uses scalar arith; f32 max uses
metal.binary_exp; i32 max/min use cmpi+select (binary_exp rejects signless i32).
Loop-carried reassociation ships for f32 sum/max and i32 sum (elementwise
maxnumf/addi lowerings); loop-carried i32 max/min and output E>1 (BN > tpb) stay
deferred. Output E==1 (launch tpb >= BN).
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
# f32 max and i32 sum/max/min. Sum uses scalar arith; f32 max uses
# metal.binary_exp maxOp; i32 max/min use cmpi+select (binary_exp rejects
# signless i32). Loop-carried f32 max/i32 sum reassociate (elementwise maxnumf/
# addi lowerings); loop-carried i32 max/min stay deferred (direct only).


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
