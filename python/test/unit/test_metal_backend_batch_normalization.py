"""Batch normalization on the Metal backend.

Runs the verbatim `leet-triton/medium-batch_normalization.py` kernel: a
three-pass per-channel normalization whose first two passes are rank-2 **axis=0
(per-column)** reduces over the batch dimension, and whose third pass broadcasts
the resulting statistics back into a tile.

It needed three fixes, each independently observable:

1. **Computed cone on an axis=0 reduce.** `lowerRank2Axis0Reduce` only accepted a
   direct masked `tt.load` (or the loop-carried accumulator the reassociation
   produces). The variance pass reduces
   `tl.where(mask, x - mean[None, :], 0.) ** 2`, which is a COMPUTED tile, so it
   hard-failed to legalize. It now re-derives the cone per (row, col) with
   `evalRank2ConeAt` — the same evaluator the axis=1 reduce uses for softmax
   cones — gated by the representative load's mask so a ragged M neither reads
   out of bounds (`g_coneAddrGuard`) nor contributes an element.

2. **`rank2ConeSupported` was missing `arith.and`/`or`.** The evaluator has
   handled them all along, so the predicate — the dry run that decides whether
   the evaluator gets to emit at all — rejected every cone whose mask is the
   ordinary `(rows < M) & (cols < N)`. Exactly the desync its sibling comment in
   `rank1ConeSupported` warns about.

3. **Two element conventions met at `tt.expand_dims`.** A rank-1 value holds
   element `localTid` in the thread; inside a rank-2 tile the thread holds column
   `(localTid*E + iv) % BLOCK_N`. Broadcast was a plain identity passthrough, so
   `mean[None, :]` gave every one of a thread's columns the mean of column
   `localTid`. Nothing crashed — batch norm just normalised each column by a
   neighbouring channel's statistics. A device load reaching the same place is
   already in the tile convention (its address came from a `tt.make_range`
   lowered inside this tile), which is why a loaded row broadcast correctly and
   only reduce-derived rows were wrong. `ExpandDimsLowering` now republishes a
   column-reduce result through a threadgroup buffer and reads back the slot the
   thread's own column needs; `preprocessAxis0Broadcasts` decides which
   expand_dims those are, pre-conversion (the reduce lowers first and a converted
   op is detached, so the same walk from the pattern would see an empty cone).

`test_expand_dims_after_column_reduce` pins (3) on its own: the smallest kernel
that stores a column reduce BOTH as a rank-1 row and broadcast into a tile. The
two disagreeing is the whole bug, and it is invisible in the rank-1 store alone.
(1) and (2) are pinned next to the rest of the axis=0 reduce coverage, in
`test_metal_backend_reduce_rank2_axis0.py`.
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


# Verbatim copy of leet-triton/medium-batch_normalization.py.
@triton.jit
def _batch_norm_kernel(
    input_ptr, gamma_ptr, beta_ptr,
    output_ptr,
    N, C,
    eps,
    BC: tl.constexpr, BN: tl.constexpr,
):
    pid = tl.program_id(0)

    mean = tl.zeros((BC,), dtype=tl.float32)
    for n_start in range(0, N, BN):
        off_n = n_start + tl.arange(0, BN)
        mask_n = off_n < N
        off_c = pid * BC + tl.arange(0, BC)
        mask_c = off_c < C
        input_vals = tl.load(input_ptr + off_n[:, None] * C + off_c[None, :], mask=mask_n[:, None] & mask_c[None, :], other=0.0)
        mean += tl.sum(input_vals, axis=0)
    mean /= N

    var = tl.zeros((BC,), dtype=tl.float32)
    for n_start in range(0, N, BN):
        off_n = n_start + tl.arange(0, BN)
        mask_n = off_n < N
        off_c = pid * BC + tl.arange(0, BC)
        mask_c = off_c < C
        input_vals = tl.load(input_ptr + off_n[:, None] * C + off_c[None, :], mask=mask_n[:, None] & mask_c[None, :], other=0.0)
        input_vals -= mean[None, :]
        input_vals = tl.where(mask_n[:, None] & mask_c[None, :], input_vals, 0.0)
        var += tl.sum(input_vals * input_vals, axis=0)
    var /= N

    inv_std_var = 1 / tl.sqrt(var + eps)

    off_c = pid * BC + tl.arange(0, BC)
    mask_c = off_c < C

    gamma = tl.load(gamma_ptr + off_c, mask=mask_c, other=0.0)
    beta = tl.load(beta_ptr + off_c, mask=mask_c, other=0.0)

    for n_start in range(0, N, BN):
        off_n = n_start + tl.arange(0, BN)
        mask_n = off_n < N
        input_vals = tl.load(input_ptr + off_n[:, None] * C + off_c[None, :], mask=mask_n[:, None] & mask_c[None, :], other=0.0)
        input_vals -= mean[None, :]
        input_vals *= inv_std_var[None, :]
        input_vals = tl.where(mask_n[:, None] & mask_c[None, :], input_vals, 0.0)

        output = gamma[None, :] * input_vals + beta[None, :]
        tl.store(output_ptr + off_n[:, None] * C + off_c[None, :], output, mask=mask_n[:, None] & mask_c[None, :])


def _run(N, C, eps=1e-5, BC=16, BN=16, num_warps=4, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(N, C, device="mps")
    gamma = torch.randn(C, device="mps")
    beta = torch.randn(C, device="mps")
    out = torch.zeros(N, C, device="mps")
    _batch_norm_kernel[(triton.cdiv(C, BC),)](
        x, gamma, beta, out, N, C, eps, BC=BC, BN=BN, num_warps=num_warps)
    torch.mps.synchronize()
    # float64 on the CPU: `var` is a sum of squared deviations, and the kernel
    # accumulates it in fp32 over N terms.
    xc = x.cpu().double()
    mean = xc.mean(dim=0)
    var = ((xc - mean) ** 2).mean(dim=0)
    ref = (xc - mean) / torch.sqrt(var + eps) * gamma.cpu().double() + beta.cpu().double()
    return out.cpu().double(), ref


# N: 1 (var == 0), ragged vs BN (7/100/33), multi-tile (256/1000). C: 1, ragged
# vs BC (5/33/40), several programs (128/256).
@pytest.mark.parametrize("N, C", [(16, 16), (64, 32), (100, 40), (7, 5), (1, 16),
                                  (128, 33), (33, 128), (256, 256), (3, 1),
                                  (1000, 17)])
def test_batch_normalization(N, C):
    got, ref = _run(N, C, seed=N * 131 + C)
    torch.testing.assert_close(got, ref, atol=1e-4, rtol=1e-4)


# BC vs BN vs tpb is what decides the tile's element-per-thread count, i.e. how
# far the rank-1 and tile column conventions diverge. BC == 8 gives E == 1 with
# tpb == 128 — the case that showed the republish cannot be skipped on E == 1.
@pytest.mark.parametrize("BC, BN", [(16, 16), (32, 16), (16, 32), (32, 32),
                                    (64, 16), (128, 16), (8, 8)])
def test_batch_normalization_block_sizes(BC, BN):
    got, ref = _run(100, 64, BC=BC, BN=BN, seed=BC * 100 + BN)
    torch.testing.assert_close(got, ref, atol=1e-4, rtol=1e-4)


# num_warps moves tpb (32/64/128/256) against a fixed BLOCK_N=16. tpb=256 is the
# other E == 1 case.
@pytest.mark.parametrize("num_warps", [1, 2, 4, 8])
def test_batch_normalization_num_warps(num_warps):
    got, ref = _run(100, 64, num_warps=num_warps, seed=num_warps)
    torch.testing.assert_close(got, ref, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("eps", [1e-9, 1e-5, 1e-1])
def test_batch_normalization_eps(eps):
    got, ref = _run(64, 32, eps=eps, seed=7)
    torch.testing.assert_close(got, ref, atol=1e-4, rtol=1e-4)


# --------------------------------------------------------------------------
# The three fixes in isolation.


@triton.jit
def _colmean_two_ways(In, RowOut, TileOut, N, C, BC: tl.constexpr, BN: tl.constexpr):
    # Fix (3): the SAME column reduce consumed as a rank-1 row and broadcast
    # into a rank-2 tile. Before the republish the row was right and the tile
    # held each value repeated E times (each thread's own `localTid` element).
    pid = tl.program_id(0)
    mean = tl.zeros((BC,), dtype=tl.float32)
    for n_start in range(0, N, BN):
        off_n = n_start + tl.arange(0, BN)
        off_c = pid * BC + tl.arange(0, BC)
        v = tl.load(In + off_n[:, None] * C + off_c[None, :],
                    mask=(off_n[:, None] < N) & (off_c[None, :] < C), other=0.0)
        mean += tl.sum(v, axis=0)
    mean /= N
    off_c = pid * BC + tl.arange(0, BC)
    mask_c = off_c < C
    tl.store(RowOut + off_c, mean, mask=mask_c)
    off_n = tl.arange(0, BN)
    tile = mean[None, :] + tl.zeros((BN, BC), dtype=tl.float32)
    tl.store(TileOut + off_n[:, None] * C + off_c[None, :], tile,
             mask=(off_n[:, None] < N) & mask_c[None, :])


@pytest.mark.parametrize("N, C, BC, BN, num_warps",
                         [(64, 32, 16, 16, 4),   # E=2
                          (64, 128, 128, 16, 4),  # E=16
                          (64, 32, 32, 16, 4),   # E=4
                          (64, 64, 8, 8, 4),     # E=1, tpb > BLOCK_N
                          (64, 32, 16, 16, 8)])  # E=1 via tpb=256
def test_expand_dims_after_column_reduce(N, C, BC, BN, num_warps):
    torch.manual_seed(N + C + BC)
    x = torch.randn(N, C, device="mps")
    row = torch.zeros(C, device="mps")
    tile = torch.zeros(N, C, device="mps")
    _colmean_two_ways[(triton.cdiv(C, BC),)](x, row, tile, N, C, BC=BC, BN=BN,
                                             num_warps=num_warps)
    torch.mps.synchronize()
    ref = x.cpu().mean(0)
    torch.testing.assert_close(row.cpu(), ref, atol=1e-5, rtol=1e-5)
    # Every row of the tile is that same vector — this is the assertion that
    # used to fail while the rank-1 store above passed.
    for r in range(min(N, BN)):
        torch.testing.assert_close(tile.cpu()[r], ref, atol=1e-5, rtol=1e-5)
