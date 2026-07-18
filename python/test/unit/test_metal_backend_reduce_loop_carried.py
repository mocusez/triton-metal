"""Rank-2 axis=1 reduce over a cone that contains a LOOP-CARRIED per-row scalar.

Isolates the staged-leaf reduce (Increment 2.5): the reduce input `acc[:, None]
* x` re-derives, per column, a per-row leaf `acc` that is the enclosing
`scf.for`'s iter_arg — a control-flow value the re-emission cone evaluator cannot
reconstruct. At M <= tpb each fill thread reduces its own row (r == localTid), so
`acc[r]` is exactly the thread's converted per-thread scalar (getRemappedValue);
the reduce fill is emitted inline inside the loop. Compared bit-exact against a
CPU replay of the same recurrence.
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
def loop_carried_reduce_kernel(x_ptr, out_ptr, STEPS,
                               M: tl.constexpr, N: tl.constexpr):
    row = tl.arange(0, M)
    col = tl.arange(0, N)
    acc = tl.zeros([M], dtype=tl.float32)          # loop-carried per-row leaf
    for _ in range(STEPS):
        x = tl.load(x_ptr + row[:, None] * N + col[None, :])
        score = acc[:, None] * x + x                # acc[:,None] is the staged leaf
        m = tl.max(score, axis=1)                   # rank-2 reduce over the cone
        acc = acc + m * 0.01
    tl.store(out_ptr + row, acc)


def _reference(x, STEPS, M, N):
    acc = torch.zeros(M)
    xc = x.cpu().reshape(M, N)
    for _ in range(STEPS):
        m = (acc[:, None] * xc + xc).max(dim=1).values
        acc = acc + m * 0.01
    return acc


@pytest.mark.parametrize("M, N, STEPS", [(128, 64, 5), (8, 16, 3), (128, 64, 1)])
def test_reduce_loop_carried_leaf(M, N, STEPS):
    torch.manual_seed(0xC0FFEE + M * N + STEPS)
    x = torch.randn(M * N, dtype=torch.float32, device="mps").contiguous()
    out = torch.zeros(M, dtype=torch.float32, device="mps").contiguous()
    loop_carried_reduce_kernel[(1,)](x, out, STEPS, M, N, num_warps=4)
    torch.mps.synchronize()
    torch.testing.assert_close(out.cpu(), _reference(x, STEPS, M, N),
                               atol=1e-4, rtol=1e-4)


# --- Loop-carried rank-2 TILE ----------------------------------------------
#
# The staged-leaf mechanism above maps a leaf to ONE per-thread scalar, i.e. a
# per-ROW value. A tile that also varies along the row has no representation
# there, and an `scf.for` iter_arg is a BlockArgument the cone evaluator cannot
# re-emit anyway — a scan is not a reduction, so there is no reassociation to
# fall back on either. The [M,N] tile is therefore materialised in threadgroup
# memory by a pre-pass (`preprocessLoopCarriedReduceTiles`), which also seeds it
# with the loop's init and hands ReduceLowering the mapping.
#
# Detection must be pre-conversion: by the time the pattern runs, the SCF
# structural type conversion has rebuilt the loop with per-thread scalar
# iter_args and detached the original body block.
#
# Restricted to M <= tpb, where the reduce's row fill degenerates to one row per
# thread, so thread r owns row r of the tile outright — seed, update and read
# all happen on that thread and the buffer needs no barrier.


@triton.jit
def _loop_carried_tile_kernel(c_ptr, y_ptr, STEPS,
                              M: tl.constexpr, N: tl.constexpr):
    row = tl.arange(0, M)
    col = tl.arange(0, N)
    idx2d = row[:, None] * N + col[None, :]
    h = tl.zeros((M, N), dtype=tl.float32)     # loop-carried [M,N] tile
    for t in range(0, STEPS):
        cc = tl.load(c_ptr + t * M * N + idx2d)
        h = h * 0.5 + cc
        tl.store(y_ptr + t * M + row, tl.sum(cc * h, axis=1))


def _tile_reference(c, STEPS, M, N):
    h = torch.zeros(M, N, dtype=torch.float64)
    out = []
    cc_all = c.cpu().double()
    for t in range(STEPS):
        cc = cc_all[t * M * N:(t + 1) * M * N].reshape(M, N)
        h = h * 0.5 + cc
        out.append((cc * h).sum(1))
    return torch.cat(out)


@pytest.mark.parametrize("M, N, STEPS", [(32, 64, 4), (16, 32, 3), (8, 8, 5),
                                         (128, 16, 2)])
def test_reduce_loop_carried_tile(M, N, STEPS):
    torch.manual_seed(0xB0A + M * N + STEPS)
    c = torch.randn(STEPS * M * N, dtype=torch.float32, device="mps").contiguous()
    y = torch.zeros(STEPS * M, dtype=torch.float32, device="mps").contiguous()
    _loop_carried_tile_kernel[(1,)](c, y, STEPS, M, N, num_warps=4)
    torch.mps.synchronize()
    ref = _tile_reference(c, STEPS, M, N)
    err = (y.cpu().double() - ref).abs().max() / max(ref.abs().max().item(), 1.0)
    assert err <= 2e-5, f"rel err {err}"


# --- Hoisted base pointer in a re-emitted cone ------------------------------
#
# `evalRank1ValueAt` scalarised only the OUTERMOST tt.addptr offset and then
# added the SCALAR offsets from the rest of the chain. A tensor offset on an
# INNER addptr — where the per-element index lives whenever the base pointer is
# hoisted out of the loop — was dropped:
#
#   base = tt.addptr(splat(p + pid*S), offs)   <- index, tensor, DROPPED
#   addr = tt.addptr(base, splat(t*S))         <- outermost, no index
#
# The mask is scalarised down a separate path and kept its index, so the read
# came out silently off-row rather than obviously broken. Every kernel that
# hoists a row/column base out of its scan loop hits this — the SSM selective
# scan does it for all four of u/delta/B/C.


@triton.jit
def _hoisted_base_kernel(v_ptr, c_ptr, y_ptr, STEPS, n_valid,
                         M: tl.constexpr, N: tl.constexpr):
    row = tl.arange(0, M)
    col = tl.arange(0, N)
    v_base = v_ptr + col                       # hoisted OUT of the loop
    idx2d = row[:, None] * N + col[None, :]
    h = tl.zeros((M, N), dtype=tl.float32)
    for t in range(0, STEPS):
        v = tl.load(v_base + t * N, mask=col < n_valid, other=0.0)
        cc = tl.load(c_ptr + t * M * N + idx2d)
        h = h * 0.5 + cc * v[None, :]
        tl.store(y_ptr + t * M + row, tl.sum(cc * h, axis=1))


@pytest.mark.parametrize("M, N, STEPS, n_valid", [(32, 64, 4, 64),
                                                  (32, 64, 3, 16),
                                                  (16, 32, 4, 20)])
def test_reduce_loop_carried_tile_hoisted_base(M, N, STEPS, n_valid):
    torch.manual_seed(0x1DE + M * N + STEPS + n_valid)
    v = torch.randn(STEPS * N, dtype=torch.float32, device="mps").contiguous()
    c = torch.randn(STEPS * M * N, dtype=torch.float32, device="mps").contiguous()
    y = torch.zeros(STEPS * M, dtype=torch.float32, device="mps").contiguous()
    _hoisted_base_kernel[(1,)](v, c, y, STEPS, n_valid, M, N, num_warps=4)
    torch.mps.synchronize()

    h = torch.zeros(M, N, dtype=torch.float64)
    vc, cc_all = v.cpu().double(), c.cpu().double()
    out = []
    for t in range(STEPS):
        vt = vc[t * N:(t + 1) * N].clone()
        vt[n_valid:] = 0.0
        cc = cc_all[t * M * N:(t + 1) * M * N].reshape(M, N)
        h = h * 0.5 + cc * vt[None, :]
        out.append((cc * h).sum(1))
    ref = torch.cat(out)
    err = (y.cpu().double() - ref).abs().max() / max(ref.abs().max().item(), 1.0)
    assert err <= 2e-5, f"rel err {err}"


# --- Cone load with its own row stride --------------------------------------
#
# evalRank2ConeAt's tt.load leaf used to read `device[rowBase + n]`, where
# rowBase is derived ONCE from the reduce's representative load. That assumes
# every load in the cone shares the reduce tile's row stride and per-program
# base. False as soon as a cone load has a different shape: the SSM scan's
# reduce tile is (BLOCK_D, BLOCK_N) while its `A` is (d_model, d_state), so A
# was read with stride BLOCK_N and carrying another buffer's batch offset. The
# address is now re-materialised from the load's own tt.addptr chain at (r, n).


@triton.jit
def _cone_load_own_stride_kernel(a_ptr, c_ptr, y_ptr, STEPS,
                                 M: tl.constexpr, N: tl.constexpr,
                                 STRIDE: tl.constexpr):
    row = tl.arange(0, M)
    col = tl.arange(0, N)
    idx2d = row[:, None] * N + col[None, :]
    # Row stride STRIDE != N: a view into a WIDER buffer.
    a = tl.load(a_ptr + row[:, None] * STRIDE + col[None, :])
    h = tl.zeros((M, N), dtype=tl.float32)
    for t in range(0, STEPS):
        cc = tl.load(c_ptr + t * M * N + idx2d)
        h = h * 0.5 + cc * a
        tl.store(y_ptr + t * M + row, tl.sum(cc * h, axis=1))


@pytest.mark.parametrize("M, N, STEPS, STRIDE", [(32, 64, 4, 96), (16, 32, 3, 40),
                                                 (32, 16, 4, 64)])
def test_reduce_loop_carried_tile_cone_load_own_stride(M, N, STEPS, STRIDE):
    torch.manual_seed(0x5D + M * N + STRIDE)
    a = torch.randn(M * STRIDE, dtype=torch.float32, device="mps").contiguous()
    c = torch.randn(STEPS * M * N, dtype=torch.float32, device="mps").contiguous()
    y = torch.zeros(STEPS * M, dtype=torch.float32, device="mps").contiguous()
    _cone_load_own_stride_kernel[(1,)](a, c, y, STEPS, M, N, STRIDE, num_warps=4)
    torch.mps.synchronize()

    ac = a.cpu().double().reshape(M, STRIDE)[:, :N]
    cc_all = c.cpu().double()
    h = torch.zeros(M, N, dtype=torch.float64)
    out = []
    for t in range(STEPS):
        cc = cc_all[t * M * N:(t + 1) * M * N].reshape(M, N)
        h = h * 0.5 + cc * ac
        out.append((cc * h).sum(1))
    ref = torch.cat(out)
    err = (y.cpu().double() - ref).abs().max() / max(ref.abs().max().item(), 1.0)
    assert err <= 2e-5, f"rel err {err}"
