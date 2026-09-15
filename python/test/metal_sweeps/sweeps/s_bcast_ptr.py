"""A 2-D tile address with a UNIFORM axis: does it give the right answer?

Triton folds an axis whose address contribution is uniform and hands the access
a `tt.broadcast` of a lower-rank pointer tensor. Every gate in the backend used
to refuse that shape (and, before the shared address predicate landed, kill the
process doing it). This sweep is what says whether admitting it is CORRECT —
because the cheap way to admit it is to widen the predicate and leave the
emitter alone, and that has produced a silent wrong answer here before.

Axes, and why each one is here:

  uniform axis {row, col}   the two spellings need opposite dense axes
  op {load, store}          the read and write sides derive the index separately
  mask {nomask, other}      `other` is the only path that must blend a value in
  dtype {f32, f16, i32}     the storage bridge differs per width
  BLOCK {16, 64}            tile extent vs threads-per-block
  num_warps {1, 2, 4, 8}    decides how many ENCODINGS a kernel has — at 1 warp
                            32 threads cannot cover a 64-element tile one
                            element each, so Triton mints a second one
  N % BLOCK {0, nonzero}    a runtime inner dim that is not a multiple of 16
                            mints a phantom encoding with the opposite order

Run:  ./drive.sh sweeps/s_bcast_ptr.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

from sweep_lib import Sweep  # noqa: E402


# ---- read a COLUMN, broadcast across the tile's columns --------------------
@triton.jit
def load_col(X, O, sx, N, K, BLOCK: tl.constexpr, OTHER: tl.constexpr):
    pi = tl.program_id(0)
    pj = tl.program_id(1)
    i = pi * BLOCK + tl.arange(0, BLOCK)
    j = pj * BLOCK + tl.arange(0, BLOCK)
    ko = tl.full((BLOCK,), K, dtype=tl.int32)
    src = X + i[:, None] * sx + ko[None, :]
    if OTHER:
        v = tl.load(src, mask=(i[:, None] < N), other=0)
    else:
        v = tl.load(src)
    tl.store(O + i[:, None] * sx + j[None, :], v,
             mask=(i[:, None] < N) & (j[None, :] < N))


# ---- read a ROW, broadcast down the tile's rows ----------------------------
@triton.jit
def load_row(X, O, sx, N, K, BLOCK: tl.constexpr, OTHER: tl.constexpr):
    pi = tl.program_id(0)
    pj = tl.program_id(1)
    i = pi * BLOCK + tl.arange(0, BLOCK)
    j = pj * BLOCK + tl.arange(0, BLOCK)
    ko = tl.full((BLOCK,), K, dtype=tl.int32)
    src = X + ko[:, None] * sx + j[None, :]
    if OTHER:
        v = tl.load(src, mask=(j[None, :] < N), other=0)
    else:
        v = tl.load(src)
    tl.store(O + i[:, None] * sx + j[None, :], v,
             mask=(i[:, None] < N) & (j[None, :] < N))


# ---- write a COLUMN: the destination's column axis is uniform --------------
@triton.jit
def store_col(X, O, sx, N, K, BLOCK: tl.constexpr, OTHER: tl.constexpr):
    pi = tl.program_id(0)
    pj = tl.program_id(1)
    i = pi * BLOCK + tl.arange(0, BLOCK)
    j = pj * BLOCK + tl.arange(0, BLOCK)
    ko = tl.full((BLOCK,), K, dtype=tl.int32)
    v = tl.load(X + i[:, None] * sx + j[None, :],
                mask=(i[:, None] < N) & (j[None, :] < N), other=0)
    # One lane per row wins, so the shape is well defined rather than a race.
    tl.store(O + i[:, None] * sx + ko[None, :], v,
             mask=(j[None, :] == 0) & (i[:, None] < N))


# ---- write a ROW: the destination's row axis is uniform --------------------
@triton.jit
def store_row(X, O, sx, N, K, BLOCK: tl.constexpr, OTHER: tl.constexpr):
    pi = tl.program_id(0)
    pj = tl.program_id(1)
    i = pi * BLOCK + tl.arange(0, BLOCK)
    j = pj * BLOCK + tl.arange(0, BLOCK)
    ko = tl.full((BLOCK,), K, dtype=tl.int32)
    v = tl.load(X + i[:, None] * sx + j[None, :],
                mask=(i[:, None] < N) & (j[None, :] < N), other=0)
    tl.store(O + ko[:, None] * sx + j[None, :], v,
             mask=(i[:, None] == 0) & (j[None, :] < N))


KERNELS = {
    "load_col": load_col,
    "load_row": load_row,
    "store_col": store_col,
    "store_row": store_row,
}

DTYPES = {
    "f32": torch.float32,
    "f16": torch.float16,
    "i32": torch.int32,
}


def reference(name, x, N, K):
    """What the kernel must produce, as a full NxN tensor."""
    out = torch.zeros_like(x)
    if name == "load_col":
        out[:N, :N] = x[:N, K:K + 1].expand(N, N)
    elif name == "load_row":
        out[:N, :N] = x[K:K + 1, :N].expand(N, N)
    elif name == "store_col":
        out[:N, K] = x[:N, 0]
    elif name == "store_row":
        out[K, :N] = x[0, :N]
    return out


def make_input(dtype, N):
    if dtype == torch.int32:
        return torch.arange(N * N, dtype=dtype).reshape(N, N)
    g = torch.Generator().manual_seed(0)
    return (torch.rand(N, N, generator=g) * 100).to(dtype)


sw = Sweep("s_bcast_ptr")

for name in ("load_col", "load_row", "store_col", "store_row"):
    for other in (False, True):
        # `other` only changes the READ side; skip the redundant store arm.
        if other and name.startswith("store"):
            continue
        for dt_name, dt in DTYPES.items():
            for BLOCK in (16, 64):
                for nw in (1, 2, 4, 8):
                    for div in ("Y", "n"):
                        N = BLOCK * 2 if div == "Y" else BLOCK * 2 - 3
                        K = N // 3
                        tag = (f"{name}_{'other' if other else 'nomask'}_"
                               f"{dt_name}_B{BLOCK}_nw{nw}_div{div}")

                        def cell(name=name, other=other, dt=dt, BLOCK=BLOCK,
                                 nw=nw, N=N, K=K):
                            x = make_input(dt, N)
                            xm = x.to("mps")
                            om = torch.zeros(N, N, dtype=dt, device="mps")
                            grid = (triton.cdiv(N, BLOCK), triton.cdiv(N, BLOCK))
                            KERNELS[name][grid](xm, om, N, N, K, BLOCK=BLOCK,
                                                OTHER=other, num_warps=nw)
                            torch.mps.synchronize()
                            got = om.cpu()
                            want = reference(name, x, N, K)
                            if torch.equal(got, want):
                                return None
                            bad = (got != want)
                            first = bad.nonzero()[0].tolist()
                            return (f"{int(bad.sum())}/{want.numel()} differ; "
                                    f"[{first[0]},{first[1]}] got="
                                    f"{got[first[0], first[1]].item()} "
                                    f"want={want[first[0], first[1]].item()}")

                        sw.run(tag, cell)

sw.finish()
