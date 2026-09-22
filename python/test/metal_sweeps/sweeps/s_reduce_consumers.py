"""What a reduce's result is allowed to BECOME.

`63d45af467` closed "rank-3 reduce, rank-2 result stored directly" and
`.omc/fuzz/rank3_reduce_rank2_store.py` holds it at 0 wrong. That round's own
lesson was that the sweep which had previously cleared the family only ever
broadcast the result back to rank 3 -- the consumer a user writes was never
swept. This sweep asks the same question one consumer further out:

  reshape   the rank-2 result flattened to rank 1 and stored with a rank-1
            address -- what `tl.reshape(tl.sum(v, ax), (-1,))` compiles to
  xbcast    the result broadcast back on a DIFFERENT axis than it was reduced
            (`s[:, None, :]` after an axis-2 reduce), against the same-axis
            broadcast as the control that is known good
  scanmix   a rank-2 column reduce broadcast back into the tile in a function
            that ALSO holds a scan phase, against the identical kernel without
            the scan

Every family carries its control arm in the same run: a WRONG that appears only
with the scan present says the phase is the discriminator, and a WRONG that
appears in both says the consumer is.

Axes: shape x reduced axis x num_warps. The tag carries E = numel/(32*warps),
the axis every previous defect in this family broke along.

    python/test/metal_sweeps/drive.sh python/test/metal_sweeps/sweeps/s_reduce_consumers.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402
from sweep_lib import Sweep  # noqa: E402


@triton.jit
def _rank3_tile(X, A: tl.constexpr, B: tl.constexpr, C: tl.constexpr,
                SRC: tl.constexpr):
    """The SAME logical tile, built the two ways a kernel can build it.

    `SRC=0` reshapes a rank-1 load -- how `tl.sort`/`tl.topk` hypercubes and
    most user code get a rank-3 tile. `SRC=1` loads it directly at rank 3,
    which is the source `.omc/fuzz/rank3_reduce.py` sweeps and the one the
    register-band fail-close (`9a10039802`) was written for. Keeping both in one
    sweep is what tells a consumer defect from a source defect.
    """
    if SRC == 0:
        return tl.reshape(tl.load(X + tl.arange(0, A * B * C)), (A, B, C))
    i = tl.arange(0, A)[:, None, None]
    j = tl.arange(0, B)[None, :, None]
    k = tl.arange(0, C)[None, None, :]
    return tl.load(X + i * (B * C) + j * C + k)


@triton.jit
def reshape_store(X, O, A: tl.constexpr, B: tl.constexpr, C: tl.constexpr,
                  AX: tl.constexpr, OUT: tl.constexpr, SRC: tl.constexpr):
    v = _rank3_tile(X, A, B, C, SRC)
    s = tl.sum(v, AX)
    tl.store(O + tl.arange(0, OUT), tl.reshape(s, (OUT,)))


@triton.jit
def rank2_store(X, O, A: tl.constexpr, B: tl.constexpr, C: tl.constexpr,
                AX: tl.constexpr, R: tl.constexpr, S: tl.constexpr,
                SRC: tl.constexpr):
    v = _rank3_tile(X, A, B, C, SRC)
    s = tl.sum(v, AX)
    r = tl.arange(0, R)
    c = tl.arange(0, S)
    tl.store(O + r[:, None] * S + c[None, :], s)


@triton.jit
def xbcast_store(X, O, A: tl.constexpr, B: tl.constexpr, C: tl.constexpr,
                 AX: tl.constexpr, D0: tl.constexpr, D1: tl.constexpr,
                 D2: tl.constexpr, SRC: tl.constexpr):
    f = tl.arange(0, A * B * C)
    v = _rank3_tile(X, A, B, C, SRC)
    s = tl.sum(v, AX)
    back = s[:, None, :] + tl.zeros((D0, D1, D2), tl.float32)
    tl.store(O + f, tl.reshape(back, (A * B * C,)))


@triton.jit
def sbcast_store(X, O, A: tl.constexpr, B: tl.constexpr, C: tl.constexpr,
                 AX: tl.constexpr, SRC: tl.constexpr):
    """The control: the result broadcast back on the axis it was reduced."""
    f = tl.arange(0, A * B * C)
    v = _rank3_tile(X, A, B, C, SRC)
    s = tl.sum(v, AX)
    if AX == 0:
        back = s[None, :, :] + tl.zeros((A, B, C), tl.float32)
    elif AX == 1:
        back = s[:, None, :] + tl.zeros((A, B, C), tl.float32)
    else:
        back = s[:, :, None] + tl.zeros((A, B, C), tl.float32)
    tl.store(O + f, tl.reshape(back, (A * B * C,)))


@triton.jit
def bc3_store(X, O, A: tl.constexpr, B: tl.constexpr, C: tl.constexpr,
              AX: tl.constexpr, SRC: tl.constexpr):
    """Same-axis broadcast, stored through a rank-3 ADDRESS.

    The discriminator between "the broadcast is wrong" and "flattening the
    result with a final tl.reshape is wrong": every other broadcast arm here
    ends in `tl.reshape(back, (n,))` and a rank-1 store, which is also the one
    thing `.omc/fuzz/rank3_reduce.py` never does.
    """
    v = _rank3_tile(X, A, B, C, SRC)
    s = tl.sum(v, AX)
    if AX == 0:
        back = s[None, :, :] + tl.zeros((A, B, C), tl.float32)
    elif AX == 1:
        back = s[:, None, :] + tl.zeros((A, B, C), tl.float32)
    else:
        back = s[:, :, None] + tl.zeros((A, B, C), tl.float32)
    i = tl.arange(0, A)[:, None, None]
    j = tl.arange(0, B)[None, :, None]
    k = tl.arange(0, C)[None, None, :]
    tl.store(O + i * (B * C) + j * C + k, back)


@triton.jit
def xbc3_store(X, O, A: tl.constexpr, B: tl.constexpr, C: tl.constexpr,
               AX: tl.constexpr, D0: tl.constexpr, D1: tl.constexpr,
               D2: tl.constexpr, SRC: tl.constexpr):
    """Cross-axis broadcast stored through a rank-3 address.

    Separates "the cross-axis broadcast is wrong" from "flattening it is":
    `xbc` is wrong at every E, including one element per thread, where the
    flatten story cannot apply.
    """
    v = _rank3_tile(X, A, B, C, SRC)
    s = tl.sum(v, AX)
    back = s[:, None, :] + tl.zeros((D0, D1, D2), tl.float32)
    i = tl.arange(0, D0)[:, None, None]
    j = tl.arange(0, D1)[None, :, None]
    k = tl.arange(0, D2)[None, None, :]
    tl.store(O + i * (D1 * D2) + j * D2 + k, back)


@triton.jit
def id3_flat(X, O, A: tl.constexpr, B: tl.constexpr, C: tl.constexpr,
             SRC: tl.constexpr):
    """No reduce at all: a rank-3 tile flattened and stored.

    The identity control for the whole flatten story. If this is green where
    the reduce arms are red, the defect is in what the REDUCE hands the
    reshape, not in reshape+rank-1 store as such.
    """
    v = _rank3_tile(X, A, B, C, SRC)
    tl.store(O + tl.arange(0, A * B * C), tl.reshape(v * 2.0, (A * B * C,)))


@triton.jit
def colbcast_scan(X, O, M: tl.constexpr, N: tl.constexpr,
                  WITH_SCAN: tl.constexpr):
    i = tl.arange(0, M)
    j = tl.arange(0, N)
    v = tl.load(X + i[:, None] * N + j[None, :])
    col = tl.sum(v, 0)
    back = col[None, :] + tl.zeros((M, N), tl.float32)
    if WITH_SCAN:
        out = tl.cumsum(v, 1) + back
    else:
        out = v + back
    tl.store(O + i[:, None] * N + j[None, :], out)


SHAPES = [(2, 4, 4), (4, 4, 4), (2, 2, 8), (4, 8, 8), (8, 8, 8), (4, 4, 16)]
WARPS = [1, 2, 4]
MN = [(8, 8), (16, 16), (32, 8), (8, 32), (16, 32)]


def cells(sw):
    for src in (0, 1):
        sname = "resh" if src == 0 else "dirl"
        for (A, B, C) in SHAPES:
            n = A * B * C
            for ax in (0, 1, 2):
                keep = [d for k, d in enumerate((A, B, C)) if k != ax]
                out = keep[0] * keep[1]
                for w in WARPS:
                    e = max(1, n // (32 * w))
                    sfx = f"{A}x{B}x{C}_w{w}_e{e}"
                    sw.run(f"flat_{sname}_ax{ax}_{sfx}",
                           lambda A=A, B=B, C=C, ax=ax, w=w, out=out, src=src:
                           _reshape(A, B, C, ax, w, out, src))
                    sw.run(f"r2st_{sname}_ax{ax}_{sfx}",
                           lambda A=A, B=B, C=C, ax=ax, w=w, keep=keep, src=src:
                           _rank2(A, B, C, ax, w, keep, src))
                    sw.run(f"xbc_{sname}_ax{ax}_{sfx}",
                           lambda A=A, B=B, C=C, ax=ax, w=w, keep=keep, src=src:
                           _xbcast(A, B, C, ax, w, keep, src))
                    sw.run(f"sbc_{sname}_ax{ax}_{sfx}",
                           lambda A=A, B=B, C=C, ax=ax, w=w, src=src:
                           _sbcast(A, B, C, ax, w, src))
                    sw.run(f"bc3st_{sname}_ax{ax}_{sfx}",
                           lambda A=A, B=B, C=C, ax=ax, w=w, src=src:
                           _bc3(A, B, C, ax, w, src))
                    sw.run(f"xbc3st_{sname}_ax{ax}_{sfx}",
                           lambda A=A, B=B, C=C, ax=ax, w=w, keep=keep, src=src:
                           _xbc3(A, B, C, ax, w, keep, src))
                if ax == 0:
                    for w2 in WARPS:
                        e2 = max(1, n // (32 * w2))
                        sw.run(f"id3flat_{sname}_{A}x{B}x{C}_w{w2}_e{e2}",
                               lambda A=A, B=B, C=C, w=w2, src=src:
                               _id3(A, B, C, w, src))
    for (M, N) in MN:
        for w in WARPS:
            e = max(1, M * N // (32 * w))
            for scan in (0, 1):
                sw.run(f"scanmix{scan}_{M}x{N}_w{w}_e{e}",
                       lambda M=M, N=N, w=w, scan=scan: _colbcast(M, N, w, scan))


def _ref(n):
    return torch.arange(n, dtype=torch.float32)


def _cmp(got, want):
    if torch.equal(got, want):
        return None
    d = int((got != want).sum())
    return f"{d}/{got.numel()} differ; got[:4]={got.flatten()[:4].tolist()} want[:4]={want.flatten()[:4].tolist()}"


def _reshape(A, B, C, ax, w, out, src):
    ref = _ref(A * B * C)
    x = ref.to("mps")
    o = torch.zeros(out, device="mps")
    reshape_store[(1,)](x, o, A, B, C, ax, out, src, num_warps=w)
    torch.mps.synchronize()
    return _cmp(o.cpu(), ref.reshape(A, B, C).sum(ax).reshape(-1))


def _rank2(A, B, C, ax, w, keep, src):
    ref = _ref(A * B * C)
    x = ref.to("mps")
    o = torch.zeros(keep[0], keep[1], device="mps")
    rank2_store[(1,)](x, o, A, B, C, ax, keep[0], keep[1], src, num_warps=w)
    torch.mps.synchronize()
    return _cmp(o.cpu(), ref.reshape(A, B, C).sum(ax))


def _xbcast(A, B, C, ax, w, keep, src):
    # `s[:, None, :]` puts the missing extent on the MIDDLE axis whatever was
    # reduced, so the destination is (keep0, missing, keep1). For ax=1 that IS
    # the axis the reduce removed, which is why the ax1 arm doubles as a second
    # same-axis control.
    missing = (A, B, C)[ax]
    d = (keep[0], missing, keep[1])
    if d[0] * d[1] * d[2] != A * B * C:
        return None
    ref = _ref(A * B * C)
    x = ref.to("mps")
    o = torch.zeros(A * B * C, device="mps")
    xbcast_store[(1,)](x, o, A, B, C, ax, d[0], d[1], d[2], src, num_warps=w)
    torch.mps.synchronize()
    want = ref.reshape(A, B, C).sum(ax)[:, None, :].expand(*d).reshape(-1)
    return _cmp(o.cpu(), want.contiguous())


def _sbcast(A, B, C, ax, w, src):
    ref = _ref(A * B * C)
    x = ref.to("mps")
    o = torch.zeros(A * B * C, device="mps")
    sbcast_store[(1,)](x, o, A, B, C, ax, src, num_warps=w)
    torch.mps.synchronize()
    want = ref.reshape(A, B, C).sum(ax).unsqueeze(ax).expand(A, B, C).reshape(-1)
    return _cmp(o.cpu(), want.contiguous())


def _bc3(A, B, C, ax, w, src):
    ref = _ref(A * B * C)
    x = ref.to("mps")
    o = torch.zeros(A * B * C, device="mps")
    bc3_store[(1,)](x, o, A, B, C, ax, src, num_warps=w)
    torch.mps.synchronize()
    want = ref.reshape(A, B, C).sum(ax).unsqueeze(ax).expand(A, B, C).reshape(-1)
    return _cmp(o.cpu(), want.contiguous())


def _xbc3(A, B, C, ax, w, keep, src):
    missing = (A, B, C)[ax]
    d = (keep[0], missing, keep[1])
    if d[0] * d[1] * d[2] != A * B * C:
        return None
    ref = _ref(A * B * C)
    x = ref.to("mps")
    o = torch.zeros(A * B * C, device="mps")
    xbc3_store[(1,)](x, o, A, B, C, ax, d[0], d[1], d[2], src, num_warps=w)
    torch.mps.synchronize()
    want = ref.reshape(A, B, C).sum(ax)[:, None, :].expand(*d).reshape(-1)
    return _cmp(o.cpu(), want.contiguous())


def _id3(A, B, C, w, src):
    ref = _ref(A * B * C)
    x = ref.to("mps")
    o = torch.zeros(A * B * C, device="mps")
    id3_flat[(1,)](x, o, A, B, C, src, num_warps=w)
    torch.mps.synchronize()
    return _cmp(o.cpu(), ref * 2.0)


def _colbcast(M, N, w, scan):
    ref = _ref(M * N).reshape(M, N)
    x = ref.to("mps")
    o = torch.zeros(M, N, device="mps")
    colbcast_scan[(1,)](x, o, M, N, scan, num_warps=w)
    torch.mps.synchronize()
    base = ref.cumsum(1) if scan else ref
    return _cmp(o.cpu(), base + ref.sum(0)[None, :])


if __name__ == "__main__":
    sw = Sweep("s_reduce_consumers")
    cells(sw)
    sw.finish()
