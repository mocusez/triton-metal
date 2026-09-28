"""Whole-tile reductions of rank >= 2 tiles, and flat reads through a broadcast.

`tl.sum(x)` with no axis is a reduce over `tl.reshape(x, (n,))`. A flat-index
cone walk that passes a rank >= 2 broadcast through unchanged reads every
2-D address with row == col, so the answer is a sum over the wrong elements.
Cells vary the op, the source cone (a load, a computed tile, an outer
product of two ranges, a masked load), rank, shape, warps and dtype, and add
order-sensitive consumers of the flattened tile: a cumsum, a store, an argmax
and a sum weighted by flat position.

Tags encode op, source, shape, warps and dtype.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402
from sweep_lib import Sweep  # noqa: E402


@triton.jit
def _max_combine(a, b):
    return tl.maximum(a, b)


@triton.jit
def kernel(X, V, Z, A: tl.constexpr, B: tl.constexpr, C: tl.constexpr,
           RANK: tl.constexpr, OP: tl.constexpr, SRC: tl.constexpr):
    if RANK == 2:
        off = tl.arange(0, A)[:, None] * B + tl.arange(0, B)[None, :]
        row = tl.arange(0, A)[:, None] + tl.zeros((A, B), tl.int32)
    else:
        off = (tl.arange(0, A)[:, None, None] * (B * C) +
               tl.arange(0, B)[None, :, None] * C + tl.arange(0, C)[None, None, :])
        row = tl.arange(0, A)[:, None, None] + tl.zeros((A, B, C), tl.int32)
    if SRC == 0:
        x = tl.load(X + off)
    elif SRC == 1:
        x = tl.load(X + off) * 3 - 1
    elif SRC == 2:
        x = (off % 13 - 6).to(X.dtype.element_ty) + tl.load(V + row)
    elif SRC == 3:
        x = tl.load(X + off, mask=off % 3 != 0, other=5)
    elif RANK == 2:  # the address spelled inline: a broadcast POINTER tile
        x = tl.load(X + tl.arange(0, A)[:, None] * B + tl.arange(0, B)[None, :])
    else:
        x = tl.load(X + tl.arange(0, A)[:, None, None] * (B * C) +
                    tl.arange(0, B)[None, :, None] * C + tl.arange(0, C)[None, None, :])
    if OP == 0:
        r = tl.sum(x)
    elif OP == 1:
        r = tl.max(x)
    elif OP == 2:
        r = tl.min(x)
    elif OP == 3:
        r = tl.reduce(x, None, _max_combine)
    elif OP == 4:
        r = tl.sum(tl.reshape(x, (A * B * C,)), 0)
    elif OP == 5:
        r = tl.argmax(tl.reshape(x, (A * B * C,)), 0)
    elif OP == 8:  # order-sensitive: each element weighted by its flat position
        r = tl.sum(tl.reshape(x, (A * B * C,)) * (tl.arange(0, A * B * C) % 7).to(x.dtype), 0)
    if OP <= 5 or OP == 8:
        tl.store(Z, r)
    elif OP == 6:
        tl.store(Z + tl.arange(0, A * B * C), tl.cumsum(tl.reshape(x, (A * B * C,)), 0))
    else:
        tl.store(Z + tl.arange(0, A * B * C), tl.reshape(x, (A * B * C,)))


OPS = ["sum", "max", "min", "custom", "flatsum", "argmax", "flatcumsum", "flatstore", "flatweighted"]
SRCS = ["load", "computed", "outer", "masked", "ptrbcast"]


def _cell(shape, w, dtype, op, src):
    rank = len(shape)
    A, B, C = (list(shape) + [1])[:3]
    n = A * B * C
    g = torch.Generator().manual_seed(n + 31 * op + 7 * src + w)
    x = (torch.randperm(n, generator=g) % 23 - 11).to(dtype)
    v = (torch.arange(A) % 5 - 2).to(dtype)
    out_n = n if op in (6, 7) else 1
    z = torch.zeros(out_n, dtype=torch.int32 if op == 5 else dtype, device="mps")
    kernel[(1,)](x.to("mps"), v.to("mps"), z, A, B, C, rank, op, src, num_warps=w)
    torch.mps.synchronize()
    off = torch.arange(n)
    row = off // (B * C)
    if src == 0:
        t = x.double()
    elif src == 1:
        t = x.double() * 3 - 1
    elif src == 2:
        t = (off % 13 - 6).double() + v.double()[row]
    elif src == 3:
        t = torch.where(off % 3 != 0, x.double(), torch.full_like(x.double(), 5))
    else:
        t = x.double()
    want = [t.sum(), t.max(), t.min(), t.max(), t.sum(), None, t.cumsum(0), t,
            (t * (off % 7).double()).sum()][op]
    got = z.cpu()
    if op == 5:
        ok = t[int(got[0])] == t.max()
        return None if ok else f"argmax {int(got[0])} -> {t[int(got[0])]} != max {t.max()}"
    want = want.to(dtype).reshape(got.shape)
    if torch.equal(got, want):
        return None
    d = int((got != want).sum())
    return f"{d}/{got.numel()} differ; got[:4]={got[:4].tolist()} want[:4]={want[:4].tolist()}"


SHAPES = [(2, 16), (4, 32), (16, 16), (8, 128), (32, 32), (64, 16),
          (2, 4, 8), (4, 8, 16), (2, 16, 32)]
WARPS = [1, 2, 4]
DTYPES = [(torch.float32, "f32"), (torch.int32, "i32"), (torch.float16, "f16")]


def cells(sw):
    for shape in SHAPES:
        for w in WARPS:
            for dtype, dn in DTYPES:
                for op, on in enumerate(OPS):
                    for src, sn in enumerate(SRCS):
                        if dn == "f16" and op == 6:
                            continue  # f16 scan is refused by name
                        if dn == "f16" and op == 8:
                            continue  # the weighted sum leaves f16's exact range
                        tag = f"{on}_{sn}_{'x'.join(map(str, shape))}_w{w}_{dn}"
                        sw.run(tag, lambda shape=shape, w=w, dtype=dtype, op=op, src=src:
                               _cell(shape, w, dtype, op, src))


if __name__ == "__main__":
    sw = Sweep("s_full_reduce")
    cells(sw)
    sw.finish()
