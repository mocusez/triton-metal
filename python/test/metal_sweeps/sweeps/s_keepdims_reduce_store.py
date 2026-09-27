"""A rank-2 reduce's result stored with its reduced axis KEPT.

`tl.max(x, 1, keep_dims=True)` stored as an `[R, 1]` column -- the shape
`tl.topk(x, 1)` lowers to, and the ordinary way to write a per-row statistic
next to its tile. Triton moves the result from the reduce's slice layout to the
store's with a slice-to-slice `ttg.convert_layout`, and since `7902b31860` that
cvt is an exchange whose publish labels the value by the source slice LAYOUT --
while the row-reduce lowering hands each thread the element `localTid % n`.
Found through a topk probe: every row received row 0's maximum.

Axes: op {sum, max} x axis {0, 1} x store {keep_dims tile, rank-1 control} x
shape x num_warps. The rank-1 store is the control: it reads the same result
without the cvt.

    python/test/metal_sweeps/drive.sh python/test/metal_sweeps/sweeps/s_keepdims_reduce_store.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402
from sweep_lib import Sweep  # noqa: E402


@triton.jit
def kernel(X, Y, R: tl.constexpr, C: tl.constexpr, OP: tl.constexpr,
           AXIS: tl.constexpr, KEEP: tl.constexpr):
    r = tl.arange(0, R)
    c = tl.arange(0, C)
    x = tl.load(X + r[:, None] * C + c[None, :])
    if AXIS == 1:
        if KEEP:
            m = tl.sum(x, 1, keep_dims=True) if OP == 0 else tl.max(x, 1, keep_dims=True)
            tl.store(Y + r[:, None] + tl.arange(0, 1)[None, :], m)
        else:
            m = tl.sum(x, 1) if OP == 0 else tl.max(x, 1)
            tl.store(Y + r, m)
    else:
        if KEEP:
            m = tl.sum(x, 0, keep_dims=True) if OP == 0 else tl.max(x, 0, keep_dims=True)
            tl.store(Y + tl.arange(0, 1)[:, None] + c[None, :], m)
        else:
            m = tl.sum(x, 0) if OP == 0 else tl.max(x, 0)
            tl.store(Y + c, m)


SHAPES = [(2, 16), (4, 32), (8, 32), (16, 16), (2, 64), (32, 8), (8, 128),
          (64, 16), (128, 8)]
WARPS = [1, 2, 4]


def _cell(r, c, w, op, axis, keep):
    torch.manual_seed(r * 31 + c + w)
    x = torch.randint(-64, 64, (r, c), device="mps").float()
    n = r if axis == 1 else c
    y = torch.zeros(n, device="mps")
    kernel[(1,)](x, y, r, c, op, axis, keep, num_warps=w)
    torch.mps.synchronize()
    xc = x.cpu()
    want = (xc.sum(axis) if op == 0 else xc.amax(axis))
    got = y.cpu()
    if torch.equal(got, want):
        return None
    return f"{int((got != want).sum())}/{n} differ; got[:4]={got[:4].tolist()} want[:4]={want[:4].tolist()}"


def cells(sw):
    for op, opn in ((0, "sum"), (1, "max")):
        for axis in (0, 1):
            for keep in (True, False):
                for (r, c) in SHAPES:
                    for w in WARPS:
                        e = r * c / (32 * w)
                        sw.run(f"{opn}_ax{axis}_{'keep' if keep else 'flat'}_{r}x{c}_w{w}_e{e:g}",
                               lambda r=r, c=c, w=w, op=op, axis=axis, keep=keep:
                               _cell(r, c, w, op, axis, keep))


if __name__ == "__main__":
    sw = Sweep("s_keepdims_reduce_store")
    cells(sw)
    sw.finish()
