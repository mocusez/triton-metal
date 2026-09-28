"""`tl.flip` across rank, axis, extent, warps, dtype and neighbours.

`tl.flip` arrives as a hypercube of xor-sum reduces between two reshapes; the
backend replaces the chain by one exchange whose read reverses the flipped
axis. Neighbours probe the labels that exchange has to agree with: a computed
source, a second flip, a reduce, an argmax or a scan in the same function,
and a flip feeding a reduce.

Tags encode form, rank, shape, axis, warps and dtype.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402
from sweep_lib import Sweep  # noqa: E402


@triton.jit
def _offsets(A: tl.constexpr, B: tl.constexpr, C: tl.constexpr, RANK: tl.constexpr):
    if RANK == 1:
        off = tl.arange(0, A)
    elif RANK == 2:
        off = tl.arange(0, A)[:, None] * B + tl.arange(0, B)[None, :]
    else:
        off = (tl.arange(0, A)[:, None, None] * (B * C) +
               tl.arange(0, B)[None, :, None] * C + tl.arange(0, C)[None, None, :])
    return off


@triton.jit
def flip_kernel(X, Y, Z, A: tl.constexpr, B: tl.constexpr, C: tl.constexpr,
                RANK: tl.constexpr, DIM: tl.constexpr, FORM: tl.constexpr):
    off = _offsets(A, B, C, RANK)
    x = tl.load(X + off)
    if FORM == 0:          # plain
        y = tl.flip(x, DIM)
    elif FORM == 1:        # computed source
        y = tl.flip(x * 2 + 1, DIM)
    elif FORM == 2:        # flip twice, once on the last axis
        y = tl.flip(tl.flip(x, DIM), RANK - 1)
    elif FORM == 3:        # plus a reduce elsewhere in the function
        y = tl.flip(x, DIM)
        tl.store(Z + tl.arange(0, 1), tl.sum(tl.reshape(x, (A * B * C,)), 0)[None])
    elif FORM == 4:        # a flip whose result is reduced
        y = tl.flip(x, DIM)
        tl.store(Z + tl.arange(0, 1), tl.max(tl.reshape(y, (A * B * C,)) * 1, 0)[None])
    elif FORM == 5:        # a scan in the same function
        y = tl.flip(x, DIM) + tl.cumsum(x, RANK - 1)
    else:                  # an argmax over the flattened source beside it
        y = tl.flip(x, DIM)
        i = tl.argmax(tl.reshape(x, (A * B * C,)), 0)
        tl.store(Z + tl.arange(0, 1), i.to(Z.dtype.element_ty)[None])
    tl.store(Y + off, y)


def _cell(shape, dim, w, dtype, form):
    rank = len(shape)
    A, B, C = (list(shape) + [1, 1])[:3]
    n = A * B * C
    g = torch.Generator().manual_seed(n * 7 + dim * 3 + w)
    if dtype.is_floating_point:
        x = (torch.randperm(n, generator=g) % 97 - 48).to(dtype).reshape(shape)
    else:
        x = (torch.randperm(n, generator=g) % 97 - 48).to(dtype).reshape(shape)
    y = torch.zeros_like(x, device="mps")
    z = torch.zeros(1, dtype=dtype, device="mps")
    flip_kernel[(1,)](x.to("mps"), y, z, A, B, C, rank, dim, form, num_warps=w)
    torch.mps.synchronize()
    xc = x
    if form == 0:
        want = xc.flip(dim)
    elif form == 1:
        want = (xc * 2 + 1).flip(dim)
    elif form == 2:
        want = xc.flip(dim).flip(rank - 1)
    elif form == 3:
        want = xc.flip(dim)
    elif form in (4, 6):
        want = xc.flip(dim)
    else:
        want = xc.flip(dim) + xc.cumsum(rank - 1).to(dtype)
    got = y.cpu()
    if not torch.equal(got, want):
        d = int((got != want).sum())
        return f"{d}/{n} differ; got[:4]={got.flatten()[:4].tolist()} want[:4]={want.flatten()[:4].tolist()}"
    if form == 3 and z.cpu()[0] != xc.sum().to(dtype):
        return f"sum {z.cpu()[0]} != {xc.sum()}"
    if form == 4 and z.cpu()[0] != xc.max():
        return f"max {z.cpu()[0]} != {xc.max()}"
    if form == 6 and xc.flatten()[int(z.cpu()[0])] != xc.max():
        return f"argmax {int(z.cpu()[0])} -> {xc.flatten()[int(z.cpu()[0])]} != max {xc.max()}"
    return None


SHAPES = [(32,), (256,), (1024,), (4, 32), (16, 16), (32, 8), (8, 128), (2, 64),
          (4, 8, 16), (2, 16, 32), (2, 4, 8), (8, 2, 16)]
WARPS = [1, 4]
DTYPES = [(torch.float32, "f32"), (torch.int32, "i32"), (torch.float16, "f16"),
          (torch.int8, "i8")]
FORMS = ["plain", "computed", "twice", "withsum", "reduced", "withscan", "withargmax"]


def cells(sw):
    for shape in SHAPES:
        for dim in range(len(shape)):
            for w in WARPS:
                for dtype, dn in DTYPES:
                    for form, fn in enumerate(FORMS):
                        if dn == "i8" and form in (3, 5):
                            continue  # i8 sum/cumsum overflow the oracle
                        if dn == "f16" and form == 5:
                            continue  # f16 scan is refused by name
                        if dn == "i8" and form == 6:
                            continue  # the index does not fit the i8 output
                        tag = f"{fn}_{'x'.join(map(str, shape))}_d{dim}_w{w}_{dn}"
                        sw.run(tag, lambda shape=shape, dim=dim, w=w, dtype=dtype, form=form:
                               _cell(shape, dim, w, dtype, form))


if __name__ == "__main__":
    sw = Sweep("s_flip")
    cells(sw)
    sw.finish()
