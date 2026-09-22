"""A scan whose SOURCE BUFFER is written by the same kernel.

Found by probing the unpinned reduce/scan diagnostics: `c = tl.cumsum(v, 0)`
followed by `tl.store(X, ...)` -- writing the very buffer `v` was loaded from,
which is what any in-place update kernel does -- answers with something that is
not the scan, silently, from two elements per thread up. `tl.sum` in the same
shape is exact, and so is a store to ANY OTHER buffer.

Axes: op x where the write lands x order x BLOCK x warps. The control arms are
the point:

  other     the same kernel writing an unrelated buffer -- isolates "a store is
            present" from "the store hits the source"
  before    the store placed BEFORE the load instead of after it, where reading
            the written values IS the correct answer
  sum       the reduce sibling of every scan cell

    python/test/metal_sweeps/drive.sh python/test/metal_sweeps/sweeps/s_scan_source_effects.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402
from sweep_lib import Sweep  # noqa: E402


@triton.jit
def rank1_kernel(X, Y, O, N: tl.constexpr, OP: tl.constexpr,
                 TARGET: tl.constexpr, ORDER: tl.constexpr):
    i = tl.arange(0, N)
    if ORDER == 0:          # write AFTER the aggregate
        v = tl.load(X + i)
        r = tl.cumsum(v, 0) if OP == 0 else (v * 0.0 + tl.sum(v, 0))
        if TARGET == 0:
            tl.store(X + i, v * 0.0)
        else:
            tl.store(Y + i, v * 0.0)
        tl.store(O + i, r)
    else:                   # write BEFORE the load
        if TARGET == 0:
            tl.store(X + i, tl.zeros((N,), tl.float32))
        else:
            tl.store(Y + i, tl.zeros((N,), tl.float32))
        v = tl.load(X + i)
        r = tl.cumsum(v, 0) if OP == 0 else (v * 0.0 + tl.sum(v, 0))
        tl.store(O + i, r)


@triton.jit
def rank2_kernel(X, Y, O, M: tl.constexpr, N: tl.constexpr, OP: tl.constexpr,
                 TARGET: tl.constexpr):
    r = tl.arange(0, M)
    c = tl.arange(0, N)
    off = r[:, None] * N + c[None, :]
    v = tl.load(X + off)
    out = tl.cumsum(v, 1) if OP == 0 else (v * 0.0 + tl.sum(v, 1)[:, None])
    if TARGET == 0:
        tl.store(X + off, v * 0.0)
    else:
        tl.store(Y + off, v * 0.0)
    tl.store(O + off, out)


BLOCKS = [32, 64, 128, 256, 1024]
WARPS = [1, 2, 4]
MN = [(8, 8), (8, 32), (16, 16), (32, 32)]


def _cmp(got, want):
    if torch.equal(got, want):
        return None
    d = int((got != want).sum())
    return (f"{d}/{got.numel()} differ; got[:4]={got.flatten()[:4].tolist()} "
            f"want[:4]={want.flatten()[:4].tolist()}")


def _rank1(n, w, op, target, order):
    ref = torch.arange(n, dtype=torch.float32)
    x = ref.clone().to("mps")
    y = torch.zeros(n, device="mps")
    o = torch.zeros(n, device="mps")
    rank1_kernel[(1,)](x, y, o, n, op, target, order, num_warps=w)
    torch.mps.synchronize()
    # ORDER=1 zeroes the source before loading it, so the aggregate is over
    # zeros -- and only when the write actually hit the source.
    src = torch.zeros(n) if (order == 1 and target == 0) else ref
    want = src.cumsum(0) if op == 0 else torch.full((n,), float(src.sum()))
    return _cmp(o.cpu(), want)


def _rank2(m, n, w, op, target):
    ref = torch.arange(m * n, dtype=torch.float32).reshape(m, n)
    x = ref.clone().reshape(-1).contiguous().to("mps")
    y = torch.zeros(m * n, device="mps")
    o = torch.zeros(m * n, device="mps")
    rank2_kernel[(1,)](x, y, o, m, n, op, target, num_warps=w)
    torch.mps.synchronize()
    want = ref.cumsum(1) if op == 0 else ref.sum(1, keepdim=True).expand(m, n)
    return _cmp(o.cpu().reshape(m, n), want.contiguous())


def cells(sw):
    for op, opname in ((0, "scan"), (1, "sum")):
        for target, tname in ((0, "self"), (1, "other")):
            for order, oname in ((0, "after"), (1, "before")):
                for n in BLOCKS:
                    for w in WARPS:
                        e = max(1, n // (32 * w))
                        sw.run(f"r1_{opname}_{tname}_{oname}_n{n}_w{w}_e{e}",
                               lambda n=n, w=w, op=op, target=target,
                               order=order: _rank1(n, w, op, target, order))
            for (m, n) in MN:
                for w in WARPS:
                    e = max(1, m * n // (32 * w))
                    sw.run(f"r2_{opname}_{tname}_{m}x{n}_w{w}_e{e}",
                           lambda m=m, n=n, w=w, op=op, target=target:
                           _rank2(m, n, w, op, target))


if __name__ == "__main__":
    sw = Sweep("s_scan_source_effects")
    cells(sw)
    sw.finish()
