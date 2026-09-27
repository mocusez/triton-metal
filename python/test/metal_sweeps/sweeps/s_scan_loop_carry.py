"""A rank >= 2 scan inside a runtime loop that carries tiles.

A multi-band scan inside a user loop runs the loop body as register-band
phases, so a tile the loop carries (an iter_arg), or one computed before it
that it or the code after it reads, has no register to live in between them.
These cells sweep the forms that carry state: an accumulator of scans, a
recurrence that feeds the scan its own previous result, a snapshot loaded
before the loop, two carries that swap, a tuple scan, and zero trips.

Tags encode form, axis, shape, warps, dtype and trip count.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402
from sweep_lib import Sweep  # noqa: E402


@triton.jit
def acc_kernel(X, Y, T, R: tl.constexpr, C: tl.constexpr, AX: tl.constexpr):
    """acc += cumsum(x_t); the base loaded before the loop is added after."""
    off = tl.arange(0, R)[:, None] * C + tl.arange(0, C)[None, :]
    base = tl.load(X + off)
    acc = base * 0
    for t in range(T):
        x = tl.load(X + t * R * C + off)
        acc += tl.cumsum(x, AX)
    tl.store(Y + off, acc + base)


@triton.jit
def recur_kernel(X, Y, T, R: tl.constexpr, C: tl.constexpr, AX: tl.constexpr):
    """The scan reads the carried tile: s = cumsum(s + x_t), stored every trip."""
    off = tl.arange(0, R)[:, None] * C + tl.arange(0, C)[None, :]
    s = tl.zeros((R, C), dtype=X.dtype.element_ty)
    for t in range(T):
        x = tl.load(X + t * R * C + off)
        s = tl.cumsum(s + x, AX)
        tl.store(Y + t * R * C + off, s)


@triton.jit
def snap_kernel(X, W, Y, T, R: tl.constexpr, C: tl.constexpr, AX: tl.constexpr):
    """A tile loaded before the loop, which then overwrites its source.

    The overwrite is built from the snapshot, not from `w`: a load reused
    after the store to Y is a separate, older refusal of the staged scan.
    """
    off = tl.arange(0, R)[:, None] * C + tl.arange(0, C)[None, :]
    snap = tl.load(X + off)
    for t in range(T):
        w = tl.load(W + t * R * C + off)
        tl.store(Y + t * R * C + off, tl.cumsum(snap + w, AX))
        tl.store(X + off, snap * 0 + t + 7)


@triton.jit
def swap_kernel(X, Y, Z, T, R: tl.constexpr, C: tl.constexpr, AX: tl.constexpr):
    """Two carries that trade places every trip."""
    off = tl.arange(0, R)[:, None] * C + tl.arange(0, C)[None, :]
    a = tl.load(X + off)
    b = a * 0 + 1
    for t in range(T):
        x = tl.load(X + (t + 1) * R * C + off)
        a, b = b + tl.cumsum(x, AX), a
    tl.store(Y + off, a)
    tl.store(Z + off, b)


@triton.jit
def _pair_add(a, b, c, d):
    return a + c, b + d


@triton.jit
def tuple_kernel(X, V, K, T, R: tl.constexpr, C: tl.constexpr, AX: tl.constexpr):
    """A tuple scan over carried (value, count)."""
    off = tl.arange(0, R)[:, None] * C + tl.arange(0, C)[None, :]
    value = tl.zeros((R, C), tl.float32)
    count = tl.zeros((R, C), tl.int32)
    for t in range(T):
        x = tl.load(X + t * R * C + off)
        value, count = tl.associative_scan((value + x, count + 1), AX, _pair_add)
        tl.store(V + t * R * C + off, value)
        tl.store(K + t * R * C + off, count)


def _cmp(got, want, exact):
    if exact:
        ok = torch.equal(got, want)
    else:
        ok = torch.allclose(got.float(), want.float(), atol=1e-2, rtol=1e-3)
    if ok:
        return None
    d = int((~torch.isclose(got.float(), want.float(), atol=1e-2, rtol=1e-3)).sum())
    return f"{d}/{got.numel()} differ; got[:4]={got.flatten()[:4].tolist()} want[:4]={want.flatten()[:4].tolist()}"


def _cumsum(u, ax, dtype):
    """float64 cumsum, or f16 rounded after every add as the kernel does."""
    if dtype != torch.float16:
        return u.double().cumsum(ax)
    u = u.half().movedim(ax, 0)
    out = torch.empty_like(u)
    run = torch.zeros_like(u[0])
    for j in range(u.shape[0]):
        run = run + u[j]
        out[j] = run
    return out.movedim(0, ax).double()


def _data(n, dtype):
    x = (torch.arange(n) % 7 - 3)
    return x.to(dtype)


def _acc(T, R, C, ax, w, dtype):
    x = _data((max(T, 1)) * R * C, dtype).reshape(max(T, 1), R, C)
    y = torch.zeros(R, C, dtype=dtype, device="mps")
    acc_kernel[(1,)](x.to("mps"), y, T, R, C, ax, num_warps=w)
    torch.mps.synchronize()
    xc = x.double()
    want = xc[:T].cumsum(ax + 1).sum(0) + xc[0]
    return _cmp(y.cpu(), want.to(dtype), dtype == torch.int32)


def _recur(T, R, C, ax, w, dtype):
    x = _data(max(T, 1) * R * C, dtype).reshape(max(T, 1), R, C)
    y = torch.full((max(T, 1), R, C), -5, dtype=dtype, device="mps")
    recur_kernel[(1,)](x.to("mps"), y, T, R, C, ax, num_warps=w)
    torch.mps.synchronize()
    want = torch.full((max(T, 1), R, C), -5, dtype=torch.float64)
    s = torch.zeros(R, C, dtype=torch.float64)
    for t in range(T):
        s = _cumsum((s + x[t].double()).to(dtype), ax, dtype)
        want[t] = s
    return _cmp(y.cpu(), want.to(dtype), dtype == torch.int32)


def _snap(T, R, C, ax, w, dtype):
    x = _data(R * C, dtype).reshape(R, C)
    wv = (_data(max(T, 1) * R * C, dtype) * 2).reshape(max(T, 1), R, C)
    y = torch.full((max(T, 1), R, C), -5, dtype=dtype, device="mps")
    snap_kernel[(1,)](x.to("mps"), wv.to("mps"), y, T, R, C, ax, num_warps=w)
    torch.mps.synchronize()
    want = torch.full((max(T, 1), R, C), -5, dtype=torch.float64)
    for t in range(T):
        want[t] = (x.double() + wv[t].double()).cumsum(ax)
    return _cmp(y.cpu(), want.to(dtype), dtype == torch.int32)


def _swap(T, R, C, ax, w, dtype):
    x = _data((T + 1) * R * C, dtype).reshape(T + 1, R, C)
    y = torch.zeros(R, C, dtype=dtype, device="mps")
    z = torch.zeros(R, C, dtype=dtype, device="mps")
    swap_kernel[(1,)](x.to("mps"), y, z, T, R, C, ax, num_warps=w)
    torch.mps.synchronize()
    a = x[0].double()
    b = a * 0 + 1
    for t in range(T):
        a, b = b + x[t + 1].double().cumsum(ax), a
    return _cmp(y.cpu(), a.to(dtype), dtype == torch.int32) or \
        _cmp(z.cpu(), b.to(dtype), dtype == torch.int32)


def _tuple(T, R, C, ax, w):
    x = _data(max(T, 1) * R * C, torch.float32).reshape(max(T, 1), R, C)
    v = torch.full((max(T, 1), R, C), -5.0, device="mps")
    k = torch.full((max(T, 1), R, C), -5, dtype=torch.int32, device="mps")
    tuple_kernel[(1,)](x.to("mps"), v, k, T, R, C, ax, num_warps=w)
    torch.mps.synchronize()
    wv = torch.full((max(T, 1), R, C), -5.0, dtype=torch.float64)
    wk = torch.full((max(T, 1), R, C), -5, dtype=torch.int64)
    value = torch.zeros(R, C, dtype=torch.float64)
    count = torch.zeros(R, C, dtype=torch.int64)
    for t in range(T):
        value = (value + x[t].double()).cumsum(ax)
        count = (count + 1).cumsum(ax)
        wv[t], wk[t] = value, count
    return _cmp(v.cpu(), wv.float(), False) or _cmp(k.cpu(), wk.int(), True)


SHAPES = [(4, 32), (16, 16), (8, 64), (32, 32)]
WARPS = [1, 4]
DTYPES = [(torch.float32, "f32"), (torch.int32, "i32"), (torch.float16, "f16")]


def cells(sw):
    for (R, C) in SHAPES:
        for w in WARPS:
            if R * C <= 32 * w:
                continue  # one element per thread: no band phases
            for ax in (0, 1):
                for T in (0, 1, 3):
                    for dtype, dn in DTYPES:
                        sfx = f"ax{ax}_{R}x{C}_w{w}_{dn}_T{T}"
                        sw.run(f"acc_{sfx}", lambda T=T, R=R, C=C, ax=ax, w=w, dtype=dtype:
                               _acc(T, R, C, ax, w, dtype))
                        sw.run(f"recur_{sfx}", lambda T=T, R=R, C=C, ax=ax, w=w, dtype=dtype:
                               _recur(T, R, C, ax, w, dtype))
                        sw.run(f"snap_{sfx}", lambda T=T, R=R, C=C, ax=ax, w=w, dtype=dtype:
                               _snap(T, R, C, ax, w, dtype))
                        sw.run(f"swap_{sfx}", lambda T=T, R=R, C=C, ax=ax, w=w, dtype=dtype:
                               _swap(T, R, C, ax, w, dtype))
                    sw.run(f"tuple_ax{ax}_{R}x{C}_w{w}_T{T}",
                           lambda T=T, R=R, C=C, ax=ax, w=w: _tuple(T, R, C, ax, w))


if __name__ == "__main__":
    sw = Sweep("s_scan_loop_carry")
    cells(sw)
    sw.finish()
