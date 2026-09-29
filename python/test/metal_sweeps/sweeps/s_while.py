"""Python `while` loops in a kernel, across the shapes `scf.while` can take.

`scf.condition` may forward values other than its block arguments, and the
after region may yield a permutation of what it received (`a, b = b, a + b`).
An emitter that gives each carried position one temp is right only when both
are the identity: the swap read an already-overwritten temp (Fibonacci(10)
came out 512), and a non-identity forward aborted the process.

Forms vary the carry pattern; cells vary the trip count (including zero),
warps and dtype. Tags encode form, trips, warps and dtype.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402
from sweep_lib import Sweep  # noqa: E402


@triton.jit
def kernel(N, X, Y, FORM: tl.constexpr, BLOCK: tl.constexpr):
    n = tl.load(N)
    if FORM == 0:      # swap: a, b = b, a + b
        a = tl.load(X) * 0
        b = a + 1
        i = 0
        while i < n:
            a, b = b, a + b
            i += 1
        tl.store(Y, a)
        tl.store(Y + 1, b)
    elif FORM == 1:    # rotate three
        a = tl.load(X)
        b = tl.load(X + 1)
        c = tl.load(X + 2)
        i = 0
        while i < n:
            a, b, c = b, c, a
            i += 1
        tl.store(Y, a)
        tl.store(Y + 1, b)
        tl.store(Y + 2, c)
    elif FORM == 2:    # carried value loaded in the body, tested in the condition
        i = 0
        v = tl.load(X)
        while v >= 0:
            i += 1
            v = tl.load(X + i)
        tl.store(Y, i.to(Y.dtype.element_ty))
        tl.store(Y + 1, v)
    elif FORM == 3:    # the condition itself loads
        i = 0
        while tl.load(X + i) >= 0:
            i += 1
        tl.store(Y, i.to(Y.dtype.element_ty))
    elif FORM == 4:    # nested, inner trip count from the outer counter
        i = 0
        s = tl.load(X) * 0
        while i < n:
            j = 0
            while j < i:
                s += tl.load(X + j)
                j += 1
            i += 1
        tl.store(Y, s)
    elif FORM == 5:    # a tensor carried with a scalar, swapped with another
        offs = tl.arange(0, BLOCK)
        p = tl.load(X + offs)
        q = p * 2
        i = 0
        while i < n:
            p, q = q, p + q
            i += 1
        tl.store(Y + offs, p)
        tl.store(Y + BLOCK + offs, q)
    elif FORM == 7:    # FORM 2 with its tested value unused afterwards: the
        i = 0          # condition forwards fewer values than the loop carries
        v = tl.load(X)
        while v >= 0:
            i += 1
            v = tl.load(X + i)
        tl.store(Y, i.to(Y.dtype.element_ty))
    else:              # two exits folded into one condition, with a swap
        a = tl.load(X) * 0
        b = a + 1
        i = 0
        while (i < n) & (b < 1000):
            a, b = b, a + b
            i += 1
        tl.store(Y, a)
        tl.store(Y + 1, b)
        tl.store(Y + 2, i.to(Y.dtype.element_ty))


def _ref(form, n, x):
    x = [int(v) if not isinstance(v, float) else v for v in x.tolist()]
    if form in (0, 6):
        a, b, i = 0, 1, 0
        while i < n and (form == 0 or b < 1000):
            a, b = b, a + b
            i += 1
        return [a, b] if form == 0 else [a, b, i]
    if form == 1:
        a, b, c = x[0], x[1], x[2]
        for _ in range(n):
            a, b, c = b, c, a
        return [a, b, c]
    if form in (2, 3, 7):
        i = 0
        while x[i] >= 0:
            i += 1
        return [i, x[i]] if form == 2 else [i]
    if form == 4:
        return [sum(x[j] for i in range(n) for j in range(i))]
    p = x[:32]
    q = [2 * v for v in p]
    for _ in range(n):
        p, q = q, [u + v for u, v in zip(p, q)]
    return p + q


def _cell(form, n, w, dtype):
    x = torch.arange(1, 65, dtype=dtype)
    if form in (2, 3, 7):
        x[min(n, 40)] = -1          # the first negative ends the loop
    else:
        x = x % 5 + 1
    y = torch.zeros(64, dtype=dtype, device="mps")
    kernel[(1,)](torch.tensor([n], dtype=torch.int32, device="mps"), x.to("mps"), y,
                 FORM=form, BLOCK=32, num_warps=w)
    torch.mps.synchronize()
    want = _ref(form, n, x)
    got = y.cpu()[:len(want)].tolist()
    if [float(v) for v in got] == [float(v) for v in want]:
        return None
    return f"got {got[:6]} want {want[:6]}"


FORMS = ["swap", "rotate3", "bodyload", "condload", "nested", "tensorswap", "twoexits",
         "unusedcarry"]
TRIPS = [0, 1, 2, 7, 20]
WARPS = [1, 4]
DTYPES = [(torch.int32, "i32"), (torch.float32, "f32")]


def cells(sw):
    for form, fn in enumerate(FORMS):
        for n in TRIPS:
            for w in WARPS:
                for dtype, dn in DTYPES:
                    if form == 5 and n > 7:
                        continue  # Fibonacci growth leaves f32's exact integers
                    tag = f"{fn}_n{n}_w{w}_{dn}"
                    sw.run(tag, lambda form=form, n=n, w=w, dtype=dtype:
                           _cell(form, n, w, dtype))


if __name__ == "__main__":
    sw = Sweep("s_while")
    cells(sw)
    sw.finish()
