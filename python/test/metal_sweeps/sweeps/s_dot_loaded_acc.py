"""`tl.dot(a, b, c)` with an accumulator that is not a zero splat.

C = A @ B + C is the ordinary way to write a GEMM with a bias or a beta term.
Loop-free, the accumulator reaches the dot through a blocked -> blocked
`ttg.convert_layout` into the dot's parent layout, and that cvt used to be
refused by name at every shape but 8x8. The scalar-dot tier now takes the init
from before the cvt. A note in `sdDotEligible` records the scalar tier once
answering a separate-buffer accumulator WRONG at two and four warps, so the
warp count is an axis here, not a constant.

Axes: accumulator form x M x N x K x num_warps.

  sep      c from its own buffer
  inplace  c loaded from the output pointer (accumulate in place)
  zero     zero init -- the control
  scaled   c = 0.5 * load(C) -- the init is computed, not a bare load
  colmajor c loaded transposed, `C + n * M + m`
  masked   ragged M/N tail with masked loads and store

    python/test/metal_sweeps/drive.sh python/test/metal_sweeps/sweeps/s_dot_loaded_acc.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402
from sweep_lib import Sweep  # noqa: E402


@triton.jit
def kernel(A, B, C, O, Mr, Nr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
           FORM: tl.constexpr):
    m = tl.arange(0, M)
    n = tl.arange(0, N)
    k = tl.arange(0, K)
    if FORM == 5:
        mm = (m[:, None] < Mr) & (n[None, :] < Nr)
        a = tl.load(A + m[:, None] * K + k[None, :], mask=(m[:, None] < Mr), other=0.0)
        b = tl.load(B + k[:, None] * N + n[None, :], mask=(n[None, :] < Nr), other=0.0)
        c = tl.load(C + m[:, None] * N + n[None, :], mask=mm, other=0.0)
        tl.store(O + m[:, None] * N + n[None, :], tl.dot(a, b, c), mask=mm)
    else:
        a = tl.load(A + m[:, None] * K + k[None, :])
        b = tl.load(B + k[:, None] * N + n[None, :])
        if FORM == 0:
            c = tl.load(C + m[:, None] * N + n[None, :])
        elif FORM == 1:
            c = tl.load(O + m[:, None] * N + n[None, :])
        elif FORM == 2:
            c = tl.zeros((M, N), tl.float32)
        elif FORM == 3:
            c = tl.load(C + m[:, None] * N + n[None, :]) * 0.5
        else:
            c = tl.load(C + n[None, :] * M + m[:, None])
        tl.store(O + m[:, None] * N + n[None, :], tl.dot(a, b, c))


FORMS = ["sep", "inplace", "zero", "scaled", "colmajor", "masked"]
SHAPES = [(8, 8, 8), (8, 16, 8), (16, 16, 16), (16, 32, 8), (32, 16, 32),
          (32, 32, 32), (64, 64, 32), (16, 16, 64), (64, 32, 16)]
WARPS = [1, 2, 4]


def _cell(form, M, N, K, w):
    torch.manual_seed(M * 7 + N * 3 + K + w)
    a = torch.randn(M, K)
    b = torch.randn(K, N)
    c = torch.randn(M, N)
    o0 = torch.randn(M, N)
    Mr, Nr = (M - 3, N - 5) if form == 5 else (M, N)
    if form == 4:
        cbuf = c.T.contiguous()
    else:
        cbuf = c
    o = o0.clone().to("mps")
    kernel[(1,)](a.to("mps"), b.to("mps"), cbuf.to("mps"), o, Mr, Nr, M, N, K, form,
                 num_warps=w)
    torch.mps.synchronize()
    acc = {0: c, 1: o0, 2: torch.zeros(M, N), 3: 0.5 * c, 4: c, 5: c}[form]
    want = a @ b + acc
    if form == 5:
        mm = (torch.arange(M)[:, None] < Mr) & (torch.arange(N)[None, :] < Nr)
        want = torch.where(mm, want, o0)
    got = o.cpu()
    if torch.allclose(got, want, atol=1e-3, rtol=1e-3):
        return None
    bad = int((~torch.isclose(got, want, atol=1e-3, rtol=1e-3)).sum())
    return f"{bad}/{M * N} differ"


def cells(sw):
    for f, name in enumerate(FORMS):
        for (M, N, K) in SHAPES:
            if name == "masked" and (M < 8 or N < 8):
                continue
            for w in WARPS:
                sw.run(f"{name}_{M}x{N}x{K}_w{w}",
                       lambda f=f, M=M, N=N, K=K, w=w: _cell(f, M, N, K, w))


if __name__ == "__main__":
    sw = Sweep("s_dot_loaded_acc")
    cells(sw)
    sw.finish()
