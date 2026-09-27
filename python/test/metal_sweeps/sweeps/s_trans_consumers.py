"""What a rank-2 `tl.trans` result may meet: another tile, a reduce, a store.

`tt.trans` is a relabel; the data movement lands on the `ttg.convert_layout`
Triton puts after it. Stored directly, that pair is exact (the diagnostic
audit's 13 trans forms). Combined elementwise with a tile the kernel LOADED
-- `tl.trans(x) + w` -- the cvt goes from the transposed layout (`order =
[0, 1]`, sizePerThread along rows) to `w`'s, and was refused by name as out
of the staged-transpose envelope.

Axes: consumer x M x N x num_warps.

  store      tl.trans(x) stored directly -- the control
  add_store  tl.trans(x) + w, stored
  add_red0   tl.sum(tl.trans(x) + w, 0)
  add_red1   tl.sum(tl.trans(x) + w, 1)
  mul_w      w * tl.trans(x), stored (operand order swapped)

    python/test/metal_sweeps/drive.sh python/test/metal_sweeps/sweeps/s_trans_consumers.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402
from sweep_lib import Sweep  # noqa: E402


@triton.jit
def kernel(X, W, Y, M: tl.constexpr, N: tl.constexpr, CONS: tl.constexpr):
    m = tl.arange(0, M)
    n = tl.arange(0, N)
    x = tl.load(X + m[:, None] * N + n[None, :])          # [M, N]
    w = tl.load(W + n[:, None] * M + m[None, :])          # [N, M]
    t = tl.trans(x)                                       # [N, M]
    off = n[:, None] * M + m[None, :]
    if CONS == 0:
        tl.store(Y + off, t)
    elif CONS == 1:
        tl.store(Y + off, t + w)
    elif CONS == 2:
        tl.store(Y + m, tl.sum(t + w, 0))
    elif CONS == 3:
        tl.store(Y + n, tl.sum(t + w, 1))
    else:
        tl.store(Y + off, w * t)


CONS = ["store", "add_store", "add_red0", "add_red1", "mul_w"]
SHAPES = [(8, 8), (16, 16), (16, 32), (32, 16), (8, 64), (64, 8), (32, 32),
          (64, 64), (4, 128)]
WARPS = [1, 2, 4]


def _cell(c, M, N, w):
    torch.manual_seed(M * 31 + N + w + c)
    x = torch.randn(M, N)
    ww = torch.randn(N, M)
    size = {0: N * M, 1: N * M, 2: M, 3: N, 4: N * M}[c]
    y = torch.zeros(size, device="mps")
    kernel[(1,)](x.to("mps"), ww.to("mps"), y, M, N, c, num_warps=w)
    torch.mps.synchronize()
    t = x.T
    want = {0: t, 1: t + ww, 2: (t + ww).sum(0), 3: (t + ww).sum(1), 4: ww * t}[c]
    got = y.cpu().reshape(want.shape)
    if torch.allclose(got, want, atol=1e-4, rtol=1e-4):
        return None
    return f"{int((~torch.isclose(got, want, atol=1e-4, rtol=1e-4)).sum())}/{want.numel()} differ"


def cells(sw):
    for c, name in enumerate(CONS):
        for (M, N) in SHAPES:
            for w in WARPS:
                e = M * N / (32 * w)
                sw.run(f"{name}_{M}x{N}_w{w}_e{e:g}",
                       lambda c=c, M=M, N=N, w=w: _cell(c, M, N, w))


if __name__ == "__main__":
    sw = Sweep("s_trans_consumers")
    cells(sw)
    sw.finish()
