"""Two rank-2 tile SHAPES in one kernel, neither derived from the other.

Every rank-2 range is decomposed against a tile: its own, when the pre-pass
found one, and otherwise the kernel's largest. Found while controlling a topk
probe: a kernel that copies a `[4, 2]` tile and, independently, a `[4, 4]` one
wrote the narrow tile as `tid / 4, tid % 4` -- the wide tile's row length --
because the own tile was only accepted when its element COUNT matched the
largest one's, and a `[R, 1]` column was never recognised as a tile at all.

Axes: narrow shape (unit column, unit row, narrow, square, tall) x wide shape x
num_warps, with each tile's E = numel / (32 * warps) in the tag. Two buffers
each, loaded, transformed, stored -- nothing moves between the tiles, so the
only way to be wrong is to address one tile with the other's shape.

    python/test/metal_sweeps/drive.sh python/test/metal_sweeps/sweeps/s_two_rank2_tiles.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402
from sweep_lib import Sweep  # noqa: E402


@triton.jit
def two_tiles(A, B, C, D, R1: tl.constexpr, C1: tl.constexpr,
              R2: tl.constexpr, C2: tl.constexpr, SHARE_ROWS: tl.constexpr):
    r1 = tl.arange(0, R1)
    c1 = tl.arange(0, C1)
    a = tl.load(A + r1[:, None] * C1 + c1[None, :])
    tl.store(B + r1[:, None] * C1 + c1[None, :], a * 2.0)
    if SHARE_ROWS:
        r2 = r1
    else:
        r2 = tl.arange(0, R2)
    c2 = tl.arange(0, C2)
    c = tl.load(C + r2[:, None] * C2 + c2[None, :])
    tl.store(D + r2[:, None] * C2 + c2[None, :], c + 1.0)


NARROW = [(4, 1), (16, 1), (64, 1), (1, 4), (1, 64), (4, 2), (8, 4), (32, 2),
          (16, 16), (64, 8)]
WIDE = [(4, 4), (16, 16), (64, 32), (8, 128)]
WARPS = [1, 2, 4]


def _cell(n, w, warps, share):
    (r1, c1), (r2, c2) = n, w
    torch.manual_seed(r1 * 131 + c1 * 7 + r2 + c2 + warps)
    a = torch.randn(r1, c1, device="mps")
    b = torch.zeros(r1, c1, device="mps")
    c = torch.randn(r2, c2, device="mps")
    d = torch.zeros(r2, c2, device="mps")
    two_tiles[(1,)](a, b, c, d, r1, c1, r2, c2, share, num_warps=warps)
    torch.mps.synchronize()
    bad = []
    if not torch.equal(b.cpu(), a.cpu() * 2):
        bad.append(f"first {int((b.cpu() != a.cpu() * 2).sum())}/{b.numel()}")
    if not torch.equal(d.cpu(), c.cpu() + 1):
        bad.append(f"second {int((d.cpu() != c.cpu() + 1).sum())}/{d.numel()}")
    return "; ".join(bad) or None


def cells(sw):
    for n in NARROW:
        for w in WIDE:
            if n == w:
                continue
            for warps in WARPS:
                for share in (False, True):
                    if share and n[0] != w[0]:
                        continue
                    e1 = n[0] * n[1] / (32 * warps)
                    e2 = w[0] * w[1] / (32 * warps)
                    sw.run(f"{n[0]}x{n[1]}_{w[0]}x{w[1]}_w{warps}_share{int(share)}"
                           f"_e{e1:g}_e{e2:g}",
                           lambda n=n, w=w, warps=warps, share=share:
                           _cell(n, w, warps, share))
                    # And the other way round: the wide tile first.
                    sw.run(f"{w[0]}x{w[1]}_{n[0]}x{n[1]}_w{warps}_share{int(share)}"
                           f"_e{e2:g}_e{e1:g}",
                           lambda n=n, w=w, warps=warps, share=share:
                           _cell(w, n, warps, share))


if __name__ == "__main__":
    sw = Sweep("s_two_rank2_tiles")
    cells(sw)
    sw.finish()
