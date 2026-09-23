"""The rank-3 `tl.permute` envelope, and where its destination stops being
addressable.

`tt.trans` is a relabel — the previous rounds measured that four separate ways —
and the op that actually moves the data is the `ttg.convert_layout` Triton puts
between the trans and the store. That cvt needs the full-tile exchange (publish,
barrier, read in a second loop), which is granted only to a function that
already has a phase, so a permute-and-store kernel was refused by name at every
shape above one element per thread.

With the phase granted unconditionally the cells split 51 exact / 19 silently
wrong, and the boundary is NOT the `E >= 8` register band it first looks like:
four of the 19 sit at E = 4, and under the full reversal `8x4x4` at one warp is
exact while `2x4x16` at one warp is not. What separates them is whether the
DESTINATION layout's flat index is the thread-major expression
(`localTid * E + iv`) that the consuming store's own address arithmetic emits --
the read half asks the layout, the store asks the tile, and only then do the two
agree.

Three arms, and the controls are what make the permute arm readable:

  permute    `tl.permute(v, perm)` stored through the permuted shape's address.
  identity   the SAME kernel with perm = (0, 1, 2). Isolates "a rank-3 tile with
             two shapes in one kernel" from "the permutation moves data". Exact
             on both binaries at every shape, including the E = 8 ones the
             permuting arm still refuses -- so the refusal is about the
             permutation, not about the rank or the register band.
  address    the SAME FUNCTION with no `tl.permute` at all: the load's index cone
             carries the permuted strides instead.

🔴 That third arm is RED, and its message is the opposite of what a control
usually carries: **56 of its 84 cells are silently wrong, and they are wrong
identically on both binaries** (A/B'd). A rank-3 load whose INNERMOST
destination axis does not carry the source's unit stride returns the wrong
elements -- all fourteen shape/warp pairs of each of the four permutations that
move a strided source axis into the last position, from E = 1 upward, while
`(1, 0, 2)` (which keeps axis 2 innermost) and the identity are exact. Every
rank-3 address probe before this one held the inner stride at 1
(`.omc/fuzz/rank3_address.py`'s `strided` mode is `i * PS + j * RS + k`), which
is why it was never seen. It is the rank-3 instance of the rank-2 defect
`3aba35ad6f` fixed -- no tier checked that a tile's inner axis is contiguous.

So the permute fix is NOT "one of two spellings reached a path that already
worked": both spellings were wrong, in different ways, and this sweep's permute
arm is the half that is now closed. The address arm is left in, red, because it
belongs to this family and is the next slice.

Axes: permutation (all six) x shape x warp count, which is what fixes E.

    python/test/metal_sweeps/drive.sh \
        python/test/metal_sweeps/sweeps/s_rank3_permute.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402
from sweep_lib import Sweep  # noqa: E402


@triton.jit
def permute_kernel(X, O, A: tl.constexpr, B: tl.constexpr, C: tl.constexpr,
                   S0: tl.constexpr, S1: tl.constexpr, S2: tl.constexpr,
                   P0: tl.constexpr, P1: tl.constexpr, P2: tl.constexpr):
    a = tl.arange(0, A)[:, None, None]
    b = tl.arange(0, B)[None, :, None]
    c = tl.arange(0, C)[None, None, :]
    v = tl.load(X + a * (B * C) + b * C + c)
    p = tl.permute(v, (P0, P1, P2))
    i0 = tl.arange(0, S0)[:, None, None]
    i1 = tl.arange(0, S1)[None, :, None]
    i2 = tl.arange(0, S2)[None, None, :]
    tl.store(O + i0 * (S1 * S2) + i1 * S2 + i2, p)


@triton.jit
def address_kernel(X, O, S0: tl.constexpr, S1: tl.constexpr, S2: tl.constexpr,
                   T0: tl.constexpr, T1: tl.constexpr, T2: tl.constexpr):
    # The same permutation with no layout op: the load's own index cone carries
    # the permuted strides. `T*` are the source strides for the destination's
    # axes, so the tile is gathered already transposed.
    i0 = tl.arange(0, S0)[:, None, None]
    i1 = tl.arange(0, S1)[None, :, None]
    i2 = tl.arange(0, S2)[None, None, :]
    v = tl.load(X + i0 * T0 + i1 * T1 + i2 * T2)
    tl.store(O + i0 * (S1 * S2) + i1 * S2 + i2, v)


PERMS = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
SHAPES = [(2, 2, 8, 1), (2, 4, 8, 1), (2, 4, 8, 2), (4, 4, 4, 1),
          (2, 4, 16, 1), (2, 4, 16, 4), (8, 4, 4, 1), (4, 8, 8, 1),
          (4, 8, 8, 2), (4, 8, 8, 4), (2, 8, 16, 1), (2, 8, 16, 2),
          (8, 8, 8, 4), (2, 16, 16, 2)]

NAMED = []


def _cmp(got, want, atol=2e-3):
    got = got.float().cpu()
    want = want.float().cpu()
    if torch.allclose(got, want, atol=atol, rtol=2e-3):
        return None
    d = (got - want).abs()
    i = int(d.argmax())
    return (f"{int((d > atol).sum())}/{got.numel()} differ; "
            f"maxdiff={d.max().item():.6g} at {i}: got "
            f"{got.flatten()[i].item():.6g} want {want.flatten()[i].item():.6g}")


def _refused(exc):
    """A named pre-conversion refusal, not a crash and not a wrong answer."""
    text = str(exc)
    return ("convert-tritongpu-to-metal failed" in text
            or "Metal backend:" in text)


def _permute(tag, a, b, c, nw, perm):
    def cell():
        torch.manual_seed(0)
        dims = (a, b, c)
        s = tuple(dims[i] for i in perm)
        x = torch.randn(a * b * c, dtype=torch.float32)
        o = torch.zeros(a * b * c, dtype=torch.float32, device="mps")
        try:
            permute_kernel[(1,)](x.to("mps"), o, a, b, c, s[0], s[1], s[2],
                                 perm[0], perm[1], perm[2], num_warps=nw)
        except Exception as e:  # noqa: BLE001
            if _refused(e):
                NAMED.append(tag)
                return None
            raise
        torch.mps.synchronize()
        want = x.reshape(a, b, c).permute(*perm)
        return _cmp(o.reshape(want.shape), want)
    return cell


def _address(tag, a, b, c, nw, perm):
    def cell():
        torch.manual_seed(0)
        dims = (a, b, c)
        src = (b * c, c, 1)
        s = tuple(dims[i] for i in perm)
        t = tuple(src[i] for i in perm)
        x = torch.randn(a * b * c, dtype=torch.float32)
        o = torch.zeros(a * b * c, dtype=torch.float32, device="mps")
        try:
            address_kernel[(1,)](x.to("mps"), o, s[0], s[1], s[2],
                                 t[0], t[1], t[2], num_warps=nw)
        except Exception as e:  # noqa: BLE001
            if _refused(e):
                NAMED.append(tag)
                return None
            raise
        torch.mps.synchronize()
        want = x.reshape(a, b, c).permute(*perm)
        return _cmp(o.reshape(want.shape), want)
    return cell


def main():
    sw = Sweep("s_rank3_permute")
    for perm in PERMS:
        p = "".join(str(d) for d in perm)
        for (a, b, c, nw) in SHAPES:
            e = (a * b * c) // (32 * nw)
            if e < 1:
                continue
            base = f"{a}x{b}x{c}_w{nw}_e{e}_p{p}"
            sw.run(f"permute_{base}", _permute(f"permute_{base}", a, b, c, nw, perm))
            sw.run(f"address_{base}", _address(f"address_{base}", a, b, c, nw, perm))
    sw.finish()
    print(f"NAMED\t{len(NAMED)}\t" + ",".join(NAMED), flush=True)


if __name__ == "__main__":
    main()
