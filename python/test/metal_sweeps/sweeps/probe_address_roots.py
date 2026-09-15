"""The address-root envelope: what may define a load/store/atomic pointer.

This is the axis the round-11 P0 decline audit did not sweep. That audit asked
"where can a `notifyMatchFailure` still reach `applyFullConversion` and kill the
process" and answered "not on any axis swept here" — it swept arithmetic/math
element types, gather/join/split, atomics and `dot_scaled`, but never the SHAPE
OF THE ADDRESS.

The address family is the worst risk/coverage ratio in the backend: 16 decline
sites inside `applyFullConversion` against 6 pre-conversion validators, and the
four main gates (`LoadLowering`, `MaskedLoadLowering`, `StoreLowering`,
`MaskedStoreLowering`) each keep their own copy of the predicate.

The cell that started this: a 2-D tile whose address has a UNIFORM axis. Triton
folds the uniform axis away and hands the load a `tt.broadcast` of an Nx1
pointer tensor, which no gate accepts. Reached by ordinary code —
`leetgpu-triton-solve/hard-all_pairs_shortest_paths.py` (Floyd-Warshall) writes
`tl.full((BLOCK_N,), k, tl.int32)[None, :]` and exits 139.

Axes: address root x op x mask form. Run:

    pixi run --frozen python python/test/metal_sweeps/sweeps/probe_address_roots.py
    REPEAT=30 pixi run --frozen python .../probe_address_roots.py <tag>
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

import probe_lib as P  # noqa: E402

P.SCRATCH = P.scratch_dir("address_roots")

# --------------------------------------------------------------------------
# Address roots. Each is a Python expression building the pointer tensor for a
# BLOCK x BLOCK f32 tile out of `X` (row stride `sx`), plus the oracle that
# says what that tile should contain.
# --------------------------------------------------------------------------

ROOTS = {
    # Control: both axes carry a per-element term, so the ptr IS a tt.addptr.
    "addptr2d": (
        "X + i[:, None] * sx + j[None, :]",
        "ref",
    ),
    # Uniform COLUMN: `ko` is a splat, so Triton folds the column axis away and
    # broadcasts a BLOCKx1 pointer tensor. The Floyd-Warshall shape.
    "bcast_uniform_col": (
        "X + i[:, None] * sx + tl.full((BLOCK,), KCOL, dtype=tl.int32)[None, :]",
        "ref[:, KCOL:KCOL + 1].expand(BLOCK, BLOCK)",
    ),
    # Uniform ROW: the mirror, a 1xBLOCK pointer tensor broadcast down.
    "bcast_uniform_row": (
        "X + tl.full((BLOCK,), KROW, dtype=tl.int32)[:, None] * sx + j[None, :]",
        "ref[KROW:KROW + 1, :].expand(BLOCK, BLOCK)",
    ),
    # Same shape, spelled with zeros+scalar instead of tl.full — a different
    # frontend path to the same folded address.
    "bcast_zeros_col": (
        "X + i[:, None] * sx + (tl.zeros((BLOCK,), dtype=tl.int32) + KCOL)[None, :]",
        "ref[:, KCOL:KCOL + 1].expand(BLOCK, BLOCK)",
    ),
    # A per-element column term that is uniform only at RUNTIME. The address is
    # a real 2-D addptr, so this must keep working however the gate is widened.
    "addptr2d_runtime_col": (
        "X + i[:, None] * sx + (j * 0 + KCOL)[None, :]",
        "ref[:, KCOL:KCOL + 1].expand(BLOCK, BLOCK)",
    ),
    # Column axis contributes nothing at all: a BLOCKx1 tile, no broadcast.
    "addptr_col_tile": (
        "X + i[:, None] * sx",
        "ref[:, 0:1]",
    ),
}

# Which roots produce a BLOCKx1 result rather than BLOCKxBLOCK.
NARROW = {"addptr_col_tile"}

LOAD_SRC = """
@triton.jit
def k(X, O, sx, KROW, KCOL, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    j = tl.arange(0, BLOCK)
    src = {root}
    v = tl.load(src{mask})
    tl.store(O + i[:, None] * {ostride} + {ocol}, v)

BLOCK, KROW, KCOL = 64, 3, 5
ref = torch.arange(BLOCK * BLOCK, dtype=torch.float32).reshape(BLOCK, BLOCK)
x = ref.to("mps")
out = torch.zeros({oshape}, device="mps")
k[(1,)](x, out, BLOCK, KROW, KCOL, BLOCK=BLOCK, num_warps={nw})
torch.mps.synchronize()
want = ({oracle}).contiguous()
got = out.cpu()
if not torch.equal(got, want):
    n = int((got != want).sum())
    print(f"WRONG {{n}}/{{want.numel()}} cells differ; first got={{got.flatten()[(got != want).flatten().nonzero()[0]].item()}}")
"""

STORE_SRC = """
@triton.jit
def k(X, O, sx, KROW, KCOL, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    j = tl.arange(0, BLOCK)
    v = tl.load(X + i[:, None] * sx + j[None, :])
    dst = {root}
    tl.store(dst, v{mask})

BLOCK, KROW, KCOL = 64, 3, 5
ref = torch.arange(BLOCK * BLOCK, dtype=torch.float32).reshape(BLOCK, BLOCK)
x = ref.to("mps")
out = torch.zeros(BLOCK, BLOCK, device="mps")
k[(1,)](x, out, BLOCK, KROW, KCOL, BLOCK=BLOCK, num_warps={nw})
torch.mps.synchronize()
# Many lanes target one address; only check that SOMETHING landed there and that
# nothing outside the addressed region was touched.
got = out.cpu()
if got.abs().sum() == 0:
    print("WRONG store wrote nothing")
"""

ATOMIC_SRC = """
@triton.jit
def k(X, O, sx, KROW, KCOL, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    j = tl.arange(0, BLOCK)
    v = tl.load(X + i[:, None] * sx + j[None, :])
    dst = {root}
    tl.atomic_add(dst, v{mask})

BLOCK, KROW, KCOL = 64, 3, 5
ref = torch.arange(BLOCK * BLOCK, dtype=torch.float32).reshape(BLOCK, BLOCK)
x = ref.to("mps")
out = torch.zeros(BLOCK, BLOCK, device="mps")
k[(1,)](x, out, BLOCK, KROW, KCOL, BLOCK=BLOCK, num_warps={nw})
torch.mps.synchronize()
got = out.cpu()
if got.abs().sum() == 0:
    print("WRONG atomic wrote nothing")
"""

MASKS = {
    "nomask": ("", ""),
    "mask": (", mask=(i[:, None] < BLOCK)", ", mask=(i[:, None] < BLOCK)"),
    "other": (", mask=(i[:, None] < BLOCK), other=0.0", ", mask=(i[:, None] < BLOCK)"),
}


def build():
    for root_name, (root, oracle) in ROOTS.items():
        narrow = root_name in NARROW
        oshape = "BLOCK, 1" if narrow else "BLOCK, BLOCK"
        ostride = "1" if narrow else "BLOCK"
        ocol = "0" if narrow else "j[None, :]"
        for mask_name, (load_mask, store_mask) in MASKS.items():
            for nw in (1, 4):
                suffix = f"{root_name}_{mask_name}_nw{nw}"
                P.PROBES[f"load_{suffix}"] = LOAD_SRC.format(
                    root=root, mask=load_mask, oracle=oracle, oshape=oshape,
                    ostride=ostride, ocol=ocol, nw=nw)
                # A store or atomic through a BLOCKx1 address is the plain tile
                # case, already covered by the load grid; skip it.
                if narrow:
                    continue
                P.PROBES[f"store_{suffix}"] = STORE_SRC.format(
                    root=root.replace("X +", "O +"), mask=store_mask, nw=nw)
                if mask_name != "other":
                    P.PROBES[f"atomic_{suffix}"] = ATOMIC_SRC.format(
                        root=root.replace("X +", "O +"), mask=store_mask, nw=nw)


build()

if __name__ == "__main__":
    sys.exit(P.main())
