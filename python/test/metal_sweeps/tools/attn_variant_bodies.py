"""Which emitter body did each cell of `s_attn_variants.py` actually run?

A correctness sweep that reports 0 wrong answers proves nothing unless the
cells reached the code the sweep was written for. `metal.fused_attention` has
four emitted shapes -- the full-tile MMA body, the chunked one, the chunked one
walking the program's rows in blocks, and a scalar correctness floor -- and a
kernel the matcher declines outright reaches none of them. This censuses the
generated MSL over the same axes, so the sweep's verdict can be read as a
statement about specific bodies.

    pixi run --frozen python python/test/metal_sweeps/tools/attn_variant_bodies.py
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, os.pardir, "sweeps"))

import torch  # noqa: E402
import triton  # noqa: E402

import s_attn_variants as S  # noqa: E402

M = 128
BN = 64


# Each body opens with a comment naming itself. Use that, not a guess at which
# temporaries are unique to which body: `_fa_kbase`, `_fa_dc` and `_fa_col` are
# emitted by ALL THREE, and reading the census off them labels everything
# chunked.
BODIES = (
    ("metal.fused_attention (chunked simdgroup dots", "mma-chunked"),
    ("metal.fused_attention (simdgroup dots", "mma-full"),
    ("metal.fused_attention (per-key", "scalar-floor"),
)


def body_of(msl):
    for marker, name in BODIES:
        if marker in msl:
            if name == "mma-chunked" and "_fa_rb" in msl:
                return "mma-row-blocked"
            return name
    return "not-fused"


def main():
    seen = {}
    for name in S.VARIANTS:
        mk_in, launch, _ref, mk_out = S.VARIANTS[name]
        for d in S.D_HEADS:
            for bm in S.BLOCK_MS:
                for nw in (1, 4):
                    bd = max(16, triton.next_power_of_2(d))
                    torch.manual_seed(0)
                    host = mk_in(M, d)
                    dev = [x.to("mps") for x in host]
                    out = mk_out(M, d) if mk_out else torch.zeros(M, d, device="mps")
                    compiled = launch(dev, out, M, d, bm, BN, bd, nw)
                    torch.mps.synchronize()
                    msl = compiled.asm["metal"]
                    if isinstance(msl, bytes):
                        msl = msl.decode()
                    body = body_of(msl)
                    # `_fa_rl` is emitted either way; only a SPLIT row
                    # spells it with a `%`.
                    lanes = "_fa_rl = _fa_tid %" in msl
                    clamp = "_fa_kend = min(_fa_kend, _fa_rowoff" in msl
                    key = (name, body, lanes, clamp)
                    seen.setdefault(key, []).append(f"d{d}_bm{bm}_nw{nw}")
    print(f"{'variant':<9} {'body':<16} {'lanes':<6} {'kclamp':<7} cells  example")
    for (name, body, lanes, clamp), cells in sorted(seen.items()):
        print(f"{name:<9} {body:<16} {str(lanes):<6} {str(clamp):<7} "
              f"{len(cells):<6} {cells[0]}")
    fused = sum(len(c) for (_n, b, _l, _c), c in seen.items() if b != "not-fused")
    total = sum(len(c) for c in seen.values())
    floor = sum(len(c) for (_n, b, _l, _c), c in seen.items() if b == "scalar-floor")
    print(f"\n[bodies] {fused}/{total} cells on metal.fused_attention, "
          f"{floor} of those on the scalar floor")
    return 0


if __name__ == "__main__":
    sys.exit(main())
