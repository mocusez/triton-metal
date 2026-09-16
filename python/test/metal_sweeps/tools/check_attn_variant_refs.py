"""Check `s_attn_variants.py`'s oracles against the Triton interpreter.

A differential sweep is only as good as the thing it differs against. The five
references in `s_attn_variants.py` were written by reading each corpus kernel,
and a misread reference reports the backend as wrong when it is right -- or,
worse, agrees with a backend that is wrong because both made the same mistake.

`TRITON_INTERPRET=1` runs the same `@triton.jit` body in Python on CPU tensors,
which is an independent implementation of exactly the kernel being swept. If
the reference and the interpreter agree at several block shapes, the reference
is the kernel. Run this whenever an oracle in that sweep is added or edited.

    pixi run --frozen python python/test/metal_sweeps/tools/check_attn_variant_refs.py
"""

import os
import sys

os.environ["TRITON_INTERPRET"] = "1"   # must precede the triton import

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, os.pardir, "sweeps"))

import torch  # noqa: E402
import triton  # noqa: E402

import s_attn_variants as S  # noqa: E402

# Small on purpose: the interpreter is a Python loop, and a misread oracle
# shows up at any shape. The block shapes vary because `sinks` derives its key
# range from BLOCK_M and BLOCK_N.
CASES = [
    (32, 16, 16, 16),
    (40, 16, 16, 16),
    (40, 32, 32, 64),
    (48, 16, 64, 16),
]


def main():
    bad = 0
    for name in S.VARIANTS:
        mk_in, launch, ref_fn, mk_out = S.VARIANTS[name]
        for (M, d, bm, bn) in CASES:
            bd = max(16, triton.next_power_of_2(d))
            torch.manual_seed(0)
            host = mk_in(M, d)
            out = (mk_out(M, d).cpu() if mk_out
                   else torch.zeros(M, d))
            launch(host, out, M, d, bm, bn, bd, 4)
            ref = ref_fn(host, M, d, bm, bn)
            err = (out.double() - ref).abs().max().item()
            tol = 1e-5 * max(1.0, ref.abs().max().item())
            verdict = "ok" if err <= tol else "MISMATCH"
            if err > tol:
                bad += 1
            print(f"{verdict:>9}  {name:<8} M={M:<4} d={d:<4} bm={bm:<3} bn={bn:<3} "
                  f"err={err:.3e} tol={tol:.1e}", flush=True)
    print(f"\n[refcheck] mismatches={bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
