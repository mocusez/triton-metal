"""A `tl.dot` result stored through a TRANSPOSED address.

Found by `probe_cvt_envelope.py` while measuring the P2.1 blocked<->blocked
`ttg.convert_layout` envelope: `tl.store(O + n * M + m, tl.dot(a, b))` -- the
output written column-major, which is how a kernel produces C^T without a
`tl.trans` -- is a silent wrong answer, at every warp count.

The two control arms are what make the cell readable, and both are green:

  rowmajor   the same dot stored `O + m * N + n`. Isolates "a dot is present"
             from "the dot's address is transposed".
  trans      `tl.store(O + nn * M + mm, tl.trans(acc))` -- the SAME function of
             the same inputs, spelled with an explicit `tl.trans`. It is exact,
             so the backend can already produce this output; only one of the two
             spellings reaches the path that does.

  nodot      an elementwise tile stored through the same transposed address, at
             the same shapes and warp counts. Exact in the probe round, so the
             transposed store address is not by itself the defect.

Axes: shape x warps x form x dtype x whether the dot sits in a K loop. The
oracle is verified against `TRITON_INTERPRET=1` (all three forms agree there).

    python/test/metal_sweeps/drive.sh \
        python/test/metal_sweeps/sweeps/s_dot_store_transposed.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402
from sweep_lib import Sweep  # noqa: E402

ROWMAJOR, COLMAJOR, TRANS = 0, 1, 2


@triton.jit
def dot_kernel(A, B, O, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
               FORM: tl.constexpr, EPI: tl.constexpr):
    m = tl.arange(0, M)[:, None]
    kk = tl.arange(0, K)[None, :]
    n = tl.arange(0, N)[None, :]
    a = tl.load(A + m * K + kk)
    b = tl.load(B + tl.arange(0, K)[:, None] * N + n)
    acc = tl.dot(a, b)
    if EPI == 1:
        acc = acc * 2.0
    if FORM == 0:
        tl.store(O + m * N + n, acc)
    elif FORM == 1:
        tl.store(O + n * M + m, acc)
    else:
        nn = tl.arange(0, N)[:, None]
        mm = tl.arange(0, M)[None, :]
        tl.store(O + nn * M + mm, tl.trans(acc))


@triton.jit
def dot_loop_kernel(A, B, O, M: tl.constexpr, N: tl.constexpr,
                    K: tl.constexpr, BK: tl.constexpr, FORM: tl.constexpr):
    m = tl.arange(0, M)[:, None]
    n = tl.arange(0, N)[None, :]
    acc = tl.zeros((M, N), dtype=tl.float32)
    for k0 in range(0, K, BK):
        kk = k0 + tl.arange(0, BK)
        a = tl.load(A + m * K + kk[None, :])
        b = tl.load(B + kk[:, None] * N + n)
        acc += tl.dot(a, b)
    if FORM == 0:
        tl.store(O + m * N + n, acc)
    elif FORM == 1:
        tl.store(O + n * M + m, acc)
    else:
        nn = tl.arange(0, N)[:, None]
        mm = tl.arange(0, M)[None, :]
        tl.store(O + nn * M + mm, tl.trans(acc))


@triton.jit
def nodot_kernel(A, B, O, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                 FORM: tl.constexpr, EPI: tl.constexpr):
    # Same shapes, same addresses, no matrix unit: the control that says the
    # transposed store address is not itself the defect.
    m = tl.arange(0, M)[:, None]
    n = tl.arange(0, N)[None, :]
    v = tl.load(A + m * N + n)
    if EPI == 1:
        v = v * 2.0
    if FORM == 0:
        tl.store(O + m * N + n, v)
    elif FORM == 1:
        tl.store(O + n * M + m, v)
    else:
        nn = tl.arange(0, N)[:, None]
        mm = tl.arange(0, M)[None, :]
        tl.store(O + nn * M + mm, tl.trans(v))


SHAPES = [(16, 16, 16), (32, 16, 16), (16, 32, 16), (32, 32, 32),
          (64, 32, 32), (16, 16, 32)]
WARPS = [1, 2, 4]
FORMS = ((ROWMAJOR, "rowmajor"), (COLMAJOR, "colmajor"), (TRANS, "trans"))


def _cmp(got, want, atol=2e-3):
    got = got.float().cpu()
    want = want.float().cpu()
    if torch.allclose(got, want, atol=atol, rtol=2e-3):
        return None
    d = (got - want).abs()
    i = int(d.argmax())
    n = int((d > atol).sum())
    return (f"{n}/{got.numel()} differ; maxdiff={d.max().item():.6g} at {i}: "
            f"got {got.flatten()[i].item():.6g} want "
            f"{want.flatten()[i].item():.6g}")


def _want(ref, form):
    return ref if form == ROWMAJOR else ref.t().contiguous()


def _dot(m, n, k, w, form, epi, dt):
    torch.manual_seed(0)
    a = torch.randn(m, k, dtype=dt)
    b = torch.randn(k, n, dtype=dt)
    ref = (a.float() @ b.float())
    if epi == 1:
        ref = ref * 2.0
    o = torch.zeros(m * n, dtype=torch.float32, device="mps")
    dot_kernel[(1,)](a.to("mps"), b.to("mps"), o, m, n, k, form, epi,
                     num_warps=w)
    torch.mps.synchronize()
    want = _want(ref, form)
    return _cmp(o.reshape(want.shape), want)


def _dot_loop(m, n, k, bk, w, form):
    torch.manual_seed(0)
    a = torch.randn(m, k, dtype=torch.float32)
    b = torch.randn(k, n, dtype=torch.float32)
    ref = a @ b
    o = torch.zeros(m * n, dtype=torch.float32, device="mps")
    dot_loop_kernel[(1,)](a.to("mps"), b.to("mps"), o, m, n, k, bk, form,
                          num_warps=w)
    torch.mps.synchronize()
    want = _want(ref, form)
    return _cmp(o.reshape(want.shape), want)


def _nodot(m, n, w, form, epi):
    torch.manual_seed(0)
    a = torch.randn(m, n, dtype=torch.float32)
    ref = a * 2.0 if epi == 1 else a
    o = torch.zeros(m * n, dtype=torch.float32, device="mps")
    nodot_kernel[(1,)](a.to("mps"), a.to("mps"), o, m, n, n, form, epi,
                       num_warps=w)
    torch.mps.synchronize()
    want = _want(ref, form)
    return _cmp(o.reshape(want.shape), want)


def cells(sw):
    for form, fname in FORMS:
        for (m, n, k) in SHAPES:
            for w in WARPS:
                e = max(1, m * n // (32 * w))
                sw.run(f"dot_{fname}_{m}x{n}x{k}_w{w}_e{e}",
                       lambda m=m, n=n, k=k, w=w, form=form:
                       _dot(m, n, k, w, form, 0, torch.float32))
        # An epilogue moves the store further from the dot; a tier that claims
        # the store only when it directly follows the dot would show up here.
        for (m, n, k) in SHAPES[:3]:
            for w in (1, 2):
                sw.run(f"epi_{fname}_{m}x{n}x{k}_w{w}",
                       lambda m=m, n=n, k=k, w=w, form=form:
                       _dot(m, n, k, w, form, 1, torch.float32))
        for (m, n, k) in ((16, 16, 16), (32, 32, 32)):
            for w in (1, 2):
                sw.run(f"f16_{fname}_{m}x{n}x{k}_w{w}",
                       lambda m=m, n=n, k=k, w=w, form=form:
                       _dot(m, n, k, w, form, 0, torch.float16))
        for (m, n, k, bk) in ((16, 16, 32, 16), (32, 32, 64, 32),
                              (32, 16, 32, 16)):
            for w in (1, 2):
                sw.run(f"loop_{fname}_{m}x{n}x{k}bk{bk}_w{w}",
                       lambda m=m, n=n, k=k, bk=bk, w=w, form=form:
                       _dot_loop(m, n, k, bk, w, form))
        for (m, n, _k) in SHAPES[:3]:
            for w in (1, 2):
                sw.run(f"nodot_{fname}_{m}x{n}_w{w}",
                       lambda m=m, n=n, w=w, form=form: _nodot(m, n, w, form, 0))


if __name__ == "__main__":
    sw = Sweep("s_dot_store_transposed")
    cells(sw)
    sw.finish()
