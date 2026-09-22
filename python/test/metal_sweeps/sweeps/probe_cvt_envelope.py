"""P2.1: where does a divergent blocked<->blocked `ttg.convert_layout` still lose?

The plan's first P2.1 item is the rank / order / `sizePerThread > 1` envelope of
`ttg.convert_layout`. Two rounds already measured parts of it and both came back
narrower than the plan's text: `.omc/fuzz/layout_breadth.py` is 51 OK / 3 NAMED
(the three are rank-3 `tl.permute`), and the thirteen `tl.trans` forms in
`probe_diag_layout_exchange.py` all compiled and were exact.

So this round probes the two things NEITHER of those did.

1. A divergent cvt whose SOURCE IS NOT A PRODUCER CONE. `normalizeDivergentCvt`
   collapses a divergent pair by re-encoding the source cone to the destination
   layout, but only for a self-contained cone (loads, elementwise, addptr /
   splat / make_range / expand_dims / broadcast, mask cmpi). Put a reduce, a
   scan, a dot or a loop in that cone and the normalization has to decline --
   and the two ways a user reaches a divergent pair at all are named in the
   backend's own comment: a MISALIGNED pointer (an MPS tensor sliced to
   `base[7:]` stays at `sizePerThread = 1` while an aligned one vectorizes) and
   a transposed store address.

2. The other five rank-3 permutations. `layout_breadth` probes `(0, 2, 1)`
   only, and `metal-rank3-permute-attempt.md` located that cell's defect in the
   rank-3 index decomposition -- which is shared by every permutation, so
   whether the refusal covers all six is a question about the guard, not about
   the fix.

Every group carries a CONTROL arm that differs in exactly one thing: the same
kernel with a pure cone, or with both pointers aligned, or storing in the load's
own order. Without it a NAMED verdict cannot be read -- the refusal might be
about the shape rather than about the divergence.

Usage:
    pixi run --frozen python python/test/metal_sweeps/sweeps/probe_cvt_envelope.py
    REPEAT=30 pixi run --frozen python .../probe_cvt_envelope.py <tag>
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

import probe_lib as P  # noqa: E402

P.SCRATCH = P.scratch_dir("cvt_envelope")

CHK = """
def chk(got, want, atol=2e-3):
    got = got.float().cpu()
    want = want.float().cpu()
    if torch.allclose(got, want, atol=atol, rtol=2e-3):
        print("EXACT")
        return
    d = (got - want).abs()
    i = int(d.argmax())
    print(f"WRONG maxdiff={d.max().item():.6g} at {i}: got "
          f"{got.flatten()[i].item():.6g} want {want.flatten()[i].item():.6g}")

def buf(n, off):
    # `off > 0` gives a base that is NOT 16-byte aligned, which is what makes
    # Triton mint a second rank-1 layout: tt.divisibility drops to 4 and the
    # load/store stops vectorizing, so the value has to be bridged by a cvt.
    t = torch.zeros(n + 8, dtype=torch.float32, device="mps")
    return t[off:off + n]
"""

# --------------------------------------------------------------------------
# Group A -- rank-1 alignment divergence. The axis is what sits in the cvt's
# SOURCE CONE: nothing (control), a reduce, a scan, a loop, an atomic's old
# value. BLOCK=256 at one warp is E=8, so a mislabelled element is visible.
# --------------------------------------------------------------------------

A_SRC = """
@triton.jit
def k(X, O, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    v = tl.load(X + i)
{body}

N = 256
x = torch.randn(N, dtype=torch.float32, device="mps")
o = buf(N, {ooff})
xa = x if {xoff} == 0 else None
if xa is None:
    src = buf(N, {xoff})
    src.copy_(x)
    xa = src
k[(1,)](xa, o, BLOCK=N, num_warps=1)
torch.mps.synchronize()
xc = x.cpu()
chk(o, {ref})
"""

A_FORMS = {
    # tag: (kernel body, reference over xc)
    "plain": ("    tl.store(O + i, v * 2.0)", "xc * 2.0"),
    "reduce": ("""    s = tl.sum(v, axis=0)
    tl.store(O + i, v - s)""", "xc - xc.sum()"),
    "scan": ("""    c = tl.cumsum(v, axis=0)
    tl.store(O + i, c)""", "xc.cumsum(0)"),
    "loop": ("""    acc = v
    for _ in range(3):
        acc = acc * 1.5
    tl.store(O + i, acc)""", "xc * (1.5 ** 3)"),
    "reduce_max": ("""    m = tl.max(v, axis=0)
    tl.store(O + i, v / m)""", "xc / xc.max()"),
}

for _form, (_body, _ref) in A_FORMS.items():
    # ooff=7: only the STORE is misaligned. xoff=7: only the LOAD is.
    # Both-aligned and both-misaligned are the controls that say whether the
    # verdict is about the divergence or about the alignment itself.
    for _tag, _xoff, _ooff in (("dststore", 0, 7), ("srcload", 7, 0),
                               ("aligned", 0, 0), ("both", 7, 7)):
        P.PROBES[f"a_{_form}_{_tag}"] = CHK + A_SRC.format(
            body=_body, xoff=_xoff, ooff=_ooff, ref=_ref)

# The same divergence with an ATOMIC as the consumer instead of a store: the
# rank-1 relabel passthrough in the cvt validator is written for `tt.store`
# users only, so an atomic is the adjacent negative of that condition.
P.PROBES["a_reduce_atomic"] = CHK + """
@triton.jit
def k(X, O, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    v = tl.load(X + i)
    s = tl.sum(v, axis=0)
    tl.atomic_add(O + i, v - s)

N = 256
x = torch.randn(N, dtype=torch.float32, device="mps")
o = buf(N, 7)
k[(1,)](x, o, BLOCK=N, num_warps=1)
torch.mps.synchronize()
xc = x.cpu()
chk(o, xc - xc.sum())
"""

P.PROBES["a_plain_atomic"] = CHK + """
@triton.jit
def k(X, O, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    tl.atomic_add(O + i, tl.load(X + i) * 2.0)

N = 256
x = torch.randn(N, dtype=torch.float32, device="mps")
o = buf(N, 7)
k[(1,)](x, o, BLOCK=N, num_warps=1)
torch.mps.synchronize()
chk(o, x.cpu() * 2.0)
"""

# --------------------------------------------------------------------------
# Group B -- rank-2 order divergence: the STORE address is column-major while
# the load is row-major, which is a transpose without a `tl.trans` op. The axis
# is again the cone: plain, reduce, scan, dot.
# --------------------------------------------------------------------------

B_SRC = """
@triton.jit
def k(X, O, M: tl.constexpr, N: tl.constexpr):
    r = tl.arange(0, M)[:, None]
    c = tl.arange(0, N)[None, :]
    v = tl.load(X + r * N + c)
{body}

M, N = {m}, {n}
x = torch.randn(M, N, dtype=torch.float32, device="mps")
o = torch.zeros(M * N, dtype=torch.float32, device="mps")
k[(1,)](x, o, M=M, N=N, num_warps={nw})
torch.mps.synchronize()
xc = x.cpu()
chk(o.reshape({outshape}), {ref})
"""

B_FORMS = {
    "plain": ("    tl.store(O + c * M + r, v * 2.0)",
              "(xc * 2.0).t().contiguous()", "(N, M)"),
    "rowsum": ("""    s = tl.sum(v, axis=1)[:, None]
    tl.store(O + c * M + r, v - s)""",
               "(xc - xc.sum(1, keepdim=True)).t().contiguous()", "(N, M)"),
    "colsum": ("""    s = tl.sum(v, axis=0)[None, :]
    tl.store(O + c * M + r, v - s)""",
               "(xc - xc.sum(0, keepdim=True)).t().contiguous()", "(N, M)"),
    "scan": ("""    s = tl.cumsum(v, axis=1)
    tl.store(O + c * M + r, s)""",
             "xc.cumsum(1).t().contiguous()", "(N, M)"),
    # The control: same tile, same cone, stored in the LOAD's order. If this
    # one is also refused the verdict is about the reduce, not the order.
    "rowsum_rowmajor": ("""    s = tl.sum(v, axis=1)[:, None]
    tl.store(O + r * N + c, v - s)""",
                        "xc - xc.sum(1, keepdim=True)", "(M, N)"),
}

for _form, (_body, _ref, _shape) in B_FORMS.items():
    for _m, _n, _nw in ((16, 16, 1), (32, 16, 1), (16, 32, 2)):
        P.PROBES[f"b_{_form}_{_m}x{_n}w{_nw}"] = CHK + B_SRC.format(
            body=_body, m=_m, n=_n, nw=_nw, ref=_ref,
            outshape=_shape.replace("M", str(_m)).replace("N", str(_n)))

P.PROBES["b_dot_colmajor"] = CHK + """
@triton.jit
def k(A, B, O, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    m = tl.arange(0, M)[:, None]
    kk = tl.arange(0, K)[None, :]
    n = tl.arange(0, N)[None, :]
    a = tl.load(A + m * K + kk)
    b = tl.load(B + tl.arange(0, K)[:, None] * N + n)
    acc = tl.dot(a, b)
    tl.store(O + n * M + m, acc)

M, N, K = 16, 16, 16
a = torch.randn(M, K, dtype=torch.float32, device="mps")
b = torch.randn(K, N, dtype=torch.float32, device="mps")
o = torch.zeros(M * N, dtype=torch.float32, device="mps")
k[(1,)](a, b, o, M=M, N=N, K=K, num_warps=1)
torch.mps.synchronize()
chk(o.reshape(N, M), (a.cpu() @ b.cpu()).t().contiguous())
"""

# --------------------------------------------------------------------------
# Group C -- the other five rank-3 permutations, plus rank 4. `layout_breadth`
# probes (0, 2, 1) only.
# --------------------------------------------------------------------------

C_SRC = """
@triton.jit
def k(X, O, A: tl.constexpr, B: tl.constexpr, C: tl.constexpr):
    a = tl.arange(0, A)[:, None, None]
    b = tl.arange(0, B)[None, :, None]
    c = tl.arange(0, C)[None, None, :]
    v = tl.load(X + a * (B * C) + b * C + c)
    p = tl.permute(v, {perm})
    oa = tl.arange(0, {d0})[:, None, None]
    ob = tl.arange(0, {d1})[None, :, None]
    oc = tl.arange(0, {d2})[None, None, :]
    tl.store(O + oa * ({d1} * {d2}) + ob * {d2} + oc, p)

A, B, C = 2, 4, 8
x = torch.randn(A, B, C, dtype=torch.float32, device="mps")
o = torch.zeros(A * B * C, dtype=torch.float32, device="mps")
k[(1,)](x, o, A=A, B=B, C=C, num_warps=1)
torch.mps.synchronize()
want = x.cpu().permute{perm}.contiguous()
chk(o.reshape(want.shape), want)
"""

_DIMS = (2, 4, 8)
for _perm in ((0, 2, 1), (1, 0, 2), (2, 1, 0), (2, 0, 1), (1, 2, 0)):
    P.PROBES["c_permute3_" + "".join(map(str, _perm))] = CHK + C_SRC.format(
        perm=_perm, d0=_DIMS[_perm[0]], d1=_DIMS[_perm[1]],
        d2=_DIMS[_perm[2]])

P.PROBES["c_permute4"] = CHK + """
@triton.jit
def k(X, O, A: tl.constexpr, B: tl.constexpr, C: tl.constexpr,
      D: tl.constexpr):
    a = tl.arange(0, A)[:, None, None, None]
    b = tl.arange(0, B)[None, :, None, None]
    c = tl.arange(0, C)[None, None, :, None]
    d = tl.arange(0, D)[None, None, None, :]
    v = tl.load(X + a * (B * C * D) + b * (C * D) + c * D + d)
    p = tl.permute(v, (0, 1, 3, 2))
    oc = tl.arange(0, D)[None, None, :, None]
    od = tl.arange(0, C)[None, None, None, :]
    tl.store(O + a * (B * C * D) + b * (C * D) + oc * C + od, p)

A, B, C, D = 2, 2, 4, 8
x = torch.randn(A, B, C, D, dtype=torch.float32, device="mps")
o = torch.zeros(A * B * C * D, dtype=torch.float32, device="mps")
k[(1,)](x, o, A=A, B=B, C=C, D=D, num_warps=1)
torch.mps.synchronize()
want = x.cpu().permute(0, 1, 3, 2).contiguous()
chk(o.reshape(want.shape), want)
"""

# The control for group C: the same rank-3 tile, no permutation. If this is
# refused too then the refusal is about rank 3, not about the permute.
P.PROBES["c_rank3_identity_store"] = CHK + """
@triton.jit
def k(X, O, A: tl.constexpr, B: tl.constexpr, C: tl.constexpr):
    a = tl.arange(0, A)[:, None, None]
    b = tl.arange(0, B)[None, :, None]
    c = tl.arange(0, C)[None, None, :]
    v = tl.load(X + a * (B * C) + b * C + c)
    tl.store(O + a * (B * C) + b * C + c, v * 2.0)

A, B, C = 2, 4, 8
x = torch.randn(A, B, C, dtype=torch.float32, device="mps")
o = torch.zeros(A * B * C, dtype=torch.float32, device="mps")
k[(1,)](x, o, A=A, B=B, C=C, num_warps=1)
torch.mps.synchronize()
chk(o.reshape(A, B, C), x.cpu() * 2.0)
"""

# A rank-3 permute whose consumer is a REDUCE rather than a store: the store is
# where the located defect lives (coordinates from one tile, strides from the
# other), so a consumer with no rank-3 address separates the two halves.
P.PROBES["c_permute3_reduce"] = CHK + """
@triton.jit
def k(X, O, A: tl.constexpr, B: tl.constexpr, C: tl.constexpr):
    a = tl.arange(0, A)[:, None, None]
    b = tl.arange(0, B)[None, :, None]
    c = tl.arange(0, C)[None, None, :]
    v = tl.load(X + a * (B * C) + b * C + c)
    p = tl.permute(v, (0, 2, 1))
    s = tl.sum(p, axis=2)
    oa = tl.arange(0, A)[:, None]
    oc = tl.arange(0, C)[None, :]
    tl.store(O + oa * C + oc, s)

A, B, C = 2, 4, 8
x = torch.randn(A, B, C, dtype=torch.float32, device="mps")
o = torch.zeros(A * C, dtype=torch.float32, device="mps")
k[(1,)](x, o, A=A, B=B, C=C, num_warps=1)
torch.mps.synchronize()
chk(o.reshape(A, C), x.cpu().permute(0, 2, 1).sum(2))
"""

# --------------------------------------------------------------------------
# Group D -- two rank-2 tiles of the SAME element count live in one kernel.
# This is the tie that `metal-rank3-permute-attempt.md` names one rank up: a
# module-wide "largest tile" answer is a coin flip, so each range has to be
# decomposed against the tile its own cone reaches.
# --------------------------------------------------------------------------

D_SRC = """
@triton.jit
def k(X, Y, O, M: tl.constexpr, N: tl.constexpr):
    r = tl.arange(0, M)[:, None]
    c = tl.arange(0, N)[None, :]
    rr = tl.arange(0, N)[:, None]
    cc = tl.arange(0, M)[None, :]
    v = tl.load(X + r * N + c)
    w = tl.load(Y + rr * M + cc)
{body}

M, N = {m}, {n}
x = torch.randn(M, N, dtype=torch.float32, device="mps")
y = torch.randn(N, M, dtype=torch.float32, device="mps")
o = torch.zeros({outn}, dtype=torch.float32, device="mps")
k[(1,)](x, y, o, M=M, N=N, num_warps={nw})
torch.mps.synchronize()
xc, yc = x.cpu(), y.cpu()
chk(o.reshape({outshape}), {ref})
"""

D_FORMS = {
    "trans_add": ("    tl.store(O + rr * M + cc, tl.trans(v) + w)",
                  "xc.t().contiguous() + yc", "(N, M)", "M * N"),
    "trans_add_rowsum": ("""    s = tl.sum(v, axis=1)[:, None]
    tl.store(O + rr * M + cc, tl.trans(v - s) + w)""",
                         "(xc - xc.sum(1, keepdim=True)).t().contiguous() + yc",
                         "(N, M)", "M * N"),
    "trans_add_reduce": ("""    t = tl.trans(v) + w
    tl.store(O + tl.arange(0, M), tl.sum(t, axis=0))""",
                         "(xc.t().contiguous() + yc).sum(0)", "(M,)", "M"),
}

for _form, (_body, _ref, _shape, _outn) in D_FORMS.items():
    for _m, _n, _nw in ((16, 8, 1), (32, 8, 1), (16, 32, 2)):
        P.PROBES[f"d_{_form}_{_m}x{_n}w{_nw}"] = CHK + D_SRC.format(
            body=_body, m=_m, n=_n, nw=_nw, ref=_ref,
            outshape=_shape.replace("M", str(_m)).replace("N", str(_n)),
            outn=_outn.replace("M", str(_m)).replace("N", str(_n)))

if __name__ == "__main__":
    raise SystemExit(P.main())
