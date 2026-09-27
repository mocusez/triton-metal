"""Sixth slice: the staged scan / rank-N reduce diagnostics nobody has reached.

What is left unprobed after five slices is the structural refusals of the
staged phase machinery:

  mixed staged scans require top-level exact-layout phases
  nested scan cannot leave tensor work outside its loop / is missing its local phases
  staged scan cannot safely partition its source and consumers
  staged scan range has no exact consumer layout
  staged rank-N reduce cannot split its phase / has no enclosing block
  retained a live tensor from a clone-only rank-1 aggregate cone

Each probe composes scans and reductions the way a kernel would -- two scans of
different shapes, a scan whose source is used again after it, a scan inside a
loop with tensor work around it, a rank-3 reduce next to a rank-2 scan, an
aggregate whose cone is reused -- and carries its own torch oracle, because
the last five slices found their real yield in the wrong answers next to the
diagnostics, not in the diagnostics.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

import probe_lib as P  # noqa: E402

P.SCRATCH = P.scratch_dir("diag_staged")

CHECK = """
def check(name, got, want, atol=1e-3):
    got = got.cpu().float(); want = want.float()
    if not torch.allclose(got, want, atol=atol, rtol=1e-3):
        n = int((~torch.isclose(got, want, atol=atol, rtol=1e-3)).sum())
        print(f"WRONG {name}: {n}/{want.numel()} differ")
"""

# Two scans of different shapes in one function: rank-1 cumsum and rank-2 row cumsum.
TWO_SCANS = CHECK + """
@triton.jit
def k(X, V, Y, Z, R: tl.constexpr, C: tl.constexpr, N: tl.constexpr):
    r = tl.arange(0, R); c = tl.arange(0, C); n = tl.arange(0, N)
    x = tl.load(X + r[:, None] * C + c[None, :])
    v = tl.load(V + n)
    tl.store(Y + r[:, None] * C + c[None, :], tl.cumsum(x, AX))
    tl.store(Z + n, tl.cumsum(v, 0))

def run(R, C, N, w):
    x = torch.randn(R, C, device="mps"); v = torch.randn(N, device="mps")
    y = torch.zeros_like(x); z = torch.zeros_like(v)
    k[(1,)](x, v, y, z, R, C, N, num_warps=w); torch.mps.synchronize()
    check("2d", y, x.cpu().cumsum(AX)); check("1d", z, v.cpu().cumsum(0))
"""

# Scan whose source is used again after the scan (x - cumsum(x)).
REUSE = CHECK + """
@triton.jit
def k(X, Y, R: tl.constexpr, C: tl.constexpr):
    r = tl.arange(0, R); c = tl.arange(0, C)
    x = tl.load(X + r[:, None] * C + c[None, :])
    s = tl.cumsum(x, AX)
    tl.store(Y + r[:, None] * C + c[None, :], x * 2.0 - s + tl.sum(x, AX, keep_dims=True))

def run(R, C, w):
    x = torch.randn(R, C, device="mps"); y = torch.zeros_like(x)
    k[(1,)](x, y, R, C, num_warps=w); torch.mps.synchronize()
    xc = x.cpu()
    check("reuse", y, xc * 2 - xc.cumsum(AX) + xc.sum(AX, keepdim=True))
"""

# Scan inside a runtime loop with tensor work outside it.
LOOP = CHECK + """
@triton.jit
def k(X, Y, T, R: tl.constexpr, C: tl.constexpr):
    r = tl.arange(0, R); c = tl.arange(0, C)
    base = tl.load(X + r[:, None] * C + c[None, :])
    acc = base * 0.0
    for t in range(T):
        x = tl.load(X + t * R * C + r[:, None] * C + c[None, :])
        acc += tl.cumsum(x, AX)
    tl.store(Y + r[:, None] * C + c[None, :], acc + base)

def run(R, C, w, T=3):
    x = torch.randn(T, R, C, device="mps"); y = torch.zeros(R, C, device="mps")
    k[(1,)](x, y, T, R, C, num_warps=w); torch.mps.synchronize()
    xc = x.cpu()
    check("loop", y, xc.cumsum(1 + AX).sum(0) + xc[0])
"""

# Rank-3 reduce and a rank-2 scan in one function.
R3_SCAN = CHECK + """
@triton.jit
def k(X, W, Y, Z, A: tl.constexpr, B: tl.constexpr, C: tl.constexpr):
    a = tl.arange(0, A); b = tl.arange(0, B); c = tl.arange(0, C)
    x = tl.load(X + a[:, None, None] * (B * C) + b[None, :, None] * C + c[None, None, :])
    w = tl.load(W + b[:, None] * C + c[None, :])
    tl.store(Y + b[:, None] * C + c[None, :], tl.sum(x, 0) + tl.cumsum(w, 1))
    tl.store(Z + b[:, None] * C + c[None, :], tl.cumsum(w, 0))

def run(A, B, C, w):
    x = torch.randn(A, B, C, device="mps"); ww = torch.randn(B, C, device="mps")
    y = torch.zeros(B, C, device="mps"); z = torch.zeros(B, C, device="mps")
    k[(1,)](x, ww, y, z, A, B, C, num_warps=w); torch.mps.synchronize()
    check("r3+scan", y, x.cpu().sum(0) + ww.cpu().cumsum(1)); check("scan0", z, ww.cpu().cumsum(0))
"""

# Atomic before a rank-3 reduce, its old value used after.
R3_ATOMIC = CHECK + """
@triton.jit
def k(X, CNT, Y, A: tl.constexpr, B: tl.constexpr, C: tl.constexpr):
    a = tl.arange(0, A); b = tl.arange(0, B); c = tl.arange(0, C)
    old = tl.atomic_add(CNT, 1.0)
    x = tl.load(X + a[:, None, None] * (B * C) + b[None, :, None] * C + c[None, None, :])
    s = tl.sum(x, 2)
    tl.store(Y + a[:, None] * B + b[None, :], s + old)

def run(A, B, C, w):
    x = torch.randn(A, B, C, device="mps"); cnt = torch.zeros(1, device="mps")
    y = torch.zeros(A, B, device="mps")
    k[(1,)](x, cnt, y, A, B, C, num_warps=w); torch.mps.synchronize()
    check("r3atomic", y, x.cpu().sum(2))
"""

# Aggregate whose source cone is reused elementwise and in a second aggregate.
AGG_REUSE = CHECK + """
@triton.jit
def k(X, Y, N: tl.constexpr):
    n = tl.arange(0, N)
    x = tl.load(X + n)
    e = tl.exp(x - tl.max(x, 0))
    s = tl.sum(e, 0)
    tl.store(Y + n, e / s + tl.cumsum(e, 0) * 0.0 + tl.min(e, 0))

def run(N, w):
    x = torch.randn(N, device="mps"); y = torch.zeros_like(x)
    k[(1,)](x, y, N, num_warps=w); torch.mps.synchronize()
    xc = x.cpu(); e = torch.exp(xc - xc.max())
    check("aggreuse", y, e / e.sum() + e.min())
"""

# Scan result broadcast into a wider tile / reshaped.
SCAN_CONSUMER = CHECK + """
@triton.jit
def k(V, Y, R: tl.constexpr, C: tl.constexpr):
    r = tl.arange(0, R); c = tl.arange(0, C)
    v = tl.load(V + c)
    s = tl.cumsum(v, 0)
    tl.store(Y + r[:, None] * C + c[None, :], s[None, :] + r[:, None].to(tl.float32))

def run(R, C, w):
    v = torch.randn(C, device="mps"); y = torch.zeros(R, C, device="mps")
    k[(1,)](v, y, R, C, num_warps=w); torch.mps.synchronize()
    check("scanbcast", y, v.cpu().cumsum(0)[None, :] + torch.arange(R).float()[:, None])
"""

for ax in (0, 1):
    for (R, C, N) in ((4, 32, 64), (16, 16, 256), (8, 64, 1024), (32, 8, 128)):
        for w in (1, 4):
            P.PROBES[f"twoscans_ax{ax}_{R}x{C}_n{N}_w{w}"] = TWO_SCANS.replace("AX", str(ax)) + f"run({R}, {C}, {N}, {w})\n"
    for (R, C) in ((4, 32), (16, 16), (8, 64), (32, 32)):
        for w in (1, 4):
            P.PROBES[f"reuse_ax{ax}_{R}x{C}_w{w}"] = REUSE.replace("AX", str(ax)) + f"run({R}, {C}, {w})\n"
            P.PROBES[f"loop_ax{ax}_{R}x{C}_w{w}"] = LOOP.replace("AX", str(ax)) + f"run({R}, {C}, {w})\n"
for (A, B, C) in ((2, 4, 8), (4, 8, 16), (2, 16, 16)):
    for w in (1, 4):
        P.PROBES[f"r3scan_{A}x{B}x{C}_w{w}"] = R3_SCAN + f"run({A}, {B}, {C}, {w})\n"
        P.PROBES[f"r3atomic_{A}x{B}x{C}_w{w}"] = R3_ATOMIC + f"run({A}, {B}, {C}, {w})\n"
for N in (32, 256, 1024):
    for w in (1, 4):
        P.PROBES[f"aggreuse_n{N}_w{w}"] = AGG_REUSE + f"run({N}, {w})\n"
for (R, C) in ((4, 32), (16, 64), (8, 256)):
    for w in (1, 4):
        P.PROBES[f"scanbcast_{R}x{C}_w{w}"] = SCAN_CONSUMER + f"run({R}, {C}, {w})\n"

if __name__ == "__main__":
    sys.exit(P.main())
