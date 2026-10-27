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

# A scalar atomic old value crosses a rank-2 scan's phase boundary.
SCAN_ATOMIC_OLD = CHECK + """
@triton.jit
def k(X, CNT, Y, R: tl.constexpr, C: tl.constexpr):
    r = tl.arange(0, R); c = tl.arange(0, C)
    old = tl.atomic_add(CNT, 1.0)
    x = tl.load(X + r[:, None] * C + c[None, :])
    tl.store(Y + r[:, None] * C + c[None, :], tl.cumsum(x, 1) + old)

def run(R, C, w):
    x = torch.randn(R, C, device="mps"); cnt = torch.zeros(1, device="mps")
    y = torch.zeros_like(x)
    k[(1,)](x, cnt, y, R, C, num_warps=w); torch.mps.synchronize()
    check("scan atomic old", y, x.cpu().cumsum(1))
"""

# The effecting value is shared by the source cone and the post-scan phase.
SCAN_EFFECT_SHARED = CHECK + """
@triton.jit
def k(X, CNT, BEFORE, Y, R: tl.constexpr, C: tl.constexpr):
    r = tl.arange(0, R); c = tl.arange(0, C)
    old = tl.atomic_add(CNT, 1.0)
    tl.store(BEFORE, old)
    x = tl.load(X + r[:, None] * C + c[None, :])
    tl.store(Y + r[:, None] * C + c[None, :], tl.cumsum(x + old, 1) + old)

def run(R, C, w):
    x = torch.randn(R, C, device="mps"); cnt = torch.zeros(1, device="mps")
    before = torch.zeros(1, device="mps"); y = torch.zeros_like(x)
    k[(1,)](x, cnt, before, y, R, C, num_warps=w); torch.mps.synchronize()
    check("scan shared effect", y, x.cpu().cumsum(1)); check("scan old store", before, torch.zeros(1))
"""

# One wide rank-1 scan feeds two rank-2 tiles with different shapes.
WIDE_TWO_CONSUMERS = CHECK + """
@triton.jit
def k(V, Y, Z, R: tl.constexpr, R2: tl.constexpr, C: tl.constexpr):
    r = tl.arange(0, R); r2 = tl.arange(0, R2); c = tl.arange(0, C)
    s = tl.cumsum(tl.load(V + c), 0)
    tl.store(Y + r[:, None] * C + c[None, :], s[None, :] + r[:, None].to(tl.float32))
    tl.store(Z + r2[:, None] * C + c[None, :], s[None, :] + 2.0 * r2[:, None].to(tl.float32))

def run(R, R2, C, w):
    v = torch.randn(C, device="mps"); y = torch.zeros(R, C, device="mps")
    z = torch.zeros(R2, C, device="mps")
    k[(1,)](v, y, z, R, R2, C, num_warps=w); torch.mps.synchronize()
    s = v.cpu().cumsum(0)
    check("wide consumer 1", y, s[None, :] + torch.arange(R).float()[:, None])
    check("wide consumer 2", z, s[None, :] + 2 * torch.arange(R2).float()[:, None])
"""

# The same wide scan broadcast is consumed through a real transpose.
WIDE_TRANS_CONSUMER = CHECK + """
@triton.jit
def k(V, Y, R: tl.constexpr, C: tl.constexpr):
    r = tl.arange(0, R); c = tl.arange(0, C)
    s = tl.cumsum(tl.load(V + c), 0)
    tile = s[None, :] + r[:, None].to(tl.float32)
    tl.store(Y + c[:, None] * R + r[None, :], tl.trans(tile))

def run(R, C, w):
    v = torch.randn(C, device="mps"); y = torch.zeros(C, R, device="mps")
    k[(1,)](v, y, R, C, num_warps=w); torch.mps.synchronize()
    want = (v.cpu().cumsum(0)[None, :] + torch.arange(R).float()[:, None]).t()
    check("wide trans consumer", y, want)
"""

# Range-only source and consumers exercise every range in a staged phase.
STAGED_RANGE_ONLY = CHECK + """
@triton.jit
def k(Y, R: tl.constexpr, C: tl.constexpr):
    r = tl.arange(0, R); c = tl.arange(0, C)
    x = (r[:, None] * C + c[None, :]).to(tl.float32)
    s = tl.cumsum(x, 1)
    tl.store(Y + r[:, None] * C + c[None, :], s + r[:, None] + c[None, :])

def run(R, C, w):
    y = torch.zeros(R, C, device="mps")
    k[(1,)](y, R, C, num_warps=w); torch.mps.synchronize()
    r = torch.arange(R).float(); c = torch.arange(C).float()
    x = r[:, None] * C + c[None, :]
    check("range only", y, x.cumsum(1) + r[:, None] + c[None, :])
"""

# A staged scan result has a transposed consumer.
STAGED_TRANS_CONSUMER = CHECK + """
@triton.jit
def k(X, Y, R: tl.constexpr, C: tl.constexpr):
    r = tl.arange(0, R); c = tl.arange(0, C)
    x = tl.load(X + r[:, None] * C + c[None, :])
    tl.store(Y + c[:, None] * R + r[None, :], tl.trans(tl.cumsum(x, 1)))

def run(R, C, w):
    x = torch.randn(R, C, device="mps"); y = torch.zeros(C, R, device="mps")
    k[(1,)](x, y, R, C, num_warps=w); torch.mps.synchronize()
    check("staged trans consumer", y, x.cpu().cumsum(1).t())
"""

# A same-shape tile with transposed addressing gives the scan a different-layout consumer.
STAGED_DIFFERENT_TILE = CHECK + """
@triton.jit
def k(X, Q, Y, R: tl.constexpr, C: tl.constexpr):
    r = tl.arange(0, R); c = tl.arange(0, C)
    x = tl.load(X + r[:, None] * C + c[None, :])
    q = tl.load(Q + c[None, :] * R + r[:, None])
    tl.store(Y + r[:, None] * C + c[None, :], tl.cumsum(x, 1) + q)

def run(R, C, w):
    x = torch.randn(R, C, device="mps"); q = torch.randn(C, R, device="mps")
    y = torch.zeros_like(x)
    k[(1,)](x, q, y, R, C, num_warps=w); torch.mps.synchronize()
    check("different tile", y, x.cpu().cumsum(1) + q.cpu().t())
"""

# An effecting value is shared before and after a staged rank-N reduction.
R3_EFFECT_SHARED = CHECK + """
@triton.jit
def k(X, CNT, BEFORE, Y, A: tl.constexpr, B: tl.constexpr, C: tl.constexpr):
    a = tl.arange(0, A); b = tl.arange(0, B); c = tl.arange(0, C)
    old = tl.atomic_add(CNT, 1.0)
    tl.store(BEFORE, old)
    x = tl.load(X + a[:, None, None] * (B * C) + b[None, :, None] * C + c[None, None, :])
    tl.store(Y + a[:, None] * B + b[None, :], tl.sum(x, 2) + old)

def run(A, B, C, w):
    x = torch.randn(A, B, C, device="mps"); cnt = torch.zeros(1, device="mps")
    before = torch.zeros(1, device="mps"); y = torch.zeros(A, B, device="mps")
    k[(1,)](x, cnt, before, y, A, B, C, num_warps=w); torch.mps.synchronize()
    check("r3 shared effect", y, x.cpu().sum(2)); check("r3 old store", before, torch.zeros(1))
"""

# Valid computed scan/reduce forms exercise the publication planner end-to-end.
SCAN_INPUT_PRODUCT = CHECK + """
@triton.jit
def combine(a, b):
    return a * b

@triton.jit
def k(X, Y, Z, R: tl.constexpr, C: tl.constexpr):
    r = tl.arange(0, R); c = tl.arange(0, C)
    x = tl.load(X + r[:, None] * C + c[None, :]) * 0.5 + 1.0
    tl.store(Y + r[:, None] * C + c[None, :], tl.cumsum(x, 1))
    tl.store(Z + c, tl.reduce(x, 0, combine))

def run(R, C, w):
    x = torch.randn(R, C, device="mps"); y = torch.zeros_like(x)
    z = torch.zeros(C, device="mps"); xc = x.cpu() * 0.5 + 1.0
    k[(1,)](x, y, z, R, C, num_warps=w); torch.mps.synchronize()
    check("scan input product", y, xc.cumsum(1)); check("column product", z, xc.prod(0))
"""

SCAN_INPUT_BITWISE = CHECK + """
@triton.jit
def combine(a, b):
    return a ^ b

@triton.jit
def k(X, Y, Z, R: tl.constexpr, C: tl.constexpr):
    r = tl.arange(0, R); c = tl.arange(0, C)
    x = tl.load(X + r[:, None] * C + c[None, :]) * 3 + 1
    tl.store(Y + r[:, None] * C + c[None, :], tl.cumsum(x, 1))
    tl.store(Z + c, tl.reduce(x, 0, combine))

def run(R, C, w):
    x = torch.randint(0, 100, (R, C), dtype=torch.int32, device="mps")
    y = torch.zeros_like(x); z = torch.zeros(C, dtype=torch.int32, device="mps")
    k[(1,)](x, y, z, R, C, num_warps=w); torch.mps.synchronize()
    xc = x.cpu() * 3 + 1; want = xc[0]
    for row in range(1, R):
        want = want ^ xc[row]
    check("scan input bitwise", y, xc.cumsum(1)); check("column xor", z, want)
"""

SCAN_INPUT_UNSIGNED = CHECK + """
@triton.jit
def k(X, Y, Z, R: tl.constexpr, C: tl.constexpr):
    r = tl.arange(0, R); c = tl.arange(0, C)
    x = tl.load(X + r[:, None] * C + c[None, :]).to(tl.uint32, bitcast=True) * 3 + 1
    tl.store(Y + r[:, None] * C + c[None, :], tl.cumsum(x, 1).to(tl.int32, bitcast=True))
    tl.store(Z + c, tl.min(x, 0).to(tl.int32, bitcast=True))

def run(R, C, w):
    x = torch.randint(0, 100, (R, C), dtype=torch.int32, device="mps")
    y = torch.zeros_like(x); z = torch.zeros(C, dtype=torch.int32, device="mps")
    k[(1,)](x, y, z, R, C, num_warps=w); torch.mps.synchronize()
    xc = x.cpu() * 3 + 1
    check("scan input unsigned", y, xc.cumsum(1)); check("column umin", z, xc.min(0).values)
"""

for w in (1, 4):
    P.PROBES[f"scan_atomic_old_8x64_w{w}"] = SCAN_ATOMIC_OLD + f"run(8, 64, {w})\n"
    P.PROBES[f"scan_effect_shared_8x64_w{w}"] = SCAN_EFFECT_SHARED + f"run(8, 64, {w})\n"
    P.PROBES[f"wide_two_consumers_4_8x256_w{w}"] = WIDE_TWO_CONSUMERS + f"run(4, 8, 256, {w})\n"
    P.PROBES[f"wide_trans_consumer_4x256_w{w}"] = WIDE_TRANS_CONSUMER + f"run(4, 256, {w})\n"
    P.PROBES[f"staged_range_only_8x64_w{w}"] = STAGED_RANGE_ONLY + f"run(8, 64, {w})\n"
    P.PROBES[f"staged_trans_consumer_8x64_w{w}"] = STAGED_TRANS_CONSUMER + f"run(8, 64, {w})\n"
    P.PROBES[f"staged_different_tile_8x64_w{w}"] = STAGED_DIFFERENT_TILE + f"run(8, 64, {w})\n"
    P.PROBES[f"r3_effect_shared_4x8x16_w{w}"] = R3_EFFECT_SHARED + f"run(4, 8, 16, {w})\n"
    P.PROBES[f"scan_input_product_8x64_w{w}"] = SCAN_INPUT_PRODUCT + f"run(8, 64, {w})\n"
    P.PROBES[f"scan_input_bitwise_8x64_w{w}"] = SCAN_INPUT_BITWISE + f"run(8, 64, {w})\n"
    P.PROBES[f"scan_input_unsigned_8x64_w{w}"] = SCAN_INPUT_UNSIGNED + f"run(8, 64, {w})\n"

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
