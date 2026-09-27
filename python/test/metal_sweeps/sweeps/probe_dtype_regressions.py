"""The session-3 fixes, re-asked in the dtypes they were not measured in.

Nearly every fix of 2026-09-27 was measured in f32: rank >= 2 sort/topk
(`601d2de0a2`, `36c1acb056`), the bitcast cone (`8dbb824479`), two independent
rank-2 tiles (`8b8df61b98`), keep-dims reduce stores (`f8ea41a105`), the
thread-major exchange (`a0ae06e5e2`), loaded dot accumulators (`c6eed1d12e`
..`339b385dce`), scan broadcasts and staged-scan neighbours (`07ef6f67e7`).
A dtype changes the storage type, the bitcast width (`tl.sort`/`tl.flip` on
f16 xor through i16), and which tier a dot takes -- each a place for a label
to go wrong again. Oracles are exact wherever the data permits (permutations,
integers, max/min); float sums get a tolerance scaled to the dtype.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

import probe_lib as P  # noqa: E402

P.SCRATCH = P.scratch_dir("dtype_regressions")

COMMON = """
DT = {"f16": torch.float16, "bf16": torch.bfloat16, "f32": torch.float32, "i32": torch.int32}
def perm(n, dt, shape):
    x = torch.randperm(n) - n // 2
    return x.to(DT[dt]).reshape(shape)
def check(name, got, want, exact=True, atol=0.0):
    got = got.cpu(); want = want.cpu()
    if exact:
        ok = torch.equal(got, want.to(got.dtype))
    else:
        ok = torch.allclose(got.float(), want.float(), atol=atol, rtol=atol)
    if not ok:
        bad = int((got.float() != want.float()).sum()) if exact else int((~torch.isclose(got.float(), want.float(), atol=atol, rtol=atol)).sum())
        print(f"WRONG {name}: {bad}/{want.numel()}")
"""

SORT2 = COMMON + """
@triton.jit
def k(X, Y, M: tl.constexpr, N: tl.constexpr, DESC: tl.constexpr):
    o = tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :]
    tl.store(Y + o, tl.sort(tl.load(X + o), descending=DESC))
def run(M, N, w, dt, desc):
    x = perm(M * N, dt, (M, N)).to("mps"); y = torch.zeros_like(x)
    k[(1,)](x, y, M, N, desc, num_warps=w); torch.mps.synchronize()
    check("sort2", y, x.cpu().sort(dim=1, descending=bool(desc)).values)
"""

TOPK2 = COMMON + """
@triton.jit
def k(X, Y, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    m = tl.arange(0, M)[:, None]
    x = tl.load(X + m * N + tl.arange(0, N)[None, :])
    tl.store(Y + m * K + tl.arange(0, K)[None, :], tl.topk(x, K))
def run(M, N, K, w, dt):
    x = perm(M * N, dt, (M, N)).to("mps"); y = torch.zeros(M, K, dtype=x.dtype, device="mps")
    k[(1,)](x, y, M, N, K, num_warps=w); torch.mps.synchronize()
    check("topk2", y, x.cpu().float().topk(K, dim=1).values)
"""

FLIP3 = COMMON + """
@triton.jit
def k(X, Y, A: tl.constexpr, B: tl.constexpr, C: tl.constexpr):
    o = (tl.arange(0, A)[:, None, None] * (B * C) + tl.arange(0, B)[None, :, None] * C
         + tl.arange(0, C)[None, None, :])
    tl.store(Y + o, tl.flip(tl.load(X + o), 0))
def run(A, B, C, w, dt):
    x = perm(A * B * C, dt, (A, B, C)).to("mps"); y = torch.zeros_like(x)
    k[(1,)](x, y, A, B, C, num_warps=w); torch.mps.synchronize()
    check("flip3", y, x.cpu().flip(0))
"""

TWO_TILES = COMMON + """
@triton.jit
def k(A, B, C, D, R1: tl.constexpr, C1: tl.constexpr, R2: tl.constexpr, C2: tl.constexpr):
    r1 = tl.arange(0, R1); c1 = tl.arange(0, C1)
    tl.store(B + r1[:, None] * C1 + c1[None, :], tl.load(A + r1[:, None] * C1 + c1[None, :]) * 2)
    r2 = tl.arange(0, R2); c2 = tl.arange(0, C2)
    tl.store(D + r2[:, None] * C2 + c2[None, :], tl.load(C + r2[:, None] * C2 + c2[None, :]) + 1)
def run(n, m, w, dt):
    a = perm(n[0] * n[1], dt, n).to("mps"); c = perm(m[0] * m[1], dt, m).to("mps")
    b = torch.zeros_like(a); d = torch.zeros_like(c)
    k[(1,)](a, b, c, d, n[0], n[1], m[0], m[1], num_warps=w); torch.mps.synchronize()
    check("first", b, a.cpu() * 2); check("second", d, c.cpu() + 1)
"""

KEEPDIMS = COMMON + """
@triton.jit
def k(X, Y, R: tl.constexpr, C: tl.constexpr, AX: tl.constexpr):
    r = tl.arange(0, R); c = tl.arange(0, C)
    x = tl.load(X + r[:, None] * C + c[None, :])
    if AX == 1:
        tl.store(Y + r[:, None] + tl.arange(0, 1)[None, :], tl.max(x, 1, keep_dims=True))
    else:
        tl.store(Y + tl.arange(0, 1)[:, None] + c[None, :], tl.max(x, 0, keep_dims=True))
def run(R, C, w, dt, ax):
    x = perm(R * C, dt, (R, C)).to("mps"); y = torch.zeros(R if ax == 1 else C, dtype=x.dtype, device="mps")
    k[(1,)](x, y, R, C, ax, num_warps=w); torch.mps.synchronize()
    check("keepdims", y, x.cpu().float().amax(ax))
"""

TRANS_RED = COMMON + """
@triton.jit
def k(X, W, Y, M: tl.constexpr, N: tl.constexpr, AX: tl.constexpr):
    m = tl.arange(0, M); n = tl.arange(0, N)
    x = tl.load(X + m[:, None] * N + n[None, :])
    w = tl.load(W + n[:, None] * M + m[None, :])
    s = tl.max(tl.trans(x) + w, AX)
    if AX == 0:
        tl.store(Y + m, s)
    else:
        tl.store(Y + n, s)
def run(M, N, w_, dt, ax):
    x = perm(M * N, dt, (M, N)).to("mps"); w = perm(M * N, dt, (N, M)).to("mps")
    y = torch.zeros(M if ax == 0 else N, dtype=x.dtype, device="mps")
    k[(1,)](x, w, y, M, N, ax, num_warps=w_); torch.mps.synchronize()
    check("transred", y, (x.cpu().float().T + w.cpu().float()).amax(ax))
"""

PERM3 = COMMON + """
@triton.jit
def k(X, Y, A: tl.constexpr, B: tl.constexpr, C: tl.constexpr,
      S0: tl.constexpr, S1: tl.constexpr, S2: tl.constexpr,
      P0: tl.constexpr, P1: tl.constexpr, P2: tl.constexpr):
    x = tl.load(X + tl.arange(0, A)[:, None, None] * (B * C) + tl.arange(0, B)[None, :, None] * C
                + tl.arange(0, C)[None, None, :])
    y = tl.permute(x, (P0, P1, P2))
    tl.store(Y + tl.arange(0, S0)[:, None, None] * (S1 * S2) + tl.arange(0, S1)[None, :, None] * S2
             + tl.arange(0, S2)[None, None, :], y)
def run(A, B, C, p, w, dt):
    x = perm(A * B * C, dt, (A, B, C)).to("mps"); y = torch.zeros(A * B * C, dtype=x.dtype, device="mps")
    s = tuple((A, B, C)[i] for i in p)
    k[(1,)](x, y, A, B, C, *s, *p, num_warps=w); torch.mps.synchronize()
    check("perm3", y.reshape(s), x.cpu().permute(*p))
"""

DOT_ACC = COMMON + """
@triton.jit
def k(A, B, C, O, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, CM: tl.constexpr):
    m = tl.arange(0, M); n = tl.arange(0, N); kk = tl.arange(0, K)
    a = tl.load(A + m[:, None] * K + kk[None, :])
    b = tl.load(B + kk[:, None] * N + n[None, :])
    if CM:
        c = tl.load(C + n[None, :] * M + m[:, None])
    else:
        c = tl.load(C + m[:, None] * N + n[None, :])
    tl.store(O + m[:, None] * N + n[None, :], tl.dot(a, b, c))
def run(M, N, K, w, dt, cm):
    if dt == "i8":
        a = torch.randint(-8, 8, (M, K), dtype=torch.int8); b = torch.randint(-8, 8, (K, N), dtype=torch.int8)
        c = torch.randint(-100, 100, (M, N), dtype=torch.int32)
        want = a.int() @ b.int() + c
    else:
        a = torch.randn(M, K).to(DT[dt]); b = torch.randn(K, N).to(DT[dt]); c = torch.randn(M, N)
        want = a.float() @ b.float() + c
    cbuf = c.T.contiguous() if cm else c
    o = torch.zeros(M, N, dtype=c.dtype, device="mps")
    k[(1,)](a.to("mps"), b.to("mps"), cbuf.to("mps"), o, M, N, K, cm, num_warps=w); torch.mps.synchronize()
    check("dotacc", o, want, exact=(dt == "i8"), atol=0.05 if dt != "i8" else 0)
"""

SCAN_BCAST = COMMON + """
@triton.jit
def k(V, Y, R: tl.constexpr, C: tl.constexpr):
    r = tl.arange(0, R); c = tl.arange(0, C)
    s = tl.cumsum(tl.load(V + c), 0)
    tl.store(Y + r[:, None] * C + c[None, :], s[None, :] + r[:, None].to(s.dtype))
def run(R, C, w, dt):
    v = perm(C, dt, (C,)).to("mps"); y = torch.zeros(R, C, dtype=v.dtype, device="mps")
    k[(1,)](v, y, R, C, num_warps=w); torch.mps.synchronize()
    want = v.cpu().float().cumsum(0)[None, :] + torch.arange(R).float()[:, None]
    check("scanbcast", y, want, exact=(dt == "i32"), atol=0.05)
"""

for dt in ("f16", "bf16", "i32"):
    for (M, N, w) in ((16, 16, 1), (8, 128, 4), (4, 32, 1), (2, 512, 1)):
        for desc in (0, 1):
            if dt == "bf16" and M * N > 256:
                continue  # bf16 cannot hold the permutation exactly
            P.PROBES[f"sort2_{dt}_{M}x{N}_w{w}_d{desc}"] = SORT2 + f"run({M}, {N}, {w}, '{dt}', {desc})\n"
    for (M, N, K, w) in ((2, 64, 2, 1), (1, 32, 2, 1), (4, 32, 4, 1), (16, 16, 4, 1), (8, 128, 16, 4)):
        if dt == "bf16" and M * N > 256:
            continue
        P.PROBES[f"topk2_{dt}_{M}x{N}k{K}_w{w}"] = TOPK2 + f"run({M}, {N}, {K}, {w}, '{dt}')\n"
    for (A, B, C, w) in ((2, 16, 32, 1), (2, 16, 32, 4), (2, 4, 8, 1)):
        if dt == "bf16" and A * B * C > 256:
            continue
        P.PROBES[f"flip3_{dt}_{A}x{B}x{C}_w{w}"] = FLIP3 + f"run({A}, {B}, {C}, {w}, '{dt}')\n"
    for (n, m, w) in (((4, 2), (4, 4), 1), ((16, 1), (16, 16), 2), ((64, 8), (16, 16), 2), ((1, 64), (8, 128), 1)):
        if dt == "bf16" and max(n[0] * n[1], m[0] * m[1]) > 256:
            continue
        P.PROBES[f"twotiles_{dt}_{n[0]}x{n[1]}_{m[0]}x{m[1]}_w{w}"] = TWO_TILES + f"run({n}, {m}, {w}, '{dt}')\n"
    for (R, C, w) in ((2, 16, 1), (4, 32, 1), (8, 32, 2), (16, 16, 4)):
        for ax in (0, 1):
            P.PROBES[f"keepdims_{dt}_{R}x{C}_w{w}_ax{ax}"] = KEEPDIMS + f"run({R}, {C}, {w}, '{dt}', {ax})\n"
    for (M, N, w) in ((16, 32, 1), (32, 32, 1), (8, 64, 1), (16, 16, 2)):
        if dt == "bf16" and M * N > 256:
            continue
        for ax in (0, 1):
            P.PROBES[f"transred_{dt}_{M}x{N}_w{w}_ax{ax}"] = TRANS_RED + f"run({M}, {N}, {w}, '{dt}', {ax})\n"
    for (A, B, C, p, w) in ((4, 8, 8, (0, 2, 1), 1), (2, 8, 16, (1, 0, 2), 1), (2, 16, 16, (2, 1, 0), 2), (2, 4, 16, (2, 1, 0), 1)):
        if dt == "bf16" and A * B * C > 256:
            continue
        P.PROBES[f"perm3_{dt}_{A}x{B}x{C}_p{''.join(map(str, p))}_w{w}"] = PERM3 + f"run({A}, {B}, {C}, {p}, {w}, '{dt}')\n"
    for (R, C, w) in ((4, 32, 1), (16, 64, 4), (2, 16, 1)):
        P.PROBES[f"scanbcast_{dt}_{R}x{C}_w{w}"] = SCAN_BCAST + f"run({R}, {C}, {w}, '{dt}')\n"
for dt in ("f16", "bf16", "i8"):
    for (M, N, K, w) in ((16, 16, 16, 1), (32, 32, 32, 4), (8, 8, 8, 2), (16, 16, 16, 4)):
        for cm in (0, 1):
            P.PROBES[f"dotacc_{dt}_{M}x{N}x{K}_w{w}_cm{cm}"] = DOT_ACC + f"run({M}, {N}, {K}, {w}, '{dt}', {cm})\n"

if __name__ == "__main__":
    sys.exit(P.main())
