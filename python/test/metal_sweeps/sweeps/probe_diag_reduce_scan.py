"""Reachability of the unpinned reduce/scan diagnostics, and the scan envelope.

Third slice of the named-diagnostic audit. The first two measured `atomic`
(8 diagnostics, 0 reachable) and `layout`/`exchange` (12 diagnostics, 1
reachable, and three families of silent wrong answers found on the way).

The 17 here are the biggest remaining bucket, and they sit on exactly the axes
the plan still lists as open for P2.2: nested multi-element scans, rank >= 2
tuple scans in wider types and combines, mixing exact and multi-phase scans in
one function, tensor iter_args, and a snapshot taken outside a loop. So every
probe carries an ORACLE: a refusal is one answer, a wrong answer is the other,
and the round before this one found the second where it was only looking for
the first.

    pixi run --frozen python python/test/metal_sweeps/sweeps/probe_diag_reduce_scan.py
    REPEAT=30 pixi run --frozen python .../probe_diag_reduce_scan.py <tag>
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

import probe_lib as P  # noqa: E402

P.SCRATCH = P.scratch_dir("diag_reduce_scan")

# --------------------------------------------------------------------------
# Reduce envelope: rank, dtype, combine, axis.
# --------------------------------------------------------------------------

P.PROBES["reduce_rank4"] = """
@triton.jit
def k(X, O, A: tl.constexpr, B: tl.constexpr, C: tl.constexpr, D: tl.constexpr):
    f = tl.arange(0, A * B * C * D)
    v = tl.reshape(tl.load(X + f), (A, B, C, D))
    s = tl.sum(v, 3)
    tl.store(O + tl.arange(0, A * B * C), tl.reshape(s, (A * B * C,)))

A, B, C, D = 2, 2, 4, 4
x = torch.arange(A*B*C*D, dtype=torch.float32, device="mps")
o = torch.zeros(A*B*C, device="mps")
k[(1,)](x, o, A, B, C, D, num_warps=1)
torch.mps.synchronize()
want = x.cpu().reshape(A,B,C,D).sum(3).reshape(-1)
if not torch.allclose(o.cpu(), want):
    print("WRONG rank4 reduce")
"""

for _dt, _torch, _tl in (("f16", "torch.float16", "tl.float16"),
                         ("bf16", "torch.bfloat16", "tl.bfloat16"),
                         ("i64", "torch.int64", "tl.int64"),
                         ("i8", "torch.int8", "tl.int8"),
                         ("i16", "torch.int16", "tl.int16")):
    P.PROBES[f"reduce_rank2_dtype_{_dt}"] = f"""
@triton.jit
def k(X, O, M: tl.constexpr, N: tl.constexpr):
    i = tl.arange(0, M)
    j = tl.arange(0, N)
    v = tl.load(X + i[:, None] * N + j[None, :])
    tl.store(O + i, tl.sum(v, 1))

M, N = 16, 8
ref = torch.ones(M * N, dtype={_torch})
x = ref.to("mps")
o = torch.zeros(M, dtype={_torch}, device="mps")
k[(1,)](x, o, M, N, num_warps=1)
torch.mps.synchronize()
want = ref.reshape(M, N).sum(1)
if not torch.equal(o.cpu().to(torch.float64), want.to(torch.float64)):
    print("WRONG dtype reduce {_dt}", o.cpu()[:4].tolist(), want[:4].tolist())
"""

COMBINES = {
    "xor": ("a ^ b", "torch.int32", "def fold(t): return t[:, 0] ^ t[:, 1] ^ t[:, 2] ^ t[:, 3] ^ t[:, 4] ^ t[:, 5] ^ t[:, 6] ^ t[:, 7]"),
    "and": ("a & b", "torch.int32", "def fold(t):\n    r = t[:, 0]\n    for c in range(1, t.shape[1]):\n        r = r & t[:, c]\n    return r"),
    "or": ("a | b", "torch.int32", "def fold(t):\n    r = t[:, 0]\n    for c in range(1, t.shape[1]):\n        r = r | t[:, c]\n    return r"),
}
for _name, (_expr, _torch, _fold) in COMBINES.items():
    P.PROBES[f"reduce_rank2_combine_{_name}"] = f"""
@triton.jit
def combine(a, b):
    return {_expr}

@triton.jit
def k(X, O, M: tl.constexpr, N: tl.constexpr):
    i = tl.arange(0, M)
    j = tl.arange(0, N)
    v = tl.load(X + i[:, None] * N + j[None, :])
    tl.store(O + i, tl.reduce(v, 1, combine))

M, N = 16, 8
ref = (torch.arange(M * N, dtype=torch.int32) * 7 + 3)
x = ref.to("mps")
o = torch.zeros(M, dtype={_torch}, device="mps")
k[(1,)](x, o, M, N, num_warps=1)
torch.mps.synchronize()
{_fold}
want = fold(ref.reshape(M, N))
if not torch.equal(o.cpu(), want):
    print("WRONG combine {_name}", o.cpu()[:4].tolist(), want[:4].tolist())
"""

P.PROBES["reduce_rank2_combine_noncommutative"] = """
@triton.jit
def combine(a, b):
    return tl.where(a != 0, a, b)

@triton.jit
def k(X, O, M: tl.constexpr, N: tl.constexpr):
    i = tl.arange(0, M)
    j = tl.arange(0, N)
    v = tl.load(X + i[:, None] * N + j[None, :])
    tl.store(O + i, tl.reduce(v, 1, combine))

M, N = 16, 8
ref = torch.zeros(M, N, dtype=torch.int32)
ref[:, 3] = torch.arange(1, M + 1, dtype=torch.int32)
ref[:, 6] = 99
x = ref.reshape(-1).contiguous().to("mps")
o = torch.zeros(M, dtype=torch.int32, device="mps")
k[(1,)](x, o, M, N, num_warps=1)
torch.mps.synchronize()
want = ref[:, 3]
if not torch.equal(o.cpu(), want):
    print("WRONG first-non-zero", o.cpu()[:4].tolist(), want[:4].tolist())
"""

P.PROBES["column_reduce_expanded_per_row"] = """
@triton.jit
def k(X, O, M: tl.constexpr, N: tl.constexpr):
    i = tl.arange(0, M)
    j = tl.arange(0, N)
    v = tl.load(X + i[:, None] * N + j[None, :])
    # A column reduce indexed BY ROW. Only writable when M == N.
    back = tl.sum(v, 0)[:, None] + tl.zeros((M, N), tl.float32)
    tl.store(O + i[:, None] * N + j[None, :], back)

M = N = 16
ref = torch.arange(M * N, dtype=torch.float32).reshape(M, N)
x = ref.to("mps")
o = torch.zeros(M, N, device="mps")
k[(1,)](x, o, M, N, num_warps=1)
torch.mps.synchronize()
want = ref.sum(0)[:, None].expand(M, N)
if not torch.allclose(o.cpu(), want):
    print("WRONG per-row column reduce")
"""

P.PROBES["column_reduce_mixed_with_load"] = """
@triton.jit
def k(X, Y, O, M: tl.constexpr, N: tl.constexpr):
    i = tl.arange(0, M)
    j = tl.arange(0, N)
    v = tl.load(X + i[:, None] * N + j[None, :])
    col = tl.sum(v, 0)                 # per-column convention
    row = tl.load(Y + tl.arange(0, N))  # per-element convention
    back = (col + row)[None, :] + tl.zeros((M, N), tl.float32)
    tl.store(O + i[:, None] * N + j[None, :], back)

M, N = 16, 8
ref = torch.arange(M * N, dtype=torch.float32).reshape(M, N)
y = torch.ones(N, device="mps")
o = torch.zeros(M, N, device="mps")
k[(1,)](ref.to("mps"), y, o, M, N, num_warps=1)
torch.mps.synchronize()
want = (ref.sum(0) + 1.0)[None, :].expand(M, N)
if not torch.allclose(o.cpu(), want):
    print("WRONG column reduce mixed with load")
"""

# --------------------------------------------------------------------------
# Scan envelope: the axes P2.2 still lists as open.
# --------------------------------------------------------------------------

for _ax in (0, 1):
    P.PROBES[f"scan_rank2_axis{_ax}"] = f"""
@triton.jit
def k(X, O, M: tl.constexpr, N: tl.constexpr):
    i = tl.arange(0, M)
    j = tl.arange(0, N)
    v = tl.load(X + i[:, None] * N + j[None, :])
    tl.store(O + i[:, None] * N + j[None, :], tl.cumsum(v, {_ax}))

M, N = 16, 16
ref = torch.arange(M * N, dtype=torch.float32).reshape(M, N)
o = torch.zeros(M, N, device="mps")
k[(1,)](ref.to("mps"), o, M, N, num_warps=1)
torch.mps.synchronize()
if not torch.allclose(o.cpu(), ref.cumsum({_ax})):
    print("WRONG rank2 scan axis {_ax}")
"""

P.PROBES["scan_rank3_axis2"] = """
@triton.jit
def k(X, O, A: tl.constexpr, B: tl.constexpr, C: tl.constexpr):
    f = tl.arange(0, A * B * C)
    v = tl.reshape(tl.load(X + f), (A, B, C))
    tl.store(O + f, tl.reshape(tl.cumsum(v, 2), (A * B * C,)))

A, B, C = 4, 4, 8
ref = torch.arange(A*B*C, dtype=torch.float32)
o = torch.zeros(A*B*C, device="mps")
k[(1,)](ref.to("mps"), o, A, B, C, num_warps=1)
torch.mps.synchronize()
want = ref.reshape(A,B,C).cumsum(2).reshape(-1)
if not torch.allclose(o.cpu(), want):
    print("WRONG rank3 scan axis2")
"""

for _dt, _torch in (("f16", "torch.float16"), ("bf16", "torch.bfloat16"),
                    ("i64", "torch.int64")):
    P.PROBES[f"scan_rank2_dtype_{_dt}"] = f"""
@triton.jit
def k(X, O, M: tl.constexpr, N: tl.constexpr):
    i = tl.arange(0, M)
    j = tl.arange(0, N)
    v = tl.load(X + i[:, None] * N + j[None, :])
    tl.store(O + i[:, None] * N + j[None, :], tl.cumsum(v, 1))

M, N = 8, 8
ref = torch.ones(M * N, dtype={_torch}).reshape(M, N)
o = torch.zeros(M, N, dtype={_torch}, device="mps")
k[(1,)](ref.reshape(-1).contiguous().to("mps"), o, M, N, num_warps=1)
torch.mps.synchronize()
want = ref.to(torch.float64).cumsum(1)
if not torch.equal(o.cpu().to(torch.float64), want):
    print("WRONG scan dtype {_dt}")
"""

P.PROBES["scan_tuple_rank2_f16"] = """
@triton.jit
def combine(a0, a1, b0, b1):
    return a0 + b0, a1 * b1

@triton.jit
def k(X, O, P_, M: tl.constexpr, N: tl.constexpr):
    i = tl.arange(0, M)
    j = tl.arange(0, N)
    v = tl.load(X + i[:, None] * N + j[None, :])
    s, p = tl.associative_scan((v, v), 1, combine)
    tl.store(O + i[:, None] * N + j[None, :], s)
    tl.store(P_ + i[:, None] * N + j[None, :], p)

M, N = 8, 8
ref = torch.ones(M, N, dtype=torch.float16)
o = torch.zeros(M, N, dtype=torch.float16, device="mps")
p = torch.zeros(M, N, dtype=torch.float16, device="mps")
k[(1,)](ref.reshape(-1).contiguous().to("mps"), o, p, M, N, num_warps=1)
torch.mps.synchronize()
if not torch.equal(o.cpu().float(), ref.float().cumsum(1)):
    print("WRONG f16 tuple scan sum")
"""

P.PROBES["scan_mixed_shapes_one_function"] = """
@triton.jit
def k(X, O, Q, M: tl.constexpr, N: tl.constexpr):
    i = tl.arange(0, M)
    j = tl.arange(0, N)
    v = tl.load(X + i[:, None] * N + j[None, :])
    tl.store(O + i[:, None] * N + j[None, :], tl.cumsum(v, 1))
    w = tl.load(X + tl.arange(0, N))
    tl.store(Q + tl.arange(0, N), tl.cumsum(w, 0))

M, N = 16, 8
ref = torch.arange(M * N, dtype=torch.float32)
o = torch.zeros(M, N, device="mps")
q = torch.zeros(N, device="mps")
k[(1,)](ref.to("mps"), o, q, M, N, num_warps=1)
torch.mps.synchronize()
if not torch.allclose(o.cpu(), ref.reshape(M, N).cumsum(1)) or \
   not torch.allclose(q.cpu(), ref[:N].cumsum(0)):
    print("WRONG mixed scan shapes")
"""

P.PROBES["scan_then_reduce_same_function"] = """
@triton.jit
def k(X, O, Q, M: tl.constexpr, N: tl.constexpr):
    i = tl.arange(0, M)
    j = tl.arange(0, N)
    v = tl.load(X + i[:, None] * N + j[None, :])
    c = tl.cumsum(v, 1)
    tl.store(O + i[:, None] * N + j[None, :], c)
    tl.store(Q + i, tl.sum(c, 1))

M, N = 16, 8
ref = torch.arange(M * N, dtype=torch.float32).reshape(M, N)
o = torch.zeros(M, N, device="mps")
q = torch.zeros(M, device="mps")
k[(1,)](ref.reshape(-1).contiguous().to("mps"), o, q, M, N, num_warps=1)
torch.mps.synchronize()
c = ref.cumsum(1)
if not torch.allclose(o.cpu(), c) or not torch.allclose(q.cpu(), c.sum(1)):
    print("WRONG scan then reduce")
"""

P.PROBES["scan_in_loop_tensor_iterarg"] = """
@triton.jit
def k(X, O, N: tl.constexpr, T: tl.constexpr):
    i = tl.arange(0, N)
    acc = tl.zeros((N,), tl.float32)
    for t in range(0, T):
        acc = acc + tl.cumsum(tl.load(X + t * N + i), 0)
    tl.store(O + i, acc)

N, T = 64, 3
ref = torch.arange(N * T, dtype=torch.float32)
o = torch.zeros(N, device="mps")
k[(1,)](ref.to("mps"), o, N, T, num_warps=1)
torch.mps.synchronize()
want = ref.reshape(T, N).cumsum(1).sum(0)
if not torch.allclose(o.cpu(), want):
    print("WRONG scan tensor iter_arg")
"""

P.PROBES["scan_in_loop_external_snapshot"] = """
@triton.jit
def k(X, Y, O, N: tl.constexpr, T: tl.constexpr):
    i = tl.arange(0, N)
    snap = tl.load(Y + i)          # taken OUTSIDE the loop, read inside
    for t in range(0, T):
        v = tl.load(X + t * N + i)
        tl.store(O + t * N + i, tl.cumsum(v, 0) + snap)

N, T = 64, 3
ref = torch.arange(N * T, dtype=torch.float32)
y = torch.ones(N, device="mps")
o = torch.zeros(N * T, device="mps")
k[(1,)](ref.to("mps"), y, o, N, T, num_warps=1)
torch.mps.synchronize()
want = (ref.reshape(T, N).cumsum(1) + 1).reshape(-1)
if not torch.allclose(o.cpu(), want):
    print("WRONG scan external snapshot")
"""

P.PROBES["scan_in_loop_with_if"] = """
@triton.jit
def k(X, O, N: tl.constexpr, T: tl.constexpr):
    i = tl.arange(0, N)
    for t in range(0, T):
        v = tl.load(X + t * N + i)
        if t % 2 == 0:
            c = tl.cumsum(v, 0)
        else:
            c = tl.cumsum(v * 2.0, 0)
        tl.store(O + t * N + i, c)

N, T = 64, 3
ref = torch.arange(N * T, dtype=torch.float32)
o = torch.zeros(N * T, device="mps")
k[(1,)](ref.to("mps"), o, N, T, num_warps=1)
torch.mps.synchronize()
r = ref.reshape(T, N)
want = torch.stack([r[t].cumsum(0) if t % 2 == 0 else (r[t] * 2).cumsum(0)
                    for t in range(T)]).reshape(-1)
if not torch.allclose(o.cpu(), want):
    print("WRONG scan under if")
"""

P.PROBES["scan_nested_rank2_in_loop"] = """
@triton.jit
def k(X, O, M: tl.constexpr, N: tl.constexpr, T: tl.constexpr):
    i = tl.arange(0, M)
    j = tl.arange(0, N)
    for t in range(0, T):
        v = tl.load(X + t * M * N + i[:, None] * N + j[None, :])
        tl.store(O + t * M * N + i[:, None] * N + j[None, :], tl.cumsum(v, 1))

M, N, T = 8, 8, 3
ref = torch.arange(M * N * T, dtype=torch.float32)
o = torch.zeros(M * N * T, device="mps")
k[(1,)](ref.to("mps"), o, M, N, T, num_warps=1)
torch.mps.synchronize()
want = ref.reshape(T, M, N).cumsum(2).reshape(-1)
if not torch.allclose(o.cpu(), want):
    print("WRONG nested rank2 scan")
"""

P.PROBES["scan_source_shared_with_store"] = """
@triton.jit
def k(X, O, Q, N: tl.constexpr):
    i = tl.arange(0, N)
    v = tl.load(X + i)
    tl.store(Q + i, v * 3.0)        # the scan's source is also stored
    tl.store(O + i, tl.cumsum(v, 0))

N = 128
ref = torch.arange(N, dtype=torch.float32)
o = torch.zeros(N, device="mps")
q = torch.zeros(N, device="mps")
k[(1,)](ref.to("mps"), o, q, N, num_warps=1)
torch.mps.synchronize()
if not torch.allclose(o.cpu(), ref.cumsum(0)) or not torch.allclose(q.cpu(), ref * 3):
    print("WRONG scan source shared with store")
"""

P.PROBES["scan_consumer_is_another_scan"] = """
@triton.jit
def k(X, O, M: tl.constexpr, N: tl.constexpr):
    i = tl.arange(0, M)
    j = tl.arange(0, N)
    v = tl.load(X + i[:, None] * N + j[None, :])
    tl.store(O + i[:, None] * N + j[None, :], tl.cumsum(tl.cumsum(v, 1), 0))

M, N = 8, 8
ref = torch.arange(M * N, dtype=torch.float32).reshape(M, N)
o = torch.zeros(M, N, device="mps")
k[(1,)](ref.reshape(-1).contiguous().to("mps"), o, M, N, num_warps=1)
torch.mps.synchronize()
if not torch.allclose(o.cpu(), ref.cumsum(1).cumsum(0)):
    print("WRONG scan of scan two axes")
"""

P.PROBES["scan_huge_row_over_budget"] = """
@triton.jit
def k(X, O, M: tl.constexpr, N: tl.constexpr):
    i = tl.arange(0, M)
    j = tl.arange(0, N)
    v = tl.load(X + i[:, None] * N + j[None, :])
    tl.store(O + i[:, None] * N + j[None, :], tl.cumsum(v, 1))

M, N = 8, 16384   # 128Ki f32 = 512 KiB, far past the 32 KiB scratch budget
ref = torch.ones(M * N, dtype=torch.float32)
o = torch.zeros(M * N, device="mps")
k[(1,)](ref.to("mps"), o, M, N, num_warps=4)
torch.mps.synchronize()
want = ref.reshape(M, N).cumsum(1).reshape(-1)
if not torch.allclose(o.cpu(), want):
    print("WRONG huge row scan")
"""

P.PROBES["reduce_huge_row_over_budget"] = """
@triton.jit
def k(X, O, M: tl.constexpr, N: tl.constexpr):
    i = tl.arange(0, M)
    j = tl.arange(0, N)
    v = tl.load(X + i[:, None] * N + j[None, :])
    tl.store(O + i, tl.sum(v, 1))

M, N = 8, 16384
ref = torch.ones(M * N, dtype=torch.float32)
o = torch.zeros(M, device="mps")
k[(1,)](ref.to("mps"), o, M, N, num_warps=4)
torch.mps.synchronize()
if not torch.allclose(o.cpu(), ref.reshape(M, N).sum(1)):
    print("WRONG huge row reduce")
"""

if __name__ == "__main__":
    sys.exit(P.main())
