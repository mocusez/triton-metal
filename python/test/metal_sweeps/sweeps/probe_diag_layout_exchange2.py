"""Second wave for the layout/tile + exchange/staging diagnostics.

Wave 1 (`probe_diag_layout_exchange.py`) settled four of the twelve: two HITs
(bf16 atomic add, rank-2 axis=1 argmin geometry) and two shadows -- both
exchange probes were refused earlier by the rank>=3 permuted-tile reduce, so
they never reached the broadcast/reshape scheduler at all.

This wave changes the SHAPE rather than repeating the question:

  * the exchange probes drop the `tl.permute` that shadowed them, and reach the
    rank-N phase through a plain rank-3 reduce or a scan instead;
  * the trans probes vary what CONSUMES the transpose, since the consumer is
    what decides the encoding TTGIR settles on;
  * the nested-scan probes put a value across the phase boundary that cannot be
    recomputed (an atomic's old value, a second scan's result);
  * the hoist probes combine a reduction LOOP with a straight-line aggregate,
    the one ordering where the two materializers interact.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

import probe_lib as P  # noqa: E402

P.SCRATCH = P.scratch_dir("diag_layout_exchange2")

# -- atomic exchange payloads the frontend still lets through ---------------

for _dtype, _torch in (("i8", "torch.int8"), ("bool", "torch.bool")):
    P.PROBES[f"atomic_xchg_{_dtype}"] = f"""
@triton.jit
def k(X, O, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    tl.atomic_xchg(O + i, tl.load(X + i))

x = torch.ones(32, dtype={_torch}, device="mps")
o = torch.zeros(32, dtype={_torch}, device="mps")
k[(1,)](x, o, BLOCK=32)
torch.mps.synchronize()
print(o.cpu()[:4])
"""

# -- tt.trans, varied by consumer -------------------------------------------

P.PROBES["trans_both_orientations_stored"] = """
@triton.jit
def k(X, O, Q, M: tl.constexpr, N: tl.constexpr):
    i = tl.arange(0, M)
    j = tl.arange(0, N)
    v = tl.load(X + i[:, None] * N + j[None, :])
    tl.store(O + i[:, None] * N + j[None, :], v)
    tl.store(Q + j[:, None] * M + i[None, :], tl.trans(v))

M, N = 16, 8
x = torch.arange(M * N, dtype=torch.float32, device="mps").reshape(M, N)
o = torch.zeros(M, N, device="mps")
q = torch.zeros(N, M, device="mps")
k[(1,)](x, o, q, M, N, num_warps=4)
torch.mps.synchronize()
if not torch.equal(o.cpu(), x.cpu()) or not torch.equal(q.cpu(), x.cpu().t().contiguous()):
    print("WRONG both orientations")
"""

P.PROBES["trans_atomic_consumer"] = """
@triton.jit
def k(X, O, M: tl.constexpr, N: tl.constexpr):
    i = tl.arange(0, M)
    j = tl.arange(0, N)
    v = tl.load(X + i[:, None] * N + j[None, :])
    tl.atomic_add(O + j[:, None] * M + i[None, :], tl.trans(v))

M, N = 16, 8
x = torch.ones(M, N, device="mps")
o = torch.zeros(N, M, device="mps")
k[(1,)](x, o, M, N, num_warps=4)
torch.mps.synchronize()
if not torch.equal(o.cpu(), torch.ones(N, M)):
    print("WRONG trans atomic")
"""

P.PROBES["trans_masked_load"] = """
@triton.jit
def k(X, O, M: tl.constexpr, N: tl.constexpr, R):
    i = tl.arange(0, M)
    j = tl.arange(0, N)
    v = tl.load(X + i[:, None] * N + j[None, :], mask=i[:, None] < R, other=0.0)
    tl.store(O + j[:, None] * M + i[None, :], tl.trans(v))

M, N, R = 16, 8, 9
x = torch.arange(M * N, dtype=torch.float32, device="mps").reshape(M, N)
o = torch.zeros(N, M, device="mps")
k[(1,)](x, o, M, N, R, num_warps=4)
torch.mps.synchronize()
want = x.cpu().clone()
want[R:] = 0
if not torch.equal(o.cpu(), want.t().contiguous()):
    print("WRONG masked trans")
"""

P.PROBES["trans_in_runtime_loop"] = """
@triton.jit
def k(X, O, M: tl.constexpr, N: tl.constexpr, T: tl.constexpr):
    i = tl.arange(0, M)
    j = tl.arange(0, N)
    for t in range(0, T):
        v = tl.load(X + t * M * N + i[:, None] * N + j[None, :])
        tl.store(O + t * M * N + j[:, None] * M + i[None, :], tl.trans(v))

M, N, T = 8, 4, 3
x = torch.arange(M * N * T, dtype=torch.float32, device="mps")
o = torch.zeros(M * N * T, device="mps")
k[(1,)](x, o, M, N, T, num_warps=1)
torch.mps.synchronize()
want = torch.cat([x.cpu().reshape(T, M, N)[t].t().reshape(-1) for t in range(T)])
if not torch.equal(o.cpu(), want):
    print("WRONG trans in loop")
"""

P.PROBES["trans_of_reshape"] = """
@triton.jit
def k(X, O, M: tl.constexpr, N: tl.constexpr):
    f = tl.arange(0, M * N)
    v = tl.reshape(tl.load(X + f), (M, N))
    i = tl.arange(0, M)
    j = tl.arange(0, N)
    tl.store(O + j[:, None] * M + i[None, :], tl.trans(v))

M, N = 16, 8
x = torch.arange(M * N, dtype=torch.float32, device="mps")
o = torch.zeros(N, M, device="mps")
k[(1,)](x, o, M, N, num_warps=4)
torch.mps.synchronize()
if not torch.equal(o.cpu(), x.cpu().reshape(M, N).t().contiguous()):
    print("WRONG trans of reshape")
"""

P.PROBES["trans_column_major_source"] = """
@triton.jit
def k(X, O, M: tl.constexpr, N: tl.constexpr):
    i = tl.arange(0, M)
    j = tl.arange(0, N)
    # Column-major address: the fast axis of the ADDRESS is the row index.
    v = tl.load(X + j[None, :] * M + i[:, None])
    tl.store(O + j[:, None] * M + i[None, :], tl.trans(v))

M, N = 16, 8
x = torch.arange(M * N, dtype=torch.float32, device="mps")
o = torch.zeros(N, M, device="mps")
k[(1,)](x, o, M, N, num_warps=4)
torch.mps.synchronize()
want = x.cpu().reshape(N, M).t().contiguous().t().contiguous()
if not torch.equal(o.cpu(), want):
    print("WRONG column-major trans")
"""

P.PROBES["trans_multiprogram"] = """
@triton.jit
def k(X, O, M: tl.constexpr, N: tl.constexpr):
    p = tl.program_id(0)
    i = tl.arange(0, M)
    j = tl.arange(0, N)
    v = tl.load(X + p * M * N + i[:, None] * N + j[None, :])
    tl.store(O + p * M * N + j[:, None] * M + i[None, :], tl.trans(v))

M, N, G = 16, 8, 4
x = torch.arange(M * N * G, dtype=torch.float32, device="mps")
o = torch.zeros(M * N * G, device="mps")
k[(G,)](x, o, M, N, num_warps=2)
torch.mps.synchronize()
want = torch.cat([x.cpu().reshape(G, M, N)[p].t().reshape(-1) for p in range(G)])
if not torch.equal(o.cpu(), want):
    print("WRONG multiprogram trans")
"""

# -- a rank-2 index range with nothing rank-2 to decompose against ----------

P.PROBES["rank2_range_mask_only_rank3_tile"] = """
@triton.jit
def k(X, O, A: tl.constexpr, B: tl.constexpr, C: tl.constexpr, R):
    f = tl.arange(0, A * B * C)
    v = tl.reshape(tl.load(X + f), (A, B, C))
    s = tl.sum(v, 2)
    a = tl.arange(0, A)
    b = tl.arange(0, B)
    # The rank-2 range exists only inside a mask, so no rank-2 DATA tile is
    # built from it.
    tl.store(O + a[:, None] * B + b[None, :], s, mask=(a[:, None] < R))

A, B, C, R = 4, 4, 4, 3
x = torch.arange(A * B * C, dtype=torch.float32, device="mps")
o = torch.zeros(A, B, device="mps")
k[(1,)](x, o, A, B, C, R, num_warps=1)
torch.mps.synchronize()
want = x.cpu().reshape(A, B, C).sum(2)
want[R:] = 0
if not torch.allclose(o.cpu(), want):
    print("WRONG rank2 mask range")
"""

P.PROBES["rank2_range_scalar_use_only"] = """
@triton.jit
def k(X, O, M: tl.constexpr, N: tl.constexpr):
    f = tl.arange(0, M * N)
    v = tl.load(X + f)
    i = tl.arange(0, M)
    j = tl.arange(0, N)
    # A rank-2 index tensor whose ONLY consumer is a rank-1 reduce back down.
    idx = i[:, None] * N + j[None, :]
    w = tl.sum(idx.to(tl.float32), 1)
    tl.store(O + i, w + tl.sum(v, 0))

M, N = 8, 4
x = torch.arange(M * N, dtype=torch.float32, device="mps")
o = torch.zeros(M, device="mps")
k[(1,)](x, o, M, N, num_warps=1)
torch.mps.synchronize()
idx = (torch.arange(M)[:, None] * N + torch.arange(N)[None, :]).float()
if not torch.allclose(o.cpu(), idx.sum(1) + x.cpu().sum()):
    print("WRONG rank2 range scalar use")
"""

# -- the exchange trio, reached WITHOUT a permute ---------------------------

P.PROBES["exchange_reshape_big_rank3_reduce"] = """
@triton.jit
def k(X, O, A: tl.constexpr, B: tl.constexpr, C: tl.constexpr):
    f = tl.arange(0, A * B * C)
    v = tl.reshape(tl.load(X + f), (A, B, C))
    s = tl.sum(v, 1)
    tl.store(O + tl.arange(0, A * C), tl.reshape(s, (A * C,)))

A, B, C = 8, 8, 8
x = torch.arange(A * B * C, dtype=torch.float32, device="mps")
o = torch.zeros(A * C, device="mps")
k[(1,)](x, o, A, B, C, num_warps=1)
torch.mps.synchronize()
if not torch.allclose(o.cpu(), x.cpu().reshape(A, B, C).sum(1).reshape(-1)):
    print("WRONG big rank3 reshape")
"""

P.PROBES["exchange_reshape_reduce_axis0_big"] = """
@triton.jit
def k(X, O, A: tl.constexpr, B: tl.constexpr, C: tl.constexpr):
    f = tl.arange(0, A * B * C)
    v = tl.reshape(tl.load(X + f), (A, B, C))
    s = tl.sum(v, 0)
    tl.store(O + tl.arange(0, B * C), tl.reshape(s, (B * C,)))

A, B, C = 8, 8, 8
x = torch.arange(A * B * C, dtype=torch.float32, device="mps")
o = torch.zeros(B * C, device="mps")
k[(1,)](x, o, A, B, C, num_warps=1)
torch.mps.synchronize()
if not torch.allclose(o.cpu(), x.cpu().reshape(A, B, C).sum(0).reshape(-1)):
    print("WRONG rank3 axis0 reshape")
"""

P.PROBES["exchange_broadcast_cross_axis"] = """
@triton.jit
def k(X, O, A: tl.constexpr, B: tl.constexpr, C: tl.constexpr):
    f = tl.arange(0, A * B * C)
    v = tl.reshape(tl.load(X + f), (A, B, C))
    s = tl.sum(v, 2)                      # [A, B]
    # Put the surviving plane back on a DIFFERENT axis than it was reduced.
    back = s[:, None, :] + tl.zeros((A, C, B), tl.float32)
    tl.store(O + f, tl.reshape(back, (A * B * C,)))

A, B, C = 4, 4, 4
x = torch.arange(A * B * C, dtype=torch.float32, device="mps")
o = torch.zeros(A * B * C, device="mps")
k[(1,)](x, o, A, B, C, num_warps=1)
torch.mps.synchronize()
want = x.cpu().reshape(A, B, C).sum(2)[:, None, :].expand(A, C, B).reshape(-1)
if not torch.allclose(o.cpu(), want):
    print("WRONG cross-axis broadcast")
"""

P.PROBES["exchange_broadcast_with_scan_phase"] = """
@triton.jit
def k(X, O, M: tl.constexpr, N: tl.constexpr):
    i = tl.arange(0, M)
    j = tl.arange(0, N)
    v = tl.load(X + i[:, None] * N + j[None, :])
    c = tl.cumsum(v, 1)                   # a rank-2 scan phase
    col = tl.sum(v, 0)                    # [N]
    back = col[None, :] + tl.zeros((M, N), tl.float32)
    tl.store(O + i[:, None] * N + j[None, :], c + back)

M, N = 16, 16
x = torch.arange(M * N, dtype=torch.float32, device="mps").reshape(M, N)
o = torch.zeros(M, N, device="mps")
k[(1,)](x, o, M, N, num_warps=1)
torch.mps.synchronize()
want = x.cpu().cumsum(1) + x.cpu().sum(0)[None, :]
if not torch.allclose(o.cpu(), want):
    print("WRONG scan phase broadcast")
"""

P.PROBES["exchange_sort_then_reshape"] = """
@triton.jit
def k(X, O, A: tl.constexpr, B: tl.constexpr):
    f = tl.arange(0, A * B)
    v = tl.reshape(tl.load(X + f), (A, B))
    s = tl.sort(v, 1)
    tl.store(O + f, tl.reshape(s, (A * B,)))

A, B = 8, 8
x = torch.randperm(A * B).float().to("mps")
o = torch.zeros(A * B, device="mps")
k[(1,)](x, o, A, B, num_warps=1)
torch.mps.synchronize()
want = x.cpu().reshape(A, B).sort(1).values.reshape(-1)
if not torch.equal(o.cpu(), want):
    print("WRONG sort reshape")
"""

# -- a value that must survive a nested scan's phase boundary ---------------

P.PROBES["nested_scan_carry_atomic_old"] = """
@triton.jit
def k(X, P_, O, N: tl.constexpr, T: tl.constexpr):
    i = tl.arange(0, N)
    for t in range(0, T):
        v = tl.load(X + t * N + i)
        old = tl.atomic_add(P_ + i, v)     # not recomputable
        c = tl.cumsum(v, 0)
        tl.store(O + t * N + i, c + old)

N, T = 64, 3
x = torch.arange(N * T, dtype=torch.float32, device="mps")
p = torch.zeros(N, device="mps")
o = torch.zeros(N * T, device="mps")
k[(1,)](x, p, o, N, T, num_warps=1)
torch.mps.synchronize()
print(float(o.cpu()[0]))
"""

P.PROBES["nested_scan_carry_scan_of_scan"] = """
@triton.jit
def k(X, O, N: tl.constexpr, T: tl.constexpr):
    i = tl.arange(0, N)
    for t in range(0, T):
        v = tl.load(X + t * N + i)
        a = tl.cumsum(v, 0)
        b = tl.cumsum(a, 0)                # second phase consumes the first
        tl.store(O + t * N + i, b)

N, T = 64, 3
x = torch.arange(N * T, dtype=torch.float32, device="mps")
o = torch.zeros(N * T, device="mps")
k[(1,)](x, o, N, T, num_warps=1)
torch.mps.synchronize()
want = x.cpu().reshape(T, N).cumsum(1).cumsum(1).reshape(-1)
if not torch.allclose(o.cpu(), want):
    print("WRONG scan of scan")
"""

P.PROBES["nested_scan_carry_masked_load"] = """
@triton.jit
def k(X, O, N: tl.constexpr, T: tl.constexpr, R):
    i = tl.arange(0, N)
    for t in range(0, T):
        m = i < R
        v = tl.load(X + t * N + i, mask=m, other=0.0)
        c = tl.cumsum(v, 0)
        tl.store(O + t * N + i, tl.where(m, c, -1.0))

N, T, R = 64, 3, 40
x = torch.arange(N * T, dtype=torch.float32, device="mps")
o = torch.zeros(N * T, device="mps")
k[(1,)](x, o, N, T, R, num_warps=1)
torch.mps.synchronize()
r = x.cpu().reshape(T, N).clone()
r[:, R:] = 0
c = r.cumsum(1)
c[:, R:] = -1
if not torch.allclose(o.cpu(), c.reshape(-1)):
    print("WRONG masked nested scan")
"""

# -- the two hoist materializers in one kernel ------------------------------

P.PROBES["hoist_loop_then_aggregate"] = """
@triton.jit
def k(X, O, N: tl.constexpr, T: tl.constexpr):
    i = tl.arange(0, N)
    acc = 0.0
    for t in range(0, T):
        acc += tl.sum(tl.load(X + t * N + i), 0)
    v = tl.load(X + i)
    s = tl.sum(v * acc, 0)
    tl.store(O + i, v / s)

N, T = 1024, 3
x = torch.rand(N * T, device="mps") + 1.0
o = torch.zeros(N, device="mps")
k[(1,)](x, o, N, T, num_warps=1)
torch.mps.synchronize()
xs = x.cpu()
want = xs[:N] / (xs[:N] * xs.sum()).sum()
if not torch.allclose(o.cpu(), want, rtol=1e-3):
    print("WRONG loop then aggregate")
"""

P.PROBES["hoist_aggregate_shared_load"] = """
@triton.jit
def k(X, O, Q, N: tl.constexpr):
    i = tl.arange(0, N)
    v = tl.load(X + i)
    s = tl.sum(v, 0)
    m = tl.max(v, 0)
    tl.store(O + i, v / s)
    tl.store(Q + i, v / m)

N = 1024
x = torch.rand(N, device="mps") + 1.0
o = torch.zeros(N, device="mps")
q = torch.zeros(N, device="mps")
k[(1,)](x, o, q, N, num_warps=1)
torch.mps.synchronize()
xs = x.cpu()
if not torch.allclose(o.cpu(), xs / xs.sum(), atol=1e-5) or \
   not torch.allclose(q.cpu(), xs / xs.max(), atol=1e-5):
    print("WRONG shared load aggregates")
"""

P.PROBES["hoist_aggregate_indirect_load"] = """
@triton.jit
def k(IDX, X, O, N: tl.constexpr):
    i = tl.arange(0, N)
    j = tl.load(IDX + i)
    v = tl.load(X + j)
    s = tl.sum(v, 0)
    tl.store(O + i, v / s)

N = 1024
idx = torch.randperm(N).to(torch.int32).to("mps")
x = torch.rand(N, device="mps") + 1.0
o = torch.zeros(N, device="mps")
k[(1,)](idx, x, o, N, num_warps=1)
torch.mps.synchronize()
g = x.cpu()[idx.cpu().long()]
if not torch.allclose(o.cpu(), g / g.sum(), atol=1e-5):
    print("WRONG indirect aggregate")
"""

P.PROBES["hoist_aggregate_program_id_cone"] = """
@triton.jit
def k(X, O, N: tl.constexpr):
    p = tl.program_id(0)
    i = tl.arange(0, N)
    v = tl.load(X + p * N + i)
    s = tl.sum(v, 0)
    tl.store(O + p * N + i, v / s)

N, G = 1024, 3
x = torch.rand(N * G, device="mps") + 1.0
o = torch.zeros(N * G, device="mps")
k[(G,)](x, o, N, num_warps=1)
torch.mps.synchronize()
xs = x.cpu().reshape(G, N)
want = (xs / xs.sum(1, keepdim=True)).reshape(-1)
if not torch.allclose(o.cpu(), want, atol=1e-5):
    print("WRONG program-id aggregate")
"""

if __name__ == "__main__":
    sys.exit(P.main())
