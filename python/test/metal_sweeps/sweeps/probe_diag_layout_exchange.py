"""Reachability of the unpinned layout/tile and exchange/staging diagnostics.

Second slice of the named-diagnostic audit (`audit_diagnostics.py`). The first
slice measured the eight `atomic` diagnostics: thirteen probes, ZERO reachable
from `@triton.jit`. The plan's warning against extrapolating that is the reason
this file exists -- a layout diagnostic is refused by a walk over TTGIR
encodings, not by a frontend dtype table, so nothing about 0/8 carries over.

Twelve diagnostics, grouped by the question each one answers:

  L36879  bf16 atomic add (Apple's compare-exchange loop)
  L36911  atomic exchange payload type
  L39441  rank-2 axis=1 argmin geometry (0 < M <= tpb)
  L40251  a tt.trans that is not a flat identity
  L40330  a rank-2 index range with no rank-2 tile to decompose against
  L37998  a value live across full-tile exchange phases (nested scan)
  L40064  a non-local tt.broadcast that cannot be phased
  L40103  a tl.reshape that changes physical element ownership
  L1087/L1101/L1211/L1225  the P4.2 tile-invariant hoist cone checks

Each probe names the shape it is about, not the diagnostic it hopes to hit: a
probe that hits a DIFFERENT named message has found a shadow, which is just as
much of a result. Run:

    pixi run --frozen python python/test/metal_sweeps/sweeps/probe_diag_layout_exchange.py
    REPEAT=30 pixi run --frozen python .../probe_diag_layout_exchange.py <tag>
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

import probe_lib as P  # noqa: E402

P.SCRATCH = P.scratch_dir("diag_layout_exchange")

# --------------------------------------------------------------------------
# atomic payload types. These two live in the `exchange/staging` bucket only
# because their text contains "exchange"; they belong to the atomic family the
# first slice probed, and the first slice recorded both as SHADOWS -- messages
# that fire in place of a more generic one. A shadow that fires is reachable.
# --------------------------------------------------------------------------

P.PROBES["atomic_add_bf16"] = """
@triton.jit
def k(X, O, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    tl.atomic_add(O + i, tl.load(X + i))

x = torch.ones(32, dtype=torch.bfloat16, device="mps")
o = torch.zeros(32, dtype=torch.bfloat16, device="mps")
k[(1,)](x, o, BLOCK=32)
torch.mps.synchronize()
print(o.cpu()[:4])
"""

P.PROBES["atomic_add_bf16_scalar"] = """
@triton.jit
def k(X, O):
    tl.atomic_add(O, tl.load(X))

x = torch.ones(1, dtype=torch.bfloat16, device="mps")
o = torch.zeros(1, dtype=torch.bfloat16, device="mps")
k[(1,)](x, o)
torch.mps.synchronize()
print(o.cpu())
"""

for _dtype, _torch in (("f16", "torch.float16"), ("bf16", "torch.bfloat16"),
                       ("i64", "torch.int64"), ("i16", "torch.int16")):
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

# --------------------------------------------------------------------------
# rank-2 axis=1 argmin geometry. The validator wants `0 < M <= tpb`, and tpb is
# 32 * num_warps, so the reachable shape is simply more rows than threads. The
# M == tpb arm is the control: if it is refused too, the message is not about
# geometry at all.
# --------------------------------------------------------------------------

ARGMIN_SRC = """
@triton.jit
def k(X, O, M: tl.constexpr, N: tl.constexpr):
    i = tl.arange(0, M)
    j = tl.arange(0, N)
    x = tl.load(X + i[:, None] * N + j[None, :])
    tl.store(O + i, tl.argmin(x, axis=1))

M, N = {M}, {N}
x = torch.randn(M, N, device="mps")
o = torch.zeros(M, dtype=torch.int32, device="mps")
k[(1,)](x, o, M, N, num_warps={nw})
torch.mps.synchronize()
want = x.cpu().argmin(dim=1).to(torch.int32)
if not torch.equal(o.cpu(), want):
    print("WRONG argmin rows differ:", int((o.cpu() != want).sum()))
"""

for _tag, _M, _nw in (("M32_nw1", 32, 1), ("M64_nw1", 64, 1),
                      ("M128_nw1", 128, 1), ("M64_nw2", 64, 2),
                      ("M256_nw4", 256, 4)):
    P.PROBES[f"argmin_ax1_{_tag}"] = ARGMIN_SRC.format(M=_M, N=8, nw=_nw)

# --------------------------------------------------------------------------
# `tt.trans` that is not a flat identity. The forwarding pattern is right only
# while the result's layout `order` was permuted along with the shape. Every
# arm below keeps the transpose OUT of a `tl.dot` (the matmul matchers consume
# those before the walk runs) and varies what the result feeds, because the
# consumer is what decides the encoding TTGIR settles on.
# --------------------------------------------------------------------------

P.PROBES["trans_store_transposed"] = """
@triton.jit
def k(X, O, M: tl.constexpr, N: tl.constexpr):
    i = tl.arange(0, M)
    j = tl.arange(0, N)
    v = tl.load(X + i[:, None] * N + j[None, :])
    tl.store(O + j[:, None] * M + i[None, :], tl.trans(v))

M, N = 16, 8
x = torch.arange(M * N, dtype=torch.float32, device="mps").reshape(M, N)
o = torch.zeros(N, M, device="mps")
k[(1,)](x, o, M, N, num_warps=4)
torch.mps.synchronize()
if not torch.equal(o.cpu(), x.cpu().t().contiguous()):
    print("WRONG trans store")
"""

P.PROBES["trans_plus_other_tile"] = """
@triton.jit
def k(X, Y, O, M: tl.constexpr, N: tl.constexpr):
    i = tl.arange(0, M)
    j = tl.arange(0, N)
    v = tl.load(X + i[:, None] * N + j[None, :])
    w = tl.load(Y + j[:, None] * M + i[None, :])
    tl.store(O + j[:, None] * M + i[None, :], tl.trans(v) + w)

M, N = 16, 8
x = torch.arange(M * N, dtype=torch.float32, device="mps").reshape(M, N)
y = torch.ones(N, M, device="mps")
o = torch.zeros(N, M, device="mps")
k[(1,)](x, y, o, M, N, num_warps=4)
torch.mps.synchronize()
if not torch.equal(o.cpu(), x.cpu().t().contiguous() + 1):
    print("WRONG trans + tile")
"""

P.PROBES["trans_big_tile_nw1"] = """
@triton.jit
def k(X, O, M: tl.constexpr, N: tl.constexpr):
    i = tl.arange(0, M)
    j = tl.arange(0, N)
    v = tl.load(X + i[:, None] * N + j[None, :])
    tl.store(O + j[:, None] * M + i[None, :], tl.trans(v))

M, N = 64, 32
x = torch.arange(M * N, dtype=torch.float32, device="mps").reshape(M, N)
o = torch.zeros(N, M, device="mps")
k[(1,)](x, o, M, N, num_warps=1)
torch.mps.synchronize()
if not torch.equal(o.cpu(), x.cpu().t().contiguous()):
    print("WRONG trans big tile")
"""

P.PROBES["trans_reduce_axis0"] = """
@triton.jit
def k(X, O, M: tl.constexpr, N: tl.constexpr):
    i = tl.arange(0, M)
    j = tl.arange(0, N)
    v = tl.load(X + i[:, None] * N + j[None, :])
    tl.store(O + j, tl.sum(tl.trans(v), 1))

M, N = 16, 8
x = torch.arange(M * N, dtype=torch.float32, device="mps").reshape(M, N)
o = torch.zeros(N, device="mps")
k[(1,)](x, o, M, N, num_warps=4)
torch.mps.synchronize()
if not torch.allclose(o.cpu(), x.cpu().sum(0)):
    print("WRONG trans reduce")
"""

P.PROBES["trans_of_trans"] = """
@triton.jit
def k(X, O, M: tl.constexpr, N: tl.constexpr):
    i = tl.arange(0, M)
    j = tl.arange(0, N)
    v = tl.load(X + i[:, None] * N + j[None, :])
    tl.store(O + i[:, None] * N + j[None, :], tl.trans(tl.trans(v) * 2.0))

M, N = 16, 8
x = torch.arange(M * N, dtype=torch.float32, device="mps").reshape(M, N)
o = torch.zeros(M, N, device="mps")
k[(1,)](x, o, M, N, num_warps=4)
torch.mps.synchronize()
if not torch.equal(o.cpu(), x.cpu() * 2):
    print("WRONG trans of trans")
"""

P.PROBES["trans_store_same_order"] = """
@triton.jit
def k(X, O, M: tl.constexpr, N: tl.constexpr):
    i = tl.arange(0, M)
    j = tl.arange(0, N)
    v = tl.load(X + i[:, None] * M + j[None, :])
    t = tl.trans(v)
    tl.store(O + i[:, None] * M + j[None, :], t)

M, N = 16, 16
x = torch.arange(M * N, dtype=torch.float32, device="mps").reshape(M, N)
o = torch.zeros(M, N, device="mps")
k[(1,)](x, o, M, N, num_warps=4)
torch.mps.synchronize()
if not torch.equal(o.cpu(), x.cpu().t().contiguous()):
    print("WRONG square trans")
"""

# --------------------------------------------------------------------------
# A rank-2 index range with no rank-2 tile to decompose against. The guard
# accepts a range whose OWN cone names a rank-2 tile; what it refuses is a
# range that has neither its own tile nor a rank-2 largest tile in the module.
# --------------------------------------------------------------------------

P.PROBES["rank2_range_rank3_tile_only"] = """
@triton.jit
def k(X, O, A: tl.constexpr, B: tl.constexpr, C: tl.constexpr):
    i = tl.arange(0, A * B * C)
    v = tl.reshape(tl.load(X + i), (A, B, C))
    s = tl.sum(v, 2)
    a = tl.arange(0, A)
    b = tl.arange(0, B)
    tl.store(O + a[:, None] * B + b[None, :], s)

A, B, C = 4, 4, 4
x = torch.arange(A * B * C, dtype=torch.float32, device="mps")
o = torch.zeros(A, B, device="mps")
k[(1,)](x, o, A, B, C, num_warps=1)
torch.mps.synchronize()
if not torch.allclose(o.cpu(), x.cpu().reshape(A, B, C).sum(2)):
    print("WRONG rank3 reduce to rank2")
"""

P.PROBES["rank2_range_rank1_tile_only"] = """
@triton.jit
def k(X, O, M: tl.constexpr, N: tl.constexpr):
    # The kernel's only DATA tile is rank-1; the rank-2 range exists solely to
    # build a store address, so nothing else in the module is rank-2.
    f = tl.arange(0, M * N)
    v = tl.load(X + f)
    i = tl.arange(0, M)
    j = tl.arange(0, N)
    tl.store(O + i[:, None] * N + j[None, :], tl.reshape(v, (M, N)))

M, N = 8, 4
x = torch.arange(M * N, dtype=torch.float32, device="mps")
o = torch.zeros(M, N, device="mps")
k[(1,)](x, o, M, N, num_warps=1)
torch.mps.synchronize()
if not torch.equal(o.cpu(), x.cpu().reshape(M, N)):
    print("WRONG rank1 to rank2 reshape")
"""

P.PROBES["rank2_range_two_shapes"] = """
@triton.jit
def k(X, Y, O, P, M: tl.constexpr, N: tl.constexpr):
    i = tl.arange(0, M)
    j = tl.arange(0, N)
    tl.store(O + i[:, None] * N + j[None, :], tl.load(X + i[:, None] * N + j[None, :]) + 1.0)
    a = tl.arange(0, N)
    b = tl.arange(0, M)
    tl.store(P + a[:, None] * M + b[None, :], tl.load(Y + a[:, None] * M + b[None, :]) * 2.0)

M, N = 16, 8
x = torch.arange(M * N, dtype=torch.float32, device="mps").reshape(M, N)
y = torch.arange(M * N, dtype=torch.float32, device="mps").reshape(N, M)
o = torch.zeros(M, N, device="mps")
p = torch.zeros(N, M, device="mps")
k[(1,)](x, y, o, p, M, N, num_warps=4)
torch.mps.synchronize()
if not torch.equal(o.cpu(), x.cpu() + 1) or not torch.equal(p.cpu(), y.cpu() * 2):
    print("WRONG two rank-2 tiles")
"""

# --------------------------------------------------------------------------
# The exact-layout exchange trio. All three need a rank-N phase in the function
# (a staged permute/reduce), so every probe here builds one first and then adds
# the op under test on top of it.
# --------------------------------------------------------------------------

P.PROBES["exchange_broadcast_after_permute"] = """
@triton.jit
def k(X, O, A: tl.constexpr, B: tl.constexpr, C: tl.constexpr):
    i = tl.arange(0, A * B * C)
    v = tl.reshape(tl.load(X + i), (A, B, C))
    t = tl.permute(v, (1, 0, 2))
    s = tl.sum(t, 2)
    # Broadcast the reduced plane back over the axis it lost.
    back = s[:, :, None] + tl.zeros((B, A, C), tl.float32)
    tl.store(O + i, tl.reshape(back, (A * B * C,)))

A, B, C = 4, 4, 4
x = torch.arange(A * B * C, dtype=torch.float32, device="mps")
o = torch.zeros(A * B * C, device="mps")
k[(1,)](x, o, A, B, C, num_warps=1)
torch.mps.synchronize()
ref = x.cpu().reshape(A, B, C).permute(1, 0, 2)
want = ref.sum(2)[:, :, None].expand(B, A, C).reshape(-1)
if not torch.allclose(o.cpu(), want):
    print("WRONG permute broadcast")
"""

P.PROBES["exchange_broadcast_rank3_axis0"] = """
@triton.jit
def k(X, O, A: tl.constexpr, B: tl.constexpr, C: tl.constexpr):
    i = tl.arange(0, A * B * C)
    v = tl.reshape(tl.load(X + i), (A, B, C))
    s = tl.sum(v, 0)
    back = s[None, :, :] + tl.zeros((A, B, C), tl.float32)
    tl.store(O + i, tl.reshape(back, (A * B * C,)))

A, B, C = 4, 4, 4
x = torch.arange(A * B * C, dtype=torch.float32, device="mps")
o = torch.zeros(A * B * C, device="mps")
k[(1,)](x, o, A, B, C, num_warps=2)
torch.mps.synchronize()
want = x.cpu().reshape(A, B, C).sum(0)[None, :, :].expand(A, B, C).reshape(-1)
if not torch.allclose(o.cpu(), want):
    print("WRONG rank3 axis0 broadcast")
"""

P.PROBES["exchange_reshape_after_reduce"] = """
@triton.jit
def k(X, O, A: tl.constexpr, B: tl.constexpr, C: tl.constexpr):
    i = tl.arange(0, A * B * C)
    v = tl.reshape(tl.load(X + i), (A, B, C))
    s = tl.sum(v, 1)          # [A, C], a slice encoding
    f = tl.reshape(s, (A * C,))
    tl.store(O + tl.arange(0, A * C), f)

A, B, C = 4, 8, 4
x = torch.arange(A * B * C, dtype=torch.float32, device="mps")
o = torch.zeros(A * C, device="mps")
k[(1,)](x, o, A, B, C, num_warps=1)
torch.mps.synchronize()
if not torch.allclose(o.cpu(), x.cpu().reshape(A, B, C).sum(1).reshape(-1)):
    print("WRONG reshape after reduce")
"""

P.PROBES["exchange_reshape_permuted_slice"] = """
@triton.jit
def k(X, O, A: tl.constexpr, B: tl.constexpr, C: tl.constexpr):
    i = tl.arange(0, A * B * C)
    v = tl.reshape(tl.load(X + i), (A, B, C))
    t = tl.permute(v, (2, 0, 1))
    s = tl.max(t, 2)          # [C, A]
    tl.store(O + tl.arange(0, A * C), tl.reshape(s, (A * C,)))

A, B, C = 4, 8, 4
x = torch.arange(A * B * C, dtype=torch.float32, device="mps")
o = torch.zeros(A * C, device="mps")
k[(1,)](x, o, A, B, C, num_warps=1)
torch.mps.synchronize()
want = x.cpu().reshape(A, B, C).permute(2, 0, 1).amax(2).reshape(-1)
if not torch.allclose(o.cpu(), want):
    print("WRONG reshape permuted slice")
"""

# The nested-scan carry check. A scan inside a runtime loop is partitioned into
# phases; a value that must survive from one phase to the next and cannot be
# recomputed is what the diagnostic is about.
P.PROBES["nested_scan_carry_load"] = """
@triton.jit
def k(X, Y, O, N: tl.constexpr, T: tl.constexpr):
    i = tl.arange(0, N)
    for t in range(0, T):
        v = tl.load(X + t * N + i)
        c = tl.cumsum(v, 0)
        w = tl.load(Y + t * N + i)
        tl.store(O + t * N + i, c + w)

N, T = 64, 3
x = torch.arange(N * T, dtype=torch.float32, device="mps")
y = torch.ones(N * T, device="mps")
o = torch.zeros(N * T, device="mps")
k[(1,)](x, y, o, N, T, num_warps=1)
torch.mps.synchronize()
want = (x.cpu().reshape(T, N).cumsum(1) + 1).reshape(-1)
if not torch.allclose(o.cpu(), want):
    print("WRONG nested scan carry")
"""

P.PROBES["nested_scan_carry_two_scans"] = """
@triton.jit
def k(X, O, N: tl.constexpr, T: tl.constexpr):
    i = tl.arange(0, N)
    for t in range(0, T):
        v = tl.load(X + t * N + i)
        a = tl.cumsum(v, 0)
        b = tl.cumsum(v * 2.0, 0)
        tl.store(O + t * N + i, a + b)

N, T = 64, 3
x = torch.arange(N * T, dtype=torch.float32, device="mps")
o = torch.zeros(N * T, device="mps")
k[(1,)](x, o, N, T, num_warps=1)
torch.mps.synchronize()
r = x.cpu().reshape(T, N)
want = (r.cumsum(1) + (r * 2).cumsum(1)).reshape(-1)
if not torch.allclose(o.cpu(), want):
    print("WRONG nested two scans")
"""

# --------------------------------------------------------------------------
# The P4.2 tile-invariant hoist. Four diagnostics guard the cone that gets
# cloned into the prologue; the selector upstream asks the same question with
# the same predicate, so the interesting probes are the ones where the cone is
# unusual enough that the two walks might disagree. Every arm needs a tile
# bigger than one element per thread, or there is no per-element loop to hoist
# out of.
# --------------------------------------------------------------------------

P.PROBES["hoist_softmax_plain"] = """
@triton.jit
def k(X, O, N: tl.constexpr):
    i = tl.arange(0, N)
    v = tl.load(X + i)
    s = tl.sum(v, 0)
    tl.store(O + i, v / s)

N = 1024
x = torch.rand(N, device="mps") + 1.0
o = torch.zeros(N, device="mps")
k[(1,)](x, o, N, num_warps=1)
torch.mps.synchronize()
if not torch.allclose(o.cpu(), x.cpu() / x.cpu().sum(), atol=1e-5):
    print("WRONG hoist softmax")
"""

P.PROBES["hoist_two_aggregates"] = """
@triton.jit
def k(X, O, N: tl.constexpr):
    i = tl.arange(0, N)
    v = tl.load(X + i)
    m = tl.max(v, 0)
    e = tl.exp(v - m)
    s = tl.sum(e, 0)
    tl.store(O + i, e / s)

N = 1024
x = torch.rand(N, device="mps")
o = torch.zeros(N, device="mps")
k[(1,)](x, o, N, num_warps=1)
torch.mps.synchronize()
e = (x.cpu() - x.cpu().max()).exp()
if not torch.allclose(o.cpu(), e / e.sum(), atol=1e-5):
    print("WRONG hoist two aggregates")
"""

P.PROBES["hoist_aggregate_after_store"] = """
@triton.jit
def k(X, O, P, N: tl.constexpr):
    i = tl.arange(0, N)
    v = tl.load(X + i)
    tl.store(P + i, v * 2.0)      # an observable write ahead of the aggregate
    s = tl.sum(v, 0)
    tl.store(O + i, v / s)

N = 1024
x = torch.rand(N, device="mps") + 1.0
o = torch.zeros(N, device="mps")
p = torch.zeros(N, device="mps")
k[(1,)](x, o, p, N, num_warps=1)
torch.mps.synchronize()
if not torch.allclose(o.cpu(), x.cpu() / x.cpu().sum(), atol=1e-5):
    print("WRONG hoist after store")
"""

P.PROBES["hoist_aggregate_idiv_cone"] = """
@triton.jit
def k(X, O, N: tl.constexpr, D):
    i = tl.arange(0, N)
    v = tl.load(X + (i // D) + (i % D))
    s = tl.sum(v, 0)
    tl.store(O + i, v / s)

N = 1024
x = torch.rand(N, device="mps") + 1.0
o = torch.zeros(N, device="mps")
k[(1,)](x, o, N, 1, num_warps=1)
torch.mps.synchronize()
print(float(o.cpu()[0]))
"""

P.PROBES["hoist_aggregate_of_aggregate"] = """
@triton.jit
def k(X, O, N: tl.constexpr):
    i = tl.arange(0, N)
    v = tl.load(X + i)
    s = tl.sum(v, 0)
    t = tl.sum(v * s, 0)
    tl.store(O + i, v / t)

N = 1024
x = torch.rand(N, device="mps") + 1.0
o = torch.zeros(N, device="mps")
k[(1,)](x, o, N, num_warps=1)
torch.mps.synchronize()
xs = x.cpu()
if not torch.allclose(o.cpu(), xs / (xs * xs.sum()).sum(), atol=1e-4):
    print("WRONG hoist aggregate of aggregate")
"""

P.PROBES["hoist_reduction_loop"] = """
@triton.jit
def k(X, O, N: tl.constexpr, T: tl.constexpr):
    i = tl.arange(0, N)
    acc = 0.0
    for t in range(0, T):
        acc += tl.sum(tl.load(X + t * N + i), 0)
    tl.store(O + i, tl.load(X + i) / acc)

N, T = 1024, 3
x = torch.rand(N * T, device="mps") + 1.0
o = torch.zeros(N, device="mps")
k[(1,)](x, o, N, T, num_warps=1)
torch.mps.synchronize()
want = x.cpu()[:N] / x.cpu().sum()
if not torch.allclose(o.cpu(), want, atol=1e-4):
    print("WRONG hoist reduction loop")
"""

if __name__ == "__main__":
    sys.exit(P.main())
