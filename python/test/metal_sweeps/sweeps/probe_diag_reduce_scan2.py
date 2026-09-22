"""Second wave: the STRUCTURAL reduce/scan diagnostics, plus the `other` bucket.

Wave 1 settled the envelope questions (dtype, combine, rank, axis) and found
two reachable diagnostics. What is left in the reduce/scan bucket is not about
what a tile CONTAINS but about how a function is SHAPED -- phases that cannot
be split, a scan whose source and consumers cannot be partitioned, a nested
scan with tensor work outside its loop. Those need whole kernel shapes, not
element types, which is why they get their own file.

The `other` bucket rides along: fp8 conversion, `tt.extern_elementwise`,
`tt.clampf` types and the `scf.for` iter_arg envelope are each one probe.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

import probe_lib as P  # noqa: E402

P.SCRATCH = P.scratch_dir("diag_reduce_scan2")

# -- staged scan / rank-N reduce phase structure ---------------------------

P.PROBES["scan_rank3_and_rank2_one_function"] = """
@triton.jit
def k(X, O, Q, A: tl.constexpr, B: tl.constexpr, C: tl.constexpr):
    f = tl.arange(0, A * B * C)
    v = tl.reshape(tl.load(X + f), (A, B, C))
    tl.store(O + f, tl.reshape(tl.cumsum(v, 2), (A * B * C,)))
    i = tl.arange(0, A)
    j = tl.arange(0, B)
    w = tl.load(X + i[:, None] * B + j[None, :])
    tl.store(Q + i[:, None] * B + j[None, :], tl.cumsum(w, 1))

A, B, C = 4, 4, 8
ref = torch.arange(A*B*C, dtype=torch.float32)
o = torch.zeros(A*B*C, device="mps")
q = torch.zeros(A*B, device="mps")
k[(1,)](ref.to("mps"), o, q, A, B, C, num_warps=1)
torch.mps.synchronize()
if not torch.allclose(o.cpu(), ref.reshape(A,B,C).cumsum(2).reshape(-1)) or \
   not torch.allclose(q.cpu(), ref[:A*B].reshape(A,B).cumsum(1).reshape(-1)):
    print("WRONG rank3+rank2 scans")
"""

P.PROBES["scan_under_divergent_if"] = """
@triton.jit
def k(X, O, N: tl.constexpr, FLAG):
    i = tl.arange(0, N)
    v = tl.load(X + i)
    if FLAG > 0:
        tl.store(O + i, tl.cumsum(v, 0))
    else:
        tl.store(O + i, v)

N = 128
ref = torch.arange(N, dtype=torch.float32)
o = torch.zeros(N, device="mps")
k[(1,)](ref.to("mps"), o, N, 1, num_warps=1)
torch.mps.synchronize()
if not torch.allclose(o.cpu(), ref.cumsum(0)):
    print("WRONG scan under if")
"""

P.PROBES["scan_source_written_after"] = """
@triton.jit
def k(X, O, N: tl.constexpr):
    i = tl.arange(0, N)
    v = tl.load(X + i)
    c = tl.cumsum(v, 0)
    tl.store(X + i, v * 0.0)     # writes the scan's own source buffer
    tl.store(O + i, c)

N = 128
ref = torch.arange(N, dtype=torch.float32)
x = ref.to("mps")
o = torch.zeros(N, device="mps")
k[(1,)](x, o, N, num_warps=1)
torch.mps.synchronize()
if not torch.allclose(o.cpu(), ref.cumsum(0)):
    print("WRONG scan whose source is overwritten")
"""

P.PROBES["scan_in_loop_with_tensor_work_outside"] = """
@triton.jit
def k(X, O, Q, N: tl.constexpr, T: tl.constexpr):
    i = tl.arange(0, N)
    outside = tl.load(X + i) * 2.0      # tensor work OUTSIDE the loop
    tl.store(Q + i, outside)
    for t in range(0, T):
        v = tl.load(X + t * N + i)
        tl.store(O + t * N + i, tl.cumsum(v, 0))

N, T = 64, 3
ref = torch.arange(N * T, dtype=torch.float32)
o = torch.zeros(N * T, device="mps")
q = torch.zeros(N, device="mps")
k[(1,)](ref.to("mps"), o, q, N, T, num_warps=1)
torch.mps.synchronize()
if not torch.allclose(o.cpu(), ref.reshape(T, N).cumsum(1).reshape(-1)) or \
   not torch.allclose(q.cpu(), ref[:N] * 2):
    print("WRONG loop scan with outside tensor work")
"""

P.PROBES["scan_in_loop_two_scans_different_shapes"] = """
@triton.jit
def k(X, O, Q, M: tl.constexpr, N: tl.constexpr, T: tl.constexpr):
    i = tl.arange(0, M)
    j = tl.arange(0, N)
    r = tl.arange(0, N)
    for t in range(0, T):
        v = tl.load(X + t * M * N + i[:, None] * N + j[None, :])
        tl.store(O + t * M * N + i[:, None] * N + j[None, :], tl.cumsum(v, 1))
        w = tl.load(X + t * N + r)
        tl.store(Q + t * N + r, tl.cumsum(w, 0))

M, N, T = 8, 8, 2
ref = torch.arange(M * N * T, dtype=torch.float32)
o = torch.zeros(M * N * T, device="mps")
q = torch.zeros(N * T, device="mps")
k[(1,)](ref.to("mps"), o, q, M, N, T, num_warps=1)
torch.mps.synchronize()
if not torch.allclose(o.cpu(), ref.reshape(T, M, N).cumsum(2).reshape(-1)):
    print("WRONG two scan shapes in a loop")
"""

P.PROBES["rank3_reduce_with_store_between_phases"] = """
@triton.jit
def k(X, O, Q, A: tl.constexpr, B: tl.constexpr, C: tl.constexpr):
    f = tl.arange(0, A * B * C)
    v = tl.reshape(tl.load(X + f), (A, B, C))
    tl.store(Q + f, tl.reshape(v * 2.0, (A * B * C,)))   # an effect in between
    s = tl.sum(v, 1)
    tl.store(O + tl.arange(0, A * C), tl.reshape(s, (A * C,)))

A, B, C = 4, 4, 4
ref = torch.arange(A*B*C, dtype=torch.float32)
o = torch.zeros(A*C, device="mps")
q = torch.zeros(A*B*C, device="mps")
k[(1,)](ref.to("mps"), o, q, A, B, C, num_warps=1)
torch.mps.synchronize()
if not torch.allclose(o.cpu(), ref.reshape(A,B,C).sum(1).reshape(-1)) or \
   not torch.allclose(q.cpu(), ref * 2):
    print("WRONG rank3 reduce with a store between phases")
"""

P.PROBES["rank3_reduce_inside_loop"] = """
@triton.jit
def k(X, O, A: tl.constexpr, B: tl.constexpr, C: tl.constexpr, T: tl.constexpr):
    f = tl.arange(0, A * B * C)
    for t in range(0, T):
        v = tl.reshape(tl.load(X + t * A * B * C + f), (A, B, C))
        s = tl.sum(v, 1)
        tl.store(O + t * A * C + tl.arange(0, A * C), tl.reshape(s, (A * C,)))

A, B, C, T = 4, 4, 4, 2
ref = torch.arange(A*B*C*T, dtype=torch.float32)
o = torch.zeros(A*C*T, device="mps")
k[(1,)](ref.to("mps"), o, A, B, C, T, num_warps=1)
torch.mps.synchronize()
want = ref.reshape(T, A, B, C).sum(2).reshape(-1)
if not torch.allclose(o.cpu(), want):
    print("WRONG rank3 reduce in a loop")
"""

P.PROBES["reduce_axis0_xor_combine"] = """
@triton.jit
def combine(a, b):
    return a ^ b

@triton.jit
def k(X, O, M: tl.constexpr, N: tl.constexpr):
    i = tl.arange(0, M)
    j = tl.arange(0, N)
    v = tl.load(X + i[:, None] * N + j[None, :])
    tl.store(O + j, tl.reduce(v, 0, combine))

M, N = 8, 16
ref = (torch.arange(M * N, dtype=torch.int32) * 5 + 1).reshape(M, N)
o = torch.zeros(N, dtype=torch.int32, device="mps")
k[(1,)](ref.reshape(-1).contiguous().to("mps"), o, M, N, num_warps=1)
torch.mps.synchronize()
want = ref[0]
for r in range(1, M):
    want = want ^ ref[r]
if not torch.equal(o.cpu(), want):
    print("WRONG axis0 xor")
"""

P.PROBES["reduce_all_axes"] = """
@triton.jit
def k(X, O, M: tl.constexpr, N: tl.constexpr):
    i = tl.arange(0, M)
    j = tl.arange(0, N)
    v = tl.load(X + i[:, None] * N + j[None, :])
    tl.store(O, tl.sum(v))

M, N = 8, 16
ref = torch.arange(M * N, dtype=torch.float32)
o = torch.zeros(1, device="mps")
k[(1,)](ref.to("mps"), o, M, N, num_warps=1)
torch.mps.synchronize()
if not torch.allclose(o.cpu(), ref.sum().reshape(1)):
    print("WRONG reduce over all axes")
"""

# -- the `other` bucket ----------------------------------------------------

for _name, _expr, _dt in (("fp8e4nv", "tl.float8e4nv", "torch.float8_e4m3fn"),
                          ("fp8e5", "tl.float8e5", "torch.float8_e5m2")):
    P.PROBES[f"fp8_cast_from_f32_{_name}"] = f"""
@triton.jit
def k(X, O, N: tl.constexpr):
    i = tl.arange(0, N)
    tl.store(O + i, tl.load(X + i).to({_expr}))

N = 64
x = torch.ones(N, dtype=torch.float32, device="mps")
o = torch.zeros(N, dtype={_dt}, device="mps")
k[(1,)](x, o, N, num_warps=1)
torch.mps.synchronize()
print(o.cpu().float()[:4].tolist())
"""

P.PROBES["fp8_cast_to_i32"] = """
@triton.jit
def k(X, O, N: tl.constexpr):
    i = tl.arange(0, N)
    tl.store(O + i, tl.load(X + i).to(tl.float8e4nv, bitcast=False).to(tl.int32))

N = 64
x = torch.ones(N, dtype=torch.float32, device="mps")
o = torch.zeros(N, dtype=torch.int32, device="mps")
k[(1,)](x, o, N, num_warps=1)
torch.mps.synchronize()
print(o.cpu()[:4].tolist())
"""

P.PROBES["extern_elementwise_unknown_symbol"] = """
@triton.jit
def k(X, O, N: tl.constexpr):
    i = tl.arange(0, N)
    v = tl.load(X + i)
    r = tl.inline_asm_elementwise("", "", [v], dtype=tl.float32, is_pure=True, pack=1)
    tl.store(O + i, r)

N = 64
x = torch.ones(N, device="mps")
o = torch.zeros(N, device="mps")
k[(1,)](x, o, N, num_warps=1)
torch.mps.synchronize()
print(o.cpu()[:2].tolist())
"""

P.PROBES["libdevice_unknown_symbol"] = """
from triton.language.extra import libdevice

@triton.jit
def k(X, O, N: tl.constexpr):
    i = tl.arange(0, N)
    tl.store(O + i, libdevice.j0(tl.load(X + i)))

N = 64
x = torch.ones(N, device="mps")
o = torch.zeros(N, device="mps")
k[(1,)](x, o, N, num_warps=1)
torch.mps.synchronize()
print(o.cpu()[:2].tolist())
"""

P.PROBES["clampf_i32"] = """
@triton.jit
def k(X, O, N: tl.constexpr):
    i = tl.arange(0, N)
    v = tl.load(X + i)
    tl.store(O + i, tl.clamp(v, -3, 3))

N = 64
x = (torch.arange(N, dtype=torch.int32) - 32).to("mps")
o = torch.zeros(N, dtype=torch.int32, device="mps")
k[(1,)](x, o, N, num_warps=1)
torch.mps.synchronize()
want = (torch.arange(N, dtype=torch.int32) - 32).clamp(-3, 3)
if not torch.equal(o.cpu(), want):
    print("WRONG i32 clamp")
"""

P.PROBES["for_iterarg_tensor"] = """
@triton.jit
def k(X, O, N: tl.constexpr, T: tl.constexpr):
    i = tl.arange(0, N)
    acc = tl.zeros((N,), tl.float32)
    for t in range(0, T):
        acc = acc * 2.0 + tl.load(X + t * N + i)
    tl.store(O + i, acc)

N, T = 64, 3
ref = torch.arange(N * T, dtype=torch.float32)
o = torch.zeros(N, device="mps")
k[(1,)](ref.to("mps"), o, N, T, num_warps=1)
torch.mps.synchronize()
r = ref.reshape(T, N)
want = torch.zeros(N)
for t in range(T):
    want = want * 2 + r[t]
if not torch.allclose(o.cpu(), want):
    print("WRONG tensor iter_arg recurrence")
"""

P.PROBES["for_iterarg_f16_scalar"] = """
@triton.jit
def k(X, O, N: tl.constexpr, T: tl.constexpr):
    acc = tl.load(X).to(tl.float16)
    for t in range(0, T):
        acc = acc + tl.load(X + t).to(tl.float16)
    tl.store(O, acc.to(tl.float32))

N, T = 8, 3
ref = torch.arange(N, dtype=torch.float32)
o = torch.zeros(1, device="mps")
k[(1,)](ref.to("mps"), o, N, T, num_warps=1)
torch.mps.synchronize()
want = ref[0] + ref[:T].sum()
if abs(float(o.cpu()[0]) - float(want)) > 1e-3:
    print("WRONG f16 scalar iter_arg", float(o.cpu()[0]), float(want))
"""

P.PROBES["gather_loop_carried"] = """
@triton.jit
def k(IDX, X, O, N: tl.constexpr, T: tl.constexpr):
    i = tl.arange(0, N)
    idx = tl.load(IDX + i)
    state = tl.load(X + i)
    for t in range(0, T):
        state = tl.gather(state, idx, 0)
    tl.store(O + i, state)

N, T = 32, 2
perm = torch.randperm(N).to(torch.int32)
ref = torch.arange(N, dtype=torch.float32)
o = torch.zeros(N, device="mps")
k[(1,)](perm.to("mps"), ref.to("mps"), o, N, T, num_warps=1)
torch.mps.synchronize()
want = ref.clone()
for _ in range(T):
    want = want[perm.long()]
if not torch.allclose(o.cpu(), want):
    print("WRONG loop-carried gather")
"""

if __name__ == "__main__":
    sys.exit(P.main())
