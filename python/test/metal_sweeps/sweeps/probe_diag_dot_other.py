"""Fourth slice: the dot/matmul, dot_scaled and remaining `other` diagnostics.

Three slices in, 111 probes have found 2 reachable diagnostics -- and two
families of silent wrong answers that no diagnostic covered. This slice takes
what is left of the buckets that have never been probed: the three canonical
multi-tile dot refusals, `dot_scaled`, the loop-carried `tl.gather` recurrence,
the resource-aware matrix launch, the per-row scratch bound, and the `scf.for`
iter_arg envelope.

Every probe carries an oracle for the same reason as the last slice: the dot
family is where this backend's worst wrong answers have lived.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

import probe_lib as P  # noqa: E402

P.SCRATCH = P.scratch_dir("diag_dot_other")

MATMUL = """
@triton.jit
def mm(A, B, C, M, N, K, sam, sak, sbk, sbn, scm, scn,
       BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
       MASKED: tl.constexpr, DIAG: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    a_ptr = A + offs_m[:, None] * sam + offs_k[None, :] * sak
    b_ptr = B + offs_k[:, None] * sbk + offs_n[None, :] * sbn
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(0, K, BK):
        if MASKED:
            a = tl.load(a_ptr, mask=(offs_k[None, :] + k) < K, other=0.0)
            b = tl.load(b_ptr, mask=(offs_k[:, None] + k) < K, other=0.0)
        else:
            a = tl.load(a_ptr)
            b = tl.load(b_ptr)
        acc += tl.dot(a, b)
        a_ptr += BK * sak
        b_ptr += BK * sbk
    c_ptr = C + offs_m[:, None] * scm + offs_n[None, :] * scn
    if DIAG:
        tl.store(c_ptr, acc, mask=(offs_m[:, None] + offs_n[None, :]) < N)
    elif MASKED:
        tl.store(c_ptr, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
    else:
        tl.store(c_ptr, acc)

def run(M, N, K, BM, BN, BK, warps, masked, diag, grid=(1, 1)):
    torch.manual_seed(0)
    a = torch.randn(M, K, device="mps")
    b = torch.randn(K, N, device="mps")
    c = torch.zeros(M, N, device="mps")
    mm[grid](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0),
             b.stride(1), c.stride(0), c.stride(1), BM, BN, BK, masked, diag,
             num_warps=warps)
    torch.mps.synchronize()
    return a, b, c
"""

# -- canonical multi-tile dot: warp partition -------------------------------

for _bm, _bn, _w in ((16, 16, 8), (16, 8, 4), (8, 16, 4), (32, 8, 8),
                     (8, 32, 8), (16, 16, 4), (64, 16, 8), (16, 64, 8)):
    P.PROBES[f"dot_warp_partition_{_bm}x{_bn}_w{_w}"] = MATMUL + f"""
M, N, K = {_bm}, {_bn}, 32
a, b, c = run(M, N, K, {_bm}, {_bn}, 8, {_w}, False, False)
want = (a.cpu().float() @ b.cpu().float())
if not torch.allclose(c.cpu(), want, atol=2e-3, rtol=2e-3):
    n = int((c.cpu() - want).abs().gt(2e-3).sum())
    print(f"WRONG warp partition {_bm}x{_bn} w{_w}: {{n}} cells")
"""

# -- masked multi-tile dot: output mask shape -------------------------------

P.PROBES["dot_masked_diagonal_output"] = MATMUL + """
M, N, K = 32, 32, 64
a, b, c = run(M, N, K, 32, 32, 16, 4, True, True)
full = (a.cpu().float() @ b.cpu().float())
i = torch.arange(M)[:, None]
j = torch.arange(N)[None, :]
want = torch.where(i + j < N, full, torch.zeros_like(full))
if not torch.allclose(c.cpu(), want, atol=2e-3, rtol=2e-3):
    n = int((c.cpu() - want).abs().gt(2e-3).sum())
    print(f"WRONG diagonal output mask: {n} cells")
"""

P.PROBES["dot_masked_runtime_k_rectangular"] = MATMUL + """
M, N, K = 48, 48, 40     # K not a multiple of BK, so the masks matter
a, b, c = run(M, N, K, 32, 32, 16, 4, True, False, grid=(2, 2))
want = (a.cpu().float() @ b.cpu().float())
if not torch.allclose(c.cpu(), want, atol=2e-3, rtol=2e-3):
    n = int((c.cpu() - want).abs().gt(2e-3).sum())
    print(f"WRONG masked runtime-K: {n} cells")
"""

P.PROBES["dot_masked_runtime_k_nonzero_other"] = """
@triton.jit
def mm(A, B, C, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    offs_m = tl.arange(0, BM)
    offs_n = tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    a_ptr = A + offs_m[:, None] * K + offs_k[None, :]
    b_ptr = B + offs_k[:, None] * N + offs_n[None, :]
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(0, K, BK):
        a = tl.load(a_ptr, mask=(offs_k[None, :] + k) < K, other=1.0)
        b = tl.load(b_ptr, mask=(offs_k[:, None] + k) < K, other=1.0)
        acc += tl.dot(a, b)
        a_ptr += BK
        b_ptr += BK * N
    tl.store(C + offs_m[:, None] * N + offs_n[None, :], acc,
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

M = N = 32
K = 40
torch.manual_seed(0)
a = torch.randn(M, K, device="mps")
b = torch.randn(K, N, device="mps")
c = torch.zeros(M, N, device="mps")
mm[(1,)](a, b, c, M, N, K, M, N, 16, num_warps=4)
torch.mps.synchronize()
want = a.cpu().float() @ b.cpu().float()
if not torch.allclose(c.cpu(), want, atol=2e-3, rtol=2e-3):
    print("WRONG nonzero-other runtime K")
"""

# -- dot_scaled -------------------------------------------------------------

P.PROBES["dot_scaled_mxfp4"] = """
@triton.jit
def k(A, SA, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    m = tl.arange(0, M)
    n = tl.arange(0, N)
    kk = tl.arange(0, K // 2)
    kf = tl.arange(0, K)
    a = tl.load(A + m[:, None] * (K // 2) + kk[None, :])
    sa = tl.load(SA + m[:, None] * (K // 32) + tl.arange(0, K // 32)[None, :])
    b = tl.load(B + kf[:, None] * N + n[None, :])
    acc = tl.dot_scaled(a, sa, "e2m1", b, None, "bf16")
    tl.store(C + m[:, None] * N + n[None, :], acc)

M, N, K = 16, 16, 64
a = torch.randint(0, 255, (M, K // 2), dtype=torch.uint8, device="mps")
sa = torch.full((M, K // 32), 127, dtype=torch.uint8, device="mps")
b = torch.randn(K, N, dtype=torch.bfloat16, device="mps")
c = torch.zeros(M, N, dtype=torch.float32, device="mps")
k[(1,)](a, sa, b, c, M, N, K, num_warps=4)
torch.mps.synchronize()
print(float(c.cpu()[0, 0]))
"""

# -- loop-carried tl.gather -------------------------------------------------

P.PROBES["gather_recurrence_runtime_bound"] = """
@triton.jit
def k(IDX, X, O, N: tl.constexpr, T):
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
    print("WRONG gather runtime bound")
"""

P.PROBES["gather_recurrence_index_changes"] = """
@triton.jit
def k(IDX, X, O, N: tl.constexpr, T: tl.constexpr):
    i = tl.arange(0, N)
    state = tl.load(X + i)
    for t in range(0, T):
        idx = tl.load(IDX + t * N + i)
        state = tl.gather(state, idx, 0)
    tl.store(O + i, state)

N, T = 32, 2
perms = torch.stack([torch.randperm(N) for _ in range(T)]).to(torch.int32)
ref = torch.arange(N, dtype=torch.float32)
o = torch.zeros(N, device="mps")
k[(1,)](perms.reshape(-1).contiguous().to("mps"), ref.to("mps"), o, N, T, num_warps=1)
torch.mps.synchronize()
want = ref.clone()
for t in range(T):
    want = want[perms[t].long()]
if not torch.allclose(o.cpu(), want):
    print("WRONG gather changing index")
"""

P.PROBES["gather_recurrence_with_arith"] = """
@triton.jit
def k(IDX, X, O, N: tl.constexpr, T: tl.constexpr):
    i = tl.arange(0, N)
    idx = tl.load(IDX + i)
    state = tl.load(X + i)
    for t in range(0, T):
        state = tl.gather(state, idx, 0) * 2.0 + 1.0
    tl.store(O + i, state)

N, T = 32, 2
perm = torch.randperm(N).to(torch.int32)
ref = torch.arange(N, dtype=torch.float32)
o = torch.zeros(N, device="mps")
k[(1,)](perm.to("mps"), ref.to("mps"), o, N, T, num_warps=1)
torch.mps.synchronize()
want = ref.clone()
for _ in range(T):
    want = want[perm.long()] * 2.0 + 1.0
if not torch.allclose(o.cpu(), want):
    print("WRONG gather with arith")
"""

# -- scratch bounds and iter_arg envelope -----------------------------------

P.PROBES["reduce_row_past_32kib"] = """
@triton.jit
def k(X, O, M: tl.constexpr, N: tl.constexpr):
    i = tl.arange(0, M)
    j = tl.arange(0, N)
    v = tl.load(X + i[:, None] * N + j[None, :])
    tl.store(O + i, tl.sum(v, 1))

M, N = 2, 65536
ref = torch.ones(M * N, dtype=torch.float32)
o = torch.zeros(M, device="mps")
k[(1,)](ref.to("mps"), o, M, N, num_warps=4)
torch.mps.synchronize()
if not torch.allclose(o.cpu(), ref.reshape(M, N).sum(1)):
    print("WRONG huge row reduce")
"""

P.PROBES["for_iterarg_i8"] = """
@triton.jit
def k(X, O, T: tl.constexpr):
    acc = tl.load(X).to(tl.int8)
    for t in range(0, T):
        acc = acc + tl.load(X + t).to(tl.int8)
    tl.store(O, acc.to(tl.int32))

T = 3
ref = torch.arange(8, dtype=torch.int32)
o = torch.zeros(1, dtype=torch.int32, device="mps")
k[(1,)](ref.to("mps"), o, T, num_warps=1)
torch.mps.synchronize()
want = int(ref[0]) + int(ref[:T].sum())
if int(o.cpu()[0]) != want:
    print("WRONG i8 iter_arg", int(o.cpu()[0]), want)
"""

P.PROBES["for_iterarg_pointer"] = """
@triton.jit
def k(X, O, N: tl.constexpr, T: tl.constexpr):
    i = tl.arange(0, N)
    p = X + i
    acc = tl.zeros((N,), tl.float32)
    for t in range(0, T):
        acc += tl.load(p)
        p += N
    tl.store(O + i, acc)

N, T = 32, 3
ref = torch.arange(N * T, dtype=torch.float32)
o = torch.zeros(N, device="mps")
k[(1,)](ref.to("mps"), o, N, T, num_warps=1)
torch.mps.synchronize()
if not torch.allclose(o.cpu(), ref.reshape(T, N).sum(0)):
    print("WRONG pointer iter_arg")
"""

P.PROBES["matrix_launch_uneven_warps"] = MATMUL + """
M, N, K = 64, 64, 64
a, b, c = run(M, N, K, 64, 64, 8, 3, False, False)
want = (a.cpu().float() @ b.cpu().float())
if not torch.allclose(c.cpu(), want, atol=2e-3, rtol=2e-3):
    print("WRONG 3-warp matmul")
"""

if __name__ == "__main__":
    sys.exit(P.main())
