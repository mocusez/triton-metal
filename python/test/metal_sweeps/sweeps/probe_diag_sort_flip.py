"""Fifth slice: the one unprobed diagnostic that names the user op it refuses.

`tl.arange has no per-element index under this layout (rank>2 broadcasts of an
arange, as tl.sort and tl.flip build, are not implemented)` is the only
unpinned refusal whose message names a user-facing op, so it is the one most
likely to be a real gap rather than defence in depth. Both ops reshape their
operand into a `[..., 2, 2, ..., 2, ...]` hypercube and xor-reduce across it,
so a rank-1 input already becomes a rank-log2(N) tile inside the op, and a
rank-2 input one rank higher again.

Axes: op {flip, sort asc/desc, topk} x input rank {1, 2, 3} x which axis x
shape x num_warps. Every probe carries its own torch oracle: the sort/topk
family has had silent wrong answers on this backend before, and a permutation
input (never arange) is the only one that tells a sort from an identity.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

import probe_lib as P  # noqa: E402

P.SCRATCH = P.scratch_dir("diag_sort_flip")

RANK1 = """
@triton.jit
def k(X, Y, N: tl.constexpr, OP: tl.constexpr):
    offs = tl.arange(0, N)
    x = tl.load(X + offs)
    if OP == 0:
        y = tl.flip(x)
    elif OP == 1:
        y = tl.sort(x)
    else:
        y = tl.sort(x, descending=True)
    tl.store(Y + offs, y)

def check(N, op, warps):
    x = torch.randperm(N, device="mps").float() - N / 2
    y = torch.empty_like(x)
    k[(1,)](x, y, N, op, num_warps=warps)
    torch.mps.synchronize()
    xc = x.cpu()
    want = [xc.flip(0), xc.sort().values, xc.sort(descending=True).values][op]
    if not torch.equal(y.cpu(), want):
        print(f"WRONG rank1 op={op} N={N} w={warps}: {int((y.cpu() != want).sum())} cells")
"""

RANK2 = """
@triton.jit
def k(X, Y, R: tl.constexpr, C: tl.constexpr, OP: tl.constexpr, DIM: tl.constexpr):
    r = tl.arange(0, R)
    c = tl.arange(0, C)
    x = tl.load(X + r[:, None] * C + c[None, :])
    if OP == 0:
        y = tl.flip(x, DIM)
    elif OP == 1:
        y = tl.sort(x)
    else:
        y = tl.sort(x, descending=True)
    tl.store(Y + r[:, None] * C + c[None, :], y)

def check(R, C, op, dim, warps):
    x = (torch.randperm(R * C, device="mps").float() - R * C / 2).reshape(R, C)
    y = torch.empty_like(x)
    k[(1,)](x, y, R, C, op, dim, num_warps=warps)
    torch.mps.synchronize()
    xc = x.cpu()
    want = [xc.flip(dim), xc.sort(dim=1).values, xc.sort(dim=1, descending=True).values][op]
    if not torch.equal(y.cpu(), want):
        print(f"WRONG rank2 op={op} dim={dim} {R}x{C} w={warps}: {int((y.cpu() != want).sum())} cells")
"""

TOPK2 = """
@triton.jit
def k(X, Y, R: tl.constexpr, C: tl.constexpr, K: tl.constexpr):
    r = tl.arange(0, R)
    c = tl.arange(0, C)
    kk = tl.arange(0, K)
    x = tl.load(X + r[:, None] * C + c[None, :])
    y = tl.topk(x, K)
    tl.store(Y + r[:, None] * K + kk[None, :], y)

def check(R, C, K, warps):
    x = (torch.randperm(R * C, device="mps").float() - R * C / 2).reshape(R, C)
    y = torch.empty(R, K, device="mps")
    k[(1,)](x, y, R, C, K, num_warps=warps)
    torch.mps.synchronize()
    want = x.cpu().topk(K, dim=1).values
    if not torch.equal(y.cpu(), want):
        print(f"WRONG topk2 {R}x{C} k={K} w={warps}: {int((y.cpu() != want).sum())} cells")
"""

RANK3 = """
@triton.jit
def k(X, Y, A: tl.constexpr, B: tl.constexpr, C: tl.constexpr, OP: tl.constexpr, DIM: tl.constexpr):
    a = tl.arange(0, A)
    b = tl.arange(0, B)
    c = tl.arange(0, C)
    off = a[:, None, None] * (B * C) + b[None, :, None] * C + c[None, None, :]
    x = tl.load(X + off)
    if OP == 0:
        y = tl.flip(x, DIM)
    else:
        y = tl.sort(x)
    tl.store(Y + off, y)

def check(A, B, C, op, dim, warps):
    x = (torch.randperm(A * B * C, device="mps").float() - A * B * C / 2).reshape(A, B, C)
    y = torch.empty_like(x)
    k[(1,)](x, y, A, B, C, op, dim, num_warps=warps)
    torch.mps.synchronize()
    xc = x.cpu()
    want = xc.flip(dim) if op == 0 else xc.sort(dim=2).values
    if not torch.equal(y.cpu(), want):
        print(f"WRONG rank3 op={op} dim={dim} {A}x{B}x{C} w={warps}: {int((y.cpu() != want).sum())} cells")
"""

for _n in (32, 256, 1024):
    for _op in (0, 1, 2):
        for _w in (1, 4):
            P.PROBES[f"r1_op{_op}_n{_n}_w{_w}"] = RANK1 + f"check({_n}, {_op}, {_w})\n"

for _r, _c in ((4, 32), (16, 16), (32, 8), (8, 128)):
    for _w in (1, 4):
        for _dim in (0, 1):
            P.PROBES[f"r2_flip_d{_dim}_{_r}x{_c}_w{_w}"] = RANK2 + f"check({_r}, {_c}, 0, {_dim}, {_w})\n"
        for _op in (1, 2):
            P.PROBES[f"r2_op{_op}_{_r}x{_c}_w{_w}"] = RANK2 + f"check({_r}, {_c}, {_op}, 1, {_w})\n"
        P.PROBES[f"r2_topk_{_r}x{_c}_w{_w}"] = TOPK2 + f"check({_r}, {_c}, 4, {_w})\n"

for _shape in ((2, 4, 8), (4, 8, 16), (2, 16, 32)):
    for _w in (1, 4):
        for _dim in (0, 1, 2):
            P.PROBES[f"r3_flip_d{_dim}_{'x'.join(map(str, _shape))}_w{_w}"] = \
                RANK3 + f"check({_shape[0]}, {_shape[1]}, {_shape[2]}, 0, {_dim}, {_w})\n"
        P.PROBES[f"r3_sort_{'x'.join(map(str, _shape))}_w{_w}"] = \
            RANK3 + f"check({_shape[0]}, {_shape[1]}, {_shape[2]}, 1, 2, {_w})\n"

if __name__ == "__main__":
    sys.exit(P.main())
