"""W2a: runtime (dynamic) K-loop `tt.dot` on the Metal backend.

The committed dot lowering (`tryUnrollCanonical3IterArgDot`) statically unrolls
the K-loop and therefore requires a *compile-time* trip count (`for _ in
range(0, K_TILES)` with `K_TILES` constexpr, ≤ 8 tiles). Kernels that write the
natural `for k in range(0, K, BLOCK_K)` form with `K` a runtime kernel argument
fell through every matcher and failed with `failed to legalize operation
'tt.dot'`.

`tryRuntimeKLoopCanonicalDot` handles that shape: it emits a fresh `scf.for`
stepping the K axis by 8 (one simdgroup 8x8 subtile per iteration) over the
runtime bound, carrying the `simdgroup_matrix` accumulator as the loop's single
iter_arg (in-place `simdgroup_multiply_accumulate`). First cut: single 8x8
output tile per program, single-warp, unmasked f32 — larger problems tile via
an 8x8 program grid. See `metal-lora-linear-fix-plan.md` (W2a).
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import triton
import triton.language as tl

libmetal = pytest.importorskip(
    "triton._C.libtriton.metal",
    reason="Metal backend pybind module not built into libtriton",
)
if not torch.backends.mps.is_available():
    pytest.skip(
        "Metal backend requires an MPS-enabled PyTorch (Apple Silicon)",
        allow_module_level=True,
    )


@triton.jit
def dyn_k_kernel(
    a_ptr, b_ptr, c_ptr, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    # Runtime trip count: K is a kernel argument, not a constexpr.
    for _k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc = tl.dot(a, b, acc, allow_tf32=False)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc)


def _run(M, N, K, *, seed=0xC0FFEE):
    torch.manual_seed(seed)
    a = torch.randn((M, K), dtype=torch.float32).contiguous()
    b = torch.randn((K, N), dtype=torch.float32).contiguous()
    c = torch.zeros((M, N), dtype=torch.float32).contiguous()
    grid = (triton.cdiv(M, 8), triton.cdiv(N, 8))
    dyn_k_kernel[grid](
        a, b, c, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=8, BLOCK_N=8, BLOCK_K=8,
        num_warps=1,
    )
    return c, a, b


@pytest.mark.parametrize("K", [8, 16, 64, 128, 256])
def test_dot_dynamic_k_single_tile(K):
    """8x8 output, runtime K a multiple of 8."""
    c, a, b = _run(8, 8, K)
    expected = torch.matmul(a, b)
    torch.testing.assert_close(c.cpu(), expected, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("M,N,K", [(16, 16, 64), (24, 8, 128), (8, 32, 32)])
def test_dot_dynamic_k_program_grid(M, N, K):
    """Larger M/N tiled across an 8x8 program grid; each program runs a
    single-tile runtime-K matmul."""
    c, a, b = _run(M, N, K)
    expected = torch.matmul(a, b)
    torch.testing.assert_close(c.cpu(), expected, atol=1e-4, rtol=1e-4)


# --- W1: transposed B operand, tl.dot(a, tl.trans(w)) -------------------------
# The B weight is stored naturally as [N, K] and transposed inside the kernel
# (no pre-transpose in torch). The transpose is folded into the simdgroup
# staged load (swapped destination index), so W stays [N, K] in memory.
@triton.jit
def dyn_k_transb_kernel(
    a_ptr, w_ptr, c_ptr, K,
    stride_am, stride_ak, stride_wn, stride_wk, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs)
        w = tl.load(w_ptrs)
        acc = tl.dot(a, tl.trans(w), acc, allow_tf32=False)
        a_ptrs += BLOCK_K * stride_ak
        w_ptrs += BLOCK_K * stride_wk
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc)


def _run_transb(M, N, K, *, seed=0xC0FFEE):
    torch.manual_seed(seed)
    a = torch.randn((M, K), dtype=torch.float32).contiguous()
    w = torch.randn((N, K), dtype=torch.float32).contiguous()   # [N,K], not pre-transposed
    c = torch.zeros((M, N), dtype=torch.float32).contiguous()
    grid = (triton.cdiv(M, 8), triton.cdiv(N, 8))
    dyn_k_transb_kernel[grid](
        a, w, c, K,
        a.stride(0), a.stride(1),
        w.stride(0), w.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=8, BLOCK_N=8, BLOCK_K=8,
        num_warps=1,
    )
    return c, a, w


@pytest.mark.parametrize("M,N,K", [(8, 8, 8), (8, 8, 128), (16, 16, 64),
                                   (24, 32, 64), (8, 16, 256)])
def test_dot_dynamic_k_transposed_b(M, N, K):
    """tl.dot(a, tl.trans(w)) with runtime K; w kept as [N, K] in memory."""
    c, a, w = _run_transb(M, N, K)
    expected = torch.matmul(a, w.t())
    torch.testing.assert_close(c.cpu(), expected, atol=1e-4, rtol=1e-4)


# --- W2b: recompute-from-IV loop shape (medium-lora_linear.py's inner loop) ---
# Addresses are rebuilt from the induction variable each iteration
# (`offs_k = k + tl.arange(...)`), so the loop carries ONLY the accumulator (no
# pointer iter_args). This is the natural hand-written matmul shape.
@triton.jit
def recompute_transb_kernel(
    x_ptr, w_ptr, c_ptr, K,
    sxm, sxk, swn, swk, scm, scn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        x = tl.load(x_ptr + offs_m[:, None] * sxm + offs_k[None, :] * sxk)  # [M,K]
        w = tl.load(w_ptr + offs_n[:, None] * swn + offs_k[None, :] * swk)  # [N,K]
        acc = tl.dot(x, tl.trans(w), acc, allow_tf32=False)
    tl.store(c_ptr + offs_m[:, None] * scm + offs_n[None, :] * scn, acc)


@pytest.mark.parametrize("M,N,K", [(8, 8, 8), (8, 8, 256), (16, 16, 64),
                                   (24, 32, 128)])
def test_dot_dynamic_k_recompute_transposed_b(M, N, K):
    """LoRA-style recompute-from-IV loop with tl.dot(x, tl.trans(w))."""
    torch.manual_seed(0xC0FFEE)
    x = torch.randn((M, K), dtype=torch.float32).contiguous()
    w = torch.randn((N, K), dtype=torch.float32).contiguous()
    c = torch.zeros((M, N), dtype=torch.float32).contiguous()
    grid = (triton.cdiv(M, 8), triton.cdiv(N, 8))
    recompute_transb_kernel[grid](
        x, w, c, K,
        x.stride(0), x.stride(1), w.stride(0), w.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=8, BLOCK_N=8, BLOCK_K=8, num_warps=1,
    )
    torch.testing.assert_close(c.cpu(), torch.matmul(x, w.t()), atol=1e-4, rtol=1e-4)


# --- W2b: multi-accumulator loop (two dots sharing x) — LoRA's inner loop -----
# Two independent matmuls that share the A operand `x` and carry two
# accumulators through one K-loop, each stored after the loop. This is the
# fused LoRA inner loop (minus the post-loop combine, which is W2c).
@triton.jit
def two_matmul_kernel(
    x_ptr, w_ptr, a_ptr, o0_ptr, o1_ptr, K,
    sxm, sxk, swn, swk, sar, sak, s0m, s0n, s1m, s1r,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_R: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_r = tl.arange(0, BLOCK_R)
    acc0 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_M, BLOCK_R), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        x = tl.load(x_ptr + offs_m[:, None] * sxm + offs_k[None, :] * sxk)
        w = tl.load(w_ptr + offs_n[:, None] * swn + offs_k[None, :] * swk)
        a = tl.load(a_ptr + offs_r[:, None] * sar + offs_k[None, :] * sak)
        acc0 = tl.dot(x, tl.trans(w), acc0, allow_tf32=False)
        acc1 = tl.dot(x, tl.trans(a), acc1, allow_tf32=False)
    tl.store(o0_ptr + offs_m[:, None] * s0m + offs_n[None, :] * s0n, acc0)
    tl.store(o1_ptr + offs_m[:, None] * s1m + offs_r[None, :] * s1r, acc1)


@pytest.mark.parametrize("M,K", [(8, 8), (8, 256), (16, 64), (32, 128)])
def test_dot_dynamic_k_two_accumulators(M, K):
    """Two dots sharing x, two accumulators, one runtime-K loop (8x8 tiles)."""
    N = R = 8
    torch.manual_seed(0xC0FFEE)
    x = torch.randn((M, K), dtype=torch.float32).contiguous()
    w = torch.randn((N, K), dtype=torch.float32).contiguous()
    a = torch.randn((R, K), dtype=torch.float32).contiguous()
    o0 = torch.zeros((M, N), dtype=torch.float32).contiguous()
    o1 = torch.zeros((M, R), dtype=torch.float32).contiguous()
    two_matmul_kernel[(triton.cdiv(M, 8),)](
        x, w, a, o0, o1, K,
        x.stride(0), x.stride(1), w.stride(0), w.stride(1),
        a.stride(0), a.stride(1), o0.stride(0), o0.stride(1),
        o1.stride(0), o1.stride(1),
        BLOCK_M=8, BLOCK_N=N, BLOCK_R=R, BLOCK_K=8, num_warps=1,
    )
    torch.testing.assert_close(o0.cpu(), torch.matmul(x, w.t()), atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(o1.cpu(), torch.matmul(x, a.t()), atol=1e-4, rtol=1e-4)


# --- W2c: fully-fused LoRA (medium-lora_linear.py's compute, unmasked) --------
# Two-accumulator loop plus the post-loop epilogue
# `acc0 += scale * tl.dot(acc1, tl.trans(b))`. The scale-and-add folds into one
# simdgroup_fused_store(acc0, acc1·trans(b), scale).
@triton.jit
def fused_lora_kernel(
    x_ptr, w_ptr, a_ptr, b_ptr, o_ptr, K, scale,
    sxm, sxk, swn, swk, sar, sak, sbn, sbr, som, son,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_R: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_r = tl.arange(0, BLOCK_R)
    acc0 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_M, BLOCK_R), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        x = tl.load(x_ptr + offs_m[:, None] * sxm + offs_k[None, :] * sxk)
        w = tl.load(w_ptr + offs_n[:, None] * swn + offs_k[None, :] * swk)
        a = tl.load(a_ptr + offs_r[:, None] * sar + offs_k[None, :] * sak)
        acc0 = tl.dot(x, tl.trans(w), acc0, allow_tf32=False)
        acc1 = tl.dot(x, tl.trans(a), acc1, allow_tf32=False)
    b = tl.load(b_ptr + offs_n[:, None] * sbn + offs_r[None, :] * sbr)
    acc0 += scale * tl.dot(acc1, tl.trans(b), allow_tf32=False)
    tl.store(o_ptr + offs_m[:, None] * som + offs_n[None, :] * son, acc0)


@pytest.mark.parametrize("M,K,scale", [(8, 8, 1.0), (8, 256, 0.25),
                                       (16, 128, 2.0), (32, 64, 0.5)])
def test_dot_dynamic_k_fused_lora(M, K, scale):
    """Fully-fused LoRA compute (8x8 tiles, single-warp, unmasked)."""
    N = R = 8
    torch.manual_seed(0xC0FFEE)
    x = torch.randn((M, K), dtype=torch.float32).contiguous()
    w = torch.randn((N, K), dtype=torch.float32).contiguous()
    a = torch.randn((R, K), dtype=torch.float32).contiguous()
    b = torch.randn((N, R), dtype=torch.float32).contiguous()
    o = torch.zeros((M, N), dtype=torch.float32).contiguous()
    fused_lora_kernel[(triton.cdiv(M, 8), 1)](
        x, w, a, b, o, K, scale,
        x.stride(0), x.stride(1), w.stride(0), w.stride(1),
        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
        o.stride(0), o.stride(1),
        BLOCK_M=8, BLOCK_N=N, BLOCK_R=R, BLOCK_K=8, num_warps=1,
    )
    ref = x @ w.t() + scale * ((x @ a.t()) @ b.t())
    torch.testing.assert_close(o.cpu(), ref, atol=1e-3, rtol=1e-3)


# --- Multi-warp: one program's tile grid is partitioned across simdgroups.
@pytest.mark.parametrize("BLK,nw", [(32, 4), (64, 8), (64, 4)])
def test_dot_dynamic_k_transposed_b_multiwarp(BLK, nw):
    """Runtime-K matmul with num_warps>1 (M-tile rows split across warps)."""
    M = N = BLK
    K = 64
    torch.manual_seed(0xC0FFEE)
    x = torch.randn((M, K), dtype=torch.float32).contiguous()
    w = torch.randn((N, K), dtype=torch.float32).contiguous()
    c = torch.zeros((M, N), dtype=torch.float32).contiguous()
    recompute_transb_kernel[(1, 1)](
        x, w, c, K, x.stride(0), x.stride(1), w.stride(0), w.stride(1),
        c.stride(0), c.stride(1), BLOCK_M=BLK, BLOCK_N=BLK, BLOCK_K=8,
        num_warps=nw)
    torch.testing.assert_close(c.cpu(), torch.matmul(x, w.t()), atol=1e-4, rtol=1e-4)


# --- The verbatim medium-lora_linear.py kernel: swizzle2d program grid, masked
#     loads + `*` store mask, BLOCK_M=64/N=128/K=32/R=16, default num_warps=4.
@triton.jit
def lora_verbatim_kernel(
    inp, W, A, B, output, M, N, K, R, scale,
    stride_im, stride_ik, stride_wn, stride_wk, stride_ar, stride_ak,
    stride_bn, stride_br, stride_om, stride_on,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr, BLOCK_SIZE_R: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_m_blocks = tl.cdiv(M, BLOCK_SIZE_M)
    num_n_blocks = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m, pid_n = tl.swizzle2d(pid // num_n_blocks, pid % num_n_blocks,
                                num_m_blocks, num_n_blocks, GROUP_SIZE)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_r = tl.arange(0, BLOCK_SIZE_R)
    acc0 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_R), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        offs_k = k + tl.arange(0, BLOCK_SIZE_K)
        mask_x = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        mask_w = (offs_n[:, None] < N) & (offs_k[None, :] < K)
        mask_a = (offs_r[:, None] < R) & (offs_k[None, :] < K)
        x = tl.load(inp + offs_m[:, None] * stride_im + offs_k[None, :] * stride_ik, mask=mask_x, other=0.0)
        w = tl.load(W + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk, mask=mask_w, other=0.0)
        a = tl.load(A + offs_r[:, None] * stride_ar + offs_k[None, :] * stride_ak, mask=mask_a, other=0.0)
        acc0 = tl.dot(x, tl.trans(w), acc0, allow_tf32=False)
        acc1 = tl.dot(x, tl.trans(a), acc1, allow_tf32=False)
    mask_b = (offs_n[:, None] < N) & (offs_r[None, :] < R)
    b = tl.load(B + offs_n[:, None] * stride_bn + offs_r[None, :] * stride_br, mask=mask_b, other=0.0)
    acc0 += scale * tl.dot(acc1, tl.trans(b), allow_tf32=False)
    mask_y = (offs_m[:, None] < M) * (offs_n[None, :] < N)
    tl.store(output + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on, acc0, mask=mask_y)


@pytest.mark.parametrize("batch,d_in,d_out,rank", [(64, 128, 128, 16),
                                                   (128, 256, 256, 16),
                                                   (100, 200, 300, 8)])
def test_metal_lora_linear_verbatim(batch, d_in, d_out, rank):
    """The exact medium-lora_linear.py kernel (swizzle2d grid, masks, BLOCK
    64/128/32/16, default num_warps) end-to-end."""
    scale = 0.5
    torch.manual_seed(0)
    x = torch.randn(batch, d_in, dtype=torch.float32).contiguous()
    W = torch.randn(d_out, d_in, dtype=torch.float32).contiguous()
    A = torch.randn(rank, d_in, dtype=torch.float32).contiguous()
    B = torch.randn(d_out, rank, dtype=torch.float32).contiguous()
    out = torch.zeros(batch, d_out, dtype=torch.float32).contiguous()
    BM, BN, BK = 64, 128, 32
    BR = max(16, triton.next_power_of_2(rank))
    grid = (triton.cdiv(batch, BM) * triton.cdiv(d_out, BN),)
    lora_verbatim_kernel[grid](
        x, W, A, B, out, batch, d_out, d_in, rank, scale,
        x.stride(0), x.stride(1), W.stride(0), W.stride(1),
        A.stride(0), A.stride(1), B.stride(0), B.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_SIZE_M=BM, BLOCK_SIZE_N=BN, BLOCK_SIZE_K=BK, BLOCK_SIZE_R=BR,
        GROUP_SIZE=4)
    ref = x @ W.t() + scale * ((x @ A.t()) @ B.t())
    torch.testing.assert_close(out.cpu(), ref, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("BM,BN,BR,nw", [(32, 32, 8, 4), (64, 64, 16, 4),
                                         (64, 128, 16, 8)])
def test_dot_dynamic_k_fused_lora_multiwarp(BM, BN, BR, nw):
    """Fused LoRA, multi-tile output partitioned across warps (M-tile rows).
    BM=64,BN=128,BR=16 is medium-lora_linear.py's block shape."""
    M, N, R, K, scale = BM, BN, BR, 64, 0.5
    torch.manual_seed(0xC0FFEE)
    x = torch.randn((M, K), dtype=torch.float32).contiguous()
    w = torch.randn((N, K), dtype=torch.float32).contiguous()
    a = torch.randn((R, K), dtype=torch.float32).contiguous()
    b = torch.randn((N, R), dtype=torch.float32).contiguous()
    o = torch.zeros((M, N), dtype=torch.float32).contiguous()
    fused_lora_kernel[(1, 1)](
        x, w, a, b, o, K, scale,
        x.stride(0), x.stride(1), w.stride(0), w.stride(1),
        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
        o.stride(0), o.stride(1),
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_R=BR, BLOCK_K=8, num_warps=nw)
    ref = x @ w.t() + scale * ((x @ a.t()) @ b.t())
    torch.testing.assert_close(o.cpu(), ref, atol=1e-2, rtol=1e-2)


# --- Multi-tile output: one program computes a (BM/8)x(BN/8) grid of 8x8 tiles.
@pytest.mark.parametrize("N,BN", [(16, 16), (32, 32)])
def test_dot_dynamic_k_transposed_b_multitile(N, BN):
    """tl.dot(x, tl.trans(w)) with a BLOCK>8 (multi-tile) output grid."""
    M, K = BN, 64
    torch.manual_seed(0xC0FFEE)
    x = torch.randn((M, K), dtype=torch.float32).contiguous()
    w = torch.randn((N, K), dtype=torch.float32).contiguous()
    c = torch.zeros((M, N), dtype=torch.float32).contiguous()
    recompute_transb_kernel[(triton.cdiv(M, BN), triton.cdiv(N, BN))](
        x, w, c, K, x.stride(0), x.stride(1), w.stride(0), w.stride(1),
        c.stride(0), c.stride(1), BLOCK_M=BN, BLOCK_N=BN, BLOCK_K=8, num_warps=1)
    torch.testing.assert_close(c.cpu(), torch.matmul(x, w.t()), atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("BM,BN,BR", [(16, 16, 8), (16, 16, 16)])
def test_dot_dynamic_k_fused_lora_multitile(BM, BN, BR):
    """Fused LoRA with a multi-tile output grid (unmasked; BLOCK 16)."""
    M, N, R, K, scale = BM, BN, BR, 64, 0.5
    torch.manual_seed(0xC0FFEE)
    x = torch.randn((M, K), dtype=torch.float32).contiguous()
    w = torch.randn((N, K), dtype=torch.float32).contiguous()
    a = torch.randn((R, K), dtype=torch.float32).contiguous()
    b = torch.randn((N, R), dtype=torch.float32).contiguous()
    o = torch.zeros((M, N), dtype=torch.float32).contiguous()
    fused_lora_kernel[(triton.cdiv(M, BM), triton.cdiv(N, BN))](
        x, w, a, b, o, K, scale,
        x.stride(0), x.stride(1), w.stride(0), w.stride(1),
        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
        o.stride(0), o.stride(1),
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_R=BR, BLOCK_K=8, num_warps=1)
    ref = x @ w.t() + scale * ((x @ a.t()) @ b.t())
    torch.testing.assert_close(o.cpu(), ref, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("M,N,R,BM,BN,BR", [(16, 16, 8, 16, 16, 8),
                                            (13, 11, 7, 16, 16, 8)])
def test_dot_dynamic_k_fused_lora_masked_multitile(M, N, R, BM, BN, BR):
    """Fused LoRA, multi-tile output + every load/store masked."""
    K, scale = 64, 0.5
    torch.manual_seed(0xC0FFEE)
    x = torch.randn((M, K), dtype=torch.float32).contiguous()
    w = torch.randn((N, K), dtype=torch.float32).contiguous()
    a = torch.randn((R, K), dtype=torch.float32).contiguous()
    b = torch.randn((N, R), dtype=torch.float32).contiguous()
    o = torch.zeros((M, N), dtype=torch.float32).contiguous()
    fused_lora_masked_kernel[(triton.cdiv(M, BM), triton.cdiv(N, BN))](
        x, w, a, b, o, M, N, K, R, scale,
        x.stride(0), x.stride(1), w.stride(0), w.stride(1),
        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
        o.stride(0), o.stride(1),
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_R=BR, BLOCK_K=8, num_warps=1)
    ref = x @ w.t() + scale * ((x @ a.t()) @ b.t())
    torch.testing.assert_close(o.cpu(), ref, atol=1e-2, rtol=1e-2)


# --- Masks: fused LoRA with every load + the store masked (LoRA verbatim, 8x8) -
@triton.jit
def fused_lora_masked_kernel(
    x_ptr, w_ptr, a_ptr, b_ptr, o_ptr, M, N, K, R, scale,
    sxm, sxk, swn, swk, sar, sak, sbn, sbr, som, son,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_R: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_r = tl.arange(0, BLOCK_R)
    acc0 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_M, BLOCK_R), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        mx = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        mw = (offs_n[:, None] < N) & (offs_k[None, :] < K)
        ma = (offs_r[:, None] < R) & (offs_k[None, :] < K)
        x = tl.load(x_ptr + offs_m[:, None] * sxm + offs_k[None, :] * sxk, mask=mx, other=0.0)
        w = tl.load(w_ptr + offs_n[:, None] * swn + offs_k[None, :] * swk, mask=mw, other=0.0)
        a = tl.load(a_ptr + offs_r[:, None] * sar + offs_k[None, :] * sak, mask=ma, other=0.0)
        acc0 = tl.dot(x, tl.trans(w), acc0, allow_tf32=False)
        acc1 = tl.dot(x, tl.trans(a), acc1, allow_tf32=False)
    mb = (offs_n[:, None] < N) & (offs_r[None, :] < R)
    b = tl.load(b_ptr + offs_n[:, None] * sbn + offs_r[None, :] * sbr, mask=mb, other=0.0)
    acc0 += scale * tl.dot(acc1, tl.trans(b), allow_tf32=False)
    my = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(o_ptr + offs_m[:, None] * som + offs_n[None, :] * son, acc0, mask=my)


@pytest.mark.parametrize("M,N,R,K", [(8, 8, 8, 8), (6, 5, 3, 20),
                                     (13, 8, 7, 40), (3, 8, 8, 17)])
def test_dot_dynamic_k_fused_lora_masked(M, N, R, K):
    """Fused LoRA with all loads + store masked (non-multiple-of-8 dims)."""
    scale = 0.5
    torch.manual_seed(0xC0FFEE)
    x = torch.randn((M, K), dtype=torch.float32).contiguous()
    w = torch.randn((N, K), dtype=torch.float32).contiguous()
    a = torch.randn((R, K), dtype=torch.float32).contiguous()
    b = torch.randn((N, R), dtype=torch.float32).contiguous()
    o = torch.zeros((M, N), dtype=torch.float32).contiguous()
    fused_lora_masked_kernel[(triton.cdiv(M, 8), triton.cdiv(N, 8))](
        x, w, a, b, o, M, N, K, R, scale,
        x.stride(0), x.stride(1), w.stride(0), w.stride(1),
        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
        o.stride(0), o.stride(1),
        BLOCK_M=8, BLOCK_N=8, BLOCK_R=8, BLOCK_K=8, num_warps=1,
    )
    ref = x @ w.t() + scale * ((x @ a.t()) @ b.t())
    torch.testing.assert_close(o.cpu(), ref, atol=1e-3, rtol=1e-3)


# --- Weighted Gram matrix: verbatim Hessian kernel from
#     leet-triton/medium-logistic_regression.py.  This is not an ordinary
#     row-major GEMM: A is X transposed in the pointer expression and B is
#     multiplied by a per-sample weight before the dot.
@triton.jit
def logistic_hessian_kernel(
    X, n_samples, n_features, W, hessian,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)
    offset_row = pid_row * BLOCK_N + tl.arange(0, BLOCK_N)
    offset_col = pid_col * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_row = offset_row < n_features
    mask_col = offset_col < n_features
    sum_h = tl.zeros((BLOCK_N, BLOCK_N), dtype=tl.float32)
    for step in range(0, n_samples, BLOCK_M):
        offset = step + tl.arange(0, BLOCK_M)
        mask = offset < n_samples
        vals_xt = tl.load(
            X + offset_row[:, None] + offset[None, :] * n_features,
            mask=mask_row[:, None] & mask[None, :], other=0.0,
        )
        vals_x = tl.load(
            X + offset[:, None] * n_features + offset_col[None, :],
            mask=mask[:, None] & mask_col[None, :], other=0.0,
        )
        vals_w = tl.load(W + offset, mask=mask, other=0.0)
        sum_h += tl.dot(vals_xt, vals_x * vals_w[:, None])
    sum_h += tl.where(
        offset_row[:, None] == offset_col[None, :], 1e-6, 0.0
    )
    tl.store(
        hessian + offset_row[:, None] * n_features + offset_col[None, :],
        sum_h,
        mask=mask_row[:, None] & mask_col[None, :],
    )


@pytest.mark.parametrize("n_samples,n_features", [(32, 8), (45, 37), (64, 32)])
def test_dot_logistic_hessian_weighted_gram(n_samples, n_features):
    torch.manual_seed(0xC0FFEE)
    x = torch.randn((n_samples, n_features), dtype=torch.float32).contiguous()
    w = torch.rand((n_samples,), dtype=torch.float32).contiguous()
    hessian = torch.zeros((n_features, n_features), dtype=torch.float32)
    logistic_hessian_kernel[
        (triton.cdiv(n_features, 32), triton.cdiv(n_features, 32))
    ](x, n_samples, n_features, w, hessian, BLOCK_M=32, BLOCK_N=32)
    ref = x.t() @ (x * w[:, None])
    ref += torch.eye(n_features, dtype=torch.float32) * 1e-6
    torch.testing.assert_close(hessian, ref, atol=2e-3, rtol=2e-3)
