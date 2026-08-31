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

import importlib.util
from pathlib import Path

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


def _load_sparse_dense_matmul_module():
    path = (Path(__file__).resolve().parent / "fixtures" / "metal_leet" /
            "medium-sparse_matrix-Dense_matrix_multiplication.py")
    assert path.is_file(), f"required Metal Leet fixture not present: {path}"
    spec = importlib.util.spec_from_file_location(
        "leet_sparse_dense_matmul", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "M,N,K,density",
    [(8, 8, 8, 0.25), (64, 64, 64, 0.1), (70, 66, 96, 0.05),
     (4, 7, 5, 0.0)],
)
def test_leet_sparse_dense_matmul_coo(M, N, K, density):
    """Sparse COO solve paths match PyTorch without a dense GPU A."""
    module = _load_sparse_dense_matmul_module()
    torch.manual_seed(0x5B00 + M * 17 + N * 3 + K)
    a_cpu = torch.randn((M, N), dtype=torch.float32)
    a_cpu *= torch.rand((M, N)) < density
    a_sparse = a_cpu.to_sparse_coo().coalesce()
    nnz = a_sparse._nnz()
    assert a_sparse.layout == torch.sparse_coo
    assert a_sparse.values().numel() == nnz
    b_cpu = torch.randn((N, K), dtype=torch.float32)
    b = b_cpu.to("mps").contiguous()
    c = torch.empty((M, K), dtype=torch.float32, device="mps")
    c_prepacked = torch.empty_like(c)

    module.solve(a_sparse, b, c, M, N, K, nnz)
    row_indices, col_indices, values = module.prepare_coo(
        a_sparse, "mps", M, N, nnz
    )
    module.solve_coo(
        row_indices,
        col_indices,
        values,
        b,
        c_prepacked,
        M,
        N,
        K,
        nnz,
    )
    torch.mps.synchronize()

    expected = a_cpu @ b_cpu
    torch.testing.assert_close(c.cpu(), expected, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(
        c_prepacked.cpu(), expected, atol=1e-3, rtol=1e-3
    )


def test_leet_sparse_dense_matmul_coalesces_duplicate_coordinates():
    module = _load_sparse_dense_matmul_module()
    indices = torch.tensor([[0, 0, 1], [1, 1, 0]], dtype=torch.int64)
    values = torch.tensor([1.25, 2.75, -3.0], dtype=torch.float32)
    a_sparse = torch.sparse_coo_tensor(indices, values, (2, 3))
    assert not a_sparse.is_coalesced()
    b_cpu = torch.tensor(
        [[2.0, -1.0], [0.5, 3.0], [-4.0, 2.0]], dtype=torch.float32
    )
    b = b_cpu.to("mps")
    c = torch.empty((2, 2), dtype=torch.float32, device="mps")
    c_prepacked = torch.empty_like(c)

    module.solve(a_sparse, b, c, 2, 3, 2, a_sparse._nnz())
    rows, cols, packed_values = module.prepare_coo(
        a_sparse, "mps", 2, 3, a_sparse._nnz()
    )
    assert packed_values.numel() == 2
    module.solve_coo(
        rows, cols, packed_values, b, c_prepacked, 2, 3, 2,
        packed_values.numel()
    )
    torch.mps.synchronize()

    expected = a_sparse.to_dense() @ b_cpu
    torch.testing.assert_close(c.cpu(), expected)
    torch.testing.assert_close(c_prepacked.cpu(), expected)


@pytest.mark.parametrize("M,N,K", [(8, 8, 8), (64, 64, 64), (70, 66, 96)])
def test_leet_sparse_dense_matmul_dense_input_compatibility(M, N, K):
    module = _load_sparse_dense_matmul_module()
    torch.manual_seed(0x5B5B + M * 17 + N * 3 + K)
    a_cpu = torch.randn((M, N), dtype=torch.float32)
    b_cpu = torch.randn((N, K), dtype=torch.float32)
    c = torch.empty((M, K), dtype=torch.float32, device="mps")

    module.solve(
        a_cpu.to("mps"),
        b_cpu.to("mps"),
        c,
        M,
        N,
        K,
        int(torch.count_nonzero(a_cpu)),
    )
    torch.mps.synchronize()

    torch.testing.assert_close(c.cpu(), a_cpu @ b_cpu, atol=1e-3, rtol=1e-3)

    if (M, N, K) == (8, 8, 8):
        a_noncontiguous = a_cpu.to("mps").T
        assert not a_noncontiguous.is_contiguous()
        with pytest.raises(ValueError, match="contiguous"):
            module.solve(
                a_noncontiguous,
                b_cpu.to("mps"),
                c,
                M,
                N,
                K,
                M * N,
            )


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


# --- Llama-block canonical masked runtime-K GEMM -----------------------------
# One program computes a 64x64 output tile, using the canonical pointer-iter_arg
# K loop from hard-llama_transformer_block.py: masked A/W loads with zero fill,
# tl.trans(W), and a masked output store. This must be handled by the canonical
# runtime-K lowering rather than the recompute or fused LoRA special cases.
@triton.jit
def masked_multitile_transb_kernel(
    a_ptr, w_ptr, c_ptr, M, N, K, w_off,
    stride_am, stride_ak, stride_wn, stride_wk, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    w_ptrs = (w_ptr + w_off + offs_n[:, None] * stride_wn
              + offs_k[None, :] * stride_wk)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k_mask = offs_k < K - k0
        a_mask = (offs_m[:, None] < M) & k_mask[None, :]
        w_mask = (offs_n[:, None] < N) & k_mask[None, :]
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)
        acc = tl.dot(a, tl.trans(w), acc, allow_tf32=False)
        a_ptrs += BLOCK_K * stride_ak
        w_ptrs += BLOCK_K * stride_wk
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def _run_masked_multitile_transb(
    M, N, K, *, num_warps, block_m=64, block_n=64, seed=0xA110
):
    torch.manual_seed(seed + M * 17 + N * 5 + K)
    a_cpu = torch.randn((M, K), dtype=torch.float32).contiguous()
    weight_base_offset = 512
    w_storage_cpu = torch.randn(
        (weight_base_offset + N * K,), dtype=torch.float32
    ).contiguous()
    w_cpu = w_storage_cpu[weight_base_offset:].view(N, K)
    a = a_cpu.to("mps")
    w_storage = w_storage_cpu.to("mps")
    w = w_storage[weight_base_offset:].view(N, K)
    sentinel = -4321.0
    c_guard = torch.full((triton.cdiv(M, block_m) * block_m,
                          triton.cdiv(N, block_n) * block_n),
                         sentinel, dtype=torch.float32,
                         device="mps").contiguous()
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
    masked_multitile_transb_kernel[grid](
        a, w_storage, c_guard, M, N, K, weight_base_offset,
        a.stride(0), a.stride(1),
        w.stride(0), w.stride(1),
        c_guard.stride(0), c_guard.stride(1),
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=32,
        num_warps=num_warps,
    )
    return c_guard, a_cpu, w_cpu, sentinel


@pytest.mark.parametrize(
    "M,N,K", [(1, 768, 512), (4, 768, 512), (16, 768, 512),
              (30, 768, 512), (64, 768, 512)]
)
def test_dot_dynamic_k_masked_multitile_transposed_b_llama_shape(M, N, K):
    c_guard, a, w, sentinel = _run_masked_multitile_transb(
        M, N, K, num_warps=4
    )
    actual = c_guard[:M, :N].cpu()
    expected = torch.matmul(a, w.t())
    torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)
    assert torch.all(c_guard[M:, :].cpu() == sentinel)
    assert torch.all(c_guard[:, N:].cpu() == sentinel)


@pytest.mark.parametrize(
    "M,N,K,num_warps,block_m,block_n",
    [(13, 70, 50, 4, 64, 64), (30, 65, 47, 2, 64, 64),
     (1, 35, 31, 1, 32, 32)],
)
def test_dot_dynamic_k_masked_multitile_transposed_b_ragged(
    M, N, K, num_warps, block_m, block_n
):
    c_guard, a, w, sentinel = _run_masked_multitile_transb(
        M, N, K, num_warps=num_warps, block_m=block_m, block_n=block_n
    )
    actual = c_guard[:M, :N].cpu()
    expected = torch.matmul(a, w.t())
    torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)
    assert torch.all(c_guard[M:, :].cpu() == sentinel)
    assert torch.all(c_guard[:, N:].cpu() == sentinel)


def test_dot_dynamic_k_masked_multitile_multiwarp_repeatability():
    """Reusable per-warp staging/store scratch stays race-free over launches."""
    M, N, K = 13, 70, 50
    for launch in range(30):
        c_guard, a, w, sentinel = _run_masked_multitile_transb(
            M, N, K, num_warps=4, seed=0xA130 + launch
        )
        torch.testing.assert_close(
            c_guard[:M, :N].cpu(), a @ w.t(), atol=1e-3, rtol=1e-3
        )
        assert torch.all(c_guard[M:, :].cpu() == sentinel)
        assert torch.all(c_guard[:, N:].cpu() == sentinel)


@triton.jit
def masked_multitile_kernel(
    a_ptr, b_ptr, c_ptr, M, N, K, b_off,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = (b_ptr + b_off + offs_k[:, None] * stride_bk
              + offs_n[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k_mask = offs_k < K - k0
        a_mask = (offs_m[:, None] < M) & k_mask[None, :]
        b_mask = k_mask[:, None] & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc = tl.dot(a, b, acc, allow_tf32=False)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


@pytest.mark.parametrize(
    "M,N,K,num_warps,block_m,block_n",
    [(13, 35, 50, 1, 32, 32), (30, 65, 47, 2, 64, 64),
     (64, 64, 64, 4, 64, 64)],
)
def test_dot_dynamic_k_masked_multitile_normal_b_ragged(
    M, N, K, num_warps, block_m, block_n
):
    torch.manual_seed(0xB110 + M * 17 + N * 5 + K)
    a_cpu = torch.randn((M, K), dtype=torch.float32).contiguous()
    b_base_offset = 13
    b_storage_cpu = torch.randn(
        (b_base_offset + K * N,), dtype=torch.float32
    ).contiguous()
    b_cpu = b_storage_cpu[b_base_offset:].view(K, N)
    a = a_cpu.to("mps")
    b_storage = b_storage_cpu.to("mps")
    b = b_storage[b_base_offset:].view(K, N)
    sentinel = -4321.0
    c_guard = torch.full(
        (triton.cdiv(M, block_m) * block_m,
         triton.cdiv(N, block_n) * block_n),
        sentinel,
        dtype=torch.float32,
        device="mps",
    ).contiguous()
    masked_multitile_kernel[(triton.cdiv(M, block_m),
                             triton.cdiv(N, block_n))](
        a, b_storage, c_guard, M, N, K, b_base_offset,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
        c_guard.stride(0), c_guard.stride(1),
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=32,
        num_warps=num_warps,
    )
    actual = c_guard[:M, :N].cpu()
    torch.testing.assert_close(actual, a_cpu @ b_cpu, atol=1e-3, rtol=1e-3)
    assert torch.all(c_guard[M:, :].cpu() == sentinel)
    assert torch.all(c_guard[:, N:].cpu() == sentinel)


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
#     python/test/unit/fixtures/metal_leet/medium-logistic_regression.py.  This is not an ordinary
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


# --- Ordinary least squares: raw single-loop Gram plus X^T y.
#     The raw LeetGPU kernel carries acc_xtx and acc_xty in the same sample
#     loop; Metal splits those independent accumulator slices before the Gram
#     scalar-dot fallback runs. The raw source only writes the diagonal XtX
#     feature block for n_features > BLOCK_N, so the raw-equivalent test below
#     intentionally stays at n_features <= 32.
@triton.jit
def ordinary_least_squares_raw_kernel(
    X, y, XtX, Xty, n_samples, n_features, stride_x_0, stride_x_1,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_col = pid_row
    offset_row = pid_row * BLOCK_N + tl.arange(0, BLOCK_N)
    offset_col = pid_col * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_row = offset_row < n_features
    mask_col = offset_col < n_features

    acc_xtx = tl.zeros((BLOCK_N, BLOCK_N), dtype=tl.float32)
    acc_xty = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for step in range(0, n_samples, BLOCK_M):
        offset_m = step + tl.arange(0, BLOCK_M)
        mask_m = offset_m < n_samples
        x_mn = tl.load(
            X + offset_m[:, None] * stride_x_0
            + offset_col[None, :] * stride_x_1,
            mask=mask_m[:, None] & mask_col[None, :],
            other=0.0,
        )
        x_nm = tl.load(
            X + offset_row[:, None] * stride_x_1
            + offset_m[None, :] * stride_x_0,
            mask=mask_row[:, None] & mask_m[None, :],
            other=0.0,
        )
        acc_xtx += tl.dot(x_nm, x_mn)
        y_m = tl.load(y + offset_m, mask=mask_m, other=0.0)
        acc_xty += tl.sum(x_nm * y_m, axis=1)

    tl.store(
        XtX + offset_row[:, None] * n_features + offset_col[None, :],
        acc_xtx,
        mask=mask_row[:, None] & mask_col[None, :],
    )
    tl.store(Xty + offset_row, acc_xty, mask=mask_row)


@pytest.mark.parametrize("n_samples,n_features", [(32, 8), (45, 16), (64, 32)])
def test_dot_ordinary_least_squares_raw_mixed_loop(n_samples, n_features):
    torch.manual_seed(0xC0FFEE)
    x = torch.randn((n_samples, n_features), dtype=torch.float32).contiguous()
    y = torch.randn((n_samples,), dtype=torch.float32).contiguous()
    xtx = torch.full((n_features, n_features), torch.nan, dtype=torch.float32)
    xty = torch.full((n_features,), torch.nan, dtype=torch.float32)

    ordinary_least_squares_raw_kernel[(triton.cdiv(n_features, 32),)](
        x, y, xtx, xty,
        n_samples, n_features,
        x.stride(0), x.stride(1),
        BLOCK_M=32, BLOCK_N=32,
    )

    torch.testing.assert_close(xtx, x.t() @ x, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(xty, x.t() @ y, atol=2e-3, rtol=2e-3)


# Corrected OLS canary for n_features > BLOCK_N. The raw LeetGPU kernel uses a
# one-dimensional program grid and leaves off-diagonal XtX blocks unwritten, so
# this separate path keeps the fixture's intended two-dimensional coverage.
@triton.jit
def ordinary_least_squares_gram_kernel(
    X, XtX, n_samples, n_features, stride_x_0, stride_x_1,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)
    offset_row = pid_row * BLOCK_N + tl.arange(0, BLOCK_N)
    offset_col = pid_col * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_row = offset_row < n_features
    mask_col = offset_col < n_features

    acc_xtx = tl.zeros((BLOCK_N, BLOCK_N), dtype=tl.float32)

    for step in range(0, n_samples, BLOCK_M):
        offset_m = step + tl.arange(0, BLOCK_M)
        mask_m = offset_m < n_samples
        x_mn = tl.load(
            X + offset_m[:, None] * stride_x_0
            + offset_col[None, :] * stride_x_1,
            mask=mask_m[:, None] & mask_col[None, :],
            other=0.0,
        )
        x_nm = tl.load(
            X + offset_row[:, None] * stride_x_1
            + offset_m[None, :] * stride_x_0,
            mask=mask_row[:, None] & mask_m[None, :],
            other=0.0,
        )
        acc_xtx += tl.dot(x_nm, x_mn)

    tl.store(
        XtX + offset_row[:, None] * n_features + offset_col[None, :],
        acc_xtx,
        mask=mask_row[:, None] & mask_col[None, :],
    )


@triton.jit
def ordinary_least_squares_xty_kernel(
    X, y, Xty, n_samples, stride_x_0, stride_x_1,
    BLOCK_M: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    offset_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offset_m < n_samples
    x_m = tl.load(X + offset_m * stride_x_0 + pid_n * stride_x_1,
                  mask=mask_m, other=0.0)
    y_m = tl.load(y + offset_m, mask=mask_m, other=0.0)
    tl.atomic_add(Xty + pid_n, tl.sum(x_m * y_m))


@pytest.mark.parametrize("n_samples,n_features", [(45, 37)])
def test_dot_ordinary_least_squares_corrected_large_feature_canary(n_samples, n_features):
    torch.manual_seed(0xC0FFEE)
    x = torch.randn((n_samples, n_features), dtype=torch.float32).contiguous()
    y = torch.randn((n_samples,), dtype=torch.float32).contiguous()
    xtx = torch.full((n_features, n_features), torch.nan, dtype=torch.float32)
    xty = torch.zeros((n_features,), dtype=torch.float32)
    gram_grid = (
        triton.cdiv(n_features, 32),
        triton.cdiv(n_features, 32),
    )
    xty_grid = (
        n_features,
        triton.cdiv(n_samples, 32),
    )

    ordinary_least_squares_gram_kernel[gram_grid](
        x, xtx,
        n_samples, n_features,
        x.stride(0), x.stride(1),
        BLOCK_M=32, BLOCK_N=32,
    )
    ordinary_least_squares_xty_kernel[xty_grid](
        x, y, xty,
        n_samples,
        x.stride(0), x.stride(1),
        BLOCK_M=32,
    )

    torch.testing.assert_close(xtx, x.t() @ x, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(xty, x.t() @ y, atol=2e-3, rtol=2e-3)


# --- Linear self-attention: verbatim hard Leet-Triton kernels.  These combine
#     a computed ELU+1 dot operand with a side reduction, atomic publication of
#     KV/Ksum, then a second computed dot + normalization kernel.  The generic
#     dot path cannot claim either whole semantic shape, so Metal uses a narrow
#     correctness fallback that preserves the source program partitioning.
@triton.jit
def matmulKV_kernel(
    Kt_ptr, V_ptr, KV_ptr, Ksum_ptr,
    M, D,
    BLOCK_M: tl.constexpr,
    NUM_ITER: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    off_D = tl.arange(0, BLOCK_D)
    mask_D = off_D < D

    acc_KV = tl.zeros((BLOCK_D, BLOCK_D), dtype=tl.float32)
    acc_K = tl.zeros((BLOCK_D,), dtype=tl.float32)

    for i in range(NUM_ITER):
        off_M = pid * BLOCK_M * NUM_ITER + i * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_M = off_M < M

        K_DM = tl.load(
            Kt_ptr + off_D[:, None] * M + off_M[None, :],
            mask=mask_D[:, None] & mask_M[None, :],
            other=float("-inf"),
        )
        K_DM = tl.where(K_DM > 0, K_DM + 1, tl.exp(K_DM))

        V_MD = tl.load(
            V_ptr + off_M[:, None] * D + off_D[None, :],
            mask=mask_M[:, None] & mask_D[None, :],
            other=0.0,
        )
        acc_KV = tl.dot(K_DM, V_MD, acc=acc_KV, allow_tf32=False)
        acc_K = acc_K + tl.sum(K_DM, axis=1)

    tl.atomic_add(
        KV_ptr + off_D[:, None] * D + off_D[None, :],
        acc_KV,
        mask=mask_D[:, None] & mask_D[None, :],
    )
    tl.atomic_add(Ksum_ptr + off_D, acc_K, mask=mask_D)


@triton.jit
def linear_attn_kernel(
    Q_ptr, KV_ptr, Ksum_ptr, out_ptr,
    M, D,
    BLOCK_M: tl.constexpr,
    NUM_ITER: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_d: tl.constexpr,
):
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    off_D = tl.arange(0, BLOCK_D)
    off_d = pid0 * BLOCK_d + tl.arange(0, BLOCK_d)
    mask_D = off_D < D
    mask_d = off_d < D

    KV_Dd = tl.load(
        KV_ptr + off_D[:, None] * D + off_d[None, :],
        mask=mask_D[:, None] & mask_d[None, :], other=0.0,
    )
    Ksum_D = tl.load(Ksum_ptr + off_D, mask=mask_D, other=0.0)
    EPS = tl.full((BLOCK_M,), 1e-5, dtype=tl.float32)

    for i in range(NUM_ITER):
        off_M = (pid1 * NUM_ITER + i) * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_M = off_M < M

        Q_MD = tl.load(
            Q_ptr + off_M[:, None] * D + off_D[None, :],
            mask=mask_M[:, None] & mask_D[None, :], other=float("-inf"),
        )
        Q_MD = tl.where(Q_MD > 0, Q_MD + 1, tl.exp(Q_MD))

        numer = tl.dot(Q_MD, KV_Dd, allow_tf32=False)
        denom = tl.sum(Q_MD * Ksum_D[None, :], axis=1) + EPS

        tl.store(
            out_ptr + off_M[:, None] * D + off_d[None, :],
            numer / denom[:, None],
            mask=mask_M[:, None] & mask_d[None, :],
        )


def _linear_phi(x):
    return torch.where(x > 0, x + 1, torch.exp(x))


def _run_linear_attention(q, k, v, out):
    M, D = q.shape
    block_d = max(16, triton.next_power_of_2(D))
    kt = k.T.contiguous()
    kv = torch.zeros((D, D), dtype=torch.float32, device=q.device)
    ksum = torch.zeros((D,), dtype=torch.float32, device=q.device)
    matmulKV_kernel[(triton.cdiv(M, 256),)](
        kt, v, kv, ksum, M, D,
        BLOCK_M=64, NUM_ITER=4, BLOCK_D=block_d, num_warps=16,
    )
    linear_attn_kernel[(triton.cdiv(D, 64), triton.cdiv(M, 64))](
        q, kv, ksum, out, M, D,
        BLOCK_M=32, NUM_ITER=2, BLOCK_D=block_d, BLOCK_d=64, num_warps=8,
    )
    return kv, ksum


@pytest.mark.parametrize("M,D", [(256, 16), (300, 32)])
def test_linear_attention_preprocess_matches_reference(M, D):
    torch.manual_seed(0x1A77 + M + D)
    k = torch.randn((M, D), dtype=torch.float32, device="mps").contiguous()
    v = torch.randn((M, D), dtype=torch.float32, device="mps").contiguous()
    kt = k.T.contiguous()
    kv = torch.zeros((D, D), dtype=torch.float32, device="mps")
    ksum = torch.zeros((D,), dtype=torch.float32, device="mps")
    matmulKV_kernel[(triton.cdiv(M, 256),)](
        kt, v, kv, ksum, M, D,
        BLOCK_M=64, NUM_ITER=4,
        BLOCK_D=max(16, triton.next_power_of_2(D)), num_warps=16,
    )
    torch.mps.synchronize()

    phi_k = _linear_phi(k.cpu())
    torch.testing.assert_close(kv.cpu(), phi_k.T @ v.cpu(), atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(ksum.cpu(), phi_k.sum(0), atol=2e-3, rtol=2e-3)


@pytest.mark.parametrize("M,D", [(64, 16), (257, 32), (300, 64), (129, 96)])
def test_linear_attention_solve_matches_reference(M, D):
    torch.manual_seed(0x1A77 + M * 3 + D)
    q = torch.randn((M, D), dtype=torch.float32, device="mps").contiguous()
    k = torch.randn((M, D), dtype=torch.float32, device="mps").contiguous()
    v = torch.randn((M, D), dtype=torch.float32, device="mps").contiguous()
    sentinel = 0x1A77
    out = torch.full((M, D), float(sentinel), dtype=torch.float32, device="mps")
    _run_linear_attention(q, k, v, out)
    torch.mps.synchronize()

    qf = _linear_phi(q.cpu())
    kf = _linear_phi(k.cpu())
    vf = v.cpu()
    kv_ref = kf.T @ vf
    expected = (qf @ kv_ref) / ((qf * kf.sum(0)).sum(1, keepdim=True) + 1e-5)
    actual = out.cpu()
    assert actual.isfinite().all()
    assert not torch.any(actual == float(sentinel))
    torch.testing.assert_close(actual, expected, atol=3e-3, rtol=3e-3)
