"""Universal tt.dot on the Metal backend.

Covers Phase 1 of `.omc/plans/metal-universal-matmul.md`:

- **AC2** (non-square multiples-of-8 f32): grid in M and/or N, K-tiled or single.
  Runtime M/N not a multiple of 8 also pass via the masked `tt.store` epilogue
  (`test_dot_f32_mn_partial_tile`).
- **AC3** (fp16 inputs, f32 accumulator): canonical-3-iter-arg path uses
  threadgroup-staged loads which implicit-cast `half`/`bfloat` -> `float`.
  bf16 single-dot passes via `rewriteSingleDot` SimdgroupLoadDeviceStaged staging
  (`test_dot_bf16_singletile`).
- **AC5** (K-loop tiled, K_TILES in {1,2,4,8}): the canonical-3-iter-arg
  matcher already covers this once the launcher arg-mask fix is in place.

Shape conventions:
- `BLOCK_M = BLOCK_N = BLOCK_K = 8` matches Apple's native simdgroup-matrix
  tile granularity (Principle 4 of the plan: tile granularity is fixed at
  hardware, kernel shape variability is encoded via the tile-grid).
- The kernel is a single canonical 3-iter-arg matmul; correctness of each
  axis (M-tile, N-tile, K-tile) is verified against `torch.matmul`.
"""
from __future__ import annotations

import os
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
def dot_universal_kernel(
    a_ptr, b_ptr, c_ptr,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    K_TILES: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, K_TILES):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc)


def _run(M, N, K, dtype_in=torch.float32, *, seed=0xC0FFEE):
    torch.manual_seed(seed)
    a = torch.randn((M, K), dtype=dtype_in).contiguous()
    b = torch.randn((K, N), dtype=dtype_in).contiguous()
    c = torch.zeros((M, N), dtype=torch.float32).contiguous()
    K_TILES = K // 8
    dot_universal_kernel[(triton.cdiv(M, 8), triton.cdiv(N, 8))](
        a, b, c,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=8, BLOCK_N=8, BLOCK_K=8, K_TILES=K_TILES,
    )
    ref = a.float() @ b.float()
    return c, ref


# ----------------------------------------------------------------------------
# AC2: non-square multiples-of-8 f32
# ----------------------------------------------------------------------------
@pytest.mark.parametrize(
    "M,N,K",
    [
        pytest.param(24, 8, 8, id="grid_m"),
        pytest.param(8, 24, 8, id="grid_n"),
        pytest.param(8, 8, 24, id="kloop_3"),
        pytest.param(16, 32, 48, id="grid_mn_kloop"),
    ],
)
def test_dot_f32_nonsquare(M, N, K):
    os.environ.setdefault(
        "TRITON_REPRODUCER_PATH", f"/tmp/dot-universal-{M}x{N}x{K}.mlir"
    )
    c, ref = _run(M, N, K, torch.float32)
    torch.testing.assert_close(c, ref, atol=1e-4, rtol=1e-4)


# AC2 masked-tail path: kernels whose static dot shape is still 8x8 but
# whose runtime (M, N) are not multiples of 8. Uses a `tl.store(c, acc,
# mask=(offs_m < M) & (offs_n < N))` so the compiler can extract M, N from
# the mask and route through the threadgroup-scratch + coop-loop predicated
# store. A/B sides are pre-padded to multiples of 8 so device loads stay
# in-bounds (the masked store discards the out-of-bounds rows/cols of acc).
@triton.jit
def dot_universal_masked_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    K_TILES: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, K_TILES):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask)


def _run_padded(M, N, K, *, seed=0xC0FFEE):
    """Allocate padded A/B/C so device loads of the tail tile stay in-bounds,
    then compare the unpadded slice against torch.matmul."""
    Mp = (M + 7) // 8 * 8
    Np = (N + 7) // 8 * 8
    torch.manual_seed(seed)
    a_pad = torch.zeros((Mp, K), dtype=torch.float32)
    b_pad = torch.zeros((K, Np), dtype=torch.float32)
    a_pad[:M, :] = torch.randn((M, K), dtype=torch.float32)
    b_pad[:, :N] = torch.randn((K, N), dtype=torch.float32)
    c_pad = torch.zeros((Mp, Np), dtype=torch.float32)
    K_TILES = K // 8
    dot_universal_masked_kernel[(Mp // 8, Np // 8)](
        a_pad, b_pad, c_pad,
        M, N,
        a_pad.stride(0), a_pad.stride(1),
        b_pad.stride(0), b_pad.stride(1),
        c_pad.stride(0), c_pad.stride(1),
        BLOCK_M=8, BLOCK_N=8, BLOCK_K=8, K_TILES=K_TILES,
    )
    ref = a_pad[:M, :].float() @ b_pad[:, :N].float()
    return c_pad[:M, :N].contiguous(), ref


@pytest.mark.parametrize(
    "M,N,K",
    [
        pytest.param(33,  8,  8, id="m_tail_only"),
        pytest.param( 8, 17,  8, id="n_tail_only"),
        pytest.param(33, 17, 16, id="both_tails_kloop_2"),
    ],
)
def test_dot_f32_mn_partial_tile(M, N, K):
    """AC2: runtime M/N not a multiple of 8. Masked tt.store triggers the
    threadgroup-scratch + coop-loop predicated emitter path."""
    os.environ.setdefault(
        "TRITON_REPRODUCER_PATH",
        f"/tmp/dot-partial-{M}x{N}x{K}.mlir",
    )
    c, ref = _run_padded(M, N, K)
    torch.testing.assert_close(c, ref, atol=1e-4, rtol=1e-4)


# ----------------------------------------------------------------------------
# AC3: fp16 / bf16 inputs with f32 accumulator
# ----------------------------------------------------------------------------
def test_dot_fp16():
    """fp16 inputs via the canonical-3-iter-arg path; threadgroup staging
    does an implicit `half -> float` cast inside the cooperative copy, so
    the simdgroup_float8x8 load from threadgroup is well-typed."""
    M = N = K = 32
    c, ref = _run(M, N, K, torch.float16)
    # K * 2^-10 = K * 9.77e-4
    tol = K * (2.0 ** -10)
    torch.testing.assert_close(c, ref, atol=tol, rtol=tol)


def test_dot_bf16_singletile():
    M = N = K = 8
    c, ref = _run(M, N, K, torch.bfloat16)
    tol = K * (2.0 ** -7)
    torch.testing.assert_close(c, ref, atol=tol, rtol=tol)


# ----------------------------------------------------------------------------
# AC5: K-loop tiled, K_TILES in {1, 2, 4, 8}
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("K_TILES", [1, 2, 4, 8])
def test_dot_kloop_tiled(K_TILES):
    """K-tiled accumulation with BLOCK_K=8, so K in {8, 16, 32, 64}. Verified
    by the existing 3-iter-arg matcher (plan-step path)."""
    M = N = 8
    K = 8 * K_TILES
    c, ref = _run(M, N, K, torch.float32, seed=0xC0FFEE + K_TILES)
    torch.testing.assert_close(c, ref, atol=1e-4, rtol=1e-4)


# ----------------------------------------------------------------------------
# AC4: multi-warp M/N tile-grid partition (Phase 2)
# ----------------------------------------------------------------------------
@triton.jit
def dot_multiwarp_kernel(
    a_ptr, b_ptr, c_ptr,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    K_TILES: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, K_TILES):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc)


def _run_multiwarp(num_warps, M=64, N=64, K=32, seed=0xAC4):
    torch.manual_seed(seed)
    a = torch.randn((M, K), dtype=torch.float32).contiguous()
    b = torch.randn((K, N), dtype=torch.float32).contiguous()
    c = torch.zeros((M, N), dtype=torch.float32).contiguous()
    dot_multiwarp_kernel[(1, 1)](
        a, b, c,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=M, BLOCK_N=N, BLOCK_K=8, K_TILES=K // 8,
        num_warps=num_warps,
    )
    ref = a.float() @ b.float()
    return c, ref


@pytest.mark.parametrize("num_warps", [1, 2, 4])
def test_dot_multiwarp(num_warps):
    """AC4: 64×64 dot with K_TILES=4, partitioned across `num_warps`
    SIMD-groups via `simdgroup_index_in_threadgroup`. factorWarps picks
    (warpsM, warpsN) maximizing min(warpsM, warpsN) under divisibility
    constraints: 1→(1,1), 2→(1,2)/(2,1), 4→(2,2)."""
    os.environ.setdefault(
        "TRITON_REPRODUCER_PATH",
        f"/tmp/dot-multiwarp-nw{num_warps}.mlir",
    )
    c, ref = _run_multiwarp(num_warps)
    torch.testing.assert_close(c, ref, atol=1e-5, rtol=1e-5)


def _capture_msl(num_warps):
    """Compile dot_multiwarp_kernel and return the emitted MSL source. The
    cache directory whose `dot_multiwarp_kernel.json` reports the matching
    `num_warps` is the only candidate, so this works even when multiple
    parametrize iterations co-exist in `~/.triton/cache`."""
    import json
    M, N, K = 64, 64, 32
    a = torch.zeros((M, K), dtype=torch.float32).contiguous()
    b = torch.zeros((K, N), dtype=torch.float32).contiguous()
    c = torch.zeros((M, N), dtype=torch.float32).contiguous()
    dot_multiwarp_kernel[(1, 1)](
        a, b, c,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=M, BLOCK_N=N, BLOCK_K=8, K_TILES=K // 8,
        num_warps=num_warps,
    )
    for root, _dirs, files in os.walk(os.path.expanduser("~/.triton/cache")):
        if "dot_multiwarp_kernel.json" not in files:
            continue
        if "dot_multiwarp_kernel.metal" not in files:
            continue
        with open(os.path.join(root, "dot_multiwarp_kernel.json")) as fh:
            if json.load(fh).get("num_warps") != num_warps:
                continue
        with open(os.path.join(root, "dot_multiwarp_kernel.metal")) as fh:
            return fh.read()
    return ""


@pytest.mark.parametrize("num_warps", [1, 2, 4])
def test_dot_multiwarp_msl_predicates(num_warps):
    """AC4-S5: the emitted MSL contains exactly
    `(8/warpsM) * (8/warpsN) * K_TILES = 256/num_warps` occurrences of
    `simdgroup_multiply_accumulate(`. For num_warps>1 the MSL also has
    `simdgroup_index_in_threadgroup` (the kernel parameter attribute) and
    per-warp slice indexing via `_stage_shared[sgid]`."""
    msl = _capture_msl(num_warps)
    assert msl, "no MSL captured (cache miss)"
    expected_mma = 256 // num_warps
    mma_count = msl.count("simdgroup_multiply_accumulate(")
    assert mma_count == expected_mma, (
        f"num_warps={num_warps}: mma count {mma_count} != expected {expected_mma}"
    )
    if num_warps > 1:
        assert "simdgroup_index_in_threadgroup" in msl, (
            "multi-warp MSL missing simdgroup_index_in_threadgroup parameter")
        assert "_stage_shared[sgid]" in msl, (
            "multi-warp MSL missing per-warp stage buffer slice _stage_shared[sgid]")
    else:
        assert "_stage_shared[" in msl
        # Single-warp Branch A: shared buffer is [elems], not [num_warps][elems].
        assert "_stage_shared[sgid]" not in msl, (
            "single-warp MSL should not slice _stage_shared by sgid")
