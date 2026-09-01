"""General GEMM with fused epilogue on the Metal backend.

Runs the verbatim `python/test/unit/fixtures/metal_leet/medium-general_matrix_multiplication.py` kernel:
a single tiled `tt.dot` (TILE=64) feeding an `alpha*(A@B) + beta*C` epilogue,
fp16 inputs / f32 accumulate / fp16 output, with masked partial tiles.

This shape matches no SIMD-group matmul matcher (they all fuse `dot -> tt.store`
with no epilogue); it is lowered by the per-thread scalar `metal.scalar_dot`
correctness fallback and bridged into the ordinary tile-loop epilogue. See
`tryScalarDotFallback` / `ScalarDotLowering` in TritonGPUToMetal.cpp.
"""
from __future__ import annotations

import importlib.util
import types
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


# Verbatim copy of python/test/unit/fixtures/metal_leet/medium-general_matrix_multiplication.py.
@triton.jit
def kernel(a, b, c, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
           alpha: tl.constexpr, beta: tl.constexpr, TILE_SIZE: tl.constexpr):
    bx = tl.program_id(0)
    by = tl.program_id(1)

    ar = tl.arange(0, TILE_SIZE)

    row = by * TILE_SIZE
    col = bx * TILE_SIZE

    iters = tl.cdiv(K, TILE_SIZE)
    output = (ar[:, None] * ar[None, :]) * 0.0
    output = tl.cast(output, tl.float32)
    ay_off = ar[:, None] + row
    bx_off = ar[None, :] + col

    ay_off_mask = ay_off < M
    bx_off_mask = bx_off < N

    for i in range(iters):
        ax_off = ar[None, :] + (i * TILE_SIZE)
        by_off = ar[:, None] + (i * TILE_SIZE)
        adata = tl.load(a + ay_off * K + ax_off, mask=(ax_off < K) & ay_off_mask, other=0.0)
        bdata = tl.load(b + by_off * N + bx_off, mask=bx_off_mask & (by_off < K), other=0.0)
        output = tl.dot(tl.cast(adata, tl.float32), tl.cast(bdata, tl.float32), acc=output)

    c_offset = c + ay_off * N + bx_off
    c_mask = ay_off_mask & bx_off_mask
    cdata = tl.load(c_offset, mask=c_mask, other=0.0)
    output = output * alpha + tl.cast(cdata, tl.float32) * beta
    output = tl.cast(output, tl.float16)
    tl.store(c_offset, output, mask=c_mask)


def _solve(a, b, c, M, N, K, alpha, beta):
    TILE_SIZE = 64
    grid = (triton.cdiv(N, TILE_SIZE), triton.cdiv(M, TILE_SIZE))
    kernel[grid](a, b, c, M=M, N=N, K=K, alpha=alpha, beta=beta, TILE_SIZE=TILE_SIZE)


def _run(M, N, K, alpha, beta, *, seed=0):
    torch.manual_seed(seed)
    a = torch.randn((M, K), dtype=torch.float16).contiguous()
    b = torch.randn((K, N), dtype=torch.float16).contiguous()
    c = torch.randn((M, N), dtype=torch.float16).contiguous()
    c_orig = c.clone()
    _solve(a, b, c, M, N, K, alpha, beta)
    ref = (alpha * (a.float() @ b.float()) + beta * c_orig.float()).to(torch.float16)
    return c, ref


@pytest.mark.parametrize(
    "M,N,K,alpha,beta",
    [
        pytest.param(64, 64, 64, 1.0, 0.0, id="square_ab"),
        pytest.param(64, 64, 64, 1.0, 1.0, id="square_gemm"),
        pytest.param(64, 64, 64, 2.0, 0.5, id="square_scaled"),
        pytest.param(64, 64, 32, 1.0, 1.0, id="k_lt_tile"),
        pytest.param(128, 128, 64, 1.0, 1.0, id="multi_tile"),
        pytest.param(128, 96, 64, 1.5, 0.5, id="nonsquare_n_tail"),
        pytest.param(100, 70, 50, 1.0, 1.0, id="ragged_all"),
    ],
)
def test_metal_general_matmul(M, N, K, alpha, beta):
    c, ref = _run(M, N, K, alpha, beta)
    # fp16 output; tolerance scales with K (accumulation error) + fp16 rounding.
    tol = max(1e-2, K * (2.0 ** -9))
    torch.testing.assert_close(c.float(), ref.float(), atol=tol, rtol=tol)


# Verbatim kernel from python/test/unit/fixtures/metal_leet/medium-int8_quantized_matmul.py.  This is a
# deliberately separate path from the generic fp16 GEMM above: the loop carries
# the int32 dot accumulator, two zero-point correction sums, and both load
# offset/mask pairs.
@triton.jit
def int8_quant_matmul_kernel(a_ptr, b_ptr, c_ptr,
                             M, N, K,
                             scale_A, scale_B, scale_C,
                             zero_point_A, zero_point_B, zero_point_C,
                             BLOCK_SIZE_M: tl.constexpr,
                             BLOCK_SIZE_N: tl.constexpr,
                             BLOCK_SIZE_K: tl.constexpr,
                             GROUPSIZE: tl.constexpr):
    hw_pid0 = tl.program_id(0)
    hw_pid1 = tl.program_id(1)

    num_programs_pid0 = tl.num_programs(0)
    num_programs_pid1 = tl.num_programs(1)

    pid0, pid1 = tl.swizzle2d(hw_pid0, hw_pid1, num_programs_pid0,
                              num_programs_pid1, GROUPSIZE)

    offset_M = pid0 * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offset_N = pid1 * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offset_K = tl.arange(0, BLOCK_SIZE_K)

    mask_M = offset_M < M
    mask_N = offset_N < N
    mask_K = offset_K < K

    offset_A = offset_M[:, None] * K + offset_K[None, :]
    mask_A = mask_M[:, None] & mask_K[None, :]

    offset_B = offset_K[:, None] * N + offset_N[None, :]
    mask_B = mask_K[:, None] & mask_N[None, :]

    accumulator = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.int32)
    accumulator_sum_A = tl.zeros([BLOCK_SIZE_M], dtype=tl.int32)
    accumulator_sum_B = tl.zeros([BLOCK_SIZE_N], dtype=tl.int32)

    scale_AB = scale_A * scale_B / scale_C

    for current_k_index in range(0, K, BLOCK_SIZE_K):
        if current_k_index + BLOCK_SIZE_K >= K:
            current_mask_K = (offset_K + current_k_index) < K
            mask_A = mask_M[:, None] & current_mask_K[None, :]
            mask_B = current_mask_K[:, None] & mask_N[None, :]

        data_A = tl.load(a_ptr + offset_A, mask=mask_A).to(tl.float32)
        data_B = tl.load(b_ptr + offset_B, mask=mask_B).to(tl.float32)

        accumulator += tl.dot(data_A, data_B).to(tl.int32)
        accumulator_sum_A += data_A.sum(1).to(tl.int32)
        accumulator_sum_B += data_B.sum(0).to(tl.int32)

        offset_A += BLOCK_SIZE_K
        offset_B += BLOCK_SIZE_K * N

    result = accumulator - (accumulator_sum_A[:, None] * zero_point_B) \
        - (accumulator_sum_B[None, :] * zero_point_A) \
        + (K * zero_point_A * zero_point_B)
    result = result.to(tl.float32) * scale_AB
    result = tl.floor(result + 0.5) + zero_point_C
    result = tl.clamp(result, -128, 127)

    offset_C = offset_M[:, None] * N + offset_N[None, :]
    mask_C = mask_M[:, None] & mask_N[None, :]

    tl.store(c_ptr + offset_C, result.to(tl.int8), mask=mask_C)


def _solve_int8_quantized(a, b, c, M, N, K, scale_A, scale_B, scale_C,
                          zero_point_A, zero_point_B, zero_point_C):
    block = 64
    grid = (triton.cdiv(M, block), triton.cdiv(N, block))
    int8_quant_matmul_kernel[grid](
        a, b, c, M, N, K, scale_A, scale_B, scale_C,
        zero_point_A, zero_point_B, zero_point_C,
        block, block, block, 8,
    )


def _clone_jit(fn, name):
    raw = fn.fn
    cloned = types.FunctionType(
        raw.__code__, raw.__globals__, name, raw.__defaults__, raw.__closure__
    )
    cloned.__annotations__ = dict(getattr(raw, "__annotations__", {}))
    cloned.__kwdefaults__ = getattr(raw, "__kwdefaults__", None)
    cloned.__module__ = raw.__module__
    return triton.jit(cloned)


renamed_int8_quant_matmul = _clone_jit(
    int8_quant_matmul_kernel, "renamed_int8_quant_matmul"
)


def _solve_renamed_int8_quantized(a, b, c, M, N, K, scale_A, scale_B, scale_C,
                                  zero_point_A, zero_point_B, zero_point_C):
    block = 64
    grid = (triton.cdiv(M, block), triton.cdiv(N, block))
    renamed_int8_quant_matmul[grid](
        a, b, c, M, N, K, scale_A, scale_B, scale_C,
        zero_point_A, zero_point_B, zero_point_C,
        block, block, block, 8,
    )


def _int8_quantized_reference(a, b, scale_A, scale_B, scale_C,
                              zero_point_A, zero_point_B, zero_point_C):
    a_i32 = a.to(torch.int32)
    b_i32 = b.to(torch.int32)
    k = a.shape[1]
    corrected = a_i32 @ b_i32
    corrected -= a_i32.sum(dim=1, keepdim=True) * zero_point_B
    corrected -= b_i32.sum(dim=0, keepdim=True) * zero_point_A
    corrected += k * zero_point_A * zero_point_B
    result = torch.floor(
        corrected.float() * (scale_A * scale_B / scale_C) + 0.5
    ) + zero_point_C
    return result.clamp(-128, 127).to(torch.int8)


def _compile_int8_quantized_metal(kernel, *, block_k=64):
    M = N = K = 64
    a = torch.empty((M, K), dtype=torch.int8, device="mps")
    b = torch.empty((K, N), dtype=torch.int8, device="mps")
    c = torch.empty((M, N), dtype=torch.int8, device="mps")
    compiled = kernel.warmup(
        a, b, c, M, N, K, 0.03125, 0.0625, 0.015625, -3, 5, -7,
        64, 64, block_k, 8, grid=(1, 1), num_warps=4,
    )
    metal = compiled.asm["metal"]
    return metal.decode() if isinstance(metal, bytes) else metal


@pytest.mark.parametrize(
    "M,N,K",
    [
        pytest.param(64, 64, 64, id="single_tile"),
        pytest.param(70, 66, 96, id="ragged_multi_tile"),
    ],
)
def test_metal_int8_quantized_matmul(M, N, K):
    scale_A, scale_B, scale_C = 0.03125, 0.0625, 0.015625
    zero_point_A, zero_point_B, zero_point_C = -3, 5, -7
    torch.manual_seed(0x18)
    a_cpu = torch.randint(-16, 17, (M, K), dtype=torch.int8)
    b_cpu = torch.randint(-16, 17, (K, N), dtype=torch.int8)
    a = a_cpu.to("mps")
    b = b_cpu.to("mps")
    c = torch.empty((M, N), dtype=torch.int8, device="mps")

    _solve_int8_quantized(
        a, b, c, M, N, K, scale_A, scale_B, scale_C,
        zero_point_A, zero_point_B, zero_point_C,
    )
    torch.mps.synchronize()

    ref = _int8_quantized_reference(
        a_cpu, b_cpu, scale_A, scale_B, scale_C,
        zero_point_A, zero_point_B, zero_point_C,
    )
    torch.testing.assert_close(c.cpu(), ref, atol=0, rtol=0)


def test_metal_int8_quantized_matmul_renamed_kernel():
    M, N, K = 64, 64, 64
    scale_A, scale_B, scale_C = 0.03125, 0.0625, 0.015625
    zero_point_A, zero_point_B, zero_point_C = -3, 5, -7
    torch.manual_seed(0x1818)
    a_cpu = torch.randint(-16, 17, (M, K), dtype=torch.int8)
    b_cpu = torch.randint(-16, 17, (K, N), dtype=torch.int8)
    a = a_cpu.to("mps")
    b = b_cpu.to("mps")
    c = torch.empty((M, N), dtype=torch.int8, device="mps")

    _solve_renamed_int8_quantized(
        a, b, c, M, N, K, scale_A, scale_B, scale_C,
        zero_point_A, zero_point_B, zero_point_C,
    )
    torch.mps.synchronize()

    ref = _int8_quantized_reference(
        a_cpu, b_cpu, scale_A, scale_B, scale_C,
        zero_point_A, zero_point_B, zero_point_C,
    )
    torch.testing.assert_close(c.cpu(), ref, atol=0, rtol=0)


def test_metal_int8_quantized_matmul_matcher_accepts_renamed_kernel():
    metal = _compile_int8_quantized_metal(renamed_int8_quant_matmul)
    assert "metal.int8_quantized_matmul scalar fallback" in metal


def test_metal_int8_quantized_matmul_matcher_rejects_adjacent_shape():
    marker = "metal.int8_quantized_matmul scalar fallback"
    try:
        metal = _compile_int8_quantized_metal(
            int8_quant_matmul_kernel, block_k=32
        )
    except RuntimeError as error:
        assert "convert-tritongpu-to-metal failed" in str(error)
    else:
        assert marker not in metal


def _load_leet_2d_fft():
    path = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "metal_leet"
        / "medium-2d_fft.py"
    )
    if not path.exists():
        pytest.skip(f"Metal Leet fixture not present: {path}")
    spec = importlib.util.spec_from_file_location("leet_medium_2d_fft", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "M,N",
    [
        pytest.param(32, 32, id="square"),
        pytest.param(35, 41, id="ragged"),
    ],
)
def test_metal_leet_2d_fft_canonical_dot(M, N):
    """Regression for the leet FFT canonical complex-matmul formulation."""
    leet_fft = _load_leet_2d_fft()
    torch.manual_seed(0x2DFF + M * 3 + N)
    signal_cpu = torch.randn((M, N, 2), dtype=torch.float32)
    signal = signal_cpu.to("mps").contiguous()
    spectrum = torch.empty_like(signal)

    leet_fft.solve(signal, spectrum, M, N)
    torch.mps.synchronize()

    ref = torch.view_as_real(
        torch.fft.fft2(torch.view_as_complex(signal_cpu.contiguous()))
    ).contiguous()
    torch.testing.assert_close(spectrum.cpu(), ref, atol=1e-2, rtol=1e-3)
