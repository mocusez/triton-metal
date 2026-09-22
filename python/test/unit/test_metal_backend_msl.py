"""End-to-end: @triton.jit -> TTIR -> TTGIR -> MSL text.

Acceptance test for the implementation notes
(AC.J1, AC.J2, AC.J3). Runs the Metal backend in-process and asserts
that the canonical unmasked vector_add lowers to MSL with the expected
substrings (same shape as the `vector_add_unmasked.mlir` lit fixture).

Skipped automatically when the metal backend's pybind module isn't
linked (e.g. a Linux build without the Metal plugin).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import re

import pytest

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource


libmetal = pytest.importorskip(
    "triton._C.libtriton.metal",
    reason="Metal backend pybind module not built into libtriton",
)

LEET_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "metal_leet"
HARD_FFT_PATH = LEET_FIXTURES / "hard-fast_fourier_transform.py"
BFS_PATH = LEET_FIXTURES / "hard-bfs_shortest_path.py"
FUSED_RMS_NORM_PATH = (
    LEET_FIXTURES / "medium-fused_residual_add_and_rms_norm.py"
)
FLASH_ATTENTION_TEST_PATH = Path(__file__).resolve().parent / (
    "test_metal_backend_flash_attention.py"
)


def _load_exact_module(path: Path, name: str):
    assert path.is_file(), f"expected exact target file to exist: {path}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_flash_attention_test_module_for_compile_only(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    return _load_exact_module(
        FLASH_ATTENTION_TEST_PATH, "metal_flash_attention_compile_only")


def _compile_to_msl(fn, signature, constexprs, *, num_warps, attrs=None):
    src = ASTSource(
        fn=fn, signature=signature, constexprs=constexprs, attrs=attrs)
    target = GPUTarget(backend="metal", arch=9, warp_size=32)
    compiled = triton.compile(src, target=target, options={"num_warps": num_warps})
    assert "metal" in compiled.asm, (
        f"expected 'metal' stage artifact; got: {sorted(compiled.asm.keys())}"
    )
    raw = compiled.asm["metal"]
    msl = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    assert msl, "MSL output is empty"
    return msl


def _compile_backward_pre_to_msl(fn):
    return _compile_to_msl(
        fn,
        {
            "Q": "*fp32", "KT": "*fp32", "VT": "*fp32", "dO": "*fp32",
            "S": "*fp32", "ST": "*fp32", "DP": "*fp32", "DPT": "*fp32",
            "Mp": "*fp32", "Lp": "*fp32", "Deltap": "*fp32",
            "stride_qm": "i32", "stride_kt": "i32", "stride_vt": "i32",
            "stride_dom": "i32", "stride_sm": "i32", "stride_stn": "i32",
            "stride_dpm": "i32", "stride_dptn": "i32", "M": "i32",
            "N_true": "i32", "D": "i32", "sm_scale": "fp32",
        },
        {"BLOCK_M": 16, "BLOCK_N": 32, "BLOCK_D": 32},
        num_warps=8,
        attrs={"N_true": {"do_not_specialize": True}},
    )


@triton.jit
def add_kernel_unmasked(
    x_ptr,
    y_ptr,
    output_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets)
    y = tl.load(y_ptr + offsets)
    tl.store(output_ptr + offsets, x + y)


def test_backward_pre_causal_mask_compiles_to_msl(monkeypatch):
    flash_attention = _load_flash_attention_test_module_for_compile_only(
        monkeypatch)
    msl = _compile_backward_pre_to_msl(
        flash_attention._causal_backward_pre_kernel())
    assert "metal.attention_backward_pre computed-tile simdgroup MMA" in msl
    assert "float score =" in msl
    assert "-INFINITY" in msl
    assert "<=" in msl
    assert "&&" in msl


def test_backward_pre_shifted_score_region_compiles_to_msl(monkeypatch):
    flash_attention = _load_flash_attention_test_module_for_compile_only(
        monkeypatch)
    msl = _compile_backward_pre_to_msl(
        flash_attention._shifted_score_backward_pre_kernel())
    assert "metal.attention_backward_pre computed-tile simdgroup MMA" in msl
    assert "float score =" in msl
    assert "-INFINITY" in msl
    assert re.search(
        r"int32_t v\d+ = \(v\d+\) \+ \(v\d+\);\n"
        r"\s*int32_t v\d+ = int32_t\(v\d+\);\n"
        r"\s*bool v\d+ = \(v\d+\) < \(v\d+\);",
        msl,
    )


@pytest.mark.parametrize(
    "factory_name",
    [
        "_wrong_recurrence_backward_pre_kernel",
    ],
)
def test_backward_pre_near_misses_still_reject_compile_only(
        monkeypatch, factory_name):
    flash_attention = _load_flash_attention_test_module_for_compile_only(
        monkeypatch)
    with pytest.raises(RuntimeError, match="convert-tritongpu-to-metal failed"):
        _compile_backward_pre_to_msl(getattr(flash_attention, factory_name)())


@triton.jit
def dot_scaled_e8m0_bf16_kernel(
    a_base,
    b_base,
    a_scale_base,
    b_scale_base,
    output_base,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a = tl.load(a_base + offs_m[:, None] * BLOCK_K + offs_k[None, :])
    b = tl.load(b_base + offs_k[:, None] * BLOCK_N + offs_n[None, :])

    SCALE_K: tl.constexpr = BLOCK_K // 32
    offs_scale_k = tl.arange(0, SCALE_K)
    a_scale = tl.load(a_scale_base + offs_m[:, None] * SCALE_K + offs_scale_k[None, :])
    b_scale = tl.load(b_scale_base + offs_n[:, None] * SCALE_K + offs_scale_k[None, :])

    result = tl.dot_scaled(
        a,
        a_scale,
        "bf16",
        b,
        b_scale,
        "bf16",
        fast_math=False,
    )
    tl.store(output_base + offs_m[:, None] * BLOCK_N + offs_n[None, :], result)


@triton.jit
def dot_scaled_e8m0_e5m2_kernel(
    a_base,
    b_base,
    a_scale_base,
    b_scale_base,
    output_base,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a = tl.load(a_base + offs_m[:, None] * BLOCK_K + offs_k[None, :])
    b = tl.load(b_base + offs_k[:, None] * BLOCK_N + offs_n[None, :])

    SCALE_K: tl.constexpr = BLOCK_K // 32
    offs_scale_k = tl.arange(0, SCALE_K)
    a_scale = tl.load(a_scale_base + offs_m[:, None] * SCALE_K + offs_scale_k[None, :])
    b_scale = tl.load(b_scale_base + offs_n[:, None] * SCALE_K + offs_scale_k[None, :])

    result = tl.dot_scaled(
        a,
        a_scale,
        "e5m2",
        b,
        b_scale,
        "e5m2",
        fast_math=False,
    )
    tl.store(output_base + offs_m[:, None] * BLOCK_N + offs_n[None, :], result)


@triton.jit
def dot_scaled_e8m0_e4m3_kernel(
    a_base,
    b_base,
    a_scale_base,
    b_scale_base,
    output_base,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a = tl.load(a_base + offs_m[:, None] * BLOCK_K + offs_k[None, :])
    b = tl.load(b_base + offs_k[:, None] * BLOCK_N + offs_n[None, :])

    SCALE_K: tl.constexpr = BLOCK_K // 32
    offs_scale_k = tl.arange(0, SCALE_K)
    a_scale = tl.load(a_scale_base + offs_m[:, None] * SCALE_K + offs_scale_k[None, :])
    b_scale = tl.load(b_scale_base + offs_n[:, None] * SCALE_K + offs_scale_k[None, :])

    result = tl.dot_scaled(
        a,
        a_scale,
        "e4m3",
        b,
        b_scale,
        "e4m3",
        fast_math=False,
    )
    tl.store(output_base + offs_m[:, None] * BLOCK_N + offs_n[None, :], result)


@triton.jit
def dot_scaled_e8m0_e2m1_kernel(
    a_base,
    b_base,
    a_scale_base,
    b_scale_base,
    output_base,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    LHS_K_PACK: tl.constexpr,
    RHS_K_PACK: tl.constexpr,
):
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    if LHS_K_PACK:
        LHS_PACKED_K: tl.constexpr = BLOCK_K // 2
        offs_lhs_packed_k = tl.arange(0, LHS_PACKED_K)
        a = tl.load(a_base + offs_m[:, None] * LHS_PACKED_K + offs_lhs_packed_k[None, :])
    else:
        PACKED_M: tl.constexpr = BLOCK_M // 2
        offs_packed_m = tl.arange(0, PACKED_M)
        a = tl.load(a_base + offs_packed_m[:, None] * BLOCK_K + offs_k[None, :])
    if RHS_K_PACK:
        RHS_PACKED_K: tl.constexpr = BLOCK_K // 2
        offs_rhs_packed_k = tl.arange(0, RHS_PACKED_K)
        b = tl.load(b_base + offs_rhs_packed_k[:, None] * BLOCK_N + offs_n[None, :])
    else:
        PACKED_N: tl.constexpr = BLOCK_N // 2
        offs_packed_n = tl.arange(0, PACKED_N)
        b = tl.load(b_base + offs_k[:, None] * PACKED_N + offs_packed_n[None, :])

    SCALE_K: tl.constexpr = BLOCK_K // 32
    offs_scale_k = tl.arange(0, SCALE_K)
    a_scale = tl.load(a_scale_base + offs_m[:, None] * SCALE_K + offs_scale_k[None, :])
    b_scale = tl.load(b_scale_base + offs_n[:, None] * SCALE_K + offs_scale_k[None, :])

    result = tl.dot_scaled(
        a,
        a_scale,
        "e2m1",
        b,
        b_scale,
        "e2m1",
        fast_math=False,
        lhs_k_pack=LHS_K_PACK,
        rhs_k_pack=RHS_K_PACK,
    )
    tl.store(output_base + offs_m[:, None] * BLOCK_N + offs_n[None, :], result)


@triton.jit
def dot_scaled_e4m3_extra_use_unsupported_kernel(
    a_base,
    b_base,
    a_scale_base,
    b_scale_base,
    output_base,
    scratch_base,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_offsets = offs_m[:, None] * BLOCK_K + offs_k[None, :]
    b_offsets = offs_k[:, None] * BLOCK_N + offs_n[None, :]
    a_raw = tl.load(a_base + a_offsets)
    b_raw = tl.load(b_base + b_offsets)
    a = a_raw.to(tl.float8e4nv, bitcast=True)
    b = b_raw.to(tl.float8e4nv, bitcast=True)
    a_roundtrip = a.to(tl.uint8, bitcast=True)

    a_scale = tl.load(a_scale_base + offs_m[:, None])
    b_scale = tl.load(b_scale_base + offs_n[:, None])
    result = tl.dot_scaled(a, a_scale, "e4m3", b, b_scale, "e4m3", fast_math=False)
    tl.store(output_base + offs_m[:, None] * BLOCK_N + offs_n[None, :], result)
    tl.store(scratch_base + a_offsets, a_roundtrip)


@triton.jit
def fp8e5_copy_unsupported_kernel(
    input_base,
    output_base,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    value = tl.load(input_base + offsets)
    tl.store(output_base + offsets, value)


@triton.jit
def fp8e4_copy_unsupported_kernel(
    input_base,
    output_base,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    value = tl.load(input_base + offsets)
    tl.store(output_base + offsets, value)


def test_triton_jit_compiles_to_msl():
    # BLOCK_SIZE=128 keeps 1 element per thread under num_warps=4 *
    # warp_size=32; that lines up with the threads-per-block invariant
    # that the existing TritonGPUToMetal lowering assumes (see
    # the implementation notes for the
    # invariant).
    signature = {
        "x_ptr": "*fp32",
        "y_ptr": "*fp32",
        "output_ptr": "*fp32",
        "BLOCK_SIZE": "constexpr",
    }
    src = ASTSource(
        fn=add_kernel_unmasked,
        signature=signature,
        constexprs={"BLOCK_SIZE": 128},
    )
    target = GPUTarget(backend="metal", arch=80, warp_size=32)
    compiled = triton.compile(src, target=target, options={"num_warps": 4})

    assert "metal" in compiled.asm, (
        f"expected 'metal' stage artifact; got: {sorted(compiled.asm.keys())}"
    )
    # `binary_ext="metal"` makes the compiler harness read the terminal
    # artifact as bytes (compiler.py reads files matching the binary_ext
    # as binary). MSL is textual; decode for the substring assertions.
    raw = compiled.asm["metal"]
    msl = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    assert msl, "MSL output is empty"

    # Match the same shape the vector_add_unmasked.mlir lit fixture
    # pins. If any of these is missing the C++ lowering regressed.
    # Post-Lmultiload-Phase-C the 1D canonical short-circuit is gone, so
    # the per-thread offset is the arithmetic-explicit
    # `pid*BLOCK + (id.x - pid*tpb)` form rather than the collapsed
    # `id.x`. The substring checks pin the new shape.
    for needle in (
        "kernel void",
        "device float",
        "thread_position_in_grid",
        "id.x",
        "(id.x - (tgid.x * 128))",
        "(tgid.x * 128) + (id.x - (tgid.x * 128))",
    ):
        assert needle in msl, f"MSL output missing required substring {needle!r}.\n--- MSL ---\n{msl}\n"


def test_bfs_shortest_path_exact_kernel_compiles_to_msl():
    module = _load_exact_module(BFS_PATH, "leet_bfs_shortest_path_msl")
    msl = _compile_to_msl(
        module._bfs_expand_kernel,
        {
            "current_frontier_ptr": "*i32",
            "next_frontier_ptr": "*i32",
            "grid_ptr": "*i32",
            "visited_ptr": "*i32",
            "counters_ptr": "*i32",
            "current_count": "i32",
            "cols": "i32",
            "rows": "i32",
            "end_idx": "i32",
            "BLOCK_SIZE": "constexpr",
        },
        {"BLOCK_SIZE": 128},
        num_warps=4,
    )

    for needle in (
        "kernel void",
        "atomic_exchange_explicit",
        "atomic_fetch_add_explicit",
        "atomic_fetch_or_explicit",
    ):
        assert needle in msl, f"BFS MSL missing required substring {needle!r}.\n--- MSL ---\n{msl}\n"
    assert msl.count("atomic_exchange_explicit") == 4, (
        "BFS cumsum lowering must consume the old-value scalar returned by each "
        "atomic exchange, not replay the exchange while filling the scan buffer."
    )
    assert "tt.scan" not in msl


def test_fused_residual_rmsnorm_exact_kernels_compile_to_msl():
    module = _load_exact_module(FUSED_RMS_NORM_PATH, "leet_fused_rms_norm_msl")
    signature = {
        "x_ptr": "*fp32",
        "res_ptr": "*fp32",
        "w_ptr": "*fp32",
        "scale_ptr": "*fp32",
        "out_ptr": "*fp32",
        "C": "i32",
        "eps": "fp32",
        "BLOCK": "constexpr",
    }

    single_msl = _compile_to_msl(
        module._add_rmsnorm_fused_kernel,
        signature,
        {"BLOCK": 8192},
        num_warps=16,
    )
    loop_msl = _compile_to_msl(
        module._add_rmsnorm_fused_loop_kernel,
        signature,
        {"BLOCK": 4096},
        num_warps=8,
    )

    for name, msl in (("single", single_msl), ("loop", loop_msl)):
        assert "kernel void" in msl, f"{name} RMSNorm MSL missing kernel entry.\n--- MSL ---\n{msl}\n"
        assert "sqrt" in msl, f"{name} RMSNorm MSL missing sqrt.\n--- MSL ---\n{msl}\n"


def test_hard_fast_fourier_transform_exact_kernel_compiles_to_msl():
    module = _load_exact_module(HARD_FFT_PATH, "leet_hard_fft_msl")
    msl = _compile_to_msl(
        module._dft_naive,
        {"x_ptr": "*fp32", "y_ptr": "*fp32", "N": "i32"},
        {"KB": 64, "NB": 64},
        num_warps=4,
    )

    for needle in ("kernel void", "cos", "sin"):
        assert needle in msl, f"hard FFT MSL missing required substring {needle!r}.\n--- MSL ---\n{msl}\n"


def test_bfs_shortest_path_host_contract_rejects_invalid_metadata():
    torch = pytest.importorskip("torch")
    module = _load_exact_module(BFS_PATH, "leet_bfs_shortest_path_contract")

    def check_raises(grid, result, rows, cols, start=(0, 0), end=(1, 2)):
        with pytest.raises(ValueError):
            module.solve(
                grid,
                result,
                rows,
                cols,
                start[0],
                start[1],
                end[0],
                end[1],
            )

    grid = torch.zeros((2, 3), dtype=torch.int32)
    result = torch.empty((1,), dtype=torch.int32)

    check_raises(grid, result, 0, 3)
    check_raises(grid, result, 3, 3)
    check_raises(grid.float(), result, 2, 3)
    check_raises(grid, result.float(), 2, 3)
    check_raises(grid, torch.empty((0,), dtype=torch.int32), 2, 3)
    check_raises(grid, torch.empty((2, 2), dtype=torch.int32)[:, 0], 2, 3)
    check_raises(grid, torch.empty((1,), dtype=torch.int32, device="meta"), 2, 3)
    check_raises(grid, result, 2, 3, start=(-1, 0))
    check_raises(grid, result, 2, 3, end=(2, 0))

    huge_rows = 46341
    huge_cols = 46341
    huge_grid = torch.empty((huge_rows, huge_cols), dtype=torch.int32, device="meta")
    huge_result = torch.empty((1,), dtype=torch.int32, device="meta")
    check_raises(huge_grid, huge_result, huge_rows, huge_cols, end=(0, 1))


def test_fused_residual_rmsnorm_host_contract_rejects_invalid_metadata():
    torch = pytest.importorskip("torch")
    module = _load_exact_module(FUSED_RMS_NORM_PATH, "leet_fused_rms_norm_contract")

    def check_raises(x, residual, weight, out, N=2, C=3, eps=1e-5):
        with pytest.raises(ValueError):
            module.solve(x, residual, weight, out, N, C, eps)

    x = torch.zeros((2, 3), dtype=torch.float32)
    residual = torch.zeros((2, 3), dtype=torch.float32)
    weight = torch.ones((3,), dtype=torch.float32)
    out = torch.empty((2, 3), dtype=torch.float32)

    check_raises(x, residual, weight, out, N=0)
    check_raises(x, residual, weight, out, C=0)
    check_raises(x, residual, weight, out, N=2.0)
    check_raises(torch.zeros((2, 4), dtype=torch.float32), residual, weight, out)
    check_raises(x, torch.zeros((1, 3), dtype=torch.float32), weight, out)
    check_raises(x, residual, torch.ones((4,), dtype=torch.float32), out)
    check_raises(x, residual, weight, torch.empty((1, 3), dtype=torch.float32))
    check_raises(x, residual, weight.to(torch.float16), out)
    check_raises(x.to(torch.int32), residual.to(torch.int32), weight.to(torch.int32), out.to(torch.int32))
    check_raises(x, residual, weight, out, eps=0.0)
    check_raises(x, residual, weight, out, eps=float("nan"))
    check_raises(x, residual, weight, out, eps="not-a-number")
    check_raises(x, residual, weight, out.t())
    check_raises(x, residual, weight, torch.empty((2, 3), dtype=torch.float32, device="meta"))


def test_dot_scaled_e8m0_bf16_compiles_to_msl():
    src = ASTSource(
        fn=dot_scaled_e8m0_bf16_kernel,
        signature={
            "a_base": "*bf16",
            "b_base": "*bf16",
            "a_scale_base": "*u8",
            "b_scale_base": "*u8",
            "output_base": "*fp32",
            "BLOCK_M": "constexpr",
            "BLOCK_N": "constexpr",
            "BLOCK_K": "constexpr",
        },
        constexprs={"BLOCK_M": 16, "BLOCK_N": 16, "BLOCK_K": 32},
    )
    target = GPUTarget(backend="metal", arch=80, warp_size=32)
    compiled = triton.compile(src, target=target, options={"num_warps": 4})

    raw = compiled.asm["metal"]
    msl = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    for needle in (
        "as_type<bfloat>",
        "bfloat(NAN)",
        "& 255",
        "== (int16_t)(255)",
        "/ (uint32_t)(32)",
    ):
        assert needle in msl, f"scaled-dot MSL output missing required substring {needle!r}.\n--- MSL ---\n{msl}\n"


def test_dot_scaled_e8m0_e5m2_compiles_to_msl():
    src = ASTSource(
        fn=dot_scaled_e8m0_e5m2_kernel,
        signature={
            "a_base": "*fp8e5",
            "b_base": "*fp8e5",
            "a_scale_base": "*u8",
            "b_scale_base": "*u8",
            "output_base": "*fp32",
            "BLOCK_M": "constexpr",
            "BLOCK_N": "constexpr",
            "BLOCK_K": "constexpr",
        },
        constexprs={"BLOCK_M": 16, "BLOCK_N": 16, "BLOCK_K": 32},
    )
    target = GPUTarget(backend="metal", arch=80, warp_size=32)
    compiled = triton.compile(src, target=target, options={"num_warps": 4})

    raw = compiled.asm["metal"]
    msl = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    for needle in (
        "as_type<half>",
        "<< 8",
        "bfloat",
        "& 255",
        "/ (uint32_t)(32)",
    ):
        assert needle in msl, f"E5M2 scaled-dot MSL output missing required substring {needle!r}.\n--- MSL ---\n{msl}\n"


def test_dot_scaled_e8m0_e4m3_compiles_to_msl():
    target = GPUTarget(backend="metal", arch=80, warp_size=32)
    for payload_type in ("*fp8e4nv", "*u8"):
        src = ASTSource(
            fn=dot_scaled_e8m0_e4m3_kernel,
            signature={
                "a_base": payload_type,
                "b_base": payload_type,
                "a_scale_base": "*u8",
                "b_scale_base": "*u8",
                "output_base": "*fp32",
                "BLOCK_M": "constexpr",
                "BLOCK_N": "constexpr",
                "BLOCK_K": "constexpr",
            },
            constexprs={"BLOCK_M": 16, "BLOCK_N": 16, "BLOCK_K": 32},
        )
        compiled = triton.compile(src, target=target, options={"num_warps": 4})

        raw = compiled.asm["metal"]
        msl = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        for needle in (
            "as_type<bfloat>",
            "bfloat(NAN)",
            "== (int16_t)(127)",
            "& 127",
            "/ (uint32_t)(32)",
        ):
            assert needle in msl, f"E4M3 scaled-dot MSL for {payload_type} missing {needle!r}.\n--- MSL ---\n{msl}\n"


def test_e4m3_decoder_reference_covers_all_payloads():
    torch = pytest.importorskip("torch")
    raw = torch.arange(256, dtype=torch.uint8)
    decoded = raw.view(torch.float8_e4m3fn).to(torch.bfloat16)
    bits = decoded.view(torch.uint16)

    expected = []
    for value in range(256):
        sign = (value & 0x80) << 8
        magnitude = value & 0x7F
        mantissa = magnitude & 0x7
        if magnitude < 8:
            subnormal_bits = (
                0x0000,
                0x3B00,
                0x3B80,
                0x3BC0,
                0x3C00,
                0x3C20,
                0x3C40,
                0x3C60,
            )
            decoded_magnitude = subnormal_bits[mantissa]
        else:
            decoded_magnitude = (magnitude << 4) + 0x3C00
        if magnitude == 0x7F:
            decoded_magnitude = 0x7FC0
        expected.append(sign | decoded_magnitude)

    expected_bits = torch.tensor(expected, dtype=torch.uint16)
    finite = ~torch.isnan(decoded.float())
    assert finite.sum().item() == 254
    assert torch.equal(bits[finite], expected_bits[finite])
    assert torch.isnan(decoded[~finite].float()).all()
    assert decoded[0x7E].float().item() == 448.0
    assert decoded[0xFE].float().item() == -448.0


@pytest.mark.parametrize("lhs_k_pack,rhs_k_pack", [(True, True), (False, True), (True, False), (False, False)])
def test_dot_scaled_e8m0_e2m1_compiles_to_msl(lhs_k_pack, rhs_k_pack):
    src = ASTSource(
        fn=dot_scaled_e8m0_e2m1_kernel,
        signature={
            "a_base": "*u8",
            "b_base": "*u8",
            "a_scale_base": "*u8",
            "b_scale_base": "*u8",
            "output_base": "*fp32",
            "BLOCK_M": "constexpr",
            "BLOCK_N": "constexpr",
            "BLOCK_K": "constexpr",
            "LHS_K_PACK": "constexpr",
            "RHS_K_PACK": "constexpr",
        },
        constexprs={
            "BLOCK_M": 16,
            "BLOCK_N": 16,
            "BLOCK_K": 32,
            "LHS_K_PACK": lhs_k_pack,
            "RHS_K_PACK": rhs_k_pack,
        },
    )
    target = GPUTarget(backend="metal", arch=80, warp_size=32)
    compiled = triton.compile(src, target=target, options={"num_warps": 4})

    raw = compiled.asm["metal"]
    msl = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    for needle in (
        "as_type<bfloat>",
        "bfloat(NAN)",
        "/ (uint32_t)(2)",
        "& 15",
        "& 7",
        "/ (uint32_t)(32)",
    ):
        assert needle in msl, f"E2M1 scaled-dot MSL missing {needle!r}.\n--- MSL ---\n{msl}\n"

    # Prove that the per-operand packing flags reach scalar_dot lowering.  The
    # generated variable numbers are intentionally ignored: only the physical
    # coordinate divided by two distinguishes K packing from outer M/N packing.
    # Each integer op spells its own signedness onto its operands
    # (`signednessCast`): the M/N coordinate divide is arith.divsi, the packing
    # divide-by-two is arith.divui. The K-packed form stays distinguishable
    # because only there is a BARE variable the unsigned divisor applies to.
    lhs_outer_address = bool(
        re.search(
            r"\(uint32_t\)\(\(\(int32_t\)\(v\d+\) / \(int32_t\)\(16\)\)\) / \(uint32_t\)\(2\)",
            msl,
        )
    )
    rhs_outer_address = "% (int32_t)(16))) / (uint32_t)(2)" in msl
    # K packing divides a BARE coordinate; outer packing divides the
    # parenthesised M/N expression above, so the two stay distinguishable.
    k_packed_address = bool(
        re.search(r"\(uint32_t\)\(v\d+\) / \(uint32_t\)\(2\)", msl)
    )
    assert lhs_outer_address is not lhs_k_pack
    assert rhs_outer_address is not rhs_k_pack
    assert k_packed_address is (lhs_k_pack or rhs_k_pack)


def test_e2m1_decoder_reference_covers_all_nibbles():
    torch = pytest.importorskip("torch")
    from triton.tools.mxfp import MXFP4Tensor

    raw = torch.arange(16, dtype=torch.uint8)
    oracle = MXFP4Tensor(size=(16,), device="cpu")
    oracle.data = raw
    decoded_bits = oracle.to(torch.float32).to(torch.bfloat16).view(torch.uint16)

    expected = []
    for nibble in range(16):
        sign = (nibble & 8) << 12
        magnitude = nibble & 7
        if magnitude == 0:
            magnitude_bits = 0
        elif magnitude == 1:
            magnitude_bits = 0x3F00
        else:
            magnitude_bits = 0x3F00 + (magnitude << 6)
        expected.append(sign | magnitude_bits)

    assert torch.equal(decoded_bits, torch.tensor(expected, dtype=torch.uint16))
    assert decoded_bits[8].item() == 0x8000


def test_generic_fp8e5_load_remains_rejected():
    src = ASTSource(
        fn=fp8e5_copy_unsupported_kernel,
        signature={
            "input_base": "*fp8e5",
            "output_base": "*fp8e5",
            "BLOCK_SIZE": "constexpr",
        },
        constexprs={"BLOCK_SIZE": 128},
    )
    target = GPUTarget(backend="metal", arch=80, warp_size=32)
    # The pybind boundary intentionally collapses pass diagnostics to the
    # stage-level failure below; the lit negative test pins the detailed E5M2
    # diagnostic emitted before dialect conversion.
    with pytest.raises(RuntimeError, match="convert-tritongpu-to-metal failed"):
        triton.compile(src, target=target, options={"num_warps": 4})


def test_e4m3_payload_bitcast_with_extra_use_remains_rejected():
    src = ASTSource(
        fn=dot_scaled_e4m3_extra_use_unsupported_kernel,
        signature={
            "a_base": "*u8",
            "b_base": "*u8",
            "a_scale_base": "*u8",
            "b_scale_base": "*u8",
            "output_base": "*fp32",
            "scratch_base": "*u8",
            "BLOCK_M": "constexpr",
            "BLOCK_N": "constexpr",
            "BLOCK_K": "constexpr",
        },
        constexprs={"BLOCK_M": 16, "BLOCK_N": 16, "BLOCK_K": 32},
    )
    target = GPUTarget(backend="metal", arch=80, warp_size=32)
    with pytest.raises(RuntimeError, match="convert-tritongpu-to-metal failed"):
        triton.compile(src, target=target, options={"num_warps": 4})


def test_generic_fp8e4_load_remains_rejected():
    src = ASTSource(
        fn=fp8e4_copy_unsupported_kernel,
        signature={
            "input_base": "*fp8e4nv",
            "output_base": "*fp8e4nv",
            "BLOCK_SIZE": "constexpr",
        },
        constexprs={"BLOCK_SIZE": 128},
    )
    target = GPUTarget(backend="metal", arch=80, warp_size=32)
    with pytest.raises(RuntimeError, match="convert-tritongpu-to-metal failed"):
        triton.compile(src, target=target, options={"num_warps": 4})


def test_e5m2_decode_embedding_covers_every_encoding():
    torch = pytest.importorskip("torch")
    raw = torch.arange(256, dtype=torch.uint8)
    e5m2 = raw.view(torch.float8_e5m2)
    f16 = e5m2.to(torch.float16)
    f16_bits = f16.view(torch.uint16).to(torch.int32)
    expected_bits = raw.to(torch.int32) * 256

    is_nan = torch.isnan(e5m2)
    is_finite = torch.isfinite(e5m2)
    assert is_finite.sum().item() == 248
    assert torch.isinf(e5m2).sum().item() == 2
    assert is_nan.sum().item() == 6
    assert torch.equal(f16_bits[~is_nan], expected_bits[~is_nan])
    assert torch.equal(torch.isnan(f16), is_nan)

    # Every finite E5M2 value is exactly representable in BF16.  NaN payload
    # and signaling bits are deliberately outside the Metal contract.
    bf16 = f16.to(torch.float32).to(torch.bfloat16)
    assert torch.equal(bf16[is_finite].to(torch.float32), e5m2[is_finite].to(torch.float32))


# --- scf.while (a Python `while` in a kernel) -----------------------------
#
# Triton lowers a Python `while` to `scf.while (inits) { cond } do { body }`.
# The emitter previously had no case for it and died on an `llvm_unreachable`
# ("translateValue: unexpected op scf.while") that aborts the process instead
# of raising. MSL has no multi-value loop carry, so each carried value becomes
# a temp declared ahead of the loop and the whole thing emits as
# `while (true) { <before>; if (!cond) break; <after>; }`.


@triton.jit
def while_accumulate_kernel(x_ptr, out_ptr, n_trips, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    n = 0
    while n < n_trips:
        acc += tl.load(x_ptr + offs)
        n += 1
    tl.store(out_ptr + offs, acc)


def test_while_loop_compiles_to_msl():
    src = ASTSource(
        fn=while_accumulate_kernel,
        signature={
            "x_ptr": "*fp32", "out_ptr": "*fp32", "n_trips": "i32",
            "BLOCK": "constexpr",
        },
        constexprs={"BLOCK": 128},
    )
    target = GPUTarget(backend="metal", arch=80, warp_size=32)
    compiled = triton.compile(src, target=target, options={"num_warps": 4})
    raw = compiled.asm["metal"]
    msl = raw.decode("utf-8") if isinstance(raw, bytes) else raw

    # The loop must be an unconditional `while (true)` with an explicit break:
    # the predicate is recomputed from the carried temps at the top of every
    # trip, which a C-style `while (<cond>)` could not express once the
    # predicate reads values the body updates.
    assert "while (true)" in msl, f"missing while(true) in MSL:\n{msl}"
    assert "break;" in msl, f"missing loop break in MSL:\n{msl}"
    # Carried values are plain temps declared BEFORE the loop, so the
    # zero-trip case yields the init unchanged.
    assert re.search(r"float v\d+ = 0\.0", msl), (
        f"carried accumulator not initialised before the loop:\n{msl}")


@pytest.mark.parametrize("n_trips", [0, 1, 3, 7])
def test_while_loop_runs_on_mps(n_trips):
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("Metal backend requires an MPS-enabled PyTorch")
    torch.manual_seed(n_trips)
    x = torch.rand(16, dtype=torch.float32, device="mps")
    out = torch.zeros(16, dtype=torch.float32, device="mps")
    while_accumulate_kernel[(1,)](x, out, n_trips, BLOCK=16)
    torch.mps.synchronize()
    # n_trips == 0 must leave the accumulator at its init, i.e. the predicate
    # is evaluated BEFORE the first body run.
    torch.testing.assert_close(out.cpu(), (x * n_trips).cpu(),
                               atol=1e-6, rtol=1e-6)


@triton.jit
def while_block_walk_kernel(x_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    """Two carried values (a tensor accumulator and a scalar offset) plus a
    masked load, i.e. the shape a hand-written `while` over blocks takes."""
    offs = tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    start = 0
    while start < n_elements:
        mask = (start + offs) < n_elements
        acc += tl.load(x_ptr + start + offs, mask=mask, other=0.0)
        start += BLOCK
    tl.store(out_ptr + offs, acc)


@pytest.mark.parametrize("n_elements", [16, 64, 100, 1024])
def test_while_loop_two_carried_values_on_mps(n_elements):
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("Metal backend requires an MPS-enabled PyTorch")
    torch.manual_seed(n_elements)
    block = 16
    x_cpu = torch.rand(n_elements, dtype=torch.float32)
    out = torch.zeros(block, dtype=torch.float32, device="mps")
    while_block_walk_kernel[(1,)](x_cpu.to("mps"), out, n_elements, BLOCK=block)
    torch.mps.synchronize()

    expected = torch.zeros(block, dtype=torch.float32)
    for start in range(0, n_elements, block):
        chunk = x_cpu[start:start + block]
        expected[:chunk.numel()] += chunk
    torch.testing.assert_close(out.cpu(), expected, atol=1e-5, rtol=1e-5)


# --- Unsupported constructs must fail, not crash --------------------------
#
# Before `validateUnsupportedOpsRejected`, these reached applyFullConversion
# with no pattern and the failed-conversion teardown killed the process:
# tt.assert and tt.gather with SIGSEGV, tt.join with an abort. In-process
# assertions cannot catch that — the crash takes pytest down with it — so the
# check runs in a subprocess and asserts on the EXIT STATUS. A negative
# returncode is a signal (crash); 1 is an ordinary Python exception.

_UNSUPPORTED_SNIPPETS = {
    # `tt.join` itself is implemented now (see the shuffle tests below); an
    # out-of-envelope join is still a rejection, and still has to be a clean
    # one. num_warps=1 puts 32 threads under a 128-element joined tile.
    "join_out_of_envelope": (
        "j = tl.join(tl.load(x_ptr + tl.arange(0, 64)),\n"
        "                tl.load(x_ptr + tl.arange(0, 64)))\n"
        "    tl.store(o_ptr + tl.arange(0, 64)[:, None] * 2\n"
        "             + tl.arange(0, 2)[None, :], j)"),
    # `tl.device_print` is implemented now (it writes into the launcher's debug
    # buffer); inline asm never will be, since its payload is PTX.
    "inline_asm": (
        "v = tl.inline_asm_elementwise('mov $0, $1;', '=r,r', [v],\n"
        "                                  dtype=tl.float32, is_pure=True,"
        " pack=1)\n"
        "    tl.store(o_ptr + i, v)"),
}
_UNSUPPORTED_NUM_WARPS = {"join_out_of_envelope": 1}


@pytest.mark.parametrize("tensor", [False, True], ids=["scalar", "tensor"])
@pytest.mark.parametrize("num_warps", [1, 4])
def test_fma_f16_fails_without_crashing(tensor, num_warps, tmp_path):
    import os
    import subprocess
    import sys

    # Compile without an MPS device. A subprocess contains the historical
    # llvm_unreachable in MSL emission and lets us distinguish it from rejection.
    script = tmp_path / "unsupported_fma.py"
    offset = "tl.arange(0, 256)" if tensor else "0"
    script.write_text(
        "import triton\n"
        "import triton.language as tl\n"
        "from triton.compiler import ASTSource\n"
        "from triton.backends.compiler import GPUTarget\n"
        "@triton.jit\n"
        "def kernel(X, Y, Z, O):\n"
        f"    offset = {offset}\n"
        "    x = tl.load(X + offset)\n"
        "    y = tl.load(Y + offset)\n"
        "    z = tl.load(Z + offset)\n"
        "    tl.store(O + offset, tl.fma(x, y, z))\n"
        "source = ASTSource(kernel, signature={k: '*fp16' for k in 'XYZO'})\n"
        "triton.compile(source, target=GPUTarget('metal', 9, 32), "
        f"options={{'num_warps': {num_warps}}})\n"
    )
    proc = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True,
        timeout=120,
        env={**os.environ, "TRITON_CACHE_DIR": str(tmp_path / "cache")},
    )
    assert proc.returncode == 1, proc.stderr
    assert "Metal backend: math.fma requires f32 operands and result" in proc.stderr
    assert "UNREACHABLE" not in proc.stderr


@pytest.mark.parametrize("name", sorted(_UNSUPPORTED_SNIPPETS))
def test_unsupported_construct_fails_without_crashing(name, tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("Metal backend requires an MPS-enabled PyTorch")
    import subprocess
    import sys

    script = tmp_path / f"unsupported_{name}.py"
    script.write_text(
        "import torch, triton\n"
        "import triton.language as tl\n"
        "@triton.jit\n"
        "def k(x_ptr, o_ptr, B: tl.constexpr):\n"
        "    i = tl.arange(0, B)\n"
        "    v = tl.load(x_ptr + i)\n"
        f"    {_UNSUPPORTED_SNIPPETS[name]}\n"
        "x = torch.rand(128, device='mps')\n"
        "o = torch.empty_like(x)\n"
        f"k[(1,)](x, o, B=16, num_warps={_UNSUPPORTED_NUM_WARPS.get(name, 4)})\n"
    )
    proc = subprocess.run([sys.executable, str(script)], capture_output=True,
                          text=True, timeout=300)
    assert proc.returncode > 0, (
        f"{name}: expected a clean non-zero exit, got returncode "
        f"{proc.returncode} (negative means the process died on a signal)\n"
        f"{proc.stderr[-2000:]}")
    assert "Metal backend:" in proc.stderr, (
        f"{name}: no backend diagnostic in stderr:\n{proc.stderr[-2000:]}")


@pytest.mark.parametrize("op", ["sqrt", "erf", "exp", "exp2", "log", "log2", "rsqrt", "sin", "cos"])
@pytest.mark.parametrize("tensor", [False, True], ids=["scalar", "tensor"])
def test_unary_math_f64_fails_without_crashing(op, tensor, tmp_path):
    import os
    import subprocess
    import sys

    # f32 buffers keep the unsupported math width independent of the buffer
    # storage contract. The explicit f64 intermediate is valid Triton input.
    script = tmp_path / "unsupported_unary_math.py"
    offset = "tl.arange(0, 256)" if tensor else "0"
    script.write_text(
        "import triton\n"
        "import triton.language as tl\n"
        "from triton.compiler import ASTSource\n"
        "from triton.backends.compiler import GPUTarget\n"
        "@triton.jit\n"
        "def kernel(X, O):\n"
        f"    offset = {offset}\n"
        "    x = tl.load(X + offset).to(tl.float64)\n"
        f"    y = tl.{op}(x).to(tl.float32)\n"
        "    tl.store(O + offset, y)\n"
        "source = ASTSource(kernel, signature={'X': '*fp32', 'O': '*fp32'})\n"
        "triton.compile(source, target=GPUTarget('metal', 9, 32), "
        "options={'num_warps': 4})\n"
    )
    proc = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, timeout=120,
        env={**os.environ, "TRITON_CACHE_DIR": str(tmp_path / "cache")},
    )
    assert proc.returncode == 1, proc.stderr
    assert f"Metal backend: math.{op} requires f32 operands and result" in proc.stderr
    assert "UNREACHABLE" not in proc.stderr


@pytest.mark.parametrize("op,input_type,output_type,expression", [
    ("extf", "fp32", "i64", "x.to(tl.float64).to(tl.int64)"),
    ("truncf", "fp64", "fp32", "x.to(tl.float32)"),
    ("sitofp", "i32", "i64", "x.to(tl.float64).to(tl.int64)"),
    ("uitofp", "u32", "i64", "x.to(tl.float64).to(tl.int64)"),
    ("fptosi", "fp64", "i32", "x.to(tl.int32)"),
    ("fptoui", "fp64", "u32", "x.to(tl.uint32)"),
])
@pytest.mark.parametrize("tensor", [False, True], ids=["scalar", "tensor"])
def test_numeric_cast_f64_fails_without_crashing(op, input_type, output_type, expression, tensor, tmp_path):
    import os
    import subprocess
    import sys

    # Preserve widening with an integer consumer. The source-f64 cases are
    # compile-only: they require no allocation of an unsupported MPS dtype.
    script = tmp_path / "unsupported_numeric_cast.py"
    offset = "tl.arange(0, 256)" if tensor else "0"
    script.write_text(
        "import triton\n"
        "import triton.language as tl\n"
        "from triton.compiler import ASTSource\n"
        "from triton.backends.compiler import GPUTarget\n"
        "@triton.jit\n"
        "def kernel(X, O):\n"
        f"    offset = {offset}\n"
        "    x = tl.load(X + offset)\n"
        f"    y = {expression}\n"
        "    tl.store(O + offset, y)\n"
        f"source = ASTSource(kernel, signature={{'X': '*{input_type}', 'O': '*{output_type}'}})\n"
        "triton.compile(source, target=GPUTarget('metal', 9, 32), "
        "options={'num_warps': 4})\n"
    )
    proc = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, timeout=120,
        env={**os.environ, "TRITON_CACHE_DIR": str(tmp_path / "cache")},
    )
    assert proc.returncode == 1, proc.stderr
    assert f"Metal backend: arith.{op} does not support f64" in proc.stderr
    assert "UNREACHABLE" not in proc.stderr


# --- tt.make_range must never fall back to a constant ---------------------
#
# MakeRangeLowering used to end with "keep the constant-0 placeholder so we
# don't break legalization of edge cases". But `tt.make_range` IS a per-element
# index, so a constant 0 is not a placeholder — it is a wrong answer that
# compiles, launches, and returns silently corrupt data.
#
# `tl.arange(0, 2)` reshaped into a rank-3 hypercube axis (exactly what
# tl.sort / tl.flip build) gets a `#ttg.linear` layout, matched none of the
# blocked / slice decompositions, and read back as 0 for EVERY element. The
# rank-2 forms below were and remain correct; the rank-3 form must now be a
# named error.


@triton.jit
def _arange_rank2_col_kernel(out_ptr, M: tl.constexpr, N: tl.constexpr):
    h = tl.zeros((M, N), dtype=tl.int32) + tl.arange(0, N)[None, :]
    tl.store(out_ptr + tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :], h)


@triton.jit
def _arange_rank2_row_kernel(out_ptr, M: tl.constexpr, N: tl.constexpr):
    h = tl.zeros((M, N), dtype=tl.int32) + tl.arange(0, M)[:, None]
    tl.store(out_ptr + tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :], h)


@triton.jit
def _arange_rank3_kernel(out_ptr, B: tl.constexpr):
    ar = tl.reshape(tl.arange(0, 2), (1, 2, 1))
    h = tl.zeros((2, 2, 2), dtype=tl.int32) + ar
    tl.store(out_ptr + tl.arange(0, B), tl.reshape(h, (B,)))


@pytest.mark.parametrize("kernel,expected", [
    (_arange_rank2_col_kernel, [p % 4 for p in range(16)]),
    (_arange_rank2_row_kernel, [p // 4 for p in range(16)]),
])
def test_arange_broadcast_rank2_values(kernel, expected):
    """The decompositions that DO exist keep producing real indices — the guard
    against over-rejecting when the constant fallback was removed."""
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("Metal backend requires an MPS-enabled PyTorch")
    out = torch.zeros(16, dtype=torch.int32, device="mps")
    kernel[(1,)](out, M=4, N=4)
    torch.mps.synchronize()
    assert out.cpu().tolist() == expected


def test_arange_broadcast_rank3_carries_real_indices():
    """This used to be a rejection: an arange reshaped into a rank-3 axis had no
    per-element index, and before that it silently read back as 0 for every
    element. It is decomposable now — the axis comes from the reshape and the
    value from the tile coordinate — so the assertion is on the VALUES.

    Zeros here would mean the constant fallback is back, which is the original
    silent wrong answer; `[0,0,1,1,0,0,1,1]` is `h[i,j,k] = j` flattened."""
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("Metal backend requires an MPS-enabled PyTorch")
    out = torch.zeros(8, dtype=torch.int32, device="mps")
    _arange_rank3_kernel[(1,)](out, B=8)
    torch.mps.synchronize()
    assert out.cpu().tolist() == [0, 0, 1, 1, 0, 0, 1, 1]


# --- tl.gather ------------------------------------------------------------
#
# A gather is an arbitrary cross-thread read, so the whole tile stages through
# the threadgroup buffer: fill, barrier, then read at the index. The barrier is
# the correctness-critical part — without it a thread can read a slot whose
# owner has not written it yet — which is why these run at num_warps > 1, where
# a missing barrier actually shows up, and assert bit-exact equality (a gather
# only copies bits, so anything but exact is a real bug).


@triton.jit
def _gather_f32_kernel(x_ptr, idx_ptr, out_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    v = tl.load(x_ptr + offs)
    ix = tl.load(idx_ptr + offs)
    tl.store(out_ptr + offs, tl.gather(v, ix, axis=0))


@triton.jit
def _gather_i32_kernel(x_ptr, idx_ptr, out_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    tl.store(out_ptr + offs,
             tl.gather(tl.load(x_ptr + offs), tl.load(idx_ptr + offs), axis=0))


@triton.jit
def _gather_computed_source_kernel(x_ptr, idx_ptr, out_ptr, BLOCK: tl.constexpr):
    """The staged source is a computed cone, not a bare load, so the fill has to
    re-derive it per logical position."""
    offs = tl.arange(0, BLOCK)
    v = tl.load(x_ptr + offs) * 2.0 + 1.0
    tl.store(out_ptr + offs, tl.gather(v, tl.load(idx_ptr + offs), axis=0))


@triton.jit(do_not_specialize=["flag"])
def _gather_source_replay_kernel(x_ptr, idx_ptr, out_ptr, flag,
                                 BLOCK: tl.constexpr, MODE: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    if MODE == "fma":
        x = tl.fma(x, x, 1.0)
    elif MODE == "rem":
        x = x % 3.0
    elif MODE == "div":
        x = tl.div_rn(x, 3.0)
    elif MODE == "if":
        if flag:
            x = x + 1.0
        else:
            x = x * 2.0
    elif MODE == "select":
        x = tl.where(x < 0.0, x * 2.0, x + 1.0)
    elif MODE == "scan":
        x = tl.cumsum(x)
    else:
        x = x * 2.0 + 1.0
    tl.store(out_ptr + offs, tl.gather(x, tl.load(idx_ptr + offs), axis=0))


@pytest.mark.parametrize("mode", ["fma", "rem", "div", "if"])
def test_gather_rank1_source_replay_is_rejected_not_crashed(mode, tmp_path):
    """A declined multi-element source must leave compilation recoverable."""
    import os
    import subprocess
    import sys
    import textwrap

    script = tmp_path / "gather_source_replay.py"
    script.write_text(textwrap.dedent(f"""
        import importlib.util
        import triton
        from triton.compiler import ASTSource
        from triton.backends.compiler import GPUTarget
        spec = importlib.util.spec_from_file_location("msl_child", {str(Path(__file__).resolve())!r})
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        target = GPUTarget("metal", 9, 32)

        def compile_case(block, mode):
            return triton.compile(ASTSource(
                module._gather_source_replay_kernel,
                signature={{"x_ptr": "*fp32", "idx_ptr": "*i32",
                           "out_ptr": "*fp32", "flag": "i32"}},
                constexprs={{"BLOCK": block, "MODE": mode}}),
                target=target, options={{"num_warps": 4}})

        for attempt in range(3):
            try:
                compile_case(256, {mode!r})
            except RuntimeError:
                pass
            else:
                raise AssertionError("unsupported gather source compiled")
        # These shapes stage the already computed value without replaying it.
        compile_case(32, {mode!r})
        compile_case(128, {mode!r})
        # Multiple elements per thread remain supported for replayable cones.
        compile_case(256, "select")
        print("RECOVERED")
    """))
    result = subprocess.run([sys.executable, str(script)], capture_output=True,
                            text=True, timeout=180,
                            env={**os.environ,
                                 "TRITON_CACHE_DIR": str(tmp_path / "cache")})
    output = result.stdout + result.stderr
    needle = "Metal backend: rank-1 gather cannot replay its source across conversion"
    assert result.returncode == 0, output
    assert output.count(needle) == 3, output
    assert "RECOVERED" in output
    assert "failed to legalize" not in output


@pytest.mark.parametrize(
    "block,num_warps,mode,flag",
    [
        pytest.param(block, 4, mode, flag, id=f"b{block}_{mode}_flag{flag}")
        for block in (32, 128)
        for mode, flag in (("fma", 1), ("rem", 1), ("div", 1), ("if", 0), ("if", 1))
    ] + [
        pytest.param(256, num_warps, mode, 1, id=f"b256_{mode}_w{num_warps}")
        for num_warps in (1, 4)
        for mode in ("select", "arithmetic", "scan")
    ],
)
def test_gather_rank1_source_replay_supported_numeric(block, num_warps, mode, flag):
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("Metal backend requires an MPS-enabled PyTorch")
    x = (torch.arange(block, dtype=torch.float32) - block // 2) * 0.125
    idx = torch.roll(torch.arange(block - 1, -1, -1), 17).to(torch.int32)
    if mode == "fma":
        source = x * x + 1.0
    elif mode == "rem":
        source = torch.fmod(x, 3.0)
    elif mode == "div":
        source = x / 3.0
    elif mode == "if":
        source = x + 1.0 if flag else x * 2.0
    elif mode == "select":
        source = torch.where(x < 0.0, x * 2.0, x + 1.0)
    elif mode == "scan":
        source = torch.cumsum(x, dim=0)
    else:
        source = x * 2.0 + 1.0
    sentinel = -4321.0
    out = torch.full((block + 8,), sentinel, dtype=torch.float32, device="mps")
    _gather_source_replay_kernel[(1,)](
        x.to("mps"), idx.to("mps"), out, flag,
        BLOCK=block, MODE=mode, num_warps=num_warps)
    torch.mps.synchronize()
    actual = out.cpu()
    torch.testing.assert_close(actual[:block], source[idx.long()], atol=1e-6, rtol=1e-6)
    assert torch.all(actual[block:] == sentinel)


@triton.jit(do_not_specialize=["flag"])
def _histogram_replay_preflight_kernel(out_ptr, flag,
                                       BLOCK: tl.constexpr,
                                       BINS: tl.constexpr,
                                       MODE: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    src = offs % BINS
    mask = offs >= 0
    if MODE == "source_if":
        if flag:
            src = src + 1
        else:
            src = src
        hist = tl.histogram(src, BINS)
    elif MODE == "mask_if":
        if flag:
            mask = src < (BINS // 2)
        else:
            mask = src >= (BINS // 2)
        hist = tl.histogram(src, BINS, mask=mask)
    else:
        hist = tl.histogram(src, BINS, mask=mask)
    tl.store(out_ptr + tl.arange(0, BINS), hist)


@pytest.mark.parametrize("mode", ["source_if", "mask_if"])
def test_histogram_tensor_if_replay_is_rejected_not_crashed(mode, tmp_path):
    """A declined tensor-yielding histogram cone must leave compilation usable."""
    import os
    import subprocess
    import sys
    import textwrap

    script = tmp_path / "histogram_replay_preflight.py"
    script.write_text(textwrap.dedent(f"""
        import importlib.util
        import triton
        from triton.compiler import ASTSource
        from triton.backends.compiler import GPUTarget
        spec = importlib.util.spec_from_file_location("msl_child", {str(Path(__file__).resolve())!r})
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        target = GPUTarget("metal", 9, 32)

        def compile_case(block, mode):
            return triton.compile(ASTSource(
                module._histogram_replay_preflight_kernel,
                signature={{"out_ptr": "*i32", "flag": "i32"}},
                constexprs={{"BLOCK": block, "BINS": 16, "MODE": mode}}),
                target=target, options={{"num_warps": 4}})

        for block in (32, 128, 256):
            try:
                compile_case(block, {mode!r})
            except RuntimeError:
                pass
            else:
                raise AssertionError("unsupported histogram cone compiled")
        for attempt in range(3):
            try:
                compile_case(256, {mode!r})
            except RuntimeError:
                pass
            else:
                raise AssertionError("unsupported histogram cone compiled")
        compile_case(256, "plain")
        print("RECOVERED")
    """))
    result = subprocess.run([sys.executable, str(script)], capture_output=True,
                            text=True, timeout=180,
                            env={**os.environ,
                                 "TRITON_CACHE_DIR": str(tmp_path / "cache")})
    output = result.stdout + result.stderr
    if mode == "source_if":
        needle = "Metal backend: tt.histogram source cone is not rank-1 evaluable"
    else:
        needle = "Metal backend: tt.histogram mask cone is not rank-1 i1 evaluable"
    assert result.returncode == 0, output
    assert output.count(needle) == 6, output
    assert "RECOVERED" in output
    assert "failed to legalize" not in output


@triton.jit(do_not_specialize=["flag"])
def _histogram_replay_supported_kernel(x_ptr, mask_ptr, out_ptr, flag,
                                       BLOCK: tl.constexpr,
                                       BINS: tl.constexpr,
                                       MODE: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    src = tl.load(x_ptr + offs)
    base_mask = tl.load(mask_ptr + offs)
    if MODE == "scalar_if":
        if flag:
            shift = 0
        else:
            shift = 1
        src = src + shift
        hist = tl.histogram(src, BINS, mask=base_mask)
        tl.store(out_ptr + tl.arange(0, BINS), hist)
    elif MODE == "where_source":
        src = tl.where((offs % 2) == flag, src + 1, src)
        hist = tl.histogram(src, BINS, mask=base_mask)
        tl.store(out_ptr + tl.arange(0, BINS), hist)
    elif MODE == "where_mask":
        if flag:
            divisor = 3
        else:
            divisor = 5
        extra = (offs % divisor) != 0
        mask = base_mask & tl.where((src >= 0) & (src < BINS), extra, base_mask)
        hist = tl.histogram(src, BINS, mask=mask)
        tl.store(out_ptr + tl.arange(0, BINS), hist)
    else:
        if flag:
            src = src + 0
            mask = base_mask & ((offs % 2) == 0)
            hist = tl.histogram(src, BINS, mask=mask)
            tl.store(out_ptr + tl.arange(0, BINS), hist)
        else:
            src = src + 1
            mask = base_mask & ((offs % 2) == 1)
            hist = tl.histogram(src, BINS, mask=mask)
            tl.store(out_ptr + tl.arange(0, BINS), hist)


@pytest.mark.parametrize(
    "block,mode,flag",
    [
        pytest.param(block, mode, flag, id=f"b{block}_{mode}_flag{flag}")
        for block in (32, 128, 256)
        for mode in ("scalar_if", "where_source", "where_mask", "inner_if")
        for flag in (0, 1)
    ],
)
def test_histogram_replay_supported_numeric(block, mode, flag):
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("Metal backend requires an MPS-enabled PyTorch")
    bins = 16
    arange = torch.arange(block, dtype=torch.int32)
    x = (arange % (bins + 3)) - 2
    base_mask = (arange % 4) != 1
    if mode == "scalar_if":
        source = x if flag else x + 1
        mask = base_mask
    elif mode == "where_source":
        source = torch.where((arange % 2) == flag, x + 1, x)
        mask = base_mask
    elif mode == "where_mask":
        source = x
        extra = (arange % (3 if flag else 5)) != 0
        in_range = (source >= 0) & (source < bins)
        mask = base_mask & torch.where(in_range, extra, base_mask)
    else:
        source = x if flag else x + 1
        mask = base_mask & ((arange % 2) == (0 if flag else 1))
    valid = mask & (source >= 0) & (source < bins)
    expected = torch.bincount(source[valid].to(torch.int64), minlength=bins)[:bins]
    sentinel = -777
    out = torch.full((bins + 4,), sentinel, dtype=torch.int32, device="mps")
    _histogram_replay_supported_kernel[(1,)](
        x.to("mps"), base_mask.to("mps"), out, flag,
        BLOCK=block, BINS=bins, MODE=mode, num_warps=4)
    torch.mps.synchronize()
    actual = out.cpu()
    torch.testing.assert_close(actual[:bins].to(torch.int64), expected,
                               atol=0, rtol=0)
    assert torch.all(actual[bins:] == sentinel)


@triton.jit(do_not_specialize=["flag"])
def _scan_source_preflight_kernel(x_ptr, out_ptr, flag,
                                  BLOCK: tl.constexpr, MODE: tl.constexpr,
                                  REVERSE: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    if MODE == "fma":
        x = tl.fma(x, x, 1.0)
        y = tl.cumsum(x, 0, reverse=REVERSE)
        tl.store(out_ptr + offs, y)
    elif MODE == "rem":
        x = x % 3.0
        y = tl.cumsum(x, 0, reverse=REVERSE)
        tl.store(out_ptr + offs, y)
    elif MODE == "div":
        x = tl.div_rn(x, 3.0)
        y = tl.cumsum(x, 0, reverse=REVERSE)
        tl.store(out_ptr + offs, y)
    elif MODE == "nested_if":
        if flag:
            x = tl.fma(x, x, 1.0)
            y = tl.cumsum(x, 0, reverse=REVERSE)
            tl.store(out_ptr + offs, y)
        else:
            x = x + 1.0
            y = tl.cumsum(x, 0, reverse=REVERSE)
            tl.store(out_ptr + offs, y)
    else:
        x = x * 2.0 + 1.0
        y = tl.cumsum(x, 0, reverse=REVERSE)
        tl.store(out_ptr + offs, y)


@pytest.mark.parametrize(
    "block,mode",
    [
        pytest.param(block, mode, id=f"b{block}_{mode}")
        for block in (32, 256)
        for mode in ("fma", "rem", "div")
    ] + [
        pytest.param(128, "nested_if", id="b128_nested_if"),
    ],
)
def test_scan_rank1_source_replay_is_rejected_not_crashed(block, mode, tmp_path):
    """A declined scan source must report the preflight diagnostic and recover."""
    import os
    import subprocess
    import sys
    import textwrap

    script = tmp_path / "scan_source_preflight.py"
    script.write_text(textwrap.dedent(f"""
        import importlib.util
        import triton
        from triton.compiler import ASTSource
        from triton.backends.compiler import GPUTarget
        spec = importlib.util.spec_from_file_location("msl_child", {str(Path(__file__).resolve())!r})
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        target = GPUTarget("metal", 9, 32)

        def compile_case(block, mode):
            return triton.compile(ASTSource(
                module._scan_source_preflight_kernel,
                signature={{"x_ptr": "*fp32", "out_ptr": "*fp32",
                           "flag": "i32"}},
                constexprs={{"BLOCK": block, "MODE": mode,
                            "REVERSE": False}}),
                target=target, options={{"num_warps": 4}})

        for attempt in range(3):
            try:
                compile_case({block}, {mode!r})
            except RuntimeError:
                pass
            else:
                raise AssertionError("unsupported scan source compiled")
        compile_case(128, "arithmetic")
        print("RECOVERED")
    """))
    result = subprocess.run([sys.executable, str(script)], capture_output=True,
                            text=True, timeout=180,
                            env={**os.environ,
                                 "TRITON_CACHE_DIR": str(tmp_path / "cache")})
    output = result.stdout + result.stderr
    needle = "Metal backend: rank-1 scan cannot replay its source across conversion"
    assert result.returncode == 0, output
    assert output.count(needle) == 3, output
    assert "RECOVERED" in output
    assert "failed to legalize" not in output


@triton.jit(do_not_specialize=["flag"])
def _scan_rank1_source_supported_kernel(x_ptr, out_ptr, flag,
                                        BLOCK: tl.constexpr,
                                        MODE: tl.constexpr,
                                        REVERSE: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    if MODE == "fma":
        src = tl.fma(x, x, 1.0)
        y = tl.cumsum(src, 0, reverse=REVERSE)
        tl.store(out_ptr + offs, y)
    elif MODE == "rem":
        src = x % 3.0
        y = tl.cumsum(src, 0, reverse=REVERSE)
        tl.store(out_ptr + offs, y)
    elif MODE == "div":
        src = tl.div_rn(x, 3.0)
        y = tl.cumsum(src, 0, reverse=REVERSE)
        tl.store(out_ptr + offs, y)
    elif MODE == "tensor_if":
        if flag:
            src = x + 1.0
        else:
            src = x * 2.0
        y = tl.cumsum(src, 0, reverse=REVERSE)
        tl.store(out_ptr + offs, y)
    elif MODE == "select":
        src = tl.where((offs % 2) == flag, x + 1.0, x * 2.0)
        y = tl.cumsum(src, 0, reverse=REVERSE)
        tl.store(out_ptr + offs, y)
    elif MODE == "scan_chain":
        first = tl.cumsum(x, 0, reverse=REVERSE)
        y = tl.cumsum(first, 0, reverse=not REVERSE)
        tl.store(out_ptr + offs, y)
    elif MODE == "cumprod":
        y = tl.cumprod(x, 0, reverse=REVERSE)
        tl.store(out_ptr + offs, y)
    elif MODE == "inside_if":
        if flag:
            src = x + 1.0
            y = tl.cumsum(src, 0, reverse=REVERSE)
            tl.store(out_ptr + offs, y)
        else:
            src = x * 2.0
            y = tl.cumsum(src, 0, reverse=REVERSE)
            tl.store(out_ptr + offs, y)
    else:
        src = x * 2.0 + 1.0
        y = tl.cumsum(src, 0, reverse=REVERSE)
        tl.store(out_ptr + offs, y)


@pytest.mark.parametrize(
    "block,mode,flag,reverse",
    [
        pytest.param(128, mode, 1, reverse,
                     id=f"b128_exact_{mode}_rev{int(reverse)}")
        for mode in ("fma", "rem", "div")
        for reverse in (False, True)
    ] + [
        pytest.param(block, "tensor_if", flag, reverse,
                     id=f"b{block}_tensor_if_flag{flag}_rev{int(reverse)}")
        for block in (32, 128, 256)
        for flag in (0, 1)
        for reverse in (False, True)
    ] + [
        pytest.param(block, mode, flag, reverse,
                     id=f"b{block}_{mode}_flag{flag}_rev{int(reverse)}")
        for block in (32, 256)
        for mode in ("select",)
        for flag in (0, 1)
        for reverse in (False, True)
    ] + [
        pytest.param(block, "arithmetic", 1, reverse,
                     id=f"b{block}_arithmetic_rev{int(reverse)}")
        for block in (32, 256)
        for reverse in (False, True)
    ] + [
        pytest.param(block, "scan_chain", 1, reverse,
                     id=f"b{block}_scan_chain_rev{int(reverse)}")
        for block in (128, 256)
        for reverse in (False, True)
    ] + [
        pytest.param(block, "cumprod", 1, False,
                     id=f"b{block}_cumprod")
        for block in (128, 256)
    ] + [
        pytest.param(128, "inside_if", flag, reverse,
                     id=f"b128_inside_if_flag{flag}_rev{int(reverse)}")
        for flag in (0, 1)
        for reverse in (False, True)
    ],
)
def test_scan_rank1_source_replay_supported_numeric(block, mode, flag, reverse):
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("Metal backend requires an MPS-enabled PyTorch")
    if mode == "cumprod":
        x = 1.0 + (torch.arange(block, dtype=torch.float32) % 5) * 0.001
    else:
        x = (torch.arange(block, dtype=torch.float32) - block // 2) * 0.03125
    if mode == "fma":
        source = x * x + 1.0
        expected = source.flip(0).cumsum(0).flip(0) if reverse else source.cumsum(0)
    elif mode == "rem":
        source = torch.fmod(x, 3.0)
        expected = source.flip(0).cumsum(0).flip(0) if reverse else source.cumsum(0)
    elif mode == "div":
        source = x / 3.0
        expected = source.flip(0).cumsum(0).flip(0) if reverse else source.cumsum(0)
    elif mode in ("tensor_if", "inside_if"):
        source = x + 1.0 if flag else x * 2.0
        expected = source.flip(0).cumsum(0).flip(0) if reverse else source.cumsum(0)
    elif mode == "select":
        arange = torch.arange(block, dtype=torch.int32)
        source = torch.where((arange % 2) == flag, x + 1.0, x * 2.0)
        expected = source.flip(0).cumsum(0).flip(0) if reverse else source.cumsum(0)
    elif mode == "scan_chain":
        first = x.flip(0).cumsum(0).flip(0) if reverse else x.cumsum(0)
        expected = first.cumsum(0) if reverse else first.flip(0).cumsum(0).flip(0)
    elif mode == "cumprod":
        expected = x.flip(0).cumprod(0).flip(0) if reverse else x.cumprod(0)
    else:
        source = x * 2.0 + 1.0
        expected = source.flip(0).cumsum(0).flip(0) if reverse else source.cumsum(0)
    sentinel = -9876.0
    out = torch.full((block + 8,), sentinel, dtype=torch.float32, device="mps")
    _scan_rank1_source_supported_kernel[(1,)](
        x.to("mps"), out, flag, BLOCK=block, MODE=mode,
        REVERSE=reverse, num_warps=4)
    torch.mps.synchronize()
    actual = out.cpu()
    torch.testing.assert_close(actual[:block], expected, atol=1e-5, rtol=1e-5)
    assert torch.all(actual[block:] == sentinel)


@triton.jit
def _gather_loop_carried_i32_kernel(x_ptr, idx_ptr, out_ptr, steps,
                                    BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    acc = tl.load(x_ptr + offs)
    for step in range(steps):
        indices = tl.load(idx_ptr + step * BLOCK + offs)
        acc = tl.gather(acc, indices, axis=0) + 3
    tl.store(out_ptr + offs, acc)


@pytest.mark.parametrize("block", [8, 32, 256, 1024])
@pytest.mark.parametrize("num_warps", [1, 2, 4])
def test_gather_f32_permutation(block, num_warps):
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("Metal backend requires an MPS-enabled PyTorch")
    torch.manual_seed(block + num_warps)
    x = torch.rand(block, dtype=torch.float32)
    idx = torch.randperm(block).to(torch.int32)
    out = torch.empty(block, dtype=torch.float32, device="mps")
    _gather_f32_kernel[(1,)](x.to("mps"), idx.to("mps"), out, BLOCK=block,
                             num_warps=num_warps)
    torch.mps.synchronize()
    assert torch.equal(out.cpu(), x[idx.long()])


@pytest.mark.parametrize("block", [32, 256])
def test_gather_i32_payload(block):
    """i32 stages through ui32; a gather only copies bits, so the round trip
    must be exact including negative values."""
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("Metal backend requires an MPS-enabled PyTorch")
    torch.manual_seed(block)
    x = torch.randint(-(2**30), 2**30, (block,), dtype=torch.int32)
    idx = torch.randperm(block).to(torch.int32)
    out = torch.empty(block, dtype=torch.int32, device="mps")
    _gather_i32_kernel[(1,)](x.to("mps"), idx.to("mps"), out, BLOCK=block)
    torch.mps.synchronize()
    assert torch.equal(out.cpu(), x[idx.long()])


@pytest.mark.parametrize("block", [32, 256])
def test_gather_computed_source(block):
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("Metal backend requires an MPS-enabled PyTorch")
    torch.manual_seed(block * 7)
    x = torch.rand(block, dtype=torch.float32)
    idx = torch.randperm(block).to(torch.int32)
    out = torch.empty(block, dtype=torch.float32, device="mps")
    _gather_computed_source_kernel[(1,)](x.to("mps"), idx.to("mps"), out,
                                         BLOCK=block)
    torch.mps.synchronize()
    torch.testing.assert_close(out.cpu(), (x * 2.0 + 1.0)[idx.long()],
                               atol=1e-6, rtol=1e-6)


def test_gather_rank1_loop_carried_source_i32():
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("Metal backend requires an MPS-enabled PyTorch")
    block, steps = 1024, 3
    x = torch.arange(block, dtype=torch.int32)
    permutations = torch.stack([
        torch.randperm(block),
        torch.arange(block - 1, -1, -1),
        torch.roll(torch.arange(block), 17),
    ]).to(torch.int32)
    out = torch.empty(block, dtype=torch.int32, device="mps")

    _gather_loop_carried_i32_kernel[(1,)](
        x.to("mps"), permutations.to("mps"), out, steps, BLOCK=block,
        num_warps=4)
    torch.mps.synchronize()

    expected = x
    for permutation in permutations:
        expected = expected[permutation.long()] + 3
    assert torch.equal(out.cpu(), expected)


# --- rank >= 3: tl.sort / tl.flip and the staged reduce --------------------
#
# `tl.sort` reshapes its tile into a `[2] * log2(N)` hypercube and drives it
# with `x ^ xor_sum(x, axis, keep_dims=True)` per axis. Two things had to exist
# for that: an arange reshaped into one cube axis has to answer with that axis's
# coordinate (it used to have no per-element index at all), and a rank >= 3
# reduce has to lower (it was rejected outright).
#
# Both regressed silently in development rather than failing, which is why
# these assert exact permutations rather than tolerances:
#   * the size-2 arange that came out `#ttg.blocked` instead of `#ttg.linear`
#     answered with the raw lane id, and the compare direction — an XOR of two
#     cube coordinates — half-sorted the input;
#   * the staged reduce fed its result to a rank-(N-1) consumer at the wrong
#     coordinates, correct for axis 0 and wrong for every other axis, because
#     dropping the SLOWEST axis happens to leave the flat index alone.


@triton.jit
def _sort_kernel(x_ptr, o_ptr, N: tl.constexpr):
    i = tl.arange(0, N)
    tl.store(o_ptr + i, tl.sort(tl.load(x_ptr + i)))


@triton.jit
def _sort_desc_kernel(x_ptr, o_ptr, N: tl.constexpr):
    i = tl.arange(0, N)
    tl.store(o_ptr + i, tl.sort(tl.load(x_ptr + i), descending=True))


@triton.jit
def _flip_kernel(x_ptr, o_ptr, N: tl.constexpr):
    i = tl.arange(0, N)
    tl.store(o_ptr + i, tl.flip(tl.load(x_ptr + i), 0))


@triton.jit
def _topk_v2_kernel(x_ptr, o_ptr, N: tl.constexpr, K: tl.constexpr,
                    DESCENDING: tl.constexpr):
    i = tl.arange(0, N)
    selected = tl.topk(tl.load(x_ptr + i), K, dim=0,
                       descending=DESCENDING)
    tl.store(o_ptr + tl.arange(0, K), selected)


@triton.jit
def _topk_rank2_v2_kernel(x_ptr, o_ptr, M: tl.constexpr, N: tl.constexpr,
                          K: tl.constexpr, DESCENDING: tl.constexpr):
    m = tl.arange(0, M)[:, None]
    n = tl.arange(0, N)[None, :]
    values = tl.load(x_ptr + m * N + n)
    selected = tl.topk(values, K, dim=1, descending=DESCENDING)
    ko = tl.arange(0, K)[None, :]
    tl.store(o_ptr + m * K + ko, selected)


@triton.jit
def _topk_rank3_v2_kernel(x_ptr, o_ptr, B: tl.constexpr, M: tl.constexpr,
                          N: tl.constexpr, K: tl.constexpr,
                          DESCENDING: tl.constexpr):
    b = tl.arange(0, B)[:, None, None]
    m = tl.arange(0, M)[None, :, None]
    n = tl.arange(0, N)[None, None, :]
    values = tl.load(x_ptr + b * (M * N) + m * N + n)
    selected = tl.topk(values, K, dim=2, descending=DESCENDING)
    ko = tl.arange(0, K)[None, None, :]
    tl.store(o_ptr + b * (M * K) + m * K + ko, selected)


@triton.jit
def _two_topk_kernel(x_ptr, o0_ptr, o1_ptr, N: tl.constexpr,
                     K0: tl.constexpr, K1: tl.constexpr):
    values = tl.load(x_ptr + tl.arange(0, N))
    first = tl.topk(values, K0, dim=0)
    second = tl.topk(values, K1, dim=0, descending=False)
    tl.store(o0_ptr + tl.arange(0, K0), first)
    tl.store(o1_ptr + tl.arange(0, K1), second)


@triton.jit
def _topk_elementwise_followup_kernel(x_ptr, o_ptr, N: tl.constexpr,
                                      K: tl.constexpr):
    values = tl.load(x_ptr + tl.arange(0, N))
    selected = tl.topk(values, K, dim=0)
    tl.store(o_ptr + tl.arange(0, K), selected * 2 - 1)


@triton.jit
def _topk_reduce_followup_kernel(x_ptr, sum_ptr, N: tl.constexpr,
                                 K: tl.constexpr):
    values = tl.load(x_ptr + tl.arange(0, N))
    selected = tl.topk(values, K, dim=0)
    tl.store(sum_ptr, tl.sum(selected, axis=0))


@triton.jit
def _mixed_sort_topk_kernel(x_ptr, sort_ptr, topk_ptr, N: tl.constexpr,
                            K: tl.constexpr):
    values = tl.load(x_ptr + tl.arange(0, N))
    tl.store(sort_ptr + tl.arange(0, N), tl.sort(values))
    tl.store(topk_ptr + tl.arange(0, K), tl.topk(values, K))


@triton.jit
def _masked_topk_v2_kernel(x_ptr, o_ptr, VALID_N: tl.constexpr,
                           BLOCK_N: tl.constexpr, K: tl.constexpr,
                           DESCENDING: tl.constexpr):
    offsets = tl.arange(0, BLOCK_N)
    padding = -float("inf") if DESCENDING else float("inf")
    values = tl.load(x_ptr + offsets, mask=offsets < VALID_N, other=padding)
    selected = tl.topk(values, K, descending=DESCENDING)
    tl.store(o_ptr + tl.arange(0, K), selected)


@pytest.mark.parametrize("N,K,num_warps,descending,dtype", [
    (32, 1, 1, True, "bf16"),
    (32, 4, 1, False, "i32"),
    (8, 4, 1, False, "fp32"),
    (16, 8, 1, True, "i32"),
    (32, 32, 4, False, "fp32"),
    (128, 8, 4, True, "fp16"),
    (256, 128, 4, False, "bf16"),
    (512, 8, 8, True, "fp32"),
    (1024, 2, 8, False, "i32"),
])
def test_topk_compile_matrix(N, K, num_warps, descending, dtype):
    msl = _compile_to_msl(
        _topk_v2_kernel,
        {"x_ptr": f"*{dtype}", "o_ptr": f"*{dtype}", "N": "constexpr",
         "K": "constexpr", "DESCENDING": "constexpr"},
        {"N": N, "K": K, "DESCENDING": descending},
        num_warps=num_warps,
    )
    assert "threadgroup" in msl
    assert "threadgroup_barrier" in msl


def test_topk_rank2_rank3_and_two_topk_compile_to_msl():
    rank2_msl = _compile_to_msl(
        _topk_rank2_v2_kernel,
        {"x_ptr": "*fp32", "o_ptr": "*fp32", "M": "constexpr",
         "N": "constexpr", "K": "constexpr", "DESCENDING": "constexpr"},
        {"M": 8, "N": 64, "K": 8, "DESCENDING": True},
        num_warps=8,
    )
    rank3_msl = _compile_to_msl(
        _topk_rank3_v2_kernel,
        {"x_ptr": "*bf16", "o_ptr": "*bf16", "B": "constexpr",
         "M": "constexpr", "N": "constexpr", "K": "constexpr",
         "DESCENDING": "constexpr"},
        {"B": 2, "M": 4, "N": 64, "K": 8, "DESCENDING": False},
        num_warps=8,
    )
    rank3_multiband_msl = _compile_to_msl(
        _topk_rank3_v2_kernel,
        {"x_ptr": "*fp32", "o_ptr": "*fp32", "B": "constexpr",
         "M": "constexpr", "N": "constexpr", "K": "constexpr",
         "DESCENDING": "constexpr"},
        {"B": 4, "M": 2, "N": 128, "K": 8, "DESCENDING": False},
        num_warps=8,
        attrs={
            (0,): [["tt.divisibility", 16]],
            (1,): [["tt.divisibility", 16]],
        },
    )
    two_msl = _compile_to_msl(
        _two_topk_kernel,
        {"x_ptr": "*fp32", "o0_ptr": "*fp32", "o1_ptr": "*fp32",
         "N": "constexpr", "K0": "constexpr", "K1": "constexpr"},
        {"N": 128, "K0": 8, "K1": 32},
        num_warps=4,
    )
    elementwise_msl = _compile_to_msl(
        _topk_elementwise_followup_kernel,
        {"x_ptr": "*fp32", "o_ptr": "*fp32",
         "N": "constexpr", "K": "constexpr"},
        {"N": 128, "K": 8},
        num_warps=4,
    )
    reduce_msl = _compile_to_msl(
        _topk_reduce_followup_kernel,
        {"x_ptr": "*fp32", "sum_ptr": "*fp32",
         "N": "constexpr", "K": "constexpr"},
        {"N": 128, "K": 8},
        num_warps=4,
    )
    mixed_msl = _compile_to_msl(
        _mixed_sort_topk_kernel,
        {"x_ptr": "*fp32", "sort_ptr": "*fp32", "topk_ptr": "*fp32",
         "N": "constexpr", "K": "constexpr"},
        {"N": 128, "K": 8},
        num_warps=4,
    )
    assert "threadgroup_barrier" in rank2_msl
    assert "threadgroup_barrier" in rank3_msl
    assert rank3_multiband_msl.count("threadgroup_barrier") >= 2
    assert two_msl.count("threadgroup_barrier") >= 2
    assert elementwise_msl.count("threadgroup_barrier") >= 2
    assert reduce_msl.count("threadgroup_barrier") >= 2
    assert mixed_msl.count("threadgroup_barrier") >= 2


@pytest.mark.parametrize("N,K,num_warps,descending", [
    (8, 4, 1, False),
    (16, 8, 1, True),
    (128, 8, 4, False),
    (512, 8, 8, True),
])
def test_topk_exact_permutation(N, K, num_warps, descending):
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("Metal backend requires an MPS-enabled PyTorch")
    torch.manual_seed(N * 97 + K)
    values = torch.randperm(N).to(torch.float32)
    out = torch.empty(K, dtype=torch.float32, device="mps")
    _topk_v2_kernel[(1,)](values.to("mps"), out, N, K, descending,
                        num_warps=num_warps)
    torch.mps.synchronize()
    expected = values.sort(descending=descending).values[:K]
    assert torch.equal(out.cpu(), expected)


_TOPK_SMALL_CASES = [
    (N, K, num_warps, descending, dtype)
    for N in (2, 4, 8, 16, 32)
    for K in (1, 2, 4, 8, 16, 32)
    if K <= N
    for num_warps in (1, 4)
    for descending in (False, True)
    for dtype in ("i32", "fp32")
]


@pytest.mark.parametrize(
    "N,K,num_warps,descending,dtype", _TOPK_SMALL_CASES)
def test_topk_small_exhaustive(N, K, num_warps, descending, dtype):
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("Metal backend requires an MPS-enabled PyTorch")
    torch.manual_seed(N * 4099 + K * 31 + num_warps)
    values = torch.randperm(N)
    values = values.to(torch.int32 if dtype == "i32" else torch.float32)
    values -= N // 2
    out = torch.empty(K, dtype=values.dtype, device="mps")
    _topk_v2_kernel[(1,)](values.to("mps"), out, N, K, descending,
                        num_warps=num_warps)
    torch.mps.synchronize()
    expected = values.sort(descending=descending).values[:K]
    assert torch.equal(out.cpu(), expected)


# Elements per thread, N / (32 * num_warps), is the axis this path breaks
# along, so the cells below deliberately span one, two and four of them at
# several k. Everything at four with k >= 4 used to return WRONG VALUES because
# the exchange allocator reused a buffer a live read still needed; those cells
# are the regression for that fix and were failing before it.
#
# The envelope is bounded by the 32 KiB threadgroup budget rather than by
# elements per thread. While the exchange scratch fits without buffer reuse
# every publish gets its own buffer and no reuse hazard exists. Above that the
# allocator falls back to two alternating slots per pool, and round-robin is
# only safe while the publishes sharing a pool alternate strictly. Keying those
# pools by element type ALONE broke that: a hypercube reduce interleaves
# two-element direction-table publishes between the tile publishes of the same
# element type, flipping the parity and handing a comparator the slot it reads.
# The last four cells below are over budget and exercise that fallback; they
# returned duplicated values until the pool key gained the element count.
@pytest.mark.parametrize("N,K,num_warps,descending,dtype", [
    (64, 2, 1, True, "fp16"),
    (128, 8, 4, False, "bf16"),
    (256, 128, 4, True, "i32"),
    (512, 8, 8, False, "fp32"),
    (1024, 2, 8, True, "i32"),
    (128, 8, 1, True, "fp32"),
    (128, 32, 1, False, "i32"),
    (256, 16, 2, False, "fp32"),
    (512, 32, 4, True, "fp32"),
    (512, 32, 4, False, "i32"),
    (512, 64, 4, True, "fp32"),
    (1024, 32, 4, True, "fp32"),
])
def test_topk_multiband_dtypes(N, K, num_warps, descending, dtype):
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("Metal backend requires an MPS-enabled PyTorch")
    torch_dtype = {
        "i32": torch.int32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[dtype]
    torch.manual_seed(N * 101 + K)
    values = torch.randperm(N).to(torch_dtype)
    if dtype == "i32":
        values -= N // 2
    out = torch.empty(K, dtype=torch_dtype, device="mps")
    _topk_v2_kernel[(1,)](values.to("mps"), out, N, K, descending,
                        num_warps=num_warps)
    torch.mps.synchronize()
    expected = values.sort(descending=descending).values[:K]
    assert torch.equal(out.cpu(), expected)


@pytest.mark.parametrize("M,N,K,num_warps,descending", [
    (2, 64, 8, 4, True),
    (8, 64, 8, 8, False),
    (16, 32, 4, 8, True),
])
def test_topk_rank2_exact_rows(M, N, K, num_warps, descending):
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("Metal backend requires an MPS-enabled PyTorch")
    values = torch.arange(M * N, dtype=torch.float32).reshape(M, N)
    values = values.flip(-1) + torch.arange(M).reshape(M, 1) / 1024
    out = torch.empty(M, K, dtype=torch.float32, device="mps")
    _topk_rank2_v2_kernel[(1,)](values.to("mps"), out, M, N, K, descending,
                              num_warps=num_warps)
    torch.mps.synchronize()
    expected = values.sort(dim=-1, descending=descending).values[..., :K]
    assert torch.equal(out.cpu(), expected)


@pytest.mark.parametrize("B,M,N,K,num_warps,descending", [
    (2, 4, 64, 8, 8, True),
    (4, 2, 128, 8, 8, False),
])
def test_topk_rank3_exact_rows(B, M, N, K, num_warps, descending):
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("Metal backend requires an MPS-enabled PyTorch")
    values = torch.arange(B * M * N, dtype=torch.float32).reshape(B, M, N)
    values = values.flip(-1) + torch.arange(B * M).reshape(B, M, 1) / 1024
    out = torch.empty(B, M, K, dtype=torch.float32, device="mps")
    _topk_rank3_v2_kernel[(1,)](values.to("mps"), out, B, M, N, K, descending,
                              num_warps=num_warps)
    torch.mps.synchronize()
    expected = values.sort(dim=-1, descending=descending).values[..., :K]
    assert torch.equal(out.cpu(), expected)


def test_two_topk_exact_results():
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("Metal backend requires an MPS-enabled PyTorch")
    N, K0, K1 = 128, 8, 32
    torch.manual_seed(9381)
    values = torch.randperm(N).to(torch.float32)
    out0 = torch.empty(K0, dtype=torch.float32, device="mps")
    out1 = torch.empty(K1, dtype=torch.float32, device="mps")
    _two_topk_kernel[(1,)](values.to("mps"), out0, out1, N, K0, K1,
                            num_warps=4)
    torch.mps.synchronize()
    assert torch.equal(out0.cpu(), values.sort(descending=True).values[:K0])
    assert torch.equal(out1.cpu(), values.sort().values[:K1])


def test_topk_followup_elementwise_and_reduce():
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("Metal backend requires an MPS-enabled PyTorch")
    N, K = 128, 8
    values = torch.randperm(N).to(torch.float32) - 64
    out = torch.empty(K, dtype=torch.float32, device="mps")
    total = torch.empty(1, dtype=torch.float32, device="mps")
    device_values = values.to("mps")
    _topk_elementwise_followup_kernel[(1,)](device_values, out, N, K,
                                             num_warps=4)
    _topk_reduce_followup_kernel[(1,)](device_values, total, N, K,
                                        num_warps=4)
    torch.mps.synchronize()
    selected = values.sort(descending=True).values[:K]
    assert torch.equal(out.cpu(), selected * 2 - 1)
    assert torch.equal(total.cpu(), selected.sum().reshape(1))


def test_mixed_sort_and_topk_results():
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("Metal backend requires an MPS-enabled PyTorch")
    N, K = 128, 8
    torch.manual_seed(611)
    values = torch.randperm(N).to(torch.float32)
    sorted_out = torch.empty(N, dtype=torch.float32, device="mps")
    topk_out = torch.empty(K, dtype=torch.float32, device="mps")
    _mixed_sort_topk_kernel[(1,)](values.to("mps"), sorted_out, topk_out,
                                   N, K, num_warps=4)
    torch.mps.synchronize()
    assert torch.equal(sorted_out.cpu(), values.sort().values)
    assert torch.equal(topk_out.cpu(),
                       values.sort(descending=True).values[:K])


@pytest.mark.parametrize("descending,expected_bits", [
    (False, [-8388608, -1055916032, -1059061760, -1063256064,
             -1063256064, -1073741824, -1073741824, -1082130432,
             -2147483648, -2147483648, 1065353216, 1073741824,
             1073741824, 1084227584, 1084227584, 2139095040]),
    (True, [2139095040, 1084227584, 1084227584, 1073741824,
            1073741824, 1065353216, -2147483648, -2147483648,
            -1082130432, -1073741824, -1073741824, -1063256064,
            -1063256064, -1059061760, -1055916032, -8388608]),
])
def test_topk_duplicates_infinities_and_signed_zero(descending, expected_bits):
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("Metal backend requires an MPS-enabled PyTorch")
    values = torch.tensor([
        float("inf"), 5.0, 5.0, 2.0, 2.0, 1.0, 0.0, -0.0,
        -1.0, -2.0, -2.0, -5.0, -5.0, -7.0, -9.0, -float("inf"),
    ], dtype=torch.float32)
    out = torch.empty_like(values, device="mps")
    _topk_v2_kernel[(1,)](values.to("mps"), out, 16, 16, descending,
                        num_warps=4)
    torch.mps.synchronize()
    # Oracle captured from this same kernel under TRITON_INTERPRET=1. PyTorch's
    # sort is not a bit-level oracle for signed-zero ties.
    assert torch.equal(
        out.cpu().view(torch.int32),
        torch.tensor(expected_bits, dtype=torch.int32),
    )


@pytest.mark.parametrize("descending,expected_bits", [
    (False, [2143289344] * 8),
    (True, [2139095040, 0, 0, -8388608,
            2139095040, 0, 0, -8388608]),
])
def test_topk_nan_matches_triton_interpreter(descending, expected_bits):
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("Metal backend requires an MPS-enabled PyTorch")
    # Oracle captured from this same kernel under TRITON_INTERPRET=1. Do not
    # substitute torch.topk: bitonic compare/XOR intentionally has different
    # NaN propagation and tie behavior.
    values = torch.tensor([
        float("nan"), 4.0, -2.0, float("nan"), float("inf"), -0.0, 0.0,
        -float("inf"),
    ], dtype=torch.float32)
    out = torch.empty_like(values, device="mps")
    _topk_v2_kernel[(1,)](values.to("mps"), out, 8, 8, descending,
                        num_warps=1)
    torch.mps.synchronize()
    assert torch.equal(
        out.cpu().view(torch.int32),
        torch.tensor(expected_bits, dtype=torch.int32),
    )


@pytest.mark.parametrize("descending", [False, True])
def test_topk_masked_padding_sentinel(descending):
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("Metal backend requires an MPS-enabled PyTorch")
    valid_n, block_n, k = 53, 64, 8
    torch.manual_seed(733)
    values = torch.randperm(valid_n).to(torch.float32) - 27
    out = torch.empty(k, dtype=torch.float32, device="mps")
    _masked_topk_v2_kernel[(1,)](values.to("mps"), out, valid_n, block_n, k,
                               descending, num_warps=4)
    torch.mps.synchronize()
    expected = values.sort(descending=descending).values[:k]
    assert torch.equal(out.cpu(), expected)


# N and num_warps are swept together so the cells span ONE through THIRTY-TWO
# elements per thread, `N / (32 * num_warps)`. That axis is the one this path
# breaks along and the matrix used to sit entirely at one element per thread,
# which is how the band-write defect survived: the final phase loop ran
# `dst.elemPerThread` trips while the store addressed
# `localTid * tile.elemPerThread + iv`, so only the first half of every band
# reached the output.
#
# The cells above four elements per thread are the regression for two further
# defects, both of which returned a PERMUTED array rather than failing:
#   * the global load and store placed element `localTid * E + iv` while the
#     hypercube scratch placed it where the blocked layout does, and the two
#     agree only while the tile fits the CTA once. A second register band
#     interleaves the bands at the CTA-tile stride, so the phase sorted a
#     permuted array and the store permuted it again (N >= 256 at one warp);
#   * Triton CSEs one size-2 axis indicator across two hypercube axes once the
#     cube needs more register bits than the indicator's `#ttg.linear` basis
#     names, so the widest merge compared along the neighbouring axis
#     (N >= 512 at one warp).
# Both are invisible at 128 and below, which is why the matrix reaches 1024.
@pytest.mark.parametrize("num_warps", [1, 4])
@pytest.mark.parametrize("N", [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
@pytest.mark.parametrize("kernel,ref", [
    (_sort_kernel, lambda x: x.sort().values),
    (_sort_desc_kernel, lambda x: x.sort(descending=True).values),
    (_flip_kernel, lambda x: x.flip(0)),
])
def test_hypercube_sort_and_flip(kernel, ref, N, num_warps):
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("Metal backend requires an MPS-enabled PyTorch")
    torch.manual_seed(N * 31 + num_warps)
    # A permutation makes any misplaced element a hard mismatch.
    x = torch.randperm(N).float()
    out = torch.zeros(N, dtype=torch.float32, device="mps")
    kernel[(1,)](x.to("mps"), out, N, num_warps=num_warps)
    torch.mps.synchronize()
    assert torch.equal(out.cpu(), ref(x))


@triton.jit
def _rank3_reduce_kernel(x_ptr, o_ptr, B: tl.constexpr, M: tl.constexpr,
                         K: tl.constexpr, AXIS: tl.constexpr):
    v = tl.load(x_ptr + tl.arange(0, B)[:, None, None] * (M * K)
                + tl.arange(0, M)[None, :, None] * K
                + tl.arange(0, K)[None, None, :])
    r = tl.sum(v, axis=AXIS)
    if AXIS == 2:
        tl.store(o_ptr + tl.arange(0, B)[:, None] * M
                 + tl.arange(0, M)[None, :], r)
    elif AXIS == 1:
        tl.store(o_ptr + tl.arange(0, B)[:, None] * K
                 + tl.arange(0, K)[None, :], r)
    else:
        tl.store(o_ptr + tl.arange(0, M)[:, None] * K
                 + tl.arange(0, K)[None, :], r)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_rank3_reduce_every_axis(axis):
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("Metal backend requires an MPS-enabled PyTorch")
    B, M, K = 2, 4, 8              # 64 elements, one per thread at num_warps=4
    torch.manual_seed(axis)
    x = torch.randn(B, M, K, dtype=torch.float32)
    shape = {0: (M, K), 1: (B, K), 2: (B, M)}[axis]
    out = torch.zeros(shape, dtype=torch.float32, device="mps")
    _rank3_reduce_kernel[(1,)](x.to("mps"), out, B, M, K, axis, num_warps=4)
    torch.mps.synchronize()
    torch.testing.assert_close(out.cpu(), x.sum(axis), atol=1e-5, rtol=1e-5)


@triton.jit
def _rank3_xor_kernel(x_ptr, o_ptr, B: tl.constexpr, M: tl.constexpr,
                      K: tl.constexpr):
    v = tl.load(x_ptr + tl.arange(0, B)[:, None, None] * (M * K)
                + tl.arange(0, M)[None, :, None] * K
                + tl.arange(0, K)[None, None, :])
    tl.store(o_ptr + tl.arange(0, B)[:, None] * K + tl.arange(0, K)[None, :],
             tl.xor_sum(v, axis=1))


def test_rank3_xor_reduce_is_bit_exact():
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("Metal backend requires an MPS-enabled PyTorch")
    B, M, K = 2, 4, 8
    torch.manual_seed(5)
    x = torch.randint(0, 1 << 20, (B, M, K), dtype=torch.int32)
    out = torch.zeros(B, K, dtype=torch.int32, device="mps")
    _rank3_xor_kernel[(1,)](x.to("mps"), out, B, M, K, num_warps=4)
    torch.mps.synchronize()
    ref = torch.zeros(B, K, dtype=torch.int32)
    for m in range(M):
        ref ^= x[:, m, :]
    assert torch.equal(out.cpu(), ref)


# --- tl.device_print / tl.device_assert -------------------------------------
#
# MSL has neither printf nor a host-visible assert channel, so both were
# rejected — leaving no way to look inside a running kernel. They now write
# fixed-size records into a trailing buffer the launcher binds, and the host
# formats them after the dispatch: word 0 is an atomically incremented count,
# then six words per record, with only the message INDEX crossing to the device.
#
# The lane recorded is the threadgroup-LOCAL one, and a tile smaller than the
# threadgroup guards its surplus lanes — without that guard an assert fired for
# every lane past the tile, on garbage those lanes never loaded.


@triton.jit
def _device_print_kernel(x_ptr, o_ptr, N: tl.constexpr):
    pid = tl.program_id(0)
    i = tl.arange(0, N)
    v = tl.load(x_ptr + pid * N + i)
    tl.device_print("v", v)
    tl.store(o_ptr + pid * N + i, v)


@triton.jit
def _device_assert_kernel(x_ptr, o_ptr, N: tl.constexpr):
    i = tl.arange(0, N)
    v = tl.load(x_ptr + i)
    tl.device_assert(v >= 0.0, "must be non-negative")
    tl.store(o_ptr + i, v)


def test_device_print_emits_one_record_per_element(capfd):
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("Metal backend requires an MPS-enabled PyTorch")
    N = 8
    x = torch.arange(2 * N, dtype=torch.float32, device="mps")
    out = torch.zeros(2 * N, dtype=torch.float32, device="mps")
    _device_print_kernel[(2,)](x, out, N)
    torch.mps.synchronize()
    assert torch.equal(out.cpu(), x.cpu())
    lines = [l for l in capfd.readouterr().err.splitlines() if l.startswith("pid (")]
    # One line per element of each program's tile, and no lines for the lanes
    # past it.
    assert len(lines) == 2 * N, lines[:4]
    assert "v: 0.0" in lines[0]
    # Sorted by (program, lane), so the log is stable run to run.
    assert lines[0].startswith("pid (0, 0, 0) idx (0)")
    assert lines[N].startswith("pid (1, 0, 0) idx (0)")


@pytest.mark.parametrize("N", [16, 32])
def test_device_assert_stays_quiet_when_it_holds(N):
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("Metal backend requires an MPS-enabled PyTorch")
    # N == 32 is a FULL tile at num_warps=1, so the predicate is unguarded —
    # which is where a `cond ^ true` that always evaluated true showed up.
    x = torch.rand(N, dtype=torch.float32, device="mps")
    out = torch.zeros(N, dtype=torch.float32, device="mps")
    _device_assert_kernel[(1,)](x, out, N, num_warps=1, debug=True)
    torch.mps.synchronize()
    assert torch.equal(out.cpu(), x.cpu())


def test_device_assert_raises_with_its_message():
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("Metal backend requires an MPS-enabled PyTorch")
    x = -torch.rand(16, dtype=torch.float32, device="mps") - 1.0
    out = torch.zeros(16, dtype=torch.float32, device="mps")
    with pytest.raises(RuntimeError, match="must be non-negative"):
        _device_assert_kernel[(1,)](x, out, 16, debug=True)
        torch.mps.synchronize()


# --- tt.call: @triton.jit(noinline=True) ------------------------------------
#
# `noinline` is a compile-time hint — it keeps Triton's own inliner off the
# callee so the IR stays small — not a semantic requirement, and this backend
# emits one MSL kernel per `tt.func` with no device-function surface. So the
# call is inlined at the IR level instead of being rejected; the MSL compiler
# inlines small functions anyway.
#
# The fixed point matters: a callee body can itself contain a call, and the copy
# cloned into the caller carries it along, so one pass over the call list left
# the nested case still holding a `tt.call`.


@triton.jit(noinline=True)
def _noinline_scale(v, k):
    return v * k + 1.0


@triton.jit(noinline=True)
def _noinline_inner(v):
    return v * 3.0


@triton.jit(noinline=True)
def _noinline_outer(v):
    return _noinline_inner(v) + 1.0


@triton.jit
def _call_once_kernel(x_ptr, o_ptr, N: tl.constexpr):
    i = tl.arange(0, N)
    acc = tl.load(x_ptr + i)
    tl.store(o_ptr + i, acc + _noinline_scale(tl.sum(acc, axis=0), 2.0))


@triton.jit
def _call_nested_kernel(x_ptr, o_ptr, N: tl.constexpr):
    i = tl.arange(0, N)
    acc = tl.load(x_ptr + i)
    tl.store(o_ptr + i, acc + _noinline_outer(tl.sum(acc, axis=0)))


@pytest.mark.parametrize("num_warps", [1, 4])
def test_noinline_call_is_inlined(num_warps):
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("Metal backend requires an MPS-enabled PyTorch")
    N = 64
    torch.manual_seed(num_warps)
    x = torch.rand(N, dtype=torch.float32)
    out = torch.zeros(N, dtype=torch.float32, device="mps")
    _call_once_kernel[(1,)](x.to("mps"), out, N, num_warps=num_warps)
    torch.mps.synchronize()
    expected = x + (x.sum() * 2.0 + 1.0)
    torch.testing.assert_close(out.cpu(), expected, atol=1e-5, rtol=1e-5)


def test_noinline_call_of_a_noinline_callee():
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("Metal backend requires an MPS-enabled PyTorch")
    N = 64
    torch.manual_seed(9)
    x = torch.rand(N, dtype=torch.float32)
    out = torch.zeros(N, dtype=torch.float32, device="mps")
    _call_nested_kernel[(1,)](x.to("mps"), out, N)
    torch.mps.synchronize()
    expected = x + (x.sum() * 3.0 + 1.0)
    torch.testing.assert_close(out.cpu(), expected, atol=1e-5, rtol=1e-5)


# --- i16 and torch.bool tensors --------------------------------------------
#
# i16 had no Metal storage type (`Metal_Type` has ui16 but no signless i16), and
# a `torch.bool` argument was rejected outright: Triton re-types the POINTER at
# every access (`ptr<i1>` -> `ptr<i8>`, since the buffer really is a byte per
# element) while the argument stays `!tt.ptr<i1>`, so the memref kept an i1
# pointee and the per-element offset ended up underneath the cast where no index
# path could see it.


@triton.jit
def _i16_add_kernel(x_ptr, y_ptr, o_ptr, N: tl.constexpr):
    i = tl.arange(0, N)
    tl.store(o_ptr + i, tl.load(x_ptr + i) + tl.load(y_ptr + i))


@triton.jit
def _i16_masked_kernel(x_ptr, o_ptr, N: tl.constexpr, M):
    i = tl.arange(0, N)
    v = tl.load(x_ptr + i, mask=i < M, other=7)
    tl.store(o_ptr + i, v * 2, mask=i < M)


@pytest.mark.parametrize("num_warps", [1, 4])
def test_i16_elementwise(num_warps):
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("Metal backend requires an MPS-enabled PyTorch")
    N = 64
    torch.manual_seed(num_warps)
    x = torch.randint(-300, 300, (N,), dtype=torch.int16)
    y = torch.randint(-300, 300, (N,), dtype=torch.int16)
    out = torch.zeros(N, dtype=torch.int16, device="mps")
    _i16_add_kernel[(1,)](x.to("mps"), y.to("mps"), out, N,
                          num_warps=num_warps)
    torch.mps.synchronize()
    assert torch.equal(out.cpu(), x + y)


def test_i16_masked_load_and_store():
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("Metal backend requires an MPS-enabled PyTorch")
    N, M = 64, 30
    torch.manual_seed(16)
    x = torch.randint(-300, 300, (N,), dtype=torch.int16)
    out = torch.zeros(N, dtype=torch.int16, device="mps")
    _i16_masked_kernel[(1,)](x.to("mps"), out, N, M)
    torch.mps.synchronize()
    expected = torch.zeros(N, dtype=torch.int16)
    expected[:M] = x[:M] * 2
    assert torch.equal(out.cpu(), expected)


@triton.jit
def _tensor_pointer_bitcast_kernel(X, Bits, Roundtrip, N: tl.constexpr):
    offsets = tl.arange(0, 128) + 1
    bits = tl.load((X + offsets).to(tl.pointer_type(tl.int32)),
                   mask=offsets < N, other=0)
    tl.store(Bits + offsets, bits, mask=offsets < N)
    tl.store((Roundtrip + offsets).to(tl.pointer_type(tl.int32)),
             bits, mask=offsets < N)


@pytest.mark.parametrize("num_warps", [1, 4])
def test_tensor_pointer_reinterpretation_rejected_before_lowering(num_warps, capfd):
    with pytest.raises(RuntimeError, match="convert-tritongpu-to-metal failed"):
        _compile_to_msl(
            _tensor_pointer_bitcast_kernel,
            {"X": "*fp32", "Bits": "*i32", "Roundtrip": "*fp32"},
            {"N": 97}, num_warps=num_warps)
    assert "tensor pointer reinterpretation requires matching storage types" in capfd.readouterr().err


@triton.jit
def _bool_copy_kernel(x_ptr, o_ptr, N: tl.constexpr):
    i = tl.arange(0, N)
    tl.store(o_ptr + i, tl.load(x_ptr + i))


@triton.jit
def _bool_predicate_kernel(x_ptr, o_ptr, N: tl.constexpr):
    i = tl.arange(0, N)
    tl.store(o_ptr + i, tl.load(x_ptr + i) > 0.0)


@triton.jit
def _bool_mask_kernel(m_ptr, a_ptr, b_ptr, o_ptr, N: tl.constexpr):
    i = tl.arange(0, N)
    tl.store(o_ptr + i,
             tl.where(tl.load(m_ptr + i), tl.load(a_ptr + i),
                      tl.load(b_ptr + i)))


@pytest.mark.parametrize("num_warps", [1, 4])
def test_bool_tensor_round_trip(num_warps):
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("Metal backend requires an MPS-enabled PyTorch")
    N = 64
    torch.manual_seed(num_warps)
    x = torch.randint(0, 2, (N,)) == 1
    out = torch.zeros(N, dtype=torch.bool, device="mps")
    _bool_copy_kernel[(1,)](x.to("mps"), out, N, num_warps=num_warps)
    torch.mps.synchronize()
    assert torch.equal(out.cpu(), x)


def test_bool_tensor_as_output_of_a_comparison():
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("Metal backend requires an MPS-enabled PyTorch")
    N = 64
    torch.manual_seed(2)
    x = torch.randn(N, dtype=torch.float32)
    out = torch.zeros(N, dtype=torch.bool, device="mps")
    _bool_predicate_kernel[(1,)](x.to("mps"), out, N)
    torch.mps.synchronize()
    assert torch.equal(out.cpu(), x > 0)


def test_bool_tensor_as_a_mask():
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("Metal backend requires an MPS-enabled PyTorch")
    N = 64
    torch.manual_seed(3)
    m = torch.randint(0, 2, (N,)) == 1
    a = torch.randn(N, dtype=torch.float32)
    b = torch.randn(N, dtype=torch.float32)
    out = torch.zeros(N, dtype=torch.float32, device="mps")
    _bool_mask_kernel[(1,)](m.to("mps"), a.to("mps"), b.to("mps"), out, N)
    torch.mps.synchronize()
    assert torch.equal(out.cpu(), torch.where(m, a, b))


# --- i64 reduce ------------------------------------------------------------
#
# "reduce dtype must be f32 or i32" was a real wall: PyTorch hands out int64 for
# every default integer tensor, so `tl.sum` over one is an ordinary thing to
# write. MSL has native `long` arithmetic and `max`, and `Metal_Type` admits
# ui64/si64, so the butterfly only had to learn the width — plus the loop
# accumulator gates, since a BLOCK > tpb reduce folds through an `scf.for` whose
# iter_arg is as wide as the tensor (the emitter hit an UNREACHABLE there).
#
# Values above 2**32 are deliberate: a 32-bit truncation anywhere would show up
# as a hard mismatch rather than as a rounding difference.


@triton.jit
def _reduce_i64_kernel(x_ptr, o_ptr, N: tl.constexpr, KIND: tl.constexpr):
    v = tl.load(x_ptr + tl.arange(0, N))
    if KIND == 0:
        tl.store(o_ptr, tl.sum(v, axis=0))
    elif KIND == 1:
        tl.store(o_ptr, tl.max(v, axis=0))
    elif KIND == 2:
        tl.store(o_ptr, tl.min(v, axis=0))
    else:
        tl.store(o_ptr, tl.xor_sum(v, axis=0))


@pytest.mark.parametrize("num_warps", [1, 4])
@pytest.mark.parametrize("N", [32, 256, 1024])
@pytest.mark.parametrize("kind", [0, 1, 2, 3])
def test_reduce_i64_matches_torch(kind, N, num_warps):
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("Metal backend requires an MPS-enabled PyTorch")
    torch.manual_seed(N + kind)
    lo = 0 if kind == 3 else -(2 ** 40)
    x = torch.randint(lo, 2 ** 40, (N,), dtype=torch.int64)
    out = torch.zeros(1, dtype=torch.int64, device="mps")
    _reduce_i64_kernel[(1,)](x.to("mps"), out, N, kind, num_warps=num_warps)
    torch.mps.synchronize()
    if kind == 0:
        expected = x.sum().item()
    elif kind == 1:
        expected = x.max().item()
    elif kind == 2:
        expected = x.min().item()
    else:
        expected = 0
        for v in x.tolist():
            expected ^= v
    assert out.cpu()[0].item() == expected


# --- rank-2 tl.gather ------------------------------------------------------
#
# The original gather is rank-1 axis-0: it stages the tile through a fill loop
# that can cover BLOCK > tpb. Any other rank or axis was a named rejection. The
# staged fallback covers them at one element per thread, reading the element the
# per-lane index names out of the published tile.


@triton.jit
def _gather_rank2_kernel(x_ptr, idx_ptr, o_ptr, M: tl.constexpr,
                         N: tl.constexpr, AXIS: tl.constexpr):
    off = tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :]
    tl.store(o_ptr + off,
             tl.gather(tl.load(x_ptr + off), tl.load(idx_ptr + off), axis=AXIS))


@pytest.mark.parametrize("num_warps", [1, 4])
@pytest.mark.parametrize("axis", [0, 1])
def test_gather_rank2_both_axes(axis, num_warps):
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("Metal backend requires an MPS-enabled PyTorch")
    M, N = 4, 8
    torch.manual_seed(axis * 7 + num_warps)
    x = torch.randn(M, N, dtype=torch.float32)
    idx = torch.randint(0, M if axis == 0 else N, (M, N), dtype=torch.int32)
    out = torch.zeros(M, N, dtype=torch.float32, device="mps")
    _gather_rank2_kernel[(1,)](x.to("mps"), idx.to("mps"), out, M, N, axis,
                               num_warps=num_warps)
    torch.mps.synchronize()
    assert torch.equal(out.cpu(), torch.gather(x, axis, idx.long()))


# --- rank-2 tl.cumsum / tl.cumprod -----------------------------------------
#
# Scan was rank-1 axis-0 only: its distributed prefix-sum template builds one
# threadgroup buffer per BLOCK-long run, which means nothing once the tile has
# rows. The staged fallback publishes the tile and lets each lane accumulate its
# own axis prefix, so both axes work. A scan's result has the same shape as its
# source, so unlike the reduce there is no output-mapping question.


@triton.jit
def _cumsum_rank2_kernel(x_ptr, o_ptr, M: tl.constexpr, N: tl.constexpr,
                         AXIS: tl.constexpr):
    off = tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :]
    tl.store(o_ptr + off, tl.cumsum(tl.load(x_ptr + off), axis=AXIS))


@triton.jit
def _cumprod_rank2_kernel(x_ptr, o_ptr, M: tl.constexpr, N: tl.constexpr,
                          AXIS: tl.constexpr):
    off = tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :]
    tl.store(o_ptr + off, tl.cumprod(tl.load(x_ptr + off), axis=AXIS))


@pytest.mark.parametrize("shape", [(4, 8), (8, 8), (8, 16)])
@pytest.mark.parametrize("num_warps", [1, 4])
@pytest.mark.parametrize("axis", [0, 1])
def test_cumsum_rank2_both_axes(shape, axis, num_warps):
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("Metal backend requires an MPS-enabled PyTorch")
    M, N = shape
    torch.manual_seed(M * 100 + N * 10 + axis * 2 + num_warps)
    x = torch.randn(M, N, dtype=torch.float32)
    out = torch.zeros(M, N, dtype=torch.float32, device="mps")
    _cumsum_rank2_kernel[(1,)](x.to("mps"), out, M, N, axis,
                               num_warps=num_warps)
    torch.mps.synchronize()
    torch.testing.assert_close(out.cpu(), x.cumsum(axis), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("shape", [(4, 8), (8, 8)])
@pytest.mark.parametrize("axis", [0, 1])
def test_cumprod_rank2_is_bit_exact(shape, axis):
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("Metal backend requires an MPS-enabled PyTorch")
    M, N = shape
    torch.manual_seed(M + N + axis)
    # Small integers keep every partial product exactly representable, so the
    # identity element (1, not 0) is pinned by an equality rather than a
    # tolerance.
    x = torch.randint(1, 4, (M, N)).float()
    out = torch.zeros(M, N, dtype=torch.float32, device="mps")
    _cumprod_rank2_kernel[(1,)](x.to("mps"), out, M, N, axis, num_warps=4)
    torch.mps.synchronize()
    assert torch.equal(out.cpu(), x.cumprod(axis))


def test_cumsum_rank2_i32_bridges_through_storage_type():
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("Metal backend requires an MPS-enabled PyTorch")
    M, N = 4, 8
    torch.manual_seed(3)
    x = torch.randint(-50, 50, (M, N), dtype=torch.int32)
    out = torch.zeros(M, N, dtype=torch.int32, device="mps")
    _cumsum_rank2_kernel[(1,)](x.to("mps"), out, M, N, 1, num_warps=4)
    torch.mps.synchronize()
    assert torch.equal(out.cpu(), x.cumsum(1, dtype=torch.int32))


@triton.jit
def _xor_scan_combine(a, b):
    return a ^ b


@triton.jit
def _xor_scan_tile_kernel(x_ptr, out_ptr, SHAPE: tl.constexpr,
                          BLOCK: tl.constexpr, AXIS: tl.constexpr,
                          REVERSE: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK).reshape(SHAPE)
    x = tl.load(x_ptr + off)
    result = tl.associative_scan(x, AXIS, _xor_scan_combine, reverse=REVERSE)
    tl.store(out_ptr + off, result)


@pytest.mark.parametrize("shape,axis,num_warps", [
    (shape, axis, nw)
    for shape in ((4, 8), (8, 8), (8, 16))
    for axis in range(len(shape))
    for nw in (1, 4)
] + [
    (shape, axis, nw) for shape in ((2, 4, 4), (4, 4, 8))
    for axis in (0, 1, 2) for nw in (1, 4)
] + [((2, 64), 1, 4), ((64, 2), 0, 4), ((64, 128), 0, 4)])
@pytest.mark.parametrize("reverse", [False, True])
def test_xor_scan_staged_axes_are_bit_exact(shape, axis, num_warps, reverse):
    import math

    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("Metal backend requires an MPS-enabled PyTorch")
    block = math.prod(shape)
    generator = torch.Generator().manual_seed(block + axis)
    x = torch.randint(-(2**31), 2**31, (3, *shape), dtype=torch.int32,
                      generator=generator)
    x.flatten()[:4] = torch.tensor([0, -1, -(2**31), 2**31 - 1])
    expected = x.clone()
    slices = expected.unbind(axis + 1)
    indices = range(len(slices) - 2, -1, -1) if reverse else range(1, len(slices))
    for i in indices:
        slices[i].bitwise_xor_(slices[i + 1 if reverse else i - 1])
    out = torch.empty_like(x, device="mps")
    _xor_scan_tile_kernel[(3,)](x.to("mps"), out, shape, block, axis, reverse,
                                num_warps=num_warps)
    torch.mps.synchronize()
    assert torch.equal(out.cpu(), expected)


@pytest.mark.parametrize("shape,block,axis,diagnostic", [
    ((32,), 32, 0, "scan combine must be add or mul"),
    ((1, 128), 128, 1, "rank >= 2 scan is implemented"),
    ((8192, 2), 16384, 1, "rank >= 2 scan is implemented"),
])
def test_xor_scan_outside_staged_envelope_is_rejected(
        shape, block, axis, diagnostic, capfd):
    with pytest.raises(Exception):
        _compile_to_msl(
            _xor_scan_tile_kernel, {"x_ptr": "*i32", "out_ptr": "*i32"},
            {"SHAPE": shape, "BLOCK": block, "AXIS": axis, "REVERSE": False},
            num_warps=1)
    assert diagnostic in capfd.readouterr().err


@triton.jit
def _staged_scan_add(a, b):
    return a + b


@triton.jit
def _staged_scan_mul(a, b):
    return a * b


@triton.jit
def _staged_scan_masked_kernel(x_ptr, out_ptr, valid, SHAPE: tl.constexpr,
                               BLOCK: tl.constexpr, AXIS: tl.constexpr,
                               REVERSE: tl.constexpr, PRODUCT: tl.constexpr):
    local = tl.arange(0, BLOCK).reshape(SHAPE)
    off = tl.program_id(0) * BLOCK + local
    x = tl.load(x_ptr + off, local < valid, other=0)
    # Compute the source before publication; preserve the input element type.
    x = (x + 1).to(x.dtype)
    if PRODUCT:
        y = tl.associative_scan(x, AXIS, _staged_scan_mul, reverse=REVERSE)
    else:
        y = tl.associative_scan(x, AXIS, _staged_scan_add, reverse=REVERSE)
    tl.store(out_ptr + off, y, local < valid)


@pytest.mark.parametrize("shape,axis,num_warps", [
    ((8, 16), 0, 1), ((8, 16), 1, 1),
    ((4, 4, 16), 0, 4), ((4, 4, 16), 1, 4), ((4, 4, 16), 2, 4),
])
@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16", "int32"])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("product", [False, True])
def test_staged_scan_multiband_masked_computed_source(
        shape, axis, num_warps, dtype, reverse, product):
    import math

    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("Metal backend requires an MPS-enabled PyTorch")
    block = math.prod(shape)
    valid = block - 7
    generator = torch.Generator().manual_seed(block + axis)
    if dtype == "int32":
        x = torch.randint(-(2**31), 2**31, (3, *shape), dtype=torch.int32,
                          generator=generator)
    else:
        # Small exact values isolate mapping/synchronization from reassociation.
        x = torch.randint(-2, 1, (3, *shape), generator=generator).to(getattr(torch, dtype))
    source = x.clone().reshape(3, block)
    source[:, valid:] = 0
    expected = (source.reshape(3, *shape) + 1).to(x.dtype)
    slices = expected.unbind(axis + 1)
    indices = range(len(slices) - 2, -1, -1) if reverse else range(1, len(slices))
    for i in indices:
        previous = slices[i + 1 if reverse else i - 1]
        if product:
            slices[i].mul_(previous)
        else:
            slices[i].add_(previous)
    expected.reshape(3, block)[:, valid:] = -13
    out = torch.full_like(x, -13, device="mps")
    _staged_scan_masked_kernel[(3,)](
        x.to("mps"), out, valid, shape, block, axis, reverse, product,
        num_warps=num_warps)
    torch.mps.synchronize()
    assert torch.equal(out.cpu(), expected)


@triton.jit
def _staged_scan_pair_kernel(x_ptr, first_ptr, second_ptr,
                             CHAINED: tl.constexpr, REVERSE: tl.constexpr):
    off = tl.program_id(0) * 128 + tl.arange(0, 128).reshape((8, 16))
    x = tl.load(x_ptr + off)
    first = tl.cumsum(x, 0, reverse=REVERSE)
    tl.store(first_ptr + off, first)
    if CHAINED:
        second = tl.cumsum(first, 1, reverse=not REVERSE)
    else:
        second = tl.cumsum(x + 3, 1, reverse=not REVERSE)
    tl.store(second_ptr + off, second)


@pytest.mark.parametrize("chained", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_staged_scan_multiband_consecutive(chained, reverse):
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("Metal backend requires an MPS-enabled PyTorch")
    x = (torch.arange(3 * 128, dtype=torch.int32) % 19 - 9).reshape(3, 8, 16)

    def scan(value, axis, backward):
        if backward:
            return value.flip((axis,)).cumsum(axis, dtype=torch.int32).flip((axis,))
        return value.cumsum(axis, dtype=torch.int32)

    first = scan(x, 1, reverse)
    second = scan(first if chained else x + 3, 2, not reverse)
    out1 = torch.empty_like(x, device="mps")
    out2 = torch.empty_like(x, device="mps")
    _staged_scan_pair_kernel[(3,)](x.to("mps"), out1, out2, chained, reverse, num_warps=1)
    torch.mps.synchronize()
    assert torch.equal(out1.cpu(), first)
    assert torch.equal(out2.cpu(), second)


@triton.jit
def _staged_scan_nested_kernel(x_ptr, out_ptr, iterations):
    off = tl.arange(0, 128).reshape((8, 16))
    for i in range(iterations):
        x = tl.load(x_ptr + i * 128 + off)
        tl.store(out_ptr + i * 128 + off, tl.cumsum(x, 1))


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16", "int32"])
@pytest.mark.parametrize("iterations", [0, 1, 7])
def test_staged_scan_multiband_nested_loop(dtype, iterations):
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("Metal backend requires an MPS-enabled PyTorch")
    x = (torch.arange(7 * 128) % 7 - 3).reshape(7, 8, 16).to(getattr(torch, dtype))
    out = torch.full_like(x, -123, device="mps")
    _staged_scan_nested_kernel[(1,)](x.to("mps"), out, iterations, num_warps=1)
    torch.mps.synchronize()
    expected = torch.full_like(x, -123)
    expected[:iterations] = x[:iterations].cumsum(2, dtype=x.dtype)
    torch.testing.assert_close(out.cpu(), expected, atol=0, rtol=0)


@triton.jit
def _staged_scan_mixed_bands_kernel(x_ptr, small_ptr, large_ptr):
    small = tl.arange(0, 4)[:, None] * 8 + tl.arange(0, 8)[None, :]
    a = tl.load(x_ptr + small)
    tl.store(small_ptr + small, tl.cumsum(a, 0))
    large = tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :]
    b = tl.load(x_ptr + large)
    tl.store(large_ptr + large, tl.cumsum(b, 1, reverse=True))


@pytest.mark.parametrize("num_warps", [1, 4])
def test_staged_scan_multiband_with_one_band_scan(num_warps):
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("Metal backend requires an MPS-enabled PyTorch")
    x = (torch.arange(256, dtype=torch.int32) % 17 - 8).reshape(16, 16)
    small = torch.empty((4, 8), dtype=torch.int32, device="mps")
    large = torch.empty_like(x, device="mps")
    _staged_scan_mixed_bands_kernel[(1,)](x.to("mps"), small, large, num_warps=num_warps)
    torch.mps.synchronize()
    assert torch.equal(small.cpu(), x.flatten()[:32].reshape(4, 8).cumsum(0, dtype=torch.int32))
    assert torch.equal(large.cpu(), x.flip((1,)).cumsum(1, dtype=torch.int32).flip((1,)))


@triton.jit
def _staged_scan_snapshot_kernel(x_ptr, out_ptr):
    off = tl.arange(0, 128).reshape((8, 16))
    x = tl.load(x_ptr + off)
    result = tl.cumsum(x + 1, 1)
    tl.store(x_ptr + off, x + 5)
    tl.store(out_ptr + off, result + x)


def test_staged_scan_multiband_load_snapshot_is_not_replayed(capfd):
    with pytest.raises(Exception):
        _compile_to_msl(_staged_scan_snapshot_kernel,
                        {"x_ptr": "*i32", "out_ptr": "*i32"}, {}, num_warps=1)
    assert "staged scan cannot safely replay a cross-phase value" in capfd.readouterr().err


@triton.jit
def _staged_scan_rank3_ranges_kernel(x_ptr, out_ptr, AXIS: tl.constexpr):
    off = (tl.arange(0, 4)[:, None, None] * 128 +
           tl.arange(0, 8)[None, :, None] * 16 +
           tl.arange(0, 16)[None, None, :])
    x = tl.load(x_ptr + off)
    tl.store(out_ptr + off, tl.cumsum(x, AXIS, reverse=True))


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("num_warps", [1, 4])
def test_staged_scan_multiband_rank3_slice_ranges(axis, num_warps):
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("Metal backend requires an MPS-enabled PyTorch")
    x = (torch.arange(512, dtype=torch.int32) % 17 - 8).reshape(4, 8, 16)
    out = torch.empty_like(x, device="mps")
    _staged_scan_rank3_ranges_kernel[(1,)](x.to("mps"), out, axis, num_warps=num_warps)
    torch.mps.synchronize()
    expected = x.flip((axis,)).cumsum(axis, dtype=torch.int32).flip((axis,))
    assert torch.equal(out.cpu(), expected)


@triton.jit
def _staged_scan_independent_snapshot_kernel(x_ptr, saved_ptr, out_ptr):
    off = tl.arange(0, 128).reshape((8, 16))
    saved = tl.load(saved_ptr + off)
    x = tl.load(x_ptr + off)
    tl.store(saved_ptr + off, tl.full((8, 16), 7, tl.int32))
    result = tl.cumsum(x, 1)
    tl.store(out_ptr + off, result + saved)


def test_staged_scan_multiband_independent_snapshot_is_not_moved(capfd):
    with pytest.raises(Exception):
        _compile_to_msl(_staged_scan_independent_snapshot_kernel,
                        {"x_ptr": "*i32", "saved_ptr": "*i32", "out_ptr": "*i32"},
                        {}, num_warps=1)
    assert "staged scan cannot move a load across a write" in capfd.readouterr().err


def test_staged_scan_multiband_masked_scratch_budget_is_checked(capfd):
    with pytest.raises(Exception):
        _compile_to_msl(
            _staged_scan_masked_kernel,
            {"x_ptr": "*fp32", "out_ptr": "*fp32", "valid": "i32"},
            {"SHAPE": (64, 128), "BLOCK": 8192, "AXIS": 0,
             "REVERSE": False, "PRODUCT": False}, num_warps=1)
    assert "exceeding the 32768-byte budget" in capfd.readouterr().err


# --- tl.join / tl.split / tl.cat / tl.interleave ---------------------------
#
# Triton keeps these per-thread by choosing a layout where one thread owns both
# columns of its row. This backend does not honour a layout's thread
# assignment — it imposes `flat = localTid*E + iv` decomposed by `order` — so
# under ITS mapping the join result's element (i, k) sits in lane i*2 + k while
# source element i sits in lane i. They are real cross-lane moves and go
# through threadgroup memory, except under the sub-tpb companion mapping, where
# rank-1 values are already held per ROW and the join is a local select.
#
# Getting that distinction wrong is a silent wrong answer, not a crash: a
# differential sweep caught join returning a0,b0,a0,b0,a1,b1,a1,b1 for
# a0,b0,a1,b1,a2,b2,a3,b3. Hence the exact-equality assertions below across
# both regimes — sub-tpb tiles (N*2 < tpb) and the exactly-full tile
# (N*2 == tpb, reached by N=16/nw=1, N=32/nw=2, N=64/nw=4).


@triton.jit
def _join_kernel(x_ptr, o_ptr, N: tl.constexpr):
    a = tl.load(x_ptr + tl.arange(0, N))
    j = tl.join(a, a + 100.0)
    tl.store(o_ptr + tl.arange(0, N)[:, None] * 2 + tl.arange(0, 2)[None, :], j)


@triton.jit
def _split_kernel(x_ptr, o_ptr, N: tl.constexpr):
    off = tl.arange(0, N)[:, None] * 2 + tl.arange(0, 2)[None, :]
    a, b = tl.split(tl.load(x_ptr + off))
    tl.store(o_ptr + tl.arange(0, N), a * 10.0 + b)


@triton.jit
def _cat_kernel(x_ptr, o_ptr, N: tl.constexpr):
    a = tl.load(x_ptr + tl.arange(0, N))
    tl.store(o_ptr + tl.arange(0, 2 * N),
             tl.cat(a, a + 100.0, can_reorder=True))


@triton.jit
def _interleave_kernel(x_ptr, o_ptr, N: tl.constexpr):
    a = tl.load(x_ptr + tl.arange(0, N))
    tl.store(o_ptr + tl.arange(0, 2 * N), tl.interleave(a, a + 100.0))


def _shuffle_inputs(N):
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("Metal backend requires an MPS-enabled PyTorch")
    return torch.arange(2 * N, dtype=torch.float32, device="mps")


@pytest.mark.parametrize("num_warps", [1, 2, 4])
@pytest.mark.parametrize("N", [1, 2, 4, 8, 16])
def test_join_interleaves_two_rank1_tiles(N, num_warps):
    torch = pytest.importorskip("torch")
    x = _shuffle_inputs(N)
    out = torch.zeros(2 * N, dtype=torch.float32, device="mps")
    _join_kernel[(1,)](x, out, N, num_warps=num_warps)
    torch.mps.synchronize()
    expected = torch.stack([x[:N], x[:N] + 100.0], dim=1).reshape(-1)
    assert torch.equal(out.cpu(), expected.cpu())


@pytest.mark.parametrize("num_warps", [1, 2, 4])
@pytest.mark.parametrize("N", [1, 2, 4, 8, 16])
def test_split_recovers_both_columns(N, num_warps):
    torch = pytest.importorskip("torch")
    x = _shuffle_inputs(N)
    out = torch.zeros(N, dtype=torch.float32, device="mps")
    _split_kernel[(1,)](x, out, N, num_warps=num_warps)
    torch.mps.synchronize()
    expected = x[0:2 * N:2] * 10.0 + x[1:2 * N:2]
    assert torch.equal(out.cpu(), expected.cpu())


@pytest.mark.parametrize("num_warps", [1, 2, 4])
@pytest.mark.parametrize("N", [2, 8, 16])
def test_cat_concatenates_rank1_tiles(N, num_warps):
    torch = pytest.importorskip("torch")
    x = _shuffle_inputs(N)
    out = torch.zeros(2 * N, dtype=torch.float32, device="mps")
    _cat_kernel[(1,)](x, out, N, num_warps=num_warps)
    torch.mps.synchronize()
    # `can_reorder=True` licenses any permutation, so the multiset is the
    # contract; the concatenation order itself is not.
    expected = torch.cat([x[:N], x[:N] + 100.0]).cpu()
    assert torch.equal(out.cpu().sort().values, expected.sort().values)


@pytest.mark.parametrize("num_warps", [1, 2, 4])
@pytest.mark.parametrize("N", [2, 8, 16])
def test_interleave_matches_torch(N, num_warps):
    torch = pytest.importorskip("torch")
    x = _shuffle_inputs(N)
    out = torch.zeros(2 * N, dtype=torch.float32, device="mps")
    _interleave_kernel[(1,)](x, out, N, num_warps=num_warps)
    torch.mps.synchronize()
    expected = torch.stack([x[:N], x[:N] + 100.0], dim=1).reshape(-1)
    assert torch.equal(out.cpu(), expected.cpu())


@triton.jit
def _join_i32_kernel(x_ptr, o_ptr, N: tl.constexpr):
    a = tl.load(x_ptr + tl.arange(0, N))
    j = tl.join(a, a + 100)
    tl.store(o_ptr + tl.arange(0, N)[:, None] * 2 + tl.arange(0, 2)[None, :], j)


def test_join_i32_payload_stages_as_ui32():
    """The metal buffer ops reject signless i32, so an integer payload has to
    bridge through ui32 on the way into and out of the threadgroup slot."""
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("Metal backend requires an MPS-enabled PyTorch")
    N = 8
    x = torch.arange(N, dtype=torch.int32, device="mps")
    out = torch.zeros(2 * N, dtype=torch.int32, device="mps")
    _join_i32_kernel[(1,)](x, out, N)
    torch.mps.synchronize()
    expected = torch.stack([x.cpu(), (x + 100).cpu()], dim=1).reshape(-1)
    assert torch.equal(out.cpu(), expected)


@triton.jit
def _max_scan_combine(a, b):
    return tl.maximum(a, b)


@triton.jit
def _running_max_scan_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    tl.store(out_ptr + offs,
             tl.associative_scan(tl.load(x_ptr + offs), 0, _max_scan_combine))


def test_unsupported_scan_combine_is_rejected_not_crashed(capfd):
    """Only add and mul scans are implemented. A running MAXIMUM is an ordinary
    thing to write, and the decline lived inside ScanLowering — i.e. inside
    applyFullConversion, where a notifyMatchFailure segfaults the caller rather
    than raising. Asserts the caller survives with a named error."""
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("Metal backend requires an MPS-enabled PyTorch")
    x = torch.rand(64, dtype=torch.float32, device="mps")
    out = torch.zeros(64, dtype=torch.float32, device="mps")
    with pytest.raises(Exception):
        _running_max_scan_kernel[(1,)](x, out, BLOCK=64)
    assert "scan combine must be add or mul" in capfd.readouterr().err

    # Still usable afterwards: a rejection that poisons the context would be no
    # better than the crash it replaced.
    out2 = torch.zeros(16, dtype=torch.int32, device="mps")
    _arange_rank2_col_kernel[(1,)](out2, M=4, N=4)
    torch.mps.synchronize()
    assert out2.cpu().tolist() == [p % 4 for p in range(16)]


@triton.jit
def _subtpb_attention_kernel(Q, K, V, Out, M, N, d, sm_scale,
                             BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr):
    """The medium-softmax_attention.py shape, with the block sizes exposed."""
    pid = tl.program_id(0)
    offs_d = tl.arange(0, BLOCK_D)
    q = tl.load(Q + pid * d + offs_d, mask=offs_d < d, other=0.0)
    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    for start_n in range(0, N, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask2 = (offs_n[:, None] < N) & (offs_d[None, :] < d)
        k = tl.load(K + offs_n[:, None] * d + offs_d[None, :], mask=mask2,
                    other=0.0)
        qk = tl.sum(q[None, :] * k, axis=1) * sm_scale
        qk = tl.where(offs_n < N, qk, -float("inf"))
        m_prev = m_i
        m_i = tl.maximum(m_prev, tl.max(qk, axis=0))
        alpha = tl.exp(m_prev - m_i)
        p = tl.exp(qk - m_i)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        v = tl.load(V + offs_n[:, None] * d + offs_d[None, :], mask=mask2,
                    other=0.0)
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
    tl.store(Out + pid * d + offs_d, acc / l_i, mask=offs_d < d)


@pytest.mark.parametrize("d,block_n,num_warps", [
    # tile SMALLER than the threadgroup — every one of these returned wrong
    # numbers, with no diagnostic, before the four sites were made to agree.
    (2, 32, 4), (2, 16, 2), (4, 16, 4), (2, 32, 2), (2, 64, 1),
    # tile at or above tpb — correct before, and must stay byte-identical.
    (8, 32, 4), (4, 32, 4), (2, 32, 1), (16, 32, 4), (8, 16, 2),
])
def test_subtpb_rank1_crossing(d, block_n, num_warps):
    """A rank-2 tile smaller than the threadgroup puts row n on thread n*cols,
    not on thread n. A value that leaves the 2-D cone and is then combined with
    a rank-1 mask therefore reads under two different mappings, and
    `medium-softmax_attention.py` returned wrong numbers at head dim 2 under
    its own default num_warps.

    Both regimes are swept together on purpose: the fix makes four sites agree
    on the mapping, and getting any subset of them right leaves a different
    pair disagreeing — which shows up as one half of this parametrization or
    the other."""
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("Metal backend requires an MPS-enabled PyTorch")
    m, n = 4, 8
    torch.manual_seed(d * 100 + block_n + num_warps)
    q = torch.randn(m, d, dtype=torch.float32)
    k = torch.randn(n, d, dtype=torch.float32)
    v = torch.randn(n, d, dtype=torch.float32)
    out = torch.zeros(m, d, dtype=torch.float32, device="mps")
    _subtpb_attention_kernel[(m,)](q.to("mps"), k.to("mps"), v.to("mps"), out,
                                   m, n, d, d ** -0.5, BLOCK_N=block_n,
                                   BLOCK_D=d, num_warps=num_warps)
    torch.mps.synchronize()
    ref = torch.softmax((q @ k.t()) * (d ** -0.5), dim=-1) @ v
    torch.testing.assert_close(out.cpu(), ref, atol=2e-4, rtol=2e-4)


@triton.jit
def _subtpb_rowsum_kernel(x_ptr, o_ptr, M, N, BM: tl.constexpr,
                          BN: tl.constexpr):
    rm = tl.arange(0, BM)
    rn = tl.arange(0, BN)
    mask = (rm[:, None] < M) & (rn[None, :] < N)
    v = tl.load(x_ptr + rm[:, None] * N + rn[None, :], mask=mask, other=0.0)
    tl.store(o_ptr + rm, tl.sum(v, axis=1), mask=rm < M)


@pytest.mark.parametrize("bm,bn,num_warps", [
    # tile smaller than the threadgroup: 32 elements on 64 threads, and so on.
    (4, 8, 2), (4, 8, 4), (4, 8, 8), (2, 16, 2), (16, 4, 4), (32, 2, 4),
    (8, 16, 8), (16, 8, 8),
    # tile at or above tpb — unchanged emission, kept in the same sweep.
    (8, 8, 1), (16, 8, 2), (32, 8, 4),
])
def test_subtpb_rowsum(bm, bn, num_warps):
    """The masked store takes a different path than the unmasked one and needs
    the same exemption. With the guard left on, `localTid < rows` let only the
    lanes below the row count store, and every one of those sits in row 0: a
    4x8 rowsum wrote row 0 and left rows 1-3 at zero."""
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("Metal backend requires an MPS-enabled PyTorch")
    torch.manual_seed(bm * 100 + bn + num_warps)
    x = torch.rand(bm, bn, dtype=torch.float32)
    o = torch.zeros(bm, dtype=torch.float32, device="mps")
    _subtpb_rowsum_kernel[(1,)](x.to("mps"), o, bm, bn, BM=bm, BN=bn,
                                num_warps=num_warps)
    torch.mps.synchronize()
    torch.testing.assert_close(o.cpu(), x.sum(1), atol=1e-5, rtol=1e-5)


@triton.jit
def _unit_axis_reduce_kernel(x_ptr, o_ptr, N, BN: tl.constexpr):
    rn = tl.arange(0, BN)
    rd = tl.arange(0, 1)
    v = tl.load(x_ptr + rn[:, None] + rd[None, :], mask=rn[:, None] < N,
                other=0.0)
    tl.store(o_ptr + rn, tl.sum(v, axis=1), mask=rn < N)


@pytest.mark.parametrize("num_warps", [1, 2, 4])
def test_unit_axis_reduce_is_rejected_not_crash(num_warps):
    """Reducing an axis of extent 1 is a no-op, but the BLOCK x 1 tile it
    leaves has no lowering for its slice-encoded rank-1 values. It used to be
    declined DURING conversion, which in this backend aborts the process and
    poisons the context for the next compile — a head-dim sweep lost the whole
    interpreter at head dim 1. Asserted as a catchable rejection."""
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("Metal backend requires an MPS-enabled PyTorch")
    x = torch.rand(32, 1, dtype=torch.float32, device="mps")
    o = torch.zeros(32, dtype=torch.float32, device="mps")
    with pytest.raises(Exception):
        _unit_axis_reduce_kernel[(1,)](x, o, 32, BN=32, num_warps=num_warps)


@triton.jit
def _topk_beyond_phase_carry_kernel(x_ptr, o_ptr, N: tl.constexpr,
                                    K: tl.constexpr):
    v = tl.topk(tl.load(x_ptr + tl.arange(0, N)), K, descending=True)
    tl.store(o_ptr + tl.arange(0, K), v)


def test_exchange_phase_carry_rejection_leaves_the_context_usable():
    """A 2048-element topk on one warp is over the threadgroup budget, so its
    exchange reuses buffers and one of its phases can no longer rematerialize a
    read. That rejection is correct. It used to be emitted from the function
    lowering, long after `inlineRegionBefore` had moved the body into the new
    kernel op, and unwinding that move segfaulted on nine of thirty runs.

    Asserting the crash directly is useless: it is intermittent, and when it
    does fire it takes the whole pytest process with it. So assert the
    PREVENTING MECHANISM instead. A rejection raised before the conversion
    starts leaves the process healthy, so a later compile in the same
    interpreter still works and still gives the right answer; a rejection
    raised during conversion teardown does not.
    """
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("Metal backend requires an MPS-enabled PyTorch")
    x = torch.rand(2048, dtype=torch.float32, device="mps")
    rejected = torch.zeros(8, dtype=torch.float32, device="mps")
    for _ in range(3):
        with pytest.raises(Exception):
            _topk_beyond_phase_carry_kernel[(1,)](x, rejected, N=2048, K=8,
                                                  num_warps=1)

    torch.manual_seed(20260908)
    values = torch.randperm(128).float()
    out = torch.zeros(128, dtype=torch.float32, device="mps")
    _sort_kernel[(1,)](values.to("mps"), out, 128, num_warps=1)
    torch.mps.synchronize()
    assert torch.equal(out.cpu(), values.sort().values)


@triton.jit
def _splat_store_1x1_kernel(x_ptr, o_ptr, M, N, BM: tl.constexpr,
                            BN: tl.constexpr):
    rm = tl.arange(0, BM)[:, None]
    rn = tl.arange(0, BN)[None, :]
    off = rm * N + rn
    mask = (rm < M) & (rn < N)
    tl.store(o_ptr + off, tl.load(x_ptr + off, mask=mask, other=0.0) * 2.0,
             mask=mask)


@pytest.mark.parametrize("bm,bn,num_warps", [
    # 1x1 is the shape that had no store pattern at all: both offsets fold to
    # zero, so the address is a bare `tt.splat` of the argument rather than a
    # `tt.addptr`.
    (1, 1, 1), (1, 1, 4),
    # Neighbours that already worked, kept so a regression here is visible as a
    # narrowing rather than a total failure.
    (1, 2, 4), (2, 1, 4), (2, 2, 4), (1, 32, 4), (32, 1, 4),
])
def test_splat_pointer_store_single_element(bm, bn, num_warps):
    """A store whose address folded down to a uniform splat.

    `tl.store(p + 0*N + 0, v)` at BLOCK=1 leaves `tt.store(tt.splat(p), v)`,
    which matched no pattern — and in this backend an unmatched op is a process
    kill during rollback, not an error. It killed the caller of the verbatim
    leet-triton RoPE kernel at head dim 2 (BLOCK_SIZE_D = next_pow2(D/2) = 1),
    sometimes with abort and sometimes with SIGSEGV.
    """
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("Metal backend requires an MPS-enabled PyTorch")
    torch.manual_seed(bm * 37 + bn)
    x = torch.rand(bm, bn, dtype=torch.float32)
    o = torch.zeros(bm, bn, dtype=torch.float32, device="mps")
    _splat_store_1x1_kernel[(1,)](x.to("mps"), o, bm, bn, BM=bm, BN=bn,
                                  num_warps=num_warps)
    torch.mps.synchronize()
    torch.testing.assert_close(o.cpu(), x * 2.0, atol=1e-6, rtol=1e-6)


@triton.jit
def _splat_masked_store_kernel(x_ptr, o_ptr, D, BD: tl.constexpr):
    rm = tl.arange(0, 1)[:, None]
    rd = tl.arange(0, BD)[None, :]
    off = rm * D + rd
    mask = rd < D // 2
    tl.store(o_ptr + off, tl.load(x_ptr + off, mask=mask, other=0.0) + 1.0,
             mask=mask)


@pytest.mark.parametrize("num_warps", [1, 2, 4])
def test_splat_pointer_store_keeps_its_mask(num_warps):
    """The masked store is a separate pattern from the unmasked one.

    This is the shape RoPE actually produced: M==1 is specialized away so the
    row half of the mask folds, but the column bound survives on a runtime `D`,
    so the 1x1 tile reaches the MASKED store. Exempting only the unmasked
    pattern left the crash exactly where it was.
    """
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("Metal backend requires an MPS-enabled PyTorch")
    torch.manual_seed(num_warps)
    x = torch.rand(1, 2, dtype=torch.float32)
    o = torch.zeros(1, 2, dtype=torch.float32, device="mps")
    _splat_masked_store_kernel[(1,)](x.to("mps"), o, 2, BD=1,
                                     num_warps=num_warps)
    torch.mps.synchronize()
    expected = torch.stack([x[:, 0] + 1.0, torch.zeros(1)], dim=1)
    torch.testing.assert_close(o.cpu(), expected, atol=1e-6, rtol=1e-6)


@triton.jit
def _splat_store_wide_kernel(x_ptr, o_ptr, BLOCK: tl.constexpr):
    z = tl.zeros((BLOCK,), tl.int32)
    tl.store(o_ptr + z, tl.load(x_ptr + z) * 2.0)


@pytest.mark.parametrize("num_warps", [1, 4])
def test_splat_pointer_store_wide_tile_is_rejected_not_crash(num_warps):
    """The same uniform address under a tile of more than one element.

    Triton reaches it — a zero offset folds away — but it is a race: eight lanes
    hold different values for one address. Declining it inside conversion took
    the process down, so it is named in the pre-pass instead.
    """
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("Metal backend requires an MPS-enabled PyTorch")
    x = torch.rand(8, dtype=torch.float32, device="mps")
    o = torch.zeros(8, dtype=torch.float32, device="mps")
    with pytest.raises(Exception):
        _splat_store_wide_kernel[(1,)](x, o, BLOCK=8, num_warps=num_warps)


@pytest.mark.parametrize("n", [1, 8, 63, 64, 65, 257, 1023, 1024, 1025, 1500])
def test_hard_fast_fourier_transform_matches_torch(n):
    """Cover the small DFT, radix-2 FFT, and Bluestein execution paths."""
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("Metal backend requires an MPS-enabled PyTorch")
    fft = _load_exact_module(HARD_FFT_PATH, "hard_fast_fourier_transform")
    torch.manual_seed(n)
    source = torch.randn(n, dtype=torch.complex64)
    signal = torch.view_as_real(source).reshape(-1).to("mps")
    spectrum = torch.empty(2 * n, dtype=torch.float32, device="mps")

    fft.solve(signal, spectrum, n)
    torch.mps.synchronize()

    actual = torch.view_as_complex(spectrum.cpu().reshape(n, 2))
    torch.testing.assert_close(actual, torch.fft.fft(source), atol=5e-4,
                               rtol=5e-4)


@triton.jit
def _floating_remainder_kernel(x_ptr, y_ptr, out_ptr, N, SCALAR: tl.constexpr, BLOCK: tl.constexpr):
    if SCALAR:
        x = tl.load(x_ptr)
        y = tl.load(y_ptr)
        tl.store(out_ptr, x % y)
    else:
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < N
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        y = tl.load(y_ptr + offsets, mask=mask, other=1.0)
        tl.store(out_ptr + offsets, x % y, mask=mask)


@pytest.mark.parametrize("scalar", [False, True])
def test_floating_remainder_compiles_to_msl(scalar):
    src = ASTSource(
        fn=_floating_remainder_kernel,
        signature={"x_ptr": "*fp32", "y_ptr": "*fp32", "out_ptr": "*fp32", "N": "i32",
                   "SCALAR": "constexpr", "BLOCK": "constexpr"},
        constexprs={"SCALAR": scalar, "BLOCK": 128},
    )
    compiled = triton.compile(src, target=GPUTarget("metal", 80, 32), options={"num_warps": 4})
    raw = compiled.asm["metal"]
    msl = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    assert "metal::precise::fmod(" in msl


@pytest.mark.parametrize("scalar,n", [(True, 1), (False, 63), (False, 128), (False, 1025)])
def test_floating_remainder_matches_torch(scalar, n):
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("Metal backend requires an MPS-enabled PyTorch")
    # Dividend sign, negative divisors, exact multiples, large quotients and
    # special values distinguish truncating remainder from floor/IEEE remainder.
    pairs = [
        (-0.0, 3.0), (0.0, -3.0), (-4.0, 2.0), (4.0, -2.0),
        (-5.5, 2.0), (5.5, -2.0), (-5.5, -2.0), (5.5, 2.0),
        (3.4028234663852886e38, 1.25), (-1073741824.0, 3.0),
        (1.0, float("inf")), (-1.0, -float("inf")),
        (float("inf"), 2.0), (-float("inf"), -2.0),
        (float("nan"), 2.0), (2.0, float("nan")), (2.0, 0.0),
    ]
    inputs = torch.tensor((pairs * triton.cdiv(n, len(pairs)))[:n], dtype=torch.float32)
    x, y = inputs[:, 0].contiguous(), inputs[:, 1].contiguous()
    out = torch.empty(n, dtype=torch.float32, device="mps")
    _floating_remainder_kernel[(triton.cdiv(n, 128),)](
        x.to("mps"), y.to("mps"), out, n, SCALAR=scalar, BLOCK=128,
    )
    torch.mps.synchronize()
    actual = out.cpu()
    expected = torch.fmod(x, y)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0, equal_nan=True)
    zero = expected == 0
    assert torch.equal(torch.signbit(actual[zero]), torch.signbit(expected[zero]))


@triton.jit
def _abs_16bit_kernel(x_ptr, out_ptr, N, DTYPE: tl.constexpr, SCALAR: tl.constexpr, BLOCK: tl.constexpr):
    if SCALAR:
        offsets = tl.program_id(0)
        bits = tl.load(x_ptr + offsets)
    else:
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        bits = tl.load(x_ptr + offsets, offsets < N, other=0)
    value = bits.to(DTYPE, bitcast=True)
    magnitude = tl.abs(value)
    if SCALAR:
        tl.store(out_ptr + offsets, magnitude.to(tl.int16, bitcast=True))
    else:
        tl.store(out_ptr + offsets, magnitude.to(tl.int16, bitcast=True), offsets < N)


@pytest.mark.parametrize("dtype", [tl.float16, tl.bfloat16])
@pytest.mark.parametrize("scalar", [False, True])
def test_abs_16bit_compiles_to_msl(dtype, scalar):
    src = ASTSource(
        fn=_abs_16bit_kernel,
        signature={"x_ptr": "*i16", "out_ptr": "*i16", "N": "i32",
                   "DTYPE": "constexpr", "SCALAR": "constexpr", "BLOCK": "constexpr"},
        constexprs={"DTYPE": dtype, "SCALAR": scalar, "BLOCK": 256},
    )
    compiled = triton.compile(src, target=GPUTarget("metal", 80, 32), options={"num_warps": 4})
    raw = compiled.asm["metal"]
    msl = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    assert "as_type<uint16_t>" in msl
    assert "32767" in msl and " & " in msl
    spelling = "half" if dtype == tl.float16 else "bfloat"
    assert f"as_type<{spelling}>" in msl


@pytest.mark.parametrize("dtype", [tl.float16, tl.bfloat16])
@pytest.mark.parametrize("scalar", [False, True])
@pytest.mark.parametrize("num_warps", [1, 4])
def test_abs_16bit_runtime_preserves_magnitude_bits(dtype, scalar, num_warps):
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("Metal backend requires an MPS-enabled PyTorch")
    # Tensor mode exhausts both signs of every normal, subnormal, zero, infinity
    # and NaN payload. Integer I/O prevents host/device float conversions from
    # hiding a changed NaN payload. The extra 37 values exercise a masked tail.
    if scalar:
        bits = torch.tensor([0, 0x8000, 1, 0x8001, 0x3c00, 0xbc00, 0x7c00, 0xfc00,
                             0x7c01, 0xfc01, 0x7f80, 0xff80, 0x7f81, 0xff81,
                             0x7fff, 0xffff], dtype=torch.int32).to(torch.int16)
    else:
        bits = torch.arange(65536 + 37, dtype=torch.int32).to(torch.int16)
    n = bits.numel()
    out = torch.empty_like(bits, device="mps")
    _abs_16bit_kernel[(n if scalar else triton.cdiv(n, 256),)](
        bits.to("mps"), out, n, DTYPE=dtype, SCALAR=scalar, BLOCK=256, num_warps=num_warps,
    )
    torch.mps.synchronize()
    assert torch.equal(out.cpu(), bits & 0x7fff)


@pytest.mark.parametrize("dtype", ["i16", "fp32", "bf16"])
@pytest.mark.parametrize("mode", ["constant", "runtime", "loaded", "none"])
def test_masked_scalar_load_compiles_with_guarded_memory_read(dtype, mode):
    from test_metal_backend_scalar_load import _masked_scalar_load_kernel

    msl = _compile_to_msl(
        _masked_scalar_load_kernel,
        {"x_ptr": f"*{dtype}", "fallback_ptr": f"*{dtype}", "out_ptr": f"*{dtype}",
         "N": "i32", "fallback": "i32", "MODE": "constexpr"},
        {"MODE": mode}, num_warps=4,
    )
    # Buffer 0 is the masked input. Its one memory read must occur inside the
    # true branch; generating a select around an unconditional read is invalid.
    assert msl.count("v0[") == 1
    assert msl.index("if (") < msl.index("v0[") < msl.index("} else {")


@triton.jit
def dot_scaled_e4m3_f16_output_kernel(
    a_base,
    b_base,
    a_scale_base,
    b_scale_base,
    output_base,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a = tl.load(a_base + offs_m[:, None] * BLOCK_K + offs_k[None, :])
    b = tl.load(b_base + offs_k[:, None] * BLOCK_N + offs_n[None, :])
    SCALE_K: tl.constexpr = BLOCK_K // 32
    offs_scale_k = tl.arange(0, SCALE_K)
    a_scale = tl.load(a_scale_base + offs_m[:, None] * SCALE_K + offs_scale_k[None, :])
    b_scale = tl.load(b_scale_base + offs_n[:, None] * SCALE_K + offs_scale_k[None, :])
    result = tl.dot_scaled(a, a_scale, "e4m3", b, b_scale, "e4m3", fast_math=False)
    # The f32 result truncates on the way into an f16 buffer, so an
    # `arith.truncf` sits between the dot and the store.
    tl.store(output_base + offs_m[:, None] * BLOCK_N + offs_n[None, :], result)


@triton.jit
def dot_scaled_e4m3_scaled_epilogue_kernel(
    a_base,
    b_base,
    a_scale_base,
    b_scale_base,
    output_base,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a = tl.load(a_base + offs_m[:, None] * BLOCK_K + offs_k[None, :])
    b = tl.load(b_base + offs_k[:, None] * BLOCK_N + offs_n[None, :])
    SCALE_K: tl.constexpr = BLOCK_K // 32
    offs_scale_k = tl.arange(0, SCALE_K)
    a_scale = tl.load(a_scale_base + offs_m[:, None] * SCALE_K + offs_scale_k[None, :])
    b_scale = tl.load(b_scale_base + offs_n[:, None] * SCALE_K + offs_scale_k[None, :])
    result = tl.dot_scaled(a, a_scale, "e4m3", b, b_scale, "e4m3", fast_math=False)
    # A multiply survives as its own op, unlike `+ bias`, which Triton folds
    # into the dot's accumulator operand and which therefore still matches.
    tl.store(output_base + offs_m[:, None] * BLOCK_N + offs_n[None, :], result * 2.0)


@pytest.mark.parametrize(
    "fn,out_ty",
    [
        (dot_scaled_e4m3_f16_output_kernel, "*fp16"),
        (dot_scaled_e4m3_scaled_epilogue_kernel, "*fp32"),
    ],
    ids=["f16_output_buffer", "multiply_epilogue"],
)
def test_scaled_dot_epilogue_between_dot_and_store_is_named(fn, out_ty, capfd):
    """The scaled-dot rewrite materializes its result at the store, so any real
    op in between keeps its original position and would consume a value defined
    after it. That used to leave `operand #0 does not dominate this use` — a
    half-applied rewrite and invalid IR, which is the state that turns into an
    intermittent process kill — instead of a named capability diagnostic.

    Asserting only on the generic `convert-tritongpu-to-metal failed` cannot
    tell those two apart, so this reads the diagnostic itself."""
    src = ASTSource(
        fn=fn,
        signature={
            "a_base": "*u8",
            "b_base": "*u8",
            "a_scale_base": "*u8",
            "b_scale_base": "*u8",
            "output_base": out_ty,
            "BLOCK_M": "constexpr",
            "BLOCK_N": "constexpr",
            "BLOCK_K": "constexpr",
        },
        constexprs={"BLOCK_M": 16, "BLOCK_N": 16, "BLOCK_K": 32},
    )
    target = GPUTarget(backend="metal", arch=80, warp_size=32)
    with pytest.raises(RuntimeError, match="convert-tritongpu-to-metal failed"):
        triton.compile(src, target=target, options={"num_warps": 4})
    err = capfd.readouterr().err
    assert "dot_scaled requires its result to reach one store directly" in err, err
    assert "does not dominate this use" not in err, err


# ---------------------------------------------------------------------------
# Loop-carried scaled-dot matcher boundary.
#
# The matcher accepts a narrow canonical K loop. Each near-miss below must exit
# with its own named diagnostic rather than a half-applied rewrite; a decline
# reaching `applyFullConversion` is a process kill, not an error.
# ---------------------------------------------------------------------------

_SCALED_LOOP_SIGNATURE = {
    "a_ptr": "*u8",
    "b_ptr": "*u8",
    "a_scale_ptr": "*u8",
    "b_scale_ptr": "*u8",
    "c_ptr": "*fp32",
    "M": "constexpr",
    "N": "constexpr",
    "K": "constexpr",
    "BLOCK_K": "constexpr",
    "SCALE_FACTOR": "constexpr",
    "LOOP_START": "constexpr",
}


@triton.jit
def _scaled_loop_canonical_kernel(a_ptr, b_ptr, a_scale_ptr, b_scale_ptr, c_ptr,
                                  M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                                  BLOCK_K: tl.constexpr, SCALE_FACTOR: tl.constexpr,
                                  LOOP_START: tl.constexpr):
    offs_m = tl.arange(0, M)
    offs_n = tl.arange(0, N)
    offs_k = tl.arange(0, BLOCK_K)
    offs_scale_k = tl.arange(0, BLOCK_K // SCALE_FACTOR)
    acc = tl.zeros((M, N), dtype=tl.float32)
    for k_start in range(LOOP_START, K, BLOCK_K):
        a = tl.load(a_ptr + offs_m[:, None] * K + (k_start + offs_k[None, :]))
        b = tl.load(b_ptr + (k_start + offs_k[:, None]) * N + offs_n[None, :])
        a_scale = tl.load(a_scale_ptr + offs_m[:, None] * (K // SCALE_FACTOR)
                          + k_start // SCALE_FACTOR + offs_scale_k[None, :])
        b_scale = tl.load(b_scale_ptr + offs_n[:, None] * (K // SCALE_FACTOR)
                          + k_start // SCALE_FACTOR + offs_scale_k[None, :])
        acc = tl.dot_scaled(a, a_scale, "e4m3", b, b_scale, "e4m3", acc, fast_math=False)
    tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc)


@triton.jit
def _scaled_loop_two_accumulators_kernel(a_ptr, b_ptr, a_scale_ptr, b_scale_ptr, c_ptr,
                                         M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                                         BLOCK_K: tl.constexpr, SCALE_FACTOR: tl.constexpr,
                                         LOOP_START: tl.constexpr):
    offs_m = tl.arange(0, M)
    offs_n = tl.arange(0, N)
    offs_k = tl.arange(0, BLOCK_K)
    offs_scale_k = tl.arange(0, BLOCK_K // SCALE_FACTOR)
    acc = tl.zeros((M, N), dtype=tl.float32)
    tally = tl.zeros((M, N), dtype=tl.float32)
    for k_start in range(LOOP_START, K, BLOCK_K):
        a = tl.load(a_ptr + offs_m[:, None] * K + (k_start + offs_k[None, :]))
        b = tl.load(b_ptr + (k_start + offs_k[:, None]) * N + offs_n[None, :])
        a_scale = tl.load(a_scale_ptr + offs_m[:, None] * (K // SCALE_FACTOR)
                          + k_start // SCALE_FACTOR + offs_scale_k[None, :])
        b_scale = tl.load(b_scale_ptr + offs_n[:, None] * (K // SCALE_FACTOR)
                          + k_start // SCALE_FACTOR + offs_scale_k[None, :])
        tally += 1.0
        acc = tl.dot_scaled(a, a_scale, "e4m3", b, b_scale, "e4m3", acc, fast_math=False)
    tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc + tally)


@triton.jit
def _scaled_loop_two_stores_kernel(a_ptr, b_ptr, a_scale_ptr, b_scale_ptr, c_ptr,
                                   M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                                   BLOCK_K: tl.constexpr, SCALE_FACTOR: tl.constexpr,
                                   LOOP_START: tl.constexpr):
    offs_m = tl.arange(0, M)
    offs_n = tl.arange(0, N)
    offs_k = tl.arange(0, BLOCK_K)
    offs_scale_k = tl.arange(0, BLOCK_K // SCALE_FACTOR)
    acc = tl.zeros((M, N), dtype=tl.float32)
    for k_start in range(LOOP_START, K, BLOCK_K):
        a = tl.load(a_ptr + offs_m[:, None] * K + (k_start + offs_k[None, :]))
        b = tl.load(b_ptr + (k_start + offs_k[:, None]) * N + offs_n[None, :])
        a_scale = tl.load(a_scale_ptr + offs_m[:, None] * (K // SCALE_FACTOR)
                          + k_start // SCALE_FACTOR + offs_scale_k[None, :])
        b_scale = tl.load(b_scale_ptr + offs_n[:, None] * (K // SCALE_FACTOR)
                          + k_start // SCALE_FACTOR + offs_scale_k[None, :])
        acc = tl.dot_scaled(a, a_scale, "e4m3", b, b_scale, "e4m3", acc, fast_math=False)
    tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc)
    tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc)


@triton.jit
def _scaled_loop_non_canonical_a_kernel(a_ptr, b_ptr, a_scale_ptr, b_scale_ptr, c_ptr,
                                        M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                                        BLOCK_K: tl.constexpr, SCALE_FACTOR: tl.constexpr,
                                        LOOP_START: tl.constexpr):
    offs_m = tl.arange(0, M)
    offs_n = tl.arange(0, N)
    offs_k = tl.arange(0, BLOCK_K)
    offs_scale_k = tl.arange(0, BLOCK_K // SCALE_FACTOR)
    acc = tl.zeros((M, N), dtype=tl.float32)
    for k_start in range(LOOP_START, K, BLOCK_K):
        # Row stride K + 1 instead of K: not the contiguous full-K form.
        a = tl.load(a_ptr + offs_m[:, None] * (K + 1) + (k_start + offs_k[None, :]))
        b = tl.load(b_ptr + (k_start + offs_k[:, None]) * N + offs_n[None, :])
        a_scale = tl.load(a_scale_ptr + offs_m[:, None] * (K // SCALE_FACTOR)
                          + k_start // SCALE_FACTOR + offs_scale_k[None, :])
        b_scale = tl.load(b_scale_ptr + offs_n[:, None] * (K // SCALE_FACTOR)
                          + k_start // SCALE_FACTOR + offs_scale_k[None, :])
        acc = tl.dot_scaled(a, a_scale, "e4m3", b, b_scale, "e4m3", acc, fast_math=False)
    tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc)


def _compile_scaled_loop(fn, *, K=64, loop_start=0):
    src = ASTSource(
        fn=fn,
        signature=_SCALED_LOOP_SIGNATURE,
        constexprs={"M": 16, "N": 16, "K": K, "BLOCK_K": 32,
                    "SCALE_FACTOR": 32, "LOOP_START": loop_start},
    )
    target = GPUTarget(backend="metal", arch=80, warp_size=32)
    triton.compile(src, target=target, options={"num_warps": 4})


def test_scaled_dot_canonical_loop_still_compiles():
    """Positive control for the four rejections below: two static iterations
    over a zero-based divisible K loop with one accumulator and one store."""
    _compile_scaled_loop(_scaled_loop_canonical_kernel)


@pytest.mark.parametrize(
    "fn,kwargs,needle",
    [
        (_scaled_loop_canonical_kernel, {"K": 48},
         "requires a K loop with nonnegative aligned divisible bounds"),
        (_scaled_loop_canonical_kernel, {"K": 32, "loop_start": -32},
         "requires a K loop with nonnegative aligned divisible bounds"),
        (_scaled_loop_two_accumulators_kernel, {},
         "loop-carried dot_scaled requires exactly one accumulator"),
        (_scaled_loop_two_stores_kernel, {},
         "loop-carried dot_scaled requires one output store"),
        (_scaled_loop_non_canonical_a_kernel, {},
         "requires canonical contiguous full-K A and B pointer arithmetic"),
    ],
    ids=["k_not_divisible", "negative_loop_start", "two_accumulators",
         "two_output_stores", "non_canonical_a_pointer"],
)
def test_scaled_dot_loop_near_miss_is_named(fn, kwargs, needle, capfd):
    """Each near-miss must name its own reason. Asserting only on the generic
    `convert-tritongpu-to-metal failed` would also accept a dominance violation
    from a half-applied rewrite, so read the diagnostic itself."""
    with pytest.raises(RuntimeError, match="convert-tritongpu-to-metal failed"):
        _compile_scaled_loop(fn, **kwargs)
    err = capfd.readouterr().err
    assert needle in err, err
    assert "does not dominate this use" not in err, err


@triton.jit
def _reduce_then_sort_kernel(X, O, N: tl.constexpr, ND: tl.constexpr,
                             AX: tl.constexpr):
    h = tl.reshape(tl.load(X + tl.arange(0, N)), [2] * ND)
    r = tl.reshape(tl.max(h, axis=AX), [N // 2])
    tl.store(O + tl.arange(0, N // 2), tl.sort(r, descending=False))


@pytest.mark.parametrize("N,num_warps,axis", [
    (64, 1, 4),
    (64, 1, 5),
    (128, 1, 5),
    (128, 1, 6),
    (128, 4, 5),
    (256, 4, 6),
    (32, 1, 4),
])
def test_reduce_then_sort_hypercube_axis(N, num_warps, axis):
    """A hypercube-axis reduce followed by a sort is the smallest shape that
    exercises exchange buffer reuse: reducing along a THREAD axis interleaves
    small direction-table publishes of the same element type between the tile
    publishes. While buffers were handed out by round-robin those tables flipped
    the parity and a compare-exchange landed on the buffer it was reading, which
    is a read-after-write inside one loop body with no barrier, so both lanes of
    a pair kept the same value. Four of these cells returned adjacent duplicates
    before the allocator stopped reusing buffers.

    This is also `tl.topk`'s inner loop, which alternates exactly these two
    steps, so it guards that path without depending on topk's own machinery."""
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("Metal backend requires an MPS-enabled PyTorch")
    nd = N.bit_length() - 1
    torch.manual_seed(N * 17 + axis)
    x = torch.randperm(N).float()
    out = torch.zeros(N // 2, device="mps")
    _reduce_then_sort_kernel[(1,)](
        x.to("mps"), out, N=N, ND=nd, AX=axis, num_warps=num_warps,
    )
    torch.mps.synchronize()
    expected = torch.sort(x.reshape([2] * nd).amax(dim=axis).reshape(-1)).values
    assert torch.equal(out.cpu(), expected)


@triton.jit
def _reduce_replay_math_kernel(X, Y, O, BLOCK: tl.constexpr,
                               OP: tl.constexpr, WRAPPED: tl.constexpr,
                               REDUCE: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + index)
    y = tl.load(Y + index)
    if OP == "sqrt_rn":
        value = tl.sqrt_rn(x)
    elif OP == "div_rn":
        value = tl.div_rn(x, y)
    elif OP == "fma":
        value = tl.fma(x, y, x)
    elif OP == "clamp":
        value = tl.clamp(x, 0.5, 1.5)
    else:
        value = tl.sqrt(x)
    if WRAPPED:
        value = tl.abs(value)
    if REDUCE == "sum":
        result = tl.sum(value, 0)
    else:
        result = tl.max(value, 0)
    tl.store(O + tl.program_id(0), result)


@triton.jit
def _reduce_replay_loop_kernel(X, Y, O, BLOCK: tl.constexpr,
                               OP: tl.constexpr, WRAPPED: tl.constexpr,
                               REDUCE: tl.constexpr):
    x = tl.load(X + tl.arange(0, BLOCK))
    state = x
    steps = tl.load(Y).to(tl.int32)
    for i in range(steps):
        # A nonlinear recurrence cannot be reassociated into scalar reductions.
        state = tl.sin(state + x)
        if OP == "loop_iter_arg":
            value = tl.abs(state) if WRAPPED else state
            result = tl.sum(value, 0) if REDUCE == "sum" else tl.max(value, 0)
            tl.store(O + i, result)
    if OP == "loop_result":
        value = tl.abs(state) if WRAPPED else state
        result = tl.sum(value, 0) if REDUCE == "sum" else tl.max(value, 0)
        tl.store(O, result)


@pytest.mark.parametrize("operation", ["sqrt_rn", "div_rn", "fma", "clamp",
                                       "loop_result", "loop_iter_arg"])
@pytest.mark.parametrize("num_warps,wrapped,reduction", [(1, False, "sum"), (4, True, "max")])
def test_reduce_rank1_unreplayable_source_fails_without_crashing(
        operation, num_warps, wrapped, reduction, tmp_path):
    import os
    import subprocess
    import sys
    import textwrap

    # Repeat in one process, then compile a supported cone. Conversion rollback
    # used to kill the interpreter; a Python exception must leave it usable.
    script = tmp_path / "reduce_replay.py"
    script.write_text(textwrap.dedent(f"""
        import importlib.util
        import triton
        from triton.compiler import ASTSource
        from triton.backends.compiler import GPUTarget
        spec = importlib.util.spec_from_file_location("msl_child", {str(Path(__file__).resolve())!r})
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        signature = {{"X": "*fp32", "Y": "*fp32", "O": "*fp32"}}
        constants = {{"BLOCK": {num_warps * 128}, "OP": {operation!r},
                     "WRAPPED": {wrapped}, "REDUCE": {reduction!r}}}
        kernel = (module._reduce_replay_loop_kernel if {operation.startswith("loop_")}
                  else module._reduce_replay_math_kernel)
        for attempt in range(3):
            try:
                triton.compile(ASTSource(kernel,
                    signature=signature, constexprs=constants),
                    target=GPUTarget("metal", 80, 32), options={{"num_warps": {num_warps}}})
            except RuntimeError:
                pass
            else:
                raise AssertionError("unsupported replay unexpectedly compiled")
        constants["OP"] = "sqrt"
        triton.compile(ASTSource(module._reduce_replay_math_kernel,
            signature=signature, constexprs=constants),
            target=GPUTarget("metal", 80, 32), options={{"num_warps": {num_warps}}})
        print("RECOVERED")
    """))
    result = subprocess.run([sys.executable, str(script)], capture_output=True,
                            text=True, timeout=120,
                            env={**os.environ, "TRITON_CACHE_DIR": str(tmp_path / "cache")})
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert output.count("rank-1 reduce cannot replay its source") == 3
    assert "RECOVERED" in output
    assert "failed to legalize" not in output


@pytest.mark.parametrize("operation", ["sqrt_rn", "div_rn", "fma", "clamp", "sqrt"])
@pytest.mark.parametrize("num_warps", [1, 4])
@pytest.mark.parametrize("reduction", ["sum", "max"])
def test_reduce_rank1_math_single_band_remains_supported(operation, num_warps, reduction):
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("requires physical MPS")
    block = num_warps * 32
    x = (torch.arange(3 * block).reshape(3, block) % 19 + 1).float() / 8
    y = (torch.arange(3 * block).reshape(3, block) % 11 + 1).float() / 8
    if operation in ("sqrt_rn", "sqrt"):
        values = x.sqrt()
    elif operation == "div_rn":
        values = x / y
    elif operation == "fma":
        values = (x.double() * y.double() + x.double()).float()
    else:
        values = x.clamp(0.5, 1.5)
    expected = values.sum(1) if reduction == "sum" else values.max(1).values
    out = torch.empty(3, device="mps")
    _reduce_replay_math_kernel[(3,)](
        x.to("mps"), y.to("mps"), out, BLOCK=block, OP=operation,
        WRAPPED=True, REDUCE=reduction, num_warps=num_warps)
    torch.mps.synchronize()
    torch.testing.assert_close(out.cpu(), expected, atol=1e-5, rtol=1e-5)


@triton.jit
def _reduce_rank1_geometry_kernel(X, O, N, BLOCK: tl.constexpr,
                                  OP: tl.constexpr, MASKED: tl.constexpr,
                                  OTHER: tl.constexpr, COMPUTED: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    if MASKED:
        values = tl.load(X + offsets, offsets < N, other=OTHER)
    else:
        values = tl.load(X + offsets)
    if COMPUTED:
        values = values + 1.0
    result = tl.sum(values, axis=0) if OP == "sum" else tl.max(values, axis=0)
    tl.store(O, result)


@triton.jit
def _reduce_rank1_mask_shape_kernel(X, Y, O, N, RUNTIME_BASE, RUNTIME_OTHER,
                                    BLOCK: tl.constexpr, OP: tl.constexpr,
                                    MASK_MODE: tl.constexpr,
                                    OTHER_MODE: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    ptr_offsets = offsets
    if MASK_MODE == "compound":
        row_ok = RUNTIME_BASE >= 0
        mask = (offsets < N) & row_ok
    elif MASK_MODE == "ge":
        mask = offsets >= N
    elif MASK_MODE == "scaled_lhs":
        mask = offsets * 2 < N
    elif MASK_MODE == "shifted_mismatch":
        mask = offsets + RUNTIME_BASE < N
    elif MASK_MODE == "shifted":
        ptr_offsets = offsets + RUNTIME_BASE
        mask = ptr_offsets < N
    elif MASK_MODE == "tensor_rhs":
        mask = offsets < tl.load(Y + offsets).to(tl.int32)
    else:
        mask = offsets < N

    if OTHER_MODE == "runtime":
        other = RUNTIME_OTHER
    elif OTHER_MODE == "tensor":
        other = offsets.to(tl.float32) * 0.25
    elif OP == "max":
        other = -float("inf")
    else:
        other = 0.0

    values = tl.load(X + ptr_offsets, mask=mask, other=other)
    result = tl.max(values, axis=0) if OP == "max" else tl.sum(values, axis=0)
    tl.store(O, result)


@triton.jit
def _gather_envelope_kernel(X, IDX, O, BLOCK: tl.constexpr):
    o = tl.arange(0, BLOCK)
    tl.store(O + o, tl.gather(tl.load(X + o), tl.load(IDX + o), 0))


@triton.jit
def _splat_store_kernel(X, O, BLOCK: tl.constexpr):
    o = tl.arange(0, BLOCK)
    # Every element of the tile lands on one address: a race, not a store.
    tl.store(O + tl.zeros((BLOCK,), tl.int32), tl.load(X + o))


@triton.jit
def _dot_scaled_envelope_kernel(A, B, AS, BS, C, M: tl.constexpr,
                                N: tl.constexpr, K: tl.constexpr,
                                FORM: tl.constexpr):
    m = tl.arange(0, M)
    n = tl.arange(0, N)
    kk = tl.arange(0, K)
    if FORM == "computed_a":
        a = tl.load(A + m[:, None] * K + kk[None, :]) ^ 1
    else:
        a = tl.load(A + m[:, None] * K + kk[None, :])
    b = tl.load(B + kk[:, None] * N + n[None, :])
    r = tl.dot_scaled(a, tl.load(AS + m[:, None]), "e4m3",
                      b, tl.load(BS + n[:, None]), "e4m3", fast_math=False)
    if FORM == "epilogue":
        tl.store(C + m[:, None] * N + n[None, :], r * 0.5)
    else:
        tl.store(C + m[:, None] * N + n[None, :], r)


@triton.jit
def _reduce_rank2_axis1_narrow_int_kernel(X, O, M: tl.constexpr,
                                          N: tl.constexpr):
    # Triton promotes an i8 or i16 sum to i32 over an `arith.extsi` cone, so the
    # source of this reduce is a COMPUTED cone rather than a direct load -- the
    # shape the rank-2 row scanner implements for f32 only.
    r = tl.arange(0, M)[:, None]
    c = tl.arange(0, N)[None, :]
    v = tl.load(X + r * N + c)
    tl.store(O + tl.arange(0, M), tl.sum(v, axis=1))


@triton.jit
def _scan_envelope_kernel(X, O, BLOCK: tl.constexpr, FORM: tl.constexpr):
    o = tl.arange(0, BLOCK)
    v = tl.load(X + o)
    if FORM == "tuple":
        val, pos = tl.associative_scan((v, o), 0, _argmax_tuple_combine)
        tl.store(O + o, val)
    else:
        tl.store(O + o, tl.cumsum(v, axis=0))


@triton.jit
def _scan_envelope_rank2_kernel(X, O, M: tl.constexpr, N: tl.constexpr,
                                AXIS: tl.constexpr):
    r = tl.arange(0, M)[:, None]
    c = tl.arange(0, N)[None, :]
    v = tl.load(X + r * N + c)
    tl.store(O + r * N + c, tl.cumsum(v, axis=AXIS))


@triton.jit
def _argmax_tuple_combine(a_val, a_idx, b_val, b_idx):
    gt = a_val > b_val
    return tl.where(gt, a_val, b_val), tl.where(gt, a_idx, b_idx)


@triton.jit
def _replayed_cone_through_if_kernel(X, Y, O, S, BLOCK: tl.constexpr,
                                     FORM: tl.constexpr):
    # The source arrives through an `scf.if` with a uniform condition. Above one
    # element per thread every one of these forms has to re-derive it per
    # element, which conversion makes impossible: it rewrites the `scf.if` and
    # its results stop being tensors.
    o = tl.arange(0, BLOCK)
    v = tl.load(X + o) if S > 0.0 else tl.load(Y + o)
    if FORM == "tuple_reduce":
        val, pos = tl.reduce((v, o), 0, _argmax_tuple_combine)
        tl.store(O, val + pos.to(tl.float32) * 0.0)
    elif FORM == "tuple_scan":
        val, pos = tl.associative_scan((v, o), 0, _argmax_tuple_combine)
        tl.store(O + o, val)
    else:
        tl.store(O, tl.max(v, axis=0))


@triton.jit
def _reduce_rank1_address_shape_kernel(X, Y, O, RUNTIME_BASE,
                                       BLOCK: tl.constexpr,
                                       OP: tl.constexpr,
                                       ADDRESS_MODE: tl.constexpr,
                                       VALUE_MODE: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    if ADDRESS_MODE == "runtime_stride":
        ptr_offsets = offsets * RUNTIME_BASE
        values = tl.load(X + ptr_offsets)
    elif ADDRESS_MODE == "constant_stride":
        ptr_offsets = offsets * 3
        values = tl.load(X + ptr_offsets)
    elif ADDRESS_MODE == "modulo":
        ptr_offsets = offsets % 7
        values = tl.load(X + ptr_offsets)
    elif ADDRESS_MODE == "gather":
        ptr_offsets = tl.load(Y + offsets).to(tl.int32)
        values = tl.load(X + ptr_offsets)
    elif ADDRESS_MODE == "shifted_tensor":
        ptr_offsets = offsets + RUNTIME_BASE
        values = tl.load(X + ptr_offsets)
    elif ADDRESS_MODE == "scalar_base":
        values = tl.load(X + RUNTIME_BASE + offsets)
    elif ADDRESS_MODE == "rich_strided":
        strided = tl.load(X + offsets * RUNTIME_BASE)
        contiguous = tl.load(X + offsets)
        values = tl.where(offsets % 2 == 0, strided, contiguous)
    elif ADDRESS_MODE == "bitand":
        # Bitwise index arithmetic is outside `indexConeSupported`, so neither
        # the single-load walker nor the general evaluator can rebuild this
        # address. It is the adjacent rejection for the replayed forms above.
        ptr_offsets = offsets & RUNTIME_BASE
        values = tl.load(X + ptr_offsets)
    else:
        values = tl.load(X + offsets)

    if VALUE_MODE == "chain":
        values = tl.exp(values)
    result = tl.max(values, axis=0) if OP == "max" else tl.sum(values, axis=0)
    tl.store(O, result)


@triton.jit
def _reduce_rank1_contiguous_masked_address_kernel(X, O, N, RUNTIME_STRIDE,
                                                   BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    ptr_offsets = offsets * RUNTIME_STRIDE
    values = tl.load(X + ptr_offsets, mask=offsets < N, other=0.0)
    tl.store(O, tl.sum(values, axis=0))


@triton.jit
def _reduce_rank1_contiguous_masked_index_kernel(
        X, O, N, K, RUNTIME_STRIDE, BLOCK: tl.constexpr,
        OFFSET_MODE: tl.constexpr, MASK_MODE: tl.constexpr,
        WIDEN: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    load_index = offsets
    if OFFSET_MODE == "bitand":
        load_index = offsets & K
    elif OFFSET_MODE == "select":
        load_index = tl.where(offsets < K, offsets, offsets // 2)
    elif OFFSET_MODE == "modulo":
        load_index = offsets % K
    elif OFFSET_MODE == "dynamic_stride":
        load_index = offsets * RUNTIME_STRIDE
    if WIDEN:
        load_index = load_index.to(tl.int64)

    mask_index = offsets
    if MASK_MODE == "bitand":
        mask_index = offsets & K
    elif MASK_MODE == "select":
        mask_index = tl.where(offsets < K, offsets, offsets // 2)
    elif MASK_MODE == "modulo":
        mask_index = offsets % K
    elif MASK_MODE == "dynamic_stride":
        mask_index = offsets * RUNTIME_STRIDE
    if WIDEN:
        mask_index = mask_index.to(tl.int64)

    values = tl.load(X + load_index, mask=mask_index < N, other=0.0)
    tl.store(O, tl.sum(values, axis=0))


@triton.jit
def _reduce_rank1_contiguous_masked_chain_kernel(
        X, O, N, K, RUNTIME_STRIDE, BLOCK: tl.constexpr, DEPTH: tl.constexpr,
        INNER_MODE: tl.constexpr, MASK_MODE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    if INNER_MODE == "stride":
        inner = offsets * RUNTIME_STRIDE
    elif INNER_MODE == "modulo":
        inner = offsets % K
    elif INNER_MODE == "bitand":
        inner = offsets & K
    else:
        inner = offsets

    row = X + pid * K + inner
    if DEPTH == 3:
        ptrs = row + offsets + RUNTIME_STRIDE
    else:
        ptrs = row + offsets

    if MASK_MODE == "compound":
        mask = (pid >= 0) & (offsets <= N)
    else:
        mask = offsets < N
    values = tl.load(ptrs, mask=mask, other=0.0)
    tl.store(O + pid, tl.sum(values, axis=0))


@pytest.mark.parametrize(
    "num_warps,operation,computed",
    [
        (1, "sum", False),
        (4, "max", False),
        (1, "sum", True),
        (4, "max", True),
    ],
    ids=["unmasked_sum_nw1", "unmasked_max_nw4",
         "computed_sum_nw1", "computed_max_nw4"],
)
def test_reduce_rank1_geometry_over_e64_fails_without_crashing(
        num_warps, operation, computed, tmp_path):
    import os
    import subprocess
    import sys
    import textwrap

    threads_per_block = num_warps * 32
    failing_block = threads_per_block * 128
    supported_block = threads_per_block * 64
    script = tmp_path / "reduce_geometry.py"
    script.write_text(textwrap.dedent(f"""
        import importlib.util
        import triton
        from triton.compiler import ASTSource
        from triton.backends.compiler import GPUTarget
        spec = importlib.util.spec_from_file_location("msl_child", {str(Path(__file__).resolve())!r})
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        def compile_case(block, operation, computed):
            triton.compile(ASTSource(
                module._reduce_rank1_geometry_kernel,
                signature={{"X": "*fp32", "O": "*fp32", "N": "i32"}},
                constexprs={{"BLOCK": block, "OP": operation, "MASKED": False,
                            "OTHER": 0.0, "COMPUTED": computed}}),
                target=GPUTarget("metal", 80, 32),
                options={{"num_warps": {num_warps}}})

        for attempt in range(3):
            try:
                compile_case({failing_block}, {operation!r}, {computed})
            except RuntimeError:
                pass
            else:
                raise AssertionError("unsupported E>64 rank-1 reduce compiled")
        compile_case({supported_block}, "sum", False)
        print("RECOVERED")
    """))
    result = subprocess.run([sys.executable, str(script)], capture_output=True,
                            text=True, timeout=120,
                            env={**os.environ,
                                 "TRITON_CACHE_DIR": str(tmp_path / "cache")})
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    needle = "rank-1 reduce E = BLOCK/tpb outside supported envelope [1, 64]"
    assert output.count(needle) == 3
    assert "RECOVERED" in output
    assert "failed to legalize" not in output


@pytest.mark.parametrize(
    "num_warps,operation",
    [
        (1, "sum"),
        (4, "max"),
    ],
    ids=["sum_nw1", "max_nw4"],
)
def test_reduce_rank1_geometry_e64_runtime_boundary(num_warps, operation):
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("Metal backend requires an MPS-enabled PyTorch")
    block = num_warps * 32 * 64
    x = ((torch.arange(block) % 113).float() - 31.0) / 17.0
    out = torch.empty((), device="mps")
    _reduce_rank1_geometry_kernel[(1,)](
        x.to("mps"), out, N=block, BLOCK=block, OP=operation, MASKED=False,
        OTHER=0.0, COMPUTED=False, num_warps=num_warps)
    torch.mps.synchronize()
    expected = x.sum() if operation == "sum" else x.max()
    torch.testing.assert_close(out.cpu(), expected, atol=1e-4, rtol=1e-5)


@pytest.mark.parametrize(
    "num_warps,elements_per_thread",
    [
        (4, 128),
        (1, 1024),
    ],
    ids=["e128_nw4", "e1024_nw1"],
)
def test_reduce_rank1_masked_zero_sum_runtime_boundary(
        num_warps, elements_per_thread):
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("Metal backend requires an MPS-enabled PyTorch")
    block = num_warps * 32 * elements_per_thread
    n = block - 17
    x = ((torch.arange(block) % 29).float() + 1.0) / 11.0
    out = torch.empty((), device="mps")
    _reduce_rank1_geometry_kernel[(1,)](
        x.to("mps"), out, N=n, BLOCK=block, OP="sum", MASKED=True,
        OTHER=0.0, COMPUTED=False, num_warps=num_warps)
    torch.mps.synchronize()
    torch.testing.assert_close(out.cpu(), x[:n].sum(), atol=1e-3, rtol=1e-5)


@pytest.mark.parametrize(
    "elements_per_thread,operation,other",
    [
        (128, "sum", 1.0),
        (128, "max", 0.0),
        (2048, "sum", 0.0),
    ],
    ids=["masked_nonzero_other_e128", "masked_max_e128", "masked_sum_e2048"],
)
def test_reduce_rank1_masked_geometry_unsupported_caps(
        elements_per_thread, operation, other, tmp_path):
    import os
    import subprocess
    import sys
    import textwrap

    block = 32 * elements_per_thread
    script = tmp_path / "reduce_masked_geometry.py"
    script.write_text(textwrap.dedent(f"""
        import importlib.util
        import triton
        from triton.compiler import ASTSource
        from triton.backends.compiler import GPUTarget
        spec = importlib.util.spec_from_file_location("msl_child", {str(Path(__file__).resolve())!r})
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        try:
            triton.compile(ASTSource(
                module._reduce_rank1_geometry_kernel,
                signature={{"X": "*fp32", "O": "*fp32", "N": "i32"}},
                constexprs={{"BLOCK": {block}, "OP": {operation!r},
                            "MASKED": True, "OTHER": {other},
                            "COMPUTED": False}}),
                target=GPUTarget("metal", 80, 32),
                options={{"num_warps": 1}})
        except RuntimeError:
            print("REJECTED")
        else:
            raise AssertionError("unsupported masked rank-1 reduce compiled")
    """))
    result = subprocess.run([sys.executable, str(script)], capture_output=True,
                            text=True, timeout=120,
                            env={**os.environ,
                                 "TRITON_CACHE_DIR": str(tmp_path / "cache")})
    err = result.stdout + result.stderr
    assert result.returncode == 0, err
    assert "rank-1 reduce E = BLOCK/tpb outside supported envelope [1, 64]" in err
    assert "REJECTED" in err
    assert "failed to legalize" not in err


# Every mask spelling the legacy fold cannot rebuild is now replayed by the
# general cone evaluator instead of being refused (their numbers are in
# test_reduce_rank1_mask_shape_matches_torch below). What remains a rejection is
# a rule that is not this path's: a masked load's `other` must be uniform across
# lanes, which `tt.load` itself enforces. A mask the EVALUATOR cannot rebuild —
# one carried by an scf.for — still fails preflight by name, pinned in
# reduce_unsupported.mlir::@reduce_rank1_masked_loop_carried_deferred.
@pytest.mark.parametrize(
    "mask_mode,other_mode,operation,needle",
    [
        ("canonical", "tensor", "sum",
         "Metal backend: a masked tt.load's `other` must be uniform across lanes"),
        ("compound", "tensor", "max",
         "Metal backend: a masked tt.load's `other` must be uniform across lanes"),
    ],
    ids=[
        "tensor_other_sum",
        "tensor_other_compound_mask_max",
    ],
)
def test_reduce_rank1_mask_shape_unsupported_fails_without_crashing(
        mask_mode, other_mode, operation, needle, tmp_path):
    import os
    import subprocess
    import sys
    import textwrap

    script = tmp_path / "reduce_mask_shape.py"
    script.write_text(textwrap.dedent(f"""
        import importlib.util
        import triton
        from triton.compiler import ASTSource
        from triton.backends.compiler import GPUTarget
        spec = importlib.util.spec_from_file_location("msl_child", {str(Path(__file__).resolve())!r})
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        target = GPUTarget("metal", 80, 32)
        signature = {{"X": "*fp32", "Y": "*fp32", "O": "*fp32",
                     "N": "i32", "RUNTIME_BASE": "i32",
                     "RUNTIME_OTHER": "fp32"}}

        def compile_case(block, mask_mode, other_mode, operation, num_warps):
            triton.compile(ASTSource(
                module._reduce_rank1_mask_shape_kernel,
                signature=signature,
                constexprs={{"BLOCK": block, "MASK_MODE": mask_mode,
                            "OTHER_MODE": other_mode, "OP": operation}}),
                target=target, options={{"num_warps": num_warps}})

        for attempt in range(3):
            try:
                compile_case(128, {mask_mode!r}, {other_mode!r},
                             {operation!r}, 1)
            except RuntimeError:
                pass
            else:
                raise AssertionError("unsupported masked rank-1 reduce compiled")
        compile_case(128, "canonical", "constant", "sum", 4)
        print("RECOVERED")
    """))
    result = subprocess.run([sys.executable, str(script)], capture_output=True,
                            text=True, timeout=120,
                            env={**os.environ,
                                 "TRITON_CACHE_DIR": str(tmp_path / "cache")})
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert output.count(needle) == 3
    assert "RECOVERED" in output
    assert "failed to legalize" not in output


@pytest.mark.parametrize(
    "address_mode,operation,value_mode",
    [
        ("bitand", "sum", "plain"),
        ("bitand", "max", "chain"),
    ],
    ids=[
        "bitand_sum",
        "bitand_chain_max",
    ],
)
def test_reduce_rank1_address_shape_unsupported_fails_without_crashing(
        address_mode, operation, value_mode, tmp_path):
    # Strided, repeated, gathered and index-space-shifted addresses are replayed
    # by the general cone evaluator (see
    # test_metal_backend_reduce_rank1_block_gt_tpb.py for their numbers); what
    # must still be refused before conversion is an index NEITHER path can
    # rebuild, three times in a row, leaving the process able to compile after.
    import os
    import subprocess
    import sys
    import textwrap

    script = tmp_path / "reduce_address_shape.py"
    script.write_text(textwrap.dedent(f"""
        import importlib.util
        import triton
        from triton.compiler import ASTSource
        from triton.backends.compiler import GPUTarget
        spec = importlib.util.spec_from_file_location("msl_child", {str(Path(__file__).resolve())!r})
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        target = GPUTarget("metal", 80, 32)
        signature = {{"X": "*fp32", "Y": "*fp32", "O": "*fp32",
                     "RUNTIME_BASE": "i32"}}

        def compile_case(block, address_mode, operation, value_mode, num_warps):
            triton.compile(ASTSource(
                module._reduce_rank1_address_shape_kernel,
                signature=signature,
                constexprs={{"BLOCK": block, "ADDRESS_MODE": address_mode,
                            "OP": operation, "VALUE_MODE": value_mode}}),
                target=target, options={{"num_warps": num_warps}})

        for attempt in range(3):
            try:
                compile_case(128, {address_mode!r}, {operation!r},
                             {value_mode!r}, 1)
            except RuntimeError:
                pass
            else:
                raise AssertionError("unsupported rank-1 reduce address compiled")
        compile_case(128, "canonical", "sum", "plain", 4)
        print("RECOVERED")
    """))
    result = subprocess.run([sys.executable, str(script)], capture_output=True,
                            text=True, timeout=120,
                            env={**os.environ,
                                 "TRITON_CACHE_DIR": str(tmp_path / "cache")})
    output = result.stdout + result.stderr
    needle = ("rank-1 reduce address cannot be replayed by the "
              "single-load-chain path nor by the general cone evaluator")
    assert result.returncode == 0, output
    assert output.count(needle) == 3
    assert "RECOVERED" in output
    assert "failed to legalize" not in output


@pytest.mark.parametrize(
    "block,offset_mode,mask_mode,widen,needle",
    [
        (128, "bitand", "canonical", False,
         "contiguous masked reduce cannot replay its load index"),
        (128, "select", "canonical", False,
         "contiguous masked reduce cannot replay its load index"),
        (128, "canonical", "bitand", False,
         "contiguous masked reduce cannot replay its mask index"),
        (128, "canonical", "select", False,
         "contiguous masked reduce cannot replay its mask index"),
        (32, "bitand", "canonical", False,
         "contiguous masked reduce cannot replay its load index"),
        (32, "canonical", "bitand", False,
         "contiguous masked reduce cannot replay its mask index"),
    ],
    ids=[
        "offset_bitand_e4",
        "offset_select_e4",
        "mask_bitand_e4",
        "mask_select_e4",
        "offset_bitand_e1",
        "mask_bitand_e1",
    ],
)
def test_reduce_rank1_contiguous_masked_index_unsupported_fails_without_crashing(
        block, offset_mode, mask_mode, widen, needle, tmp_path):
    import os
    import subprocess
    import sys
    import textwrap

    script = tmp_path / "reduce_contiguous_masked_index.py"
    script.write_text(textwrap.dedent(f"""
        import importlib.util
        import triton
        from triton.compiler import ASTSource
        from triton.backends.compiler import GPUTarget
        spec = importlib.util.spec_from_file_location("msl_child", {str(Path(__file__).resolve())!r})
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        target = GPUTarget("metal", 80, 32)
        signature = {{"X": "*fp32", "O": "*fp32", "N": "i32",
                     "K": "i32", "RUNTIME_STRIDE": "i32"}}

        def compile_case(block, offset_mode, mask_mode, widen):
            triton.compile(ASTSource(
                module._reduce_rank1_contiguous_masked_index_kernel,
                signature=signature,
                constexprs={{"BLOCK": block, "OFFSET_MODE": offset_mode,
                            "MASK_MODE": mask_mode, "WIDEN": widen}}),
                target=target, options={{"num_warps": 1}})

        for attempt in range(3):
            try:
                compile_case({block}, {offset_mode!r}, {mask_mode!r}, {widen})
            except RuntimeError:
                pass
            else:
                raise AssertionError("unsupported contiguous masked rank-1 reduce compiled")
        compile_case(128, "modulo", "canonical", False)
        print("RECOVERED")
    """))
    result = subprocess.run([sys.executable, str(script)], capture_output=True,
                            text=True, timeout=120,
                            env={**os.environ,
                                 "TRITON_CACHE_DIR": str(tmp_path / "cache")})
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert output.count(needle) == 3
    assert "RECOVERED" in output
    assert "failed to legalize" not in output


@pytest.mark.parametrize(
    "mask_mode,other_mode,operation,n,runtime_base,runtime_other",
    [
        ("compound", "constant", "max", 17, 1, -9.0),
        ("compound", "constant", "sum", 17, -1, -9.0),
        ("ge", "constant", "max", 11, 1, -9.0),
        ("scaled_lhs", "constant", "max", 19, 1, -9.0),
        ("shifted_mismatch", "constant", "max", 23, 5, -9.0),
        ("shifted", "constant", "sum", 29, 3, -9.0),
        ("tensor_rhs", "constant", "max", 31, 1, -9.0),
        ("canonical", "runtime", "sum", 21, 1, -3.5),
    ],
    ids=[
        "compound_mask_max",
        "compound_row_false_sum",
        "ge_mask_max",
        "scaled_lhs_mask_max",
        "shifted_mask_mismatch_max",
        "shifted_mask_sum",
        "tensor_rhs_mask_max",
        "runtime_other_sum",
    ],
)
@pytest.mark.parametrize(
    "block,num_warps",
    [(32, 1), (128, 1), (256, 2)],
    ids=["e1", "e4", "e4_w2"],
)
def test_reduce_rank1_mask_shape_matches_torch(
        mask_mode, other_mode, operation, n, runtime_base, runtime_other,
        block, num_warps):
    # At one element per thread the band's converted scalar is read directly and
    # none of the mask rules run; above it the source is replayed per element,
    # which is where the legacy fold's `cmpi slt(range, splat)` envelope used to
    # refuse everything below. The two geometries must agree.
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("Metal backend requires an MPS-enabled PyTorch")
    x_extent = block + max(runtime_base, 0) + 8
    x = ((torch.arange(x_extent) % 23).float() - 7.0) / 5.0
    y = torch.full((block,), float(n), dtype=torch.float32)
    if mask_mode == "tensor_rhs":
        y = (torch.arange(block, dtype=torch.float32) % 5) * 3.0 + 5.0
    offsets = torch.arange(block)
    if mask_mode == "compound":
        mask = (offsets < n) & (runtime_base >= 0)
    elif mask_mode == "ge":
        mask = offsets >= n
    elif mask_mode == "scaled_lhs":
        mask = offsets * 2 < n
    elif mask_mode == "shifted_mismatch":
        mask = offsets + runtime_base < n
    elif mask_mode == "shifted":
        mask = offsets + runtime_base < n
    elif mask_mode == "tensor_rhs":
        mask = offsets < y.to(torch.int32)
    else:
        mask = offsets < n
    ptr_offsets = offsets + runtime_base if mask_mode == "shifted" else offsets
    if other_mode == "runtime":
        other = torch.full((block,), runtime_other)
    elif other_mode == "tensor":
        other = offsets.to(torch.float32) * 0.25
    elif operation == "max":
        other = torch.full((block,), -float("inf"))
    else:
        other = torch.zeros(block)
    values = torch.where(mask, x[ptr_offsets], other)
    expected = values.max() if operation == "max" else values.sum()
    out = torch.empty((), device="mps")

    def launch():
        _reduce_rank1_mask_shape_kernel[(1,)](
            x.to("mps"), y.to("mps"), out, n, runtime_base, runtime_other,
            BLOCK=block, MASK_MODE=mask_mode, OTHER_MODE=other_mode,
            OP=operation, num_warps=num_warps)

    # `tensor_rhs` spells its bound as an `arith.fptosi` of a loaded f32 tensor.
    # That cast was the last mask cone the evaluator could not rebuild, and it
    # was refused above one element per thread until the evaluator and its
    # dry-run predicate learned it together.
    launch()
    torch.mps.synchronize()
    torch.testing.assert_close(out.cpu(), expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    "block,num_warps,operation,n",
    [
        (512, 4, "sum", 503),
        (512, 4, "max", 497),
    ],
    ids=["runtime_sum_e4", "runtime_max_e4"],
)
def test_reduce_rank1_canonical_mask_multiband_runtime_bound_matches_torch(
        block, num_warps, operation, n):
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("Metal backend requires an MPS-enabled PyTorch")
    x = ((torch.arange(block) % 37).float() - 13.0) / 7.0
    y = torch.zeros(block, dtype=torch.float32)
    out = torch.empty((), device="mps")
    _reduce_rank1_mask_shape_kernel[(1,)](
        x.to("mps"), y.to("mps"), out, n, 0, 0.0,
        BLOCK=block, MASK_MODE="canonical", OTHER_MODE="constant",
        OP=operation, num_warps=num_warps)
    torch.mps.synchronize()
    expected = x[:n].sum() if operation == "sum" else x[:n].max()
    torch.testing.assert_close(out.cpu(), expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    "address_mode,operation,block,num_warps,runtime_base",
    [
        ("runtime_stride", "sum", 32, 1, 3),
        ("constant_stride", "max", 32, 1, 0),
        ("modulo", "sum", 32, 1, 0),
        ("gather", "max", 32, 1, 0),
        ("shifted_tensor", "sum", 32, 1, 5),
        ("scalar_base", "sum", 512, 4, 7),
        ("rich_strided", "sum", 512, 4, 3),
        ("rich_strided", "max", 512, 4, 3),
    ],
    ids=[
        "single_band_runtime_stride_sum",
        "single_band_constant_stride_max",
        "single_band_modulo_sum",
        "single_band_gather_max",
        "single_band_shifted_tensor_sum",
        "multiband_scalar_base_sum",
        "rich_cone_strided_sum",
        "rich_cone_strided_max",
    ],
)
def test_reduce_rank1_address_shape_supported_controls_match_torch(
        address_mode, operation, block, num_warps, runtime_base):
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("Metal backend requires an MPS-enabled PyTorch")
    offsets = torch.arange(block)
    y = (offsets * 5 + 11).remainder(block * 4).float()
    x = ((torch.arange(block * 4 + runtime_base + 32) % 41).float() - 17.0) / 9.0
    if address_mode == "runtime_stride":
        ptr_offsets = offsets * runtime_base
    elif address_mode == "constant_stride":
        ptr_offsets = offsets * 3
    elif address_mode == "modulo":
        ptr_offsets = offsets % 7
    elif address_mode == "gather":
        ptr_offsets = y.to(torch.int64)
    elif address_mode == "shifted_tensor":
        ptr_offsets = offsets + runtime_base
    elif address_mode == "scalar_base":
        ptr_offsets = offsets + runtime_base
    elif address_mode == "rich_strided":
        ptr_offsets = torch.where(offsets % 2 == 0, offsets * runtime_base, offsets)
    else:
        ptr_offsets = offsets
    values = x[ptr_offsets]
    expected = values.max() if operation == "max" else values.sum()
    out = torch.empty((), device="mps")
    _reduce_rank1_address_shape_kernel[(1,)](
        x.to("mps"), y.to("mps"), out, runtime_base, BLOCK=block,
        ADDRESS_MODE=address_mode, OP=operation, VALUE_MODE="plain",
        num_warps=num_warps)
    torch.mps.synchronize()
    torch.testing.assert_close(out.cpu(), expected, atol=1e-5, rtol=1e-5)


def test_reduce_rank1_shifted_canonical_mask_multiband_matches_torch():
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("Metal backend requires an MPS-enabled PyTorch")
    block = 512
    base = 9
    n = block + base - 23
    x = ((torch.arange(block + base + 4) % 37).float() - 13.0) / 7.0
    y = torch.zeros(block, dtype=torch.float32)
    out = torch.empty((), device="mps")
    _reduce_rank1_mask_shape_kernel[(1,)](
        x.to("mps"), y.to("mps"), out, n, base, 0.0,
        BLOCK=block, MASK_MODE="shifted", OTHER_MODE="constant",
        OP="max", num_warps=4)
    torch.mps.synchronize()
    torch.testing.assert_close(out.cpu(), x[base:n].max(), atol=1e-5, rtol=1e-5)


def test_reduce_rank1_contiguous_masked_strided_sum_matches_torch():
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("Metal backend requires an MPS-enabled PyTorch")
    block = 4096
    n = block - 19
    stride = 3
    x = ((torch.arange(block * stride) % 29).float() + 1.0) / 11.0
    out = torch.empty((), device="mps")
    _reduce_rank1_contiguous_masked_address_kernel[(1,)](
        x.to("mps"), out, n, stride, BLOCK=block, num_warps=1)
    torch.mps.synchronize()
    expected = x[torch.arange(n) * stride].sum()
    torch.testing.assert_close(out.cpu(), expected, atol=1e-3, rtol=1e-5)


@pytest.mark.parametrize(
    "block,offset_mode,mask_mode,widen,n,k,runtime_stride",
    [
        (128, "modulo", "canonical", False, 113, 7, 3),
        (128, "dynamic_stride", "canonical", False, 119, 5, 2),
        (128, "canonical", "modulo", False, 5, 7, 3),
        (128, "canonical", "dynamic_stride", False, 113, 7, 1),
        (16, "bitand", "canonical", False, 11, 7, 3),
        (16, "select", "canonical", False, 13, 7, 3),
        (16, "canonical", "bitand", False, 5, 7, 3),
        (16, "canonical", "select", False, 7, 9, 3),
        (128, "bitand", "canonical", True, 17, 31, 3),
        (128, "canonical", "select", True, 23, 17, 3),
    ],
    ids=[
        "offset_modulo_e4",
        "offset_dynamic_stride_e4",
        "mask_modulo_e4",
        "mask_dynamic_stride_e4",
        "block16_offset_bitand",
        "block16_offset_select",
        "block16_mask_bitand",
        "block16_mask_select",
        "widened_offset_bitand_e4",
        "widened_mask_select_e4",
    ],
)
def test_reduce_rank1_contiguous_masked_index_supported_controls_match_torch(
        block, offset_mode, mask_mode, widen, n, k, runtime_stride):
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("Metal backend requires an MPS-enabled PyTorch")

    offsets = torch.arange(block, dtype=torch.int64)
    if offset_mode == "bitand":
        load_index = offsets & k
    elif offset_mode == "select":
        load_index = torch.where(offsets < k, offsets, offsets // 2)
    elif offset_mode == "modulo":
        load_index = offsets.remainder(k)
    elif offset_mode == "dynamic_stride":
        load_index = offsets * runtime_stride
    else:
        load_index = offsets

    if mask_mode == "bitand":
        mask_index = offsets & k
    elif mask_mode == "select":
        mask_index = torch.where(offsets < k, offsets, offsets // 2)
    elif mask_mode == "modulo":
        mask_index = offsets.remainder(k)
    elif mask_mode == "dynamic_stride":
        mask_index = offsets * runtime_stride
    else:
        mask_index = offsets

    extent = int(max(load_index.max(), mask_index.max()).item()) + 8
    x = ((torch.arange(extent) % 43).float() - 19.0) / 11.0
    values = torch.where(mask_index < n, x[load_index], torch.zeros(block))
    expected = values.sum()
    out = torch.empty((), device="mps")
    _reduce_rank1_contiguous_masked_index_kernel[(1,)](
        x.to("mps"), out, n, k, runtime_stride, BLOCK=block,
        OFFSET_MODE=offset_mode, MASK_MODE=mask_mode, WIDEN=widen,
        num_warps=1)
    torch.mps.synchronize()
    torch.testing.assert_close(out.cpu(), expected, atol=1e-5, rtol=1e-5)


def _contiguous_masked_chain_reference(torch, block, depth, inner_mode,
                                       mask_mode, n, k, runtime_stride,
                                       programs, x):
    offsets = torch.arange(block)
    if inner_mode == "stride":
        inner = offsets * runtime_stride
    elif inner_mode == "modulo":
        inner = offsets.remainder(k)
    elif inner_mode == "bitand":
        inner = offsets & k
    else:
        inner = offsets
    mask = offsets <= n if mask_mode == "compound" else offsets < n
    rows = []
    for pid in range(programs):
        index = pid * k + inner + offsets
        if depth == 3:
            index = index + runtime_stride
        rows.append(torch.where(mask, x[index], torch.zeros(block)).sum())
    return torch.stack(rows)


@pytest.mark.parametrize(
    "block,num_warps,depth,inner_mode,mask_mode",
    [
        (128, 1, 2, "canonical", "simple"),
        (128, 1, 2, "stride", "simple"),
        (128, 1, 3, "stride", "compound"),
        (128, 1, 2, "modulo", "simple"),
        (32, 1, 2, "stride", "simple"),
        (32, 1, 3, "canonical", "compound"),
        (512, 4, 2, "stride", "simple"),
        (512, 4, 3, "modulo", "compound"),
        (2048, 2, 2, "canonical", "simple"),
    ],
    ids=[
        "depth2_canonical_e4",
        "depth2_stride_e4",
        "depth3_stride_compound_e4",
        "depth2_modulo_e4",
        "depth2_stride_e1",
        "depth3_canonical_compound_e1",
        "depth2_stride_nw4",
        "depth3_modulo_compound_nw4",
        "depth2_canonical_block2048",
    ],
)
def test_reduce_rank1_contiguous_masked_chain_matches_torch(
        block, num_warps, depth, inner_mode, mask_mode):
    """Every tt.addptr level of the load's chain must reach the address.

    `p = X + rows; tl.load(p + cols)` stacks two tensor offsets. Replaying only
    the outermost one reads `X + cols` and still returns a plausible number, so
    these cases compare against torch rather than against a rejection.
    """
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("Metal backend requires an MPS-enabled PyTorch")

    programs = 3
    n = max(1, block - 19)
    k = 7
    runtime_stride = 3
    extent = programs * k + block * (runtime_stride + 1) + runtime_stride + 8
    x = ((torch.arange(extent) % 43).float() - 19.0) / 11.0
    expected = _contiguous_masked_chain_reference(
        torch, block, depth, inner_mode, mask_mode, n, k, runtime_stride,
        programs, x)
    out = torch.empty(programs, device="mps")
    _reduce_rank1_contiguous_masked_chain_kernel[(programs,)](
        x.to("mps"), out, n, k, runtime_stride, BLOCK=block, DEPTH=depth,
        INNER_MODE=inner_mode, MASK_MODE=mask_mode, num_warps=num_warps)
    torch.mps.synchronize()
    tolerance = 2e-3 if block >= 512 else 1e-5
    torch.testing.assert_close(out.cpu(), expected, atol=tolerance, rtol=1e-5)


@pytest.mark.parametrize(
    "block,depth",
    [(128, 2), (32, 2), (128, 3)],
    ids=["inner_bitand_e4", "inner_bitand_e1", "inner_bitand_depth3"],
)
def test_reduce_rank1_contiguous_masked_chain_unsupported_fails_without_crashing(
        block, depth, tmp_path):
    """An inner level is proved before conversion, exactly like the outer one."""
    import os
    import subprocess
    import sys
    import textwrap

    script = tmp_path / "reduce_contiguous_masked_chain.py"
    script.write_text(textwrap.dedent(f"""
        import importlib.util
        import triton
        from triton.compiler import ASTSource
        from triton.backends.compiler import GPUTarget
        spec = importlib.util.spec_from_file_location("msl_child", {str(Path(__file__).resolve())!r})
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        target = GPUTarget("metal", 80, 32)
        signature = {{"X": "*fp32", "O": "*fp32", "N": "i32",
                     "K": "i32", "RUNTIME_STRIDE": "i32"}}

        def compile_case(block, depth, inner_mode):
            triton.compile(ASTSource(
                module._reduce_rank1_contiguous_masked_chain_kernel,
                signature=signature,
                constexprs={{"BLOCK": block, "DEPTH": depth,
                            "INNER_MODE": inner_mode, "MASK_MODE": "simple"}}),
                target=target, options={{"num_warps": 1}})

        for attempt in range(3):
            try:
                compile_case({block}, {depth}, "bitand")
            except RuntimeError:
                pass
            else:
                raise AssertionError("unsupported chain level compiled")
        compile_case({block}, {depth}, "stride")
        print("RECOVERED")
    """))
    result = subprocess.run([sys.executable, str(script)], capture_output=True,
                            text=True, timeout=120,
                            env={**os.environ,
                                 "TRITON_CACHE_DIR": str(tmp_path / "cache")})
    output = result.stdout + result.stderr
    needle = "contiguous masked reduce cannot replay its load index"
    assert result.returncode == 0, output
    assert output.count(needle) == 3
    assert "RECOVERED" in output
    assert "failed to legalize" not in output


@triton.jit
def _reduce_rank3_double_kernel(X, O, D0: tl.constexpr, D1: tl.constexpr,
                                D2: tl.constexpr, FORM: tl.constexpr):
    i = tl.arange(0, D0)[:, None, None]
    j = tl.arange(0, D1)[None, :, None]
    k = tl.arange(0, D2)[None, None, :]
    v = tl.load(X + i * D1 * D2 + j * D2 + k)
    if FORM == "sum_sum_inner_middle":
        tl.store(O + tl.arange(0, D0), tl.sum(tl.sum(v, axis=2), axis=1))
    elif FORM == "sum_sum_inner_outer":
        tl.store(O + tl.arange(0, D1), tl.sum(tl.sum(v, axis=2), axis=0))
    elif FORM == "max_of_sum":
        tl.store(O + tl.arange(0, D0), tl.max(tl.sum(v, axis=2), axis=1))
    else:  # single inner reduce — the supported neighbour
        r = tl.sum(v, axis=2)
        tl.store(O + tl.arange(0, D0)[:, None] * D1
                 + tl.arange(0, D1)[None, :], r)


@pytest.mark.parametrize(
    "form",
    ["sum_sum_inner_middle", "sum_sum_inner_outer", "max_of_sum"],
    ids=["sum_sum_inner_middle", "sum_sum_inner_outer", "max_of_sum"],
)
def test_reduce_rank3_double_fails_without_crashing(form, tmp_path):
    """The rank-2 result of an inner reduce carries a slice encoding.

    Both rank-2 specializations need a blocked tile and the staged rank-N path
    declines it, so the shape has no lowering. It used to reach conversion and
    surface as an unnamed `failed to legalize operation 'tt.reduce'` after the
    kernel signature had already been rewritten.
    """
    import os
    import subprocess
    import sys
    import textwrap

    # 2x4x4: a shape whose layout stays thread-major under this compile path, so
    # the INNER rank-3 reduce is accepted and the refusal under test is the
    # rank-2 one above it. A tile whose layout wraps into a second register band
    # is refused earlier, by the rank >= 3 envelope.
    script = tmp_path / "reduce_rank3_double.py"
    script.write_text(textwrap.dedent(f"""
        import importlib.util
        import triton
        from triton.compiler import ASTSource
        from triton.backends.compiler import GPUTarget
        spec = importlib.util.spec_from_file_location("msl_child", {str(Path(__file__).resolve())!r})
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        target = GPUTarget("metal", 80, 32)
        signature = {{"X": "*fp32", "O": "*fp32"}}

        def compile_case(form):
            triton.compile(ASTSource(
                module._reduce_rank3_double_kernel,
                signature=signature,
                constexprs={{"D0": 2, "D1": 4, "D2": 4, "FORM": form}}),
                target=target, options={{"num_warps": 1}})

        for attempt in range(3):
            try:
                compile_case({form!r})
            except RuntimeError:
                pass
            else:
                raise AssertionError("rank-3 double reduce compiled")
        compile_case("single_inner")
        print("RECOVERED")
    """))
    result = subprocess.run([sys.executable, str(script)], capture_output=True,
                            text=True, timeout=120,
                            env={**os.environ,
                                 "TRITON_CACHE_DIR": str(tmp_path / "cache")})
    output = result.stdout + result.stderr
    needle = "rank-2 reduce requires a blocked source tile"
    assert result.returncode == 0, output
    assert output.count(needle) == 3
    assert "RECOVERED" in output
    assert "failed to legalize" not in output


def test_reduce_rank3_single_inner_still_matches_torch():
    """The supported neighbour must keep working: the refusal is for the
    rank-2 reduce ABOVE an inner one, not for the inner reduce itself."""
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("Metal backend requires an MPS-enabled PyTorch")
    d0, d1, d2 = 2, 4, 8
    x = ((torch.arange(d0 * d1 * d2) % 43).float() - 19.0) / 11.0
    out = torch.empty(d0 * d1, device="mps")
    _reduce_rank3_double_kernel[(1,)](
        x.to("mps"), out, D0=d0, D1=d1, D2=d2, FORM="single_inner",
        num_warps=1)
    torch.mps.synchronize()
    expected = x.reshape(d0, d1, d2).sum(dim=2).flatten()
    torch.testing.assert_close(out.cpu(), expected, atol=1e-4, rtol=1e-4)


@triton.jit
def _reduce_rank3_axis_kernel(X, O, D0: tl.constexpr, D1: tl.constexpr,
                              D2: tl.constexpr, AXIS: tl.constexpr):
    i = tl.arange(0, D0)[:, None, None]
    j = tl.arange(0, D1)[None, :, None]
    k = tl.arange(0, D2)[None, None, :]
    off = i * D1 * D2 + j * D2 + k
    v = tl.load(X + off)
    if AXIS == 2:
        r = tl.sum(v, axis=2)[:, :, None]
    elif AXIS == 1:
        r = tl.sum(v, axis=1)[:, None, :]
    else:
        r = tl.sum(v, axis=0)[None, :, :]
    tl.store(O + off, r + 0.0 * v)


@pytest.mark.parametrize(
    "d0,d1,d2,num_warps,axis",
    [
        (8, 4, 4, 1, 2), (8, 4, 4, 1, 1), (8, 4, 4, 1, 0),
        (2, 4, 4, 1, 2), (2, 8, 32, 4, 1), (4, 8, 16, 4, 0),
    ],
    ids=["e4_axis2", "e4_axis1", "e4_axis0", "small_axis2", "wide_axis1",
         "wide_axis0"],
)
def test_reduce_rank3_single_band_matches_torch(d0, d1, d2, num_warps, axis):
    """One register band per thread: the layout's flat index and the
    `localTid * elemPerThread + iv` order the rest of the emitter uses are the
    same mapping, so the staged fold groups the elements it addressed."""
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("Metal backend requires an MPS-enabled PyTorch")
    n = d0 * d1 * d2
    x = ((torch.arange(n) % 43).float() - 19.0) / 11.0
    out = torch.empty(n, device="mps")
    _reduce_rank3_axis_kernel[(1,)](x.to("mps"), out, D0=d0, D1=d1, D2=d2,
                                    AXIS=axis, num_warps=num_warps)
    torch.mps.synchronize()
    tile = x.reshape(d0, d1, d2)
    expected = tile.sum(dim=axis, keepdim=True).expand(d0, d1, d2).flatten()
    torch.testing.assert_close(out.cpu(), expected, atol=2e-3, rtol=1e-4)


@pytest.mark.parametrize(
    "d0,d1,d2,num_warps,axis",
    [
        (4, 8, 16, 1, 2), (4, 8, 16, 1, 1), (4, 8, 16, 1, 0),
        (4, 8, 16, 2, 1), (4, 4, 64, 4, 0), (4, 4, 64, 1, 2),
        (2, 8, 32, 1, 1),
    ],
    ids=["e16_axis2", "e16_axis1", "e16_axis0", "e8_axis1", "e8_axis0",
         "e32_axis2", "e16_mid_axis1"],
)
def test_reduce_rank3_multiband_matches_torch(d0, d1, d2, num_warps, axis):
    """A layout that wraps into a second register band carries two labels.

    The staged scratch used to be indexed by the LAYOUT's flat index while the
    value in it was the element `localTid * elemPerThread + iv` names, so the
    fold reduced a group the kernel never addressed and returned a plausible
    number. The scratch is now relabelled into that same thread-major order, so
    one label serves the addresses, the publish and the fold.
    """
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("Metal backend requires an MPS-enabled PyTorch")
    n = d0 * d1 * d2
    x = ((torch.arange(n) % 43).float() - 19.0) / 11.0
    out = torch.empty(n, device="mps")
    _reduce_rank3_axis_kernel[(1,)](x.to("mps"), out, D0=d0, D1=d1, D2=d2,
                                    AXIS=axis, num_warps=num_warps)
    torch.mps.synchronize()
    tile = x.reshape(d0, d1, d2)
    expected = tile.sum(dim=axis, keepdim=True).expand(d0, d1, d2).flatten()
    torch.testing.assert_close(out.cpu(), expected, atol=2e-3, rtol=1e-4)


@triton.jit
def _reduce_rank3_source_kernel(X, O, PLANE_STRIDE, ROW_STRIDE, BOUND, OTHER,
                                D0: tl.constexpr, D1: tl.constexpr,
                                D2: tl.constexpr, AXIS: tl.constexpr,
                                MODE: tl.constexpr):
    pid = tl.program_id(0)
    i = tl.arange(0, D0)[:, None, None]
    j = tl.arange(0, D1)[None, :, None]
    k = tl.arange(0, D2)[None, None, :]
    if MODE == "strided":
        off = i * PLANE_STRIDE + j * ROW_STRIDE + k
    elif MODE == "multiprog":
        off = pid * PLANE_STRIDE + i * D1 * D2 + j * D2 + k
    else:
        off = i * D1 * D2 + j * D2 + k
    if MODE == "masked":
        v = tl.load(X + off, mask=k < BOUND, other=OTHER)
    else:
        v = tl.load(X + off)
    if MODE == "cone":
        v = v * 2.0 + 1.0
    elif MODE == "cone_exp":
        v = tl.exp(v * 0.25)
    if AXIS == 2:
        r = tl.sum(v, axis=2)[:, :, None]
    elif AXIS == 1:
        r = tl.sum(v, axis=1)[:, None, :]
    else:
        r = tl.sum(v, axis=0)[None, :, :]
    out = i * D1 * D2 + j * D2 + k
    tl.store(O + pid * (D0 * D1 * D2) + out, r + 0.0 * v)


@pytest.mark.parametrize(
    "d0,d1,d2,num_warps,axis,mode",
    [
        # A computed cone reaches the fold through elementwise arith, so its
        # values still carry the loads' thread-major addresses and the scratch
        # has to be relabelled for them too. Missing that left every
        # multi-band cone reducing the wrong group.
        (4, 8, 16, 1, 2, "cone"),
        (4, 8, 16, 1, 1, "cone"),
        (4, 8, 16, 1, 0, "cone"),
        (4, 4, 64, 2, 2, "cone_exp"),
        # The address itself: a tile that is a strided sub-block of a larger
        # array, a per-program base, and a masked load carrying `other`.
        (4, 8, 16, 1, 2, "strided"),
        (4, 8, 16, 1, 1, "strided"),
        (2, 8, 32, 4, 0, "strided"),
        (4, 8, 16, 1, 2, "multiprog"),
        (4, 8, 16, 1, 2, "masked"),
        (4, 8, 16, 2, 1, "masked"),
    ],
    ids=["cone_axis2", "cone_axis1", "cone_axis0", "cone_exp_axis2",
         "strided_axis2", "strided_axis1", "strided_axis0_nw4",
         "multiprog_axis2", "masked_axis2", "masked_axis1"],
)
def test_reduce_rank3_source_and_address_match_torch(d0, d1, d2, num_warps,
                                                     axis, mode):
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("Metal backend requires an MPS-enabled PyTorch")
    plane_stride, row_stride = d1 * d2 * 3, d2 * 2
    bound, other = max(1, d2 - 5), 5.0
    programs = 3 if mode == "multiprog" else 1
    extent = programs * plane_stride * d0 + d0 * plane_stride + d1 * row_stride \
        + d2 + 64
    x = ((torch.arange(extent) % 43).float() - 19.0) / 11.0
    out = torch.empty(programs * d0 * d1 * d2, device="mps")
    _reduce_rank3_source_kernel[(programs,)](
        x.to("mps"), out, plane_stride, row_stride, bound, other, D0=d0, D1=d1,
        D2=d2, AXIS=axis, MODE=mode, num_warps=num_warps)
    torch.mps.synchronize()

    i = torch.arange(d0)[:, None, None]
    j = torch.arange(d1)[None, :, None]
    k = torch.arange(d2)[None, None, :]
    expected = []
    for pid in range(programs):
        if mode == "strided":
            index = i * plane_stride + j * row_stride + k
        elif mode == "multiprog":
            index = pid * plane_stride + i * d1 * d2 + j * d2 + k
        else:
            index = i * d1 * d2 + j * d2 + k
        tile = x[index]
        if mode == "masked":
            tile = torch.where(k < bound, tile, torch.full_like(tile, other))
        if mode == "cone":
            tile = tile * 2.0 + 1.0
        elif mode == "cone_exp":
            tile = torch.exp(tile * 0.25)
        expected.append(
            tile.sum(dim=axis, keepdim=True).expand(d0, d1, d2).flatten())
    torch.testing.assert_close(out.cpu(), torch.cat(expected), atol=3e-3,
                               rtol=1e-4)


@triton.jit
def _reduce_rank4_kernel(X, O, D0: tl.constexpr, D1: tl.constexpr,
                         D2: tl.constexpr, D3: tl.constexpr,
                         AXIS: tl.constexpr):
    a = tl.arange(0, D0)[:, None, None, None]
    b = tl.arange(0, D1)[None, :, None, None]
    c = tl.arange(0, D2)[None, None, :, None]
    d = tl.arange(0, D3)[None, None, None, :]
    off = a * D1 * D2 * D3 + b * D2 * D3 + c * D3 + d
    v = tl.load(X + off)
    if AXIS == 3:
        r = tl.sum(v, axis=3)[:, :, :, None]
    else:
        r = tl.sum(v, axis=0)[None, :, :, :]
    tl.store(O + off, r + 0.0 * v)


@triton.jit
def _reduce_rank3_loop_carried_kernel(X, O, T, PLANE_STRIDE,
                                      D0: tl.constexpr, D1: tl.constexpr,
                                      D2: tl.constexpr, AXIS: tl.constexpr):
    i = tl.arange(0, D0)[:, None, None]
    j = tl.arange(0, D1)[None, :, None]
    k = tl.arange(0, D2)[None, None, :]
    off = i * D1 * D2 + j * D2 + k
    acc = tl.zeros((D0, D1, D2), dtype=tl.float32)
    for t in range(0, T):
        acc += tl.load(X + t * PLANE_STRIDE + off)
    if AXIS == 2:
        r = tl.sum(acc, axis=2)[:, :, None]
    else:
        r = tl.sum(acc, axis=0)[None, :, :]
    tl.store(O + off, r + 0.0 * acc)


@pytest.mark.parametrize(
    "kernel,constexprs,needle",
    [
        ("_reduce_rank4_kernel",
         '{"D0": 2, "D1": 4, "D2": 4, "D3": 8, "AXIS": 3}',
         "rank >= 4 tiles are not implemented"),
        ("_reduce_rank4_kernel",
         '{"D0": 2, "D1": 4, "D2": 4, "D3": 8, "AXIS": 0}',
         "rank >= 4 tiles are not implemented"),
        ("_reduce_rank3_loop_carried_kernel",
         '{"D0": 4, "D1": 8, "D2": 16, "AXIS": 2}',
         "cannot hoist its source before the phase boundary"),
        ("_reduce_rank3_loop_carried_kernel",
         '{"D0": 4, "D1": 8, "D2": 16, "AXIS": 0}',
         "cannot hoist its source before the phase boundary"),
    ],
    ids=["rank4_axis3", "rank4_axis0", "loop_carried_axis2",
         "loop_carried_axis0"],
)
def test_reduce_rank4_and_loop_carried_fail_without_crashing(
        kernel, constexprs, needle, tmp_path):
    """Both shapes are out of envelope; both must SAY so before conversion.

    A rank >= 4 tile has no per-element range lowering at all — even a plain
    elementwise copy of one fails — and used to report only `failed to legalize
    operation 'tt.make_range'`. A rank-3 reduce over a loop-carried accumulator
    used to fail the pass with NO diagnostic whatsoever, because the phase
    pre-pass returned failure and its caller only signalled.
    """
    import os
    import subprocess
    import sys
    import textwrap

    signature = ('{"X": "*fp32", "O": "*fp32"}' if "rank4" in kernel else
                 '{"X": "*fp32", "O": "*fp32", "T": "i32", '
                 '"PLANE_STRIDE": "i32"}')
    script = tmp_path / "reduce_rank4_loop.py"
    script.write_text(textwrap.dedent(f"""
        import importlib.util
        import triton
        from triton.compiler import ASTSource
        from triton.backends.compiler import GPUTarget
        spec = importlib.util.spec_from_file_location("msl_child", {str(Path(__file__).resolve())!r})
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        target = GPUTarget("metal", 80, 32)

        def compile_case():
            triton.compile(ASTSource(
                getattr(module, {kernel!r}),
                signature={signature},
                constexprs={constexprs}),
                target=target, options={{"num_warps": 1}})

        def compile_supported():
            triton.compile(ASTSource(
                module._reduce_rank3_axis_kernel,
                signature={{"X": "*fp32", "O": "*fp32"}},
                constexprs={{"D0": 2, "D1": 4, "D2": 4, "AXIS": 2}}),
                target=target, options={{"num_warps": 1}})

        for attempt in range(3):
            try:
                compile_case()
            except RuntimeError:
                pass
            else:
                raise AssertionError("out-of-envelope shape compiled")
        compile_supported()
        print("RECOVERED")
    """))
    result = subprocess.run([sys.executable, str(script)], capture_output=True,
                            text=True, timeout=180,
                            env={**os.environ,
                                 "TRITON_CACHE_DIR": str(tmp_path / "cache")})
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert output.count(needle) >= 3, output
    assert "RECOVERED" in output
    assert "failed to legalize" not in output


@triton.jit
def _rank3_mul(a, b):
    return a * b


@triton.jit
def _rank3_and(a, b):
    return a & b


@triton.jit
def _rank3_xor(a, b):
    return a ^ b


@triton.jit
def _rank3_larger_square(a, b):
    return tl.where(a * a > b * b, a, b)


@triton.jit
def _reduce_rank3_combine_kernel(X, O, BOUND, OTHER, D0: tl.constexpr,
                                 D1: tl.constexpr, D2: tl.constexpr,
                                 AXIS: tl.constexpr, OP: tl.constexpr,
                                 MASKED: tl.constexpr):
    i = tl.arange(0, D0)[:, None, None]
    j = tl.arange(0, D1)[None, :, None]
    k = tl.arange(0, D2)[None, None, :]
    off = i * D1 * D2 + j * D2 + k
    if MASKED:
        v = tl.load(X + off, mask=k < BOUND, other=OTHER)
    else:
        v = tl.load(X + off)
    if OP == "max":
        r = tl.max(v, axis=AXIS)
    elif OP == "min":
        r = tl.min(v, axis=AXIS)
    elif OP == "prod":
        r = tl.reduce(v, AXIS, _rank3_mul)
    elif OP == "and":
        r = tl.reduce(v, AXIS, _rank3_and)
    elif OP == "xor":
        r = tl.reduce(v, AXIS, _rank3_xor)
    elif OP == "custom":
        r = tl.reduce(v, AXIS, _rank3_larger_square)
    else:
        r = tl.sum(v, axis=AXIS)
    if AXIS == 2:
        rb = r[:, :, None]
    elif AXIS == 1:
        rb = r[:, None, :]
    else:
        rb = r[None, :, :]
    tl.store(O + off, rb + v * 0)


def _rank3_combine_reference(torch, tile, axis, op):
    if op == "sum":
        return tile.sum(axis)
    if op == "max":
        return tile.amax(axis)
    if op == "min":
        return tile.amin(axis)
    if op == "prod":
        return tile.prod(axis)
    moved = tile.movedim(axis, -1)
    acc = moved[..., 0].clone()
    for step in range(1, moved.shape[-1]):
        nxt = moved[..., step]
        if op == "and":
            acc = acc & nxt
        elif op == "xor":
            acc = acc ^ nxt
        else:
            acc = torch.where(acc * acc > nxt * nxt, acc, nxt)
    return acc


@pytest.mark.parametrize(
    "d0,d1,d2,num_warps,axis,op,dtype,masked",
    [
        # The staged path only ever had `add` under test. Every other combine,
        # in the multi-band regime the scratch relabel governs.
        (4, 8, 16, 1, 2, "prod", "f32", False),
        (4, 8, 16, 1, 1, "min", "f32", False),
        (4, 8, 16, 1, 0, "custom", "f32", False),
        (4, 4, 64, 1, 2, "max", "f32", False),
        (4, 8, 16, 1, 2, "xor", "i32", False),
        (4, 8, 16, 1, 1, "and", "i32", False),
        (4, 8, 16, 4, 0, "prod", "i32", False),
        (8, 4, 4, 1, 2, "custom", "f32", False),
        # A masked lane's `other`, deliberately away from every identity: with
        # 0 and a sum, a dropped `other` is invisible.
        (4, 8, 16, 1, 2, "max", "f32", True),
        (4, 8, 16, 1, 1, "prod", "f32", True),
        (4, 8, 16, 4, 0, "min", "f32", True),
    ],
    ids=["prod_axis2", "min_axis1", "custom_axis0", "max_wide", "xor_i32",
         "and_i32", "prod_i32_nw4", "custom_single_band", "masked_max",
         "masked_prod", "masked_min_nw4"],
)
def test_reduce_rank3_combines_match_torch(d0, d1, d2, num_warps, axis, op,
                                           dtype, masked):
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("Metal backend requires an MPS-enabled PyTorch")
    n = d0 * d1 * d2
    if dtype == "i32":
        if op in ("and", "xor"):
            x = (torch.arange(n) * 2654435761 % 4096).to(torch.int32)
        else:
            x = ((torch.arange(n) % 3) - 1).to(torch.int32)
    else:
        base = ((torch.arange(n) % 43).float() - 19.0) / 11.0
        x = 1.0 + base * 0.03 if op == "prod" else base
    bound = max(1, d2 - 5)
    other = {"max": 7.5, "min": -7.5, "prod": 1.5, "sum": 3.25}.get(op, 0.0)

    out = torch.empty(n, dtype=x.dtype, device="mps")
    _reduce_rank3_combine_kernel[(1,)](
        x.to("mps"), out, bound, other, D0=d0, D1=d1, D2=d2, AXIS=axis, OP=op,
        MASKED=masked, num_warps=num_warps)
    torch.mps.synchronize()

    tile = x.reshape(d0, d1, d2)
    if masked:
        keep = torch.arange(d2)[None, None, :] < bound
        tile = torch.where(keep, tile, torch.full((d0, d1, d2), other))
    expected = _rank3_combine_reference(torch, tile, axis, op)
    expected = expected.unsqueeze(axis).expand(d0, d1, d2).flatten()
    if dtype == "i32":
        assert torch.equal(out.cpu(), expected)
    else:
        torch.testing.assert_close(out.cpu(), expected, atol=3e-3, rtol=1e-3)


@pytest.mark.parametrize(
    "form,needle",
    [
        ("tuple_reduce",
         "multi-value tt.reduce is implemented for the canonical rank-1 argmax"),
        ("tuple_scan",
         "tl.associative_scan carrying more than one value is implemented"),
        ("rank1_reduce", "rank-1 reduce cannot replay its source"),
    ],
    ids=["tuple_reduce", "tuple_scan", "rank1_reduce"],
)
def test_replayed_cone_through_if_is_rejected_not_crashed(form, needle,
                                                          tmp_path):
    # A cone that has to be replayed per element and passes through an `scf.if`
    # used to pass the pre-conversion validator and then be declined by the
    # pattern, because the same predicate answers differently once the `scf.if`
    # has been converted. That decline is inside `applyFullConversion`: it took
    # the process down with SIGSEGV every run for the two tuple forms and an
    # intermittent SIGABRT/SIGTRAP for the plain rank-1 one. The refusal now
    # happens before conversion, so it is catchable, named, and repeatable --
    # and the same kernel at ONE element per thread still compiles afterwards,
    # which is both the control and the proof that the context survived.
    import os
    import subprocess
    import sys
    import textwrap

    script = tmp_path / "replayed_cone_through_if.py"
    script.write_text(textwrap.dedent(f"""
        import importlib.util
        import triton
        from triton.compiler import ASTSource
        from triton.backends.compiler import GPUTarget
        spec = importlib.util.spec_from_file_location("msl_child", {str(Path(__file__).resolve())!r})
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        target = GPUTarget("metal", 80, 32)
        signature = {{"X": "*fp32", "Y": "*fp32", "O": "*fp32", "S": "fp32"}}

        def compile_case(block, num_warps):
            triton.compile(ASTSource(
                module._replayed_cone_through_if_kernel,
                signature=signature,
                constexprs={{"BLOCK": block, "FORM": {form!r}}}),
                target=target, options={{"num_warps": num_warps}})

        for attempt in range(3):
            try:
                compile_case(128, 1)     # four elements per thread
            except RuntimeError:
                pass
            else:
                raise AssertionError("a replayed scf.if cone compiled")
        compile_case(64, 2)              # one element per thread: supported
        print("RECOVERED")
    """))
    result = subprocess.run([sys.executable, str(script)], capture_output=True,
                            text=True, timeout=180,
                            env={**os.environ,
                                 "TRITON_CACHE_DIR": str(tmp_path / "cache")})
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert output.count(needle) == 3, output
    assert "RECOVERED" in output
    assert "failed to legalize" not in output


# `validateScanSupport` carries ten fail-closed diagnostics, none of which had a
# fixture. Probing them first (`.omc/fuzz/scan_envelope.py`, 20 shapes) sorted
# them into what a user can actually reach and what nothing can express: a
# non-power-of-two thread block is refused by the Triton frontend
# ("num_warps must be a power of 2"), and an empty or non-tpb-multiple tile
# cannot be spelled because `tl.arange` requires a power-of-two extent. Those
# stay unpinned on purpose. These six are reachable, and each is checked for the
# property that matters more than the wording: three refusals in a row leave the
# process able to compile.
@pytest.mark.parametrize(
    "form,needle",
    [
        ("f16", "scan is implemented for f32 and i32 only"),
        ("i64", "scan is implemented for f32 and i32 only"),
        ("e128", "rank-1 scan requires a power-of-two 1..64 elements per thread"),
        ("rank2_wide_axis1",
         "rank >= 2 scan is implemented for a blocked tile"),
        ("rank2_wide_axis0",
         "rank >= 2 scan is implemented for a blocked tile"),
        ("tuple_f16",
         "tl.associative_scan carrying more than one value is implemented"),
        ("tuple_e128",
         "tl.associative_scan carrying more than one value is implemented"),
    ],
    ids=["f16", "i64", "e128", "rank2_wide_axis1", "rank2_wide_axis0",
         "tuple_f16", "tuple_e128"],
)
def test_scan_envelope_unsupported_fails_without_crashing(form, needle,
                                                          tmp_path):
    import os
    import subprocess
    import sys
    import textwrap

    cases = {
        "f16": ("_scan_envelope_kernel", '{"X": "*fp16", "O": "*fp16"}',
                '{"BLOCK": 128, "FORM": "plain"}', 1),
        "i64": ("_scan_envelope_kernel", '{"X": "*i64", "O": "*i64"}',
                '{"BLOCK": 128, "FORM": "plain"}', 1),
        "e128": ("_scan_envelope_kernel", '{"X": "*fp32", "O": "*fp32"}',
                 '{"BLOCK": 4096, "FORM": "plain"}', 1),
        "rank2_wide_axis1": ("_scan_envelope_rank2_kernel",
                             '{"X": "*fp32", "O": "*fp32"}',
                             '{"M": 4, "N": 128, "AXIS": 1}', 1),
        "rank2_wide_axis0": ("_scan_envelope_rank2_kernel",
                             '{"X": "*fp32", "O": "*fp32"}',
                             '{"M": 128, "N": 4, "AXIS": 0}', 1),
        "tuple_f16": ("_scan_envelope_kernel", '{"X": "*fp16", "O": "*fp16"}',
                      '{"BLOCK": 128, "FORM": "tuple"}', 1),
        "tuple_e128": ("_scan_envelope_kernel", '{"X": "*fp32", "O": "*fp32"}',
                       '{"BLOCK": 4096, "FORM": "tuple"}', 1),
    }
    kernel, signature, constexprs, num_warps = cases[form]

    script = tmp_path / "scan_envelope.py"
    script.write_text(textwrap.dedent(f"""
        import importlib.util
        import triton
        from triton.compiler import ASTSource
        from triton.backends.compiler import GPUTarget
        spec = importlib.util.spec_from_file_location("msl_child", {str(Path(__file__).resolve())!r})
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        target = GPUTarget("metal", 80, 32)

        for attempt in range(3):
            try:
                triton.compile(ASTSource(
                    getattr(module, {kernel!r}),
                    signature={signature},
                    constexprs={constexprs}),
                    target=target, options={{"num_warps": {num_warps}}})
            except RuntimeError:
                pass
            else:
                raise AssertionError("an unsupported scan compiled")
        # The supported neighbour, in the same process.
        triton.compile(ASTSource(
            module._scan_envelope_kernel,
            signature={{"X": "*fp32", "O": "*fp32"}},
            constexprs={{"BLOCK": 128, "FORM": "plain"}}),
            target=target, options={{"num_warps": 1}})
        print("RECOVERED")
    """))
    result = subprocess.run([sys.executable, str(script)], capture_output=True,
                            text=True, timeout=180,
                            env={**os.environ,
                                 "TRITON_CACHE_DIR": str(tmp_path / "cache")})
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert output.count(needle) == 3, output
    assert "RECOVERED" in output
    assert "failed to legalize" not in output


@pytest.mark.parametrize("dtype", ["*i8", "*i16"], ids=["i8", "i16"])
def test_reduce_rank2_axis1_narrow_int_is_rejected_not_crashed(dtype,
                                                               tmp_path):
    # `tl.sum(x, axis=1)` over an i8 or i16 tile used to pass the validator and
    # decline INSIDE the conversion, which killed the process: SIGSEGV on every
    # run and occasionally SIGABRT. The validator listed the combines it covered
    # (i32 product, extrema, bitwise; f16/bf16) and `arith.addi` was not among
    # them, while the lowering declines every non-f32 computed cone. The
    # validator now asks the lowering's own question.
    import os
    import subprocess
    import sys
    import textwrap

    script = tmp_path / "reduce_narrow_int.py"
    script.write_text(textwrap.dedent(f"""
        import importlib.util
        import triton
        from triton.compiler import ASTSource
        from triton.backends.compiler import GPUTarget
        spec = importlib.util.spec_from_file_location("msl_child", {str(Path(__file__).resolve())!r})
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        target = GPUTarget("metal", 80, 32)

        for attempt in range(3):
            try:
                triton.compile(ASTSource(
                    module._reduce_rank2_axis1_narrow_int_kernel,
                    signature={{"X": {dtype!r}, "O": "*i32"}},
                    constexprs={{"M": 8, "N": 32}}),
                    target=target, options={{"num_warps": 1}})
            except RuntimeError:
                pass
            else:
                raise AssertionError("a narrow-int rank-2 axis=1 sum compiled")
        # The f32 neighbour is a computed cone too, and it is supported.
        triton.compile(ASTSource(
            module._reduce_rank2_axis1_narrow_int_kernel,
            signature={{"X": "*fp32", "O": "*fp32"}},
            constexprs={{"M": 8, "N": 32}}),
            target=target, options={{"num_warps": 1}})
        print("RECOVERED")
    """))
    result = subprocess.run([sys.executable, str(script)], capture_output=True,
                            text=True, timeout=180,
                            env={**os.environ,
                                 "TRITON_CACHE_DIR": str(tmp_path / "cache")})
    output = result.stdout + result.stderr
    needle = "rank-2 axis=1 reduce over a computed cone is implemented for f32 only"
    assert result.returncode == 0, output
    assert output.count(needle) == 3, output
    assert "RECOVERED" in output
    assert "failed to legalize" not in output


# The `tl.dot_scaled` matcher and the unsupported-op validator were probed as
# whole families (`.omc/fuzz/scaled_dot_and_unsupported.py`, 16 kernels):
# 10 named refusals, 6 supported shapes exact, no wrong answers and no signals.
# Four of the ten refusals already had fixtures. These five did not, and each is
# reachable from an ordinary kernel.
@pytest.mark.parametrize(
    "case,needle",
    [
        ("gather_e128", "rank-1 gather requires a power-of-two 1..64 elements"),
        ("splat_store", "an unmasked store through a uniform"),
        ("sd_unaligned_m", "requires static M/N multiples of 8"),
        ("sd_computed_a", "requires direct matrix loads"),
        ("sd_epilogue", "an epilogue between the dot and the store"),
    ],
    ids=["gather_e128", "splat_store", "sd_unaligned_m", "sd_computed_a",
         "sd_epilogue"],
)
def test_scaled_dot_and_unsupported_op_refusals(case, needle, tmp_path):
    import os
    import subprocess
    import sys
    import textwrap

    cases = {
        "gather_e128": ("_gather_envelope_kernel",
                        '{"X": "*fp32", "IDX": "*i32", "O": "*fp32"}',
                        '{"BLOCK": 4096}'),
        "splat_store": ("_splat_store_kernel",
                        '{"X": "*fp32", "O": "*fp32"}', '{"BLOCK": 64}'),
        "sd_unaligned_m": ("_dot_scaled_envelope_kernel",
                           '{"A": "*u8", "B": "*u8", "AS": "*u8", "BS": "*u8",'
                           ' "C": "*fp32"}',
                           '{"M": 4, "N": 16, "K": 32, "FORM": "plain"}'),
        "sd_computed_a": ("_dot_scaled_envelope_kernel",
                          '{"A": "*u8", "B": "*u8", "AS": "*u8", "BS": "*u8",'
                          ' "C": "*fp32"}',
                          '{"M": 16, "N": 16, "K": 32, "FORM": "computed_a"}'),
        "sd_epilogue": ("_dot_scaled_envelope_kernel",
                        '{"A": "*u8", "B": "*u8", "AS": "*u8", "BS": "*u8",'
                        ' "C": "*fp32"}',
                        '{"M": 16, "N": 16, "K": 32, "FORM": "epilogue"}'),
    }
    kernel, signature, constexprs = cases[case]

    script = tmp_path / "scaled_dot_refusal.py"
    script.write_text(textwrap.dedent(f"""
        import importlib.util
        import triton
        from triton.compiler import ASTSource
        from triton.backends.compiler import GPUTarget
        spec = importlib.util.spec_from_file_location("msl_child", {str(Path(__file__).resolve())!r})
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        target = GPUTarget("metal", 80, 32)

        for attempt in range(3):
            try:
                triton.compile(ASTSource(
                    getattr(module, {kernel!r}),
                    signature={signature},
                    constexprs={constexprs}),
                    target=target, options={{"num_warps": 1}})
            except RuntimeError:
                pass
            else:
                raise AssertionError("an unsupported shape compiled")
        # A supported scaled dot in the same process.
        triton.compile(ASTSource(
            module._dot_scaled_envelope_kernel,
            signature={{"A": "*u8", "B": "*u8", "AS": "*u8", "BS": "*u8",
                       "C": "*fp32"}},
            constexprs={{"M": 16, "N": 16, "K": 32, "FORM": "plain"}}),
            target=target, options={{"num_warps": 1}})
        print("RECOVERED")
    """))
    result = subprocess.run([sys.executable, str(script)], capture_output=True,
                            text=True, timeout=180,
                            env={**os.environ,
                                 "TRITON_CACHE_DIR": str(tmp_path / "cache")})
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert output.count(needle) == 3, output
    assert "RECOVERED" in output
    assert "failed to legalize" not in output


@triton.jit
def _affine_scan_combine(a, x, b, y):
    return a * b, b * x + y


@triton.jit
def _segmented_scan_combine(a, x, b, y):
    return tl.where(y, b, a + b), x | y


@triton.jit(do_not_specialize=["flag"])
def _canonical_tuple_scan_kernel(x_ptr, y_ptr, out_ptr, aux_ptr, flag,
                                 BLOCK: tl.constexpr, MODE: tl.constexpr,
                                 KIND: tl.constexpr, REVERSE: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    y = tl.load(y_ptr + offs)
    if MODE == "fma":
        z = tl.fma(x, x, 1.0)
    elif MODE == "rem":
        z = x % 3.0
    elif MODE == "div":
        z = tl.div_rn(x, 3.0)
    elif MODE == "tensor_if":
        if flag:
            z = x + 1.0
        else:
            z = x * 2.0
    else:
        z = x * 2.0 + 1.0
    if KIND == "affine":
        u, v = tl.associative_scan((z, y), 0, _affine_scan_combine,
                                   reverse=REVERSE)
        tl.store(out_ptr + offs, u)
        tl.store(aux_ptr + offs, v)
    else:
        u, v = tl.associative_scan((z, y > 0.0), 0, _segmented_scan_combine,
                                   reverse=REVERSE)
        tl.store(out_ptr + offs, u)
        tl.store(aux_ptr + offs, v.to(tl.float32))


# BLOCK 256 over four warps is two elements per thread, which the general
# tuple scan replays as well, so nothing can take it; the segmented form
# carries an i1 the general tuple scan has no slot for at any block size.
@pytest.mark.parametrize(
    "block,kind,mode",
    [(256, "affine", "fma"), (256, "affine", "tensor_if"),
     (128, "segmented", "fma"), (32, "segmented", "tensor_if")],
)
def test_canonical_tuple_scan_unreplayable_source_is_rejected_not_crashed(
        block, kind, mode, tmp_path):
    """The canonical affine and segmented scans replay both of their sources.

    `validateScanSupport` returns early for these two forms because they have
    their own lowerings, so nothing asked about the source, and the lowering's
    own question was asked from inside `applyFullConversion` -- the rollback
    site. A sweep of BLOCK x source x form measured 41 of 60 cells killing the
    process (29 SIGSEGV, 6 SIGTRAP, 6 SIGABRT) and named only 7.
    """
    import os
    import subprocess
    import sys
    import textwrap

    script = tmp_path / "canonical_tuple_scan_preflight.py"
    script.write_text(textwrap.dedent(f"""
        import importlib.util
        import triton
        from triton.compiler import ASTSource
        from triton.backends.compiler import GPUTarget
        spec = importlib.util.spec_from_file_location("msl_child", {str(Path(__file__).resolve())!r})
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        target = GPUTarget("metal", 9, 32)

        def compile_case(block, kind, mode):
            return triton.compile(ASTSource(
                module._canonical_tuple_scan_kernel,
                signature={{"x_ptr": "*fp32", "y_ptr": "*fp32",
                           "out_ptr": "*fp32", "aux_ptr": "*fp32",
                           "flag": "i32"}},
                constexprs={{"BLOCK": block, "MODE": mode, "KIND": kind,
                            "REVERSE": False}}),
                target=target, options={{"num_warps": 4}})

        for attempt in range(3):
            try:
                compile_case({block}, {kind!r}, {mode!r})
            except RuntimeError:
                pass
            else:
                raise AssertionError("unsupported tuple scan source compiled")
        compile_case(128, {kind!r}, "arithmetic")
        print("RECOVERED")
    """))
    result = subprocess.run([sys.executable, str(script)], capture_output=True,
                            text=True, timeout=180,
                            env={**os.environ,
                                 "TRITON_CACHE_DIR": str(tmp_path / "cache")})
    output = result.stdout + result.stderr
    needle = ("Metal backend: a canonical tuple scan replays each source at a "
              "logical position")
    assert result.returncode == 0, output
    assert output.count(needle) == 3, output
    assert "RECOVERED" in output
    assert "failed to legalize" not in output


def _affine_scan_reference(a, b, reverse):
    """The inclusive affine scan, one combine at a time, in float64."""
    import torch

    a = a.to(torch.float64)
    b = b.to(torch.float64)
    ua, ub = torch.empty_like(a), torch.empty_like(b)
    carry_a = carry_b = None
    for i in (range(a.numel() - 1, -1, -1) if reverse else range(a.numel())):
        if carry_a is None:
            carry_a, carry_b = a[i].clone(), b[i].clone()
        else:
            # combine(carry, element): the carry's range comes first, and the
            # combine is not commutative, so this order is the answer.
            carry_a, carry_b = carry_a * a[i], a[i] * carry_b + b[i]
        ua[i], ub[i] = carry_a, carry_b
    return ua, ub


@pytest.mark.parametrize("block", [32, 128])
@pytest.mark.parametrize("mode", ["fma", "rem", "div", "tensor_if"])
@pytest.mark.parametrize("reverse", [False, True])
def test_canonical_affine_scan_unreplayable_source_is_exact(block, mode,
                                                            reverse):
    """At one element per thread the general tuple scan takes these over.

    It reads the converted scalars instead of replaying the source, so a
    source the dedicated primitive cannot rebuild is computed rather than
    refused. The answer is what matters: turning a crash into a wrong answer
    would be worse than the crash.
    """
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("Metal backend requires an MPS-enabled PyTorch")
    flag = 1
    # The affine monoid MULTIPLIES the first slot across the whole tile, so
    # keep every source near zero: at BLOCK 128 a factor of 1.05 each is 1e2.
    scale = 0.3 if block == 32 else 0.02
    x = torch.linspace(-scale, scale, block, dtype=torch.float32)
    y = torch.linspace(-0.5, 0.5, block, dtype=torch.float32)
    if mode == "fma":
        z = x * x + 1.0
    elif mode == "rem":
        z = torch.fmod(x, 3.0)  # Triton's float `%` is fmod, not remainder
    elif mode == "div":
        z = x / 3.0
    else:
        z = x + 1.0 if flag else x * 2.0
    expected_u, expected_v = _affine_scan_reference(z, y, reverse)
    sentinel = -4321.0
    out = torch.full((block + 8,), sentinel, dtype=torch.float32, device="mps")
    aux = torch.full((block + 8,), sentinel, dtype=torch.float32, device="mps")
    _canonical_tuple_scan_kernel[(1,)](
        x.to("mps"), y.to("mps"), out, aux, flag, BLOCK=block, MODE=mode,
        KIND="affine", REVERSE=reverse, num_warps=4)
    torch.mps.synchronize()
    torch.testing.assert_close(out.cpu()[:block], expected_u.float(),
                               atol=1e-4, rtol=2e-3)
    torch.testing.assert_close(aux.cpu()[:block], expected_v.float(),
                               atol=1e-4, rtol=2e-3)
    assert torch.all(out.cpu()[block:] == sentinel)
    assert torch.all(aux.cpu()[block:] == sentinel)


@pytest.mark.parametrize(
    "mode,claimed", [("arithmetic", True), ("fma", False)])
def test_canonical_affine_scan_tier_census(mode, claimed):
    """Which tier ran is not visible in the answer; census the emitted body.

    Both tiers compute the same inclusive affine scan, so only the presence of
    the dedicated primitive separates them. A replayable source must keep it.
    """
    msl = _compile_to_msl(
        _canonical_tuple_scan_kernel,
        {"x_ptr": "*fp32", "y_ptr": "*fp32", "out_ptr": "*fp32",
         "aux_ptr": "*fp32", "flag": "i32"},
        {"BLOCK": 128, "MODE": mode, "KIND": "affine", "REVERSE": False},
        num_warps=4)
    assert ("metal.threadgroup_affine_prefix_scan" in msl) is claimed, msl[:400]


# ---------------------------------------------------------------------------
# Prefix negation over a binary expression.
#
# `metal.binary_exp` brackets its OPERANDS but not ITSELF: it emits
# `(lhs) OP (rhs)`. Every other context that embeds an expression supplies its
# own brackets -- a call's `(`, an index's `[`, the right-hand side of an
# assignment -- so the omission is invisible there. A prefix operator supplies
# none: `arith.negf` emits `(-X)`, and with X unbracketed the minus binds to
# the LEFT operand alone, so `-(a - b)` computes `-a - b`.
#
# `-(a * b)` and `-(a / b)` stay correct BY ACCIDENT, because negation commutes
# with multiplication. A probe built from products therefore proves nothing;
# the sum and difference forms are the only discriminating ones, and
# `tl.sigmoid(a - b)` reaches the same defect through the `exp(-x)` in its
# expansion.
# ---------------------------------------------------------------------------


@triton.jit
def _prefix_negation_kernel(x_ptr, y_ptr, out_ptr, BLOCK: tl.constexpr,
                            FORM: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    y = tl.load(y_ptr + offs)
    if FORM == "neg_sub":
        v = -(x - y)
    elif FORM == "neg_add":
        v = -(x + y)
    elif FORM == "neg_mul":
        v = -(x * y)
    elif FORM == "neg_div":
        v = -(x / y)
    elif FORM == "exp_neg_sub":
        v = tl.exp(-(x - y))
    else:
        v = tl.sigmoid(x - y)
    tl.store(out_ptr + offs, v)


def _compile_prefix_negation(form, block=128):
    return _compile_to_msl(
        _prefix_negation_kernel,
        {"x_ptr": "*fp32", "y_ptr": "*fp32", "out_ptr": "*fp32"},
        {"BLOCK": block, "FORM": form},
        num_warps=4)


# `(-` followed by a single bracketed term and then an infix +/- is the defect
# spelled out: the minus has taken the left operand and left the right one
# outside. A correct emission puts a `(` directly after the minus.
_MINUS_BINDS_LEFT = re.compile(r"\(-\([^()]*\)\s*[-+]\s")


@pytest.mark.parametrize("form", ["neg_sub", "neg_add", "exp_neg_sub",
                                  "sigmoid_sub"])
def test_prefix_negation_brackets_its_whole_operand(form):
    """Compile-only: the minus must not bind to the left operand alone."""
    msl = _compile_prefix_negation(form)
    hit = _MINUS_BINDS_LEFT.search(msl)
    assert hit is None, (
        f"prefix minus bound to the left operand only: {hit.group(0)!r}\n"
        f"--- MSL ---\n{msl}\n" if hit else "")


@pytest.mark.parametrize("block", [128, 256])
@pytest.mark.parametrize("form", ["neg_sub", "neg_add", "neg_mul", "neg_div",
                                  "exp_neg_sub", "sigmoid_sub"])
def test_prefix_negation_matches_torch(form, block):
    """Runtime: the four reachable forms, plus the two accidentally-correct
    controls that must not regress."""
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("requires an MPS-enabled PyTorch")
    torch.manual_seed(0x5EED + block)
    # Away from zero and from each other, so a dropped sign cannot coincide
    # with the right answer.
    x = torch.rand(block, dtype=torch.float32) + 1.0
    y = torch.rand(block, dtype=torch.float32) + 1.0
    out = torch.zeros(block, dtype=torch.float32, device="mps")
    _prefix_negation_kernel[(1,)](x.to("mps"), y.to("mps"), out,
                                  BLOCK=block, FORM=form, num_warps=4)
    torch.mps.synchronize()
    expected = {
        "neg_sub": lambda: -(x - y),
        "neg_add": lambda: -(x + y),
        "neg_mul": lambda: -(x * y),
        "neg_div": lambda: -(x / y),
        "exp_neg_sub": lambda: torch.exp(-(x - y)),
        "sigmoid_sub": lambda: torch.sigmoid(x - y),
    }[form]()
    torch.testing.assert_close(out.cpu(), expected, atol=1e-5, rtol=1e-5)
