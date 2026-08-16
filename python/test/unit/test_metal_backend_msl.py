"""End-to-end: @triton.jit -> TTIR -> TTGIR -> MSL text.

Acceptance test for `.omc/specs/deep-interview-metal-jit-to-msl-text.md`
(AC.J1, AC.J2, AC.J3). Runs the Metal backend in-process and asserts
that the canonical unmasked vector_add lowers to MSL with the expected
substrings (same shape as the `vector_add_unmasked.mlir` lit fixture).

Skipped automatically when the metal backend's pybind module isn't
linked (e.g. a Linux build without the Metal plugin).
"""

from __future__ import annotations

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
    # `.omc/specs/deep-interview-metal-masked-loadstore.md` for the
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
    # `id.x`. See `.omc/specs/deep-interview-lmultiload-phase-c-
    # makerange.md`. The substring checks pin the new shape.
    for needle in (
        "kernel void",
        "device float",
        "thread_position_in_grid",
        "id.x",
        "(id.x - (tgid.x * 128))",
        "(tgid.x * 128) + (id.x - (tgid.x * 128))",
    ):
        assert needle in msl, f"MSL output missing required substring {needle!r}.\n--- MSL ---\n{msl}\n"


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
        "== 255",
        "/ 32",
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
        "/ 32",
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
            "== 127",
            "& 127",
            "/ 32",
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
        "/ 2",
        "& 15",
        "& 7",
        "/ 32",
    ):
        assert needle in msl, f"E2M1 scaled-dot MSL missing {needle!r}.\n--- MSL ---\n{msl}\n"

    # Prove that the per-operand packing flags reach scalar_dot lowering.  The
    # generated variable numbers are intentionally ignored: only the physical
    # coordinate divided by two distinguishes K packing from outer M/N packing.
    lhs_outer_address = bool(re.search(r"\(\(v\d+ / 16\) / 2\)", msl))
    rhs_outer_address = "% 16) / 2)" in msl
    k_packed_address = bool(re.search(r"\(v\d+ / 2\)", msl))
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
    "join": "tl.store(o_ptr + i, tl.sum(tl.join(v, v), axis=1))",
    "gather": "tl.store(o_ptr + i, tl.gather(v, i, axis=0))",
    "device_print": "tl.device_print('v', v)\n    tl.store(o_ptr + i, v)",
}


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
        "x = torch.rand(16, device='mps')\n"
        "o = torch.empty_like(x)\n"
        "k[(1,)](x, o, B=16)\n"
    )
    proc = subprocess.run([sys.executable, str(script)], capture_output=True,
                          text=True, timeout=300)
    assert proc.returncode > 0, (
        f"{name}: expected a clean non-zero exit, got returncode "
        f"{proc.returncode} (negative means the process died on a signal)\n"
        f"{proc.stderr[-2000:]}")
    assert "Metal backend:" in proc.stderr, (
        f"{name}: no backend diagnostic in stderr:\n{proc.stderr[-2000:]}")
