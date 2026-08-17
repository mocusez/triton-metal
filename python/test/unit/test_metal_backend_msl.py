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
    # `tt.join` itself is implemented now (see the shuffle tests below); an
    # out-of-envelope join is still a rejection, and still has to be a clean
    # one. num_warps=1 puts 32 threads under a 128-element joined tile.
    "join_out_of_envelope": (
        "j = tl.join(tl.load(x_ptr + tl.arange(0, 64)),\n"
        "                tl.load(x_ptr + tl.arange(0, 64)))\n"
        "    tl.store(o_ptr + tl.arange(0, 64)[:, None] * 2\n"
        "             + tl.arange(0, 2)[None, :], j)"),
    "device_print": "tl.device_print('v', v)\n    tl.store(o_ptr + i, v)",
}
_UNSUPPORTED_NUM_WARPS = {"join_out_of_envelope": 1}


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


@pytest.mark.parametrize("num_warps", [1, 4])
@pytest.mark.parametrize("N", [2, 4, 8, 16, 32])
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


@pytest.mark.parametrize("num_warps", [1, 4])
@pytest.mark.parametrize("axis", [0, 1])
def test_cumsum_rank2_both_axes(axis, num_warps):
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("Metal backend requires an MPS-enabled PyTorch")
    M, N = 4, 8
    torch.manual_seed(axis * 10 + num_warps)
    x = torch.randn(M, N, dtype=torch.float32)
    out = torch.zeros(M, N, dtype=torch.float32, device="mps")
    _cumsum_rank2_kernel[(1,)](x.to("mps"), out, M, N, axis,
                               num_warps=num_warps)
    torch.mps.synchronize()
    torch.testing.assert_close(out.cpu(), x.cumsum(axis), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("axis", [0, 1])
def test_cumprod_rank2_is_bit_exact(axis):
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("Metal backend requires an MPS-enabled PyTorch")
    M, N = 4, 8
    torch.manual_seed(axis)
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
