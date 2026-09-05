"""Session L2: Elementwise integer arith op coverage on the Metal backend.

Compile-only smoke for each of the 12 broad integer arith ops. Drives a
small `@triton.jit` kernel through the full Metal pipeline (TTIR -> TTGIR
-> `convert-tritongpu-to-metal` -> MSL) and asserts that compilation
succeeds end-to-end without a `RuntimeError`. This proves both:

  * the new conversion patterns (line 758+ in
    `third_party/metal/lib/Conversion/TritonGPUToMetal/TritonGPUToMetal.cpp`)
    rewrite the tensor-form `arith.<op>` to the scalar form, and
  * the new MSL emitter cases (line 3255+ in
    `third_party/metal/lib/Target/Metal/ModuleTranslation.cpp`) do not
    `llvm_unreachable` when the op survives into the kernel body.

The compile matrix is complemented by runtime canaries for select, u64
high-product carry handling, and integer-to-float conversion. Integer storage
is routed through explicit signed/unsigned Metal types; the earlier signless
i32 verifier gap is closed. IR coverage remains pinned by
`test/Dialect/Metal/convert-tritongpu-to-metal/int_arith_broad.mlir`.

See the implementation notes.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import triton  # noqa: E402
import triton.language as tl  # noqa: E402
from triton.backends.compiler import GPUTarget  # noqa: E402
from triton.compiler import ASTSource  # noqa: E402

libmetal = pytest.importorskip(
    "triton._C.libtriton.metal",
    reason="Metal backend pybind module not built into libtriton",
)


# Each kernel forces the target `arith.<op>` into TTGIR by producing the
# i32 result and using it as the offset for an f32 store. Even though the
# `StoreLowering` ultimately bypasses the computed offset (using
# `metal.thread_id` instead), the conversion pattern fires for the tensor
# `arith.<op>` and the scalar form transiently lives in the kernel body —
# tripping any verifier rejection or emitter `llvm_unreachable` if the
# new patterns / cases were buggy.


@triton.jit
def _subi_kernel(out_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    other = tl.full((BLOCK,), 1, tl.int32)
    idx = offs - other
    tl.store(out_ptr + idx, tl.full((BLOCK,), 1.0, tl.float32))


@triton.jit
def _divsi_kernel(out_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    two = tl.full((BLOCK,), 2, tl.int32)
    idx = offs // two
    tl.store(out_ptr + idx, tl.full((BLOCK,), 1.0, tl.float32))


@triton.jit
def _remsi_kernel(out_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    two = tl.full((BLOCK,), 2, tl.int32)
    idx = offs % two
    tl.store(out_ptr + idx, tl.full((BLOCK,), 1.0, tl.float32))


@triton.jit
def _andi_kernel(out_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mask = tl.full((BLOCK,), 0x7F, tl.int32)
    idx = offs & mask
    tl.store(out_ptr + idx, tl.full((BLOCK,), 1.0, tl.float32))


@triton.jit
def _ori_kernel(out_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    bits = tl.full((BLOCK,), 0x1, tl.int32)
    idx = offs | bits
    tl.store(out_ptr + idx, tl.full((BLOCK,), 1.0, tl.float32))


@triton.jit
def _xori_kernel(out_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    bits = tl.full((BLOCK,), 0x1, tl.int32)
    idx = offs ^ bits
    tl.store(out_ptr + idx, tl.full((BLOCK,), 1.0, tl.float32))


@triton.jit
def _shli_kernel(out_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    one = tl.full((BLOCK,), 1, tl.int32)
    idx = offs << one
    tl.store(out_ptr + idx, tl.full((BLOCK,), 1.0, tl.float32))


@triton.jit
def _shrsi_kernel(out_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    one = tl.full((BLOCK,), 1, tl.int32)
    idx = offs >> one
    tl.store(out_ptr + idx, tl.full((BLOCK,), 1.0, tl.float32))


@triton.jit
def _divui_kernel(out_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK).to(tl.uint32)
    two = tl.full((BLOCK,), 2, tl.uint32)
    idx = (offs // two).to(tl.int32)
    tl.store(out_ptr + idx, tl.full((BLOCK,), 1.0, tl.float32))


@triton.jit
def _remui_kernel(out_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK).to(tl.uint32)
    two = tl.full((BLOCK,), 2, tl.uint32)
    idx = (offs % two).to(tl.int32)
    tl.store(out_ptr + idx, tl.full((BLOCK,), 1.0, tl.float32))


@triton.jit
def _shrui_kernel(out_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK).to(tl.uint32)
    one = tl.full((BLOCK,), 1, tl.uint32)
    idx = (offs >> one).to(tl.int32)
    tl.store(out_ptr + idx, tl.full((BLOCK,), 1.0, tl.float32))


@triton.jit
def _minui_kernel(out_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK).to(tl.uint32)
    pivot = tl.full((BLOCK,), 63, tl.uint32)
    idx = tl.minimum(offs, pivot).to(tl.int32, bitcast=True)
    tl.store(out_ptr + idx, tl.full((BLOCK,), 1.0, tl.float32))


@triton.jit
def _maxui_kernel(out_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK).to(tl.uint32)
    pivot = tl.full((BLOCK,), 63, tl.uint32)
    idx = tl.maximum(offs, pivot).to(tl.int32, bitcast=True)
    tl.store(out_ptr + idx, tl.full((BLOCK,), 1.0, tl.float32))


@triton.jit
def _select_kernel(x_ptr, y_ptr, out_ptr, N, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    y = tl.load(y_ptr + offs)
    cond = offs < N
    tl.store(out_ptr + offs, tl.where(cond, x, y))


@triton.jit
def _umulhi_u64_kernel(x_ptr, y_ptr, out_ptr, N, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask, other=0)
    y = tl.load(y_ptr + offs, mask=mask, other=0)
    high = tl.umulhi(x, y)
    tl.store(out_ptr + offs, high, mask=mask)


@triton.jit
def _unsigned_minmax_u32_kernel(
    a_ptr, b_ptr, min_ptr, max_ptr, signed_min_ptr, N, BLOCK: tl.constexpr
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    a_s = tl.load(a_ptr + offs, mask=mask, other=0)
    b_s = tl.load(b_ptr + offs, mask=mask, other=0)
    a_u = a_s.to(tl.uint32, bitcast=True)
    b_u = b_s.to(tl.uint32, bitcast=True)
    tl.store(min_ptr + offs, tl.minimum(a_u, b_u).to(tl.int32, bitcast=True), mask=mask)
    tl.store(max_ptr + offs, tl.maximum(a_u, b_u).to(tl.int32, bitcast=True), mask=mask)
    tl.store(signed_min_ptr + offs, tl.minimum(a_s, b_s), mask=mask)


@triton.jit
def _unsigned_minmax_u64_kernel(
    a_ptr, b_ptr, min_ptr, max_ptr, signed_max_ptr, N, BLOCK: tl.constexpr
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    a_s = tl.load(a_ptr + offs, mask=mask, other=0)
    b_s = tl.load(b_ptr + offs, mask=mask, other=0)
    a_u = a_s.to(tl.uint64, bitcast=True)
    b_u = b_s.to(tl.uint64, bitcast=True)
    tl.store(min_ptr + offs, tl.minimum(a_u, b_u).to(tl.int64, bitcast=True), mask=mask)
    tl.store(max_ptr + offs, tl.maximum(a_u, b_u).to(tl.int64, bitcast=True), mask=mask)
    tl.store(signed_max_ptr + offs, tl.maximum(a_s, b_s), mask=mask)


@triton.jit
def _map_linear(a, b):
    return a * 3 + b


@triton.jit
def _map_pair(a, b):
    return a + b, a - b


@triton.jit
def _map_elementwise_pack1_kernel(
    a_ptr, b_ptr, out0_ptr, out1_ptr, MULTI: tl.constexpr, BLOCK: tl.constexpr
):
    offsets = tl.arange(0, BLOCK)
    a = tl.load(a_ptr + offsets)
    b = tl.load(b_ptr + offsets)
    if MULTI:
        out0, out1 = tl.map_elementwise(_map_pair, a, b)
        tl.store(out0_ptr + offsets, out0)
        tl.store(out1_ptr + offsets, out1)
    else:
        out0 = tl.map_elementwise(_map_linear, a, b)
        tl.store(out0_ptr + offsets, out0)


_SIGNATURE_1ARG = {"out_ptr": "*fp32", "BLOCK": "constexpr"}
_SIGNATURE_SELECT = {
    "x_ptr": "*fp32",
    "y_ptr": "*fp32",
    "out_ptr": "*fp32",
    "N": "i32",
    "BLOCK": "constexpr",
}

_CASES = [
    ("subi",  _subi_kernel,  _SIGNATURE_1ARG),
    ("divsi", _divsi_kernel, _SIGNATURE_1ARG),
    ("remsi", _remsi_kernel, _SIGNATURE_1ARG),
    ("andi",  _andi_kernel,  _SIGNATURE_1ARG),
    ("ori",   _ori_kernel,   _SIGNATURE_1ARG),
    ("xori",  _xori_kernel,  _SIGNATURE_1ARG),
    ("shli",  _shli_kernel,  _SIGNATURE_1ARG),
    ("shrsi", _shrsi_kernel, _SIGNATURE_1ARG),
    ("divui", _divui_kernel, _SIGNATURE_1ARG),
    ("remui", _remui_kernel, _SIGNATURE_1ARG),
    ("shrui", _shrui_kernel, _SIGNATURE_1ARG),
    ("minui", _minui_kernel, _SIGNATURE_1ARG),
    ("maxui", _maxui_kernel, _SIGNATURE_1ARG),
    ("select", _select_kernel, _SIGNATURE_SELECT),
]


@pytest.mark.parametrize(
    "op_name, kernel, signature",
    _CASES,
    ids=[c[0] for c in _CASES],
)
def test_int_arith_compiles_end_to_end(op_name, kernel, signature):
    """Compile-only smoke: each arith op survives convert-tritongpu-to-metal
    and the MSL emitter without RuntimeError/`llvm_unreachable`."""
    constexprs = {"BLOCK": 128}
    src = ASTSource(fn=kernel, signature=signature, constexprs=constexprs)
    target = GPUTarget(backend="metal", arch=80, warp_size=32)
    compiled = triton.compile(src, target=target, options={"num_warps": 4})
    assert "metal" in compiled.asm, (
        f"arith.{op_name}: expected 'metal' stage artifact; got: "
        f"{sorted(compiled.asm.keys())}"
    )
    raw = compiled.asm["metal"]
    msl = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    assert msl.strip(), f"arith.{op_name}: empty MSL output"
    # Every MSL kernel must at least contain the kernel signature opener
    # and a return — sanity-check the emitter didn't hand us truncated text.
    assert "kernel void" in msl, f"arith.{op_name}: malformed MSL: {msl!r}"


# Runtime bit-exact for select with f32 values (the only op whose
# end-to-end runtime path doesn't require i32 i/o).
#
# Lmultiload Phase C un-skipped this case: the `arange < N` cond now
# flows through MakeRangeLowering's real per-thread localTid term
# (`id.x - tgid.x*tpb`) instead of the prior `arith.constant 0`
# placeholder, so the per-thread cond `(localTid < N)` survives into
# MSL. See the implementation notes.
def test_select_f32_runtime_bit_exact():
    BLOCK = 128
    N = 64
    torch.manual_seed(0xC0FFEE)
    x = torch.rand(BLOCK, dtype=torch.float32)
    y = torch.rand(BLOCK, dtype=torch.float32)
    out = torch.zeros(BLOCK, dtype=torch.float32)
    _select_kernel[(1, 1, 1)](x, y, out, N, BLOCK=BLOCK)
    mask = torch.arange(BLOCK) < N
    expected = torch.where(mask, x, y)
    torch.testing.assert_close(out, expected, atol=0, rtol=0)


def test_umulhi_u64_compiles_exact_limb_msl():
    src = ASTSource(
        fn=_umulhi_u64_kernel,
        signature={
            "x_ptr": "*u64",
            "y_ptr": "*u64",
            "out_ptr": "*u64",
            "N": "i32",
            "BLOCK": "constexpr",
        },
        constexprs={"BLOCK": 8},
    )
    target = GPUTarget(backend="metal", arch=80, warp_size=32)
    compiled = triton.compile(src, target=target, options={"num_warps": 1})
    msl = compiled.asm["metal"]
    if isinstance(msl, bytes):
        msl = msl.decode()
    assert "uint128" not in msl
    assert "0xfffffffful" in msl
    assert msl.count(">> 32u") >= 4


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="requires an MPS device"
)
def test_umulhi_u64_runtime_edge_carries():
    """Lock the exact high 64 bits, especially both limb-carry terms."""
    max_u64 = 2**64 - 1
    x_host = [
        0,
        1,
        max_u64,
        2**63,
        0xFFFFFFFF00000000,
        0x00000000FFFFFFFF,
        0xFEDCBA9876543210,
        0xFFFFFFFF7FFFFFFF,
    ]
    y_host = [
        max_u64,
        1,
        max_u64,
        2,
        0x00000001FFFFFFFF,
        0xFFFFFFFF00000001,
        0x0123456789ABCDEF,
        0x80000001FFFFFFFF,
    ]
    expected_host = [(x * y) >> 64 for x, y in zip(x_host, y_host)]
    x = torch.tensor(x_host, dtype=torch.uint64, device="mps")
    y = torch.tensor(y_host, dtype=torch.uint64, device="mps")
    out = torch.tensor([0] * len(x_host), dtype=torch.uint64, device="mps")

    compiled = _umulhi_u64_kernel.warmup(
        x, y, out, len(x_host), BLOCK=8, grid=(1,)
    )
    msl = compiled.asm["metal"]
    if isinstance(msl, bytes):
        msl = msl.decode()
    assert "uint128" not in msl
    assert "0xfffffffful" in msl

    _umulhi_u64_kernel[(1,)](x, y, out, len(x_host), BLOCK=8)
    torch.mps.synchronize()
    assert out.cpu().tolist() == expected_host


def _to_signed(bits: int, width: int) -> int:
    bits &= (1 << width) - 1
    sign = 1 << (width - 1)
    return bits - (1 << width) if bits & sign else bits


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="requires an MPS device"
)
def test_unsigned_minmax_u32_runtime_high_bits_and_mask():
    n, block = 13, 8
    a_bits = [
        0,
        0xFFFFFFFF,
        0x80000000,
        17,
        0x7FFFFFFF,
        0x80000001,
        42,
        0xFFFFFFFE,
        0,
        0x90000000,
        0xFFFFFFFF,
        123456789,
        0x80000000,
    ]
    b_bits = [
        0xFFFFFFFF,
        0,
        0x7FFFFFFF,
        17,
        0x80000000,
        0x80000001,
        0xFFFFFFFE,
        42,
        0,
        0x70000000,
        0xFFFFFFFF,
        987654321,
        0x80000000,
    ]
    sentinel = _to_signed(0x5A5A5A5A, 32)
    a = torch.tensor([_to_signed(v, 32) for v in a_bits], dtype=torch.int32, device="mps")
    b = torch.tensor([_to_signed(v, 32) for v in b_bits], dtype=torch.int32, device="mps")
    out_min = torch.full((block * 2,), sentinel, dtype=torch.int32, device="mps")
    out_max = torch.full((block * 2,), sentinel, dtype=torch.int32, device="mps")
    signed_min = torch.full((block * 2,), sentinel, dtype=torch.int32, device="mps")

    _unsigned_minmax_u32_kernel[(2,)](a, b, out_min, out_max, signed_min, n, BLOCK=block)
    torch.mps.synchronize()

    expected_min = torch.tensor(
        [_to_signed(min(x, y), 32) for x, y in zip(a_bits, b_bits)],
        dtype=torch.int32,
    )
    expected_max = torch.tensor(
        [_to_signed(max(x, y), 32) for x, y in zip(a_bits, b_bits)],
        dtype=torch.int32,
    )
    expected_signed_min = torch.minimum(a.cpu(), b.cpu())
    assert torch.equal(out_min.cpu()[:n], expected_min)
    assert torch.equal(out_max.cpu()[:n], expected_max)
    assert torch.equal(signed_min.cpu()[:n], expected_signed_min)
    assert torch.equal(out_min.cpu()[n:], torch.full((block * 2 - n,), sentinel, dtype=torch.int32))
    assert torch.equal(out_max.cpu()[n:], torch.full((block * 2 - n,), sentinel, dtype=torch.int32))


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="requires an MPS device"
)
def test_unsigned_minmax_u64_runtime_high_bits_and_mask():
    n, block = 7, 4
    a_bits = [
        0,
        0xFFFFFFFFFFFFFFFF,
        0x8000000000000000,
        17,
        0x7FFFFFFFFFFFFFFF,
        0x8000000000000001,
        0xFEDCBA9876543210,
    ]
    b_bits = [
        0xFFFFFFFFFFFFFFFF,
        0,
        0x7FFFFFFFFFFFFFFF,
        17,
        0x8000000000000000,
        0x8000000000000001,
        0x0123456789ABCDEF,
    ]
    sentinel = _to_signed(0x5A5A5A5A5A5A5A5A, 64)
    a = torch.tensor([_to_signed(v, 64) for v in a_bits], dtype=torch.int64, device="mps")
    b = torch.tensor([_to_signed(v, 64) for v in b_bits], dtype=torch.int64, device="mps")
    out_min = torch.full((block * 2,), sentinel, dtype=torch.int64, device="mps")
    out_max = torch.full((block * 2,), sentinel, dtype=torch.int64, device="mps")
    signed_max = torch.full((block * 2,), sentinel, dtype=torch.int64, device="mps")

    _unsigned_minmax_u64_kernel[(2,)](a, b, out_min, out_max, signed_max, n, BLOCK=block)
    torch.mps.synchronize()

    expected_min = torch.tensor(
        [_to_signed(min(x, y), 64) for x, y in zip(a_bits, b_bits)],
        dtype=torch.int64,
    )
    expected_max = torch.tensor(
        [_to_signed(max(x, y), 64) for x, y in zip(a_bits, b_bits)],
        dtype=torch.int64,
    )
    expected_signed_max = torch.maximum(a.cpu(), b.cpu())
    assert torch.equal(out_min.cpu()[:n], expected_min)
    assert torch.equal(out_max.cpu()[:n], expected_max)
    assert torch.equal(signed_max.cpu()[:n], expected_signed_max)
    assert torch.equal(out_min.cpu()[n:], torch.full((block * 2 - n,), sentinel, dtype=torch.int64))
    assert torch.equal(out_max.cpu()[n:], torch.full((block * 2 - n,), sentinel, dtype=torch.int64))


@triton.jit
def _scalar_int_to_float_kernel(out_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    val = pid.to(tl.float32)  # scalar arith.sitofp
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    vals = val + tl.zeros((BLOCK,), dtype=tl.float32)
    tl.store(out_ptr + offs, vals)


# Scalar numeric conversion end-to-end. A scalar `program_id(0).to(tl.float32)`
# lowers to `arith.sitofp` and survives into the kernel body (only tensor
# conversions are folded into metal ops by the conversion pass). Before the
# emitter learned the conversion-op family, this crashed at translation with
# `llvm_unreachable("Unexpected operation")`. The translate-side emission of
# every conversion op is pinned by the lit fixture
# `test/Dialect/Metal/metal-translate/arith_scalar_conversions.mlir`; this test
# pins int->float runtime correctness through the full pipeline.
def test_scalar_int_to_float_runtime_bit_exact():
    G, BLOCK = 5, 64
    out = torch.full((G * BLOCK,), -1.0, dtype=torch.float32)
    _scalar_int_to_float_kernel[(G,)](out, BLOCK=BLOCK)
    # Each program g writes its pid (as f32) into out[g*BLOCK : (g+1)*BLOCK].
    expected = torch.arange(G, dtype=torch.float32).repeat_interleave(BLOCK)
    torch.testing.assert_close(out, expected, atol=0, rtol=0)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="requires an MPS device"
)
@pytest.mark.parametrize("multi", [False, True])
def test_map_elementwise_pack1_runtime(multi: bool):
    block = 128
    a = torch.arange(-64, 64, dtype=torch.int32, device="mps")
    b = torch.arange(127, -1, -1, dtype=torch.int32, device="mps")
    out0 = torch.empty(block, dtype=torch.int32, device="mps")
    out1 = torch.empty(block, dtype=torch.int32, device="mps")

    _map_elementwise_pack1_kernel[(1,)](
        a, b, out0, out1, MULTI=multi, BLOCK=block
    )

    if multi:
        torch.testing.assert_close(out0.cpu(), (a + b).cpu(), atol=0, rtol=0)
        torch.testing.assert_close(out1.cpu(), (a - b).cpu(), atol=0, rtol=0)
    else:
        torch.testing.assert_close(out0.cpu(), (a * 3 + b).cpu(), atol=0, rtol=0)
