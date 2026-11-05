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
def _map_compare(a, b):
    if a < b:
        return -1
    elif a == b:
        return 0
    else:
        return 1


@triton.jit
def _map_early_pair(a, b):
    if a < 0:
        return b - a, b + a
    return a + b, a - b


@triton.jit
def _map_scalar_loops(a, b):
    sum_value = a
    for _ in range(4):
        sum_value = sum_value * 2 + b
    product = b
    i = 0
    while i < 3:
        product = product + a - i
        i += 1
    return sum_value, product


@triton.jit
def _map_pack2_cross(a0, a1, b0, b1):
    return a0 * b1 + a1, b0 - a0, a1 + b0, b1 - a1


@triton.jit
def _map_pack2_cfg(a0, a1, b0, b1):
    if a0 * b1 < a1 + b0:
        return a1 - b0, a0 + b1, b1 - a0, b0 - a1
    return a0 + b1, a1 - b0, b0 - a1, b1 - a0


@triton.jit
def _map_pack4_cross(a0, a1, a2, a3, b0, b1, b2, b3):
    return (
        a0 * b3 + a1,
        a1 * b2 - a2,
        a2 + b1,
        a3 - b0,
    )


@triton.jit
def _map_elementwise_pack_kernel(
    a_ptr, b_ptr, out0_ptr, out1_ptr, PACK: tl.constexpr, BLOCK: tl.constexpr
):
    offsets = tl.arange(0, BLOCK)
    a = tl.load(a_ptr + offsets)
    b = tl.load(b_ptr + offsets)
    if PACK == 2:
        out0, out1 = tl.map_elementwise(_map_pack2_cross, a, b, pack=2)
        tl.store(out1_ptr + offsets, out1)
    else:
        tl.static_assert(PACK == 4)
        out0, = tl.map_elementwise(_map_pack4_cross, a, b, pack=4)
    tl.store(out0_ptr + offsets, out0)


@triton.jit
def _map_elementwise_pack_cfg_kernel(
    a_ptr, b_ptr, out0_ptr, out1_ptr, BLOCK: tl.constexpr
):
    offsets = tl.arange(0, BLOCK)
    a = tl.load(a_ptr + offsets)
    b = tl.load(b_ptr + offsets)
    out0, out1 = tl.map_elementwise(_map_pack2_cfg, a, b, pack=2)
    tl.store(out0_ptr + offsets, out0)
    tl.store(out1_ptr + offsets, out1)


@triton.jit
def _map_elementwise_two_pack_kernel(
    a_ptr, b_ptr, out0_ptr, out1_ptr, BLOCK: tl.constexpr
):
    offsets = tl.arange(0, BLOCK)
    a = tl.load(a_ptr + offsets)
    b = tl.load(b_ptr + offsets)
    x0, x1 = tl.map_elementwise(_map_pack2_cross, a, b, pack=2)
    out0, out1 = tl.map_elementwise(_map_pack2_cfg, x0, x1, pack=2)
    tl.store(out0_ptr + offsets, out0)
    tl.store(out1_ptr + offsets, out1)


@triton.jit
def _map_elementwise_pack_rank2_kernel(
    a_ptr,
    b_ptr,
    out0_ptr,
    out1_ptr,
    M,
    N,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = tl.arange(0, BM)[:, None]
    cols = tl.arange(0, BN)[None, :]
    offsets = pid * BM * BN + rows * BN + cols
    mask = (rows < M) & (cols < N)
    a = tl.load(a_ptr + offsets, mask=mask, other=1)
    b = tl.load(b_ptr + offsets, mask=mask, other=2)
    out0, out1 = tl.map_elementwise(_map_pack2_cross, a, b, pack=2)
    tl.store(out0_ptr + offsets, out0, mask=mask)
    tl.store(out1_ptr + offsets, out1, mask=mask)


@triton.jit
def _map_elementwise_pack_rank3_kernel(out_ptr):
    a = tl.full((2, 2, 32), 3, tl.int32)
    b = tl.full((2, 2, 32), 5, tl.int32)
    out0, out1 = tl.map_elementwise(_map_pack2_cross, a, b, pack=2)
    offsets = tl.arange(0, 128).reshape(2, 2, 32)
    tl.store(out_ptr + offsets, out0)


@triton.jit
def _map_elementwise_pack_loop_kernel(a_ptr, b_ptr, out_ptr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    a = tl.load(a_ptr + offsets)
    b = tl.load(b_ptr + offsets)
    acc = tl.zeros((BLOCK,), tl.int32)
    for _ in range(2):
        out0, out1 = tl.map_elementwise(_map_pack2_cross, a, b, pack=2)
        acc += out0
    tl.store(out_ptr + offsets, acc)


@triton.jit
def _map_elementwise_pack_while_kernel(a_ptr, b_ptr, out_ptr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    a = tl.load(a_ptr + offsets)
    b = tl.load(b_ptr + offsets)
    acc = tl.zeros((BLOCK,), tl.int32)
    i = 0
    while i < 2:
        out0, out1 = tl.map_elementwise(_map_pack2_cross, a, b, pack=2)
        acc += out0
        i += 1
    tl.store(out_ptr + offsets, acc)


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


@triton.jit
def _map_elementwise_cfg_kernel(
    a_ptr, b_ptr, out0_ptr, out1_ptr, MODE: tl.constexpr, BLOCK: tl.constexpr
):
    offsets = tl.arange(0, BLOCK)
    a = tl.load(a_ptr + offsets)
    b = tl.load(b_ptr + offsets)
    if MODE == "compare":
        out0 = tl.map_elementwise(_map_compare, a, b)
        tl.store(out0_ptr + offsets, out0)
    elif MODE == "early":
        out0, out1 = tl.map_elementwise(_map_early_pair, a, b)
        tl.store(out0_ptr + offsets, out0)
        tl.store(out1_ptr + offsets, out1)
    else:
        tl.static_assert(MODE == "loops")
        out0, out1 = tl.map_elementwise(_map_scalar_loops, a, b)
        tl.store(out0_ptr + offsets, out0)
        tl.store(out1_ptr + offsets, out1)


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


@pytest.mark.parametrize("mode", ["compare", "early", "loops"])
@pytest.mark.parametrize("num_warps", [1, 4])
def test_map_elementwise_pack1_cfg_runtime(mode, num_warps):
    block = 128
    a = torch.arange(-64, 64, dtype=torch.int32, device="mps")
    b = torch.arange(127, -1, -1, dtype=torch.int32, device="mps")
    out0 = torch.empty(block, dtype=torch.int32, device="mps")
    out1 = torch.empty(block, dtype=torch.int32, device="mps")

    _map_elementwise_cfg_kernel[(1,)](
        a, b, out0, out1, MODE=mode, BLOCK=block, num_warps=num_warps
    )

    a_cpu = a.cpu()
    b_cpu = b.cpu()
    if mode == "compare":
        expected0 = (a_cpu > b_cpu).to(torch.int32) - (b_cpu > a_cpu).to(torch.int32)
        torch.testing.assert_close(out0.cpu(), expected0, atol=0, rtol=0)
    elif mode == "early":
        negative = a_cpu < 0
        expected0 = torch.where(negative, b_cpu - a_cpu, a_cpu + b_cpu)
        expected1 = torch.where(negative, b_cpu + a_cpu, a_cpu - b_cpu)
        torch.testing.assert_close(out0.cpu(), expected0, atol=0, rtol=0)
        torch.testing.assert_close(out1.cpu(), expected1, atol=0, rtol=0)
    else:
        expected0 = a_cpu * 16 + b_cpu * 15
        expected1 = b_cpu + a_cpu * 3 - 3
        torch.testing.assert_close(out0.cpu(), expected0, atol=0, rtol=0)
        torch.testing.assert_close(out1.cpu(), expected1, atol=0, rtol=0)


@pytest.mark.parametrize("pack", [2, 4])
@pytest.mark.parametrize("block,num_warps", [(128, 1), (512, 4)])
def test_map_elementwise_pack_cross_runtime(pack, block, num_warps):
    a = torch.arange(block, dtype=torch.int32, device="mps") - block // 2
    b = torch.arange(block, dtype=torch.int32, device="mps") % 17 - 8
    out0 = torch.full((block,), -1, dtype=torch.int32, device="mps")
    out1 = torch.full((block,), -1, dtype=torch.int32, device="mps")

    _map_elementwise_pack_kernel[(1,)](
        a, b, out0, out1, PACK=pack, BLOCK=block, num_warps=num_warps
    )

    a_cpu = a.cpu()
    b_cpu = b.cpu()
    if pack == 2:
        a0, a1 = a_cpu.reshape(-1, 2).unbind(dim=1)
        b0, b1 = b_cpu.reshape(-1, 2).unbind(dim=1)
        expected0 = torch.stack((a0 * b1 + a1, b0 - a0), dim=1).reshape(-1)
        expected1 = torch.stack((a1 + b0, b1 - a1), dim=1).reshape(-1)
        torch.testing.assert_close(out1.cpu(), expected1, atol=0, rtol=0)
    else:
        a0, a1, a2, a3 = a_cpu.reshape(-1, 4).unbind(dim=1)
        b0, b1, b2, b3 = b_cpu.reshape(-1, 4).unbind(dim=1)
        expected0 = torch.stack(
            (a0 * b3 + a1, a1 * b2 - a2, a2 + b1, a3 - b0), dim=1
        ).reshape(-1)
    torch.testing.assert_close(out0.cpu(), expected0, atol=0, rtol=0)


@pytest.mark.parametrize("block,num_warps", [(128, 1), (512, 4)])
def test_map_elementwise_pack_cfg_runtime(block, num_warps):
    a = torch.arange(block, dtype=torch.int32, device="mps") - block // 2
    b = torch.arange(block, dtype=torch.int32, device="mps") % 17 - 8
    out0 = torch.empty(block, dtype=torch.int32, device="mps")
    out1 = torch.empty(block, dtype=torch.int32, device="mps")

    _map_elementwise_pack_cfg_kernel[(1,)](
        a, b, out0, out1, BLOCK=block, num_warps=num_warps
    )

    a0, a1 = a.cpu().reshape(-1, 2).unbind(dim=1)
    b0, b1 = b.cpu().reshape(-1, 2).unbind(dim=1)
    cond = a0 * b1 < a1 + b0
    first0 = torch.where(cond, a1 - b0, a0 + b1)
    second0 = torch.where(cond, a0 + b1, a1 - b0)
    first1 = torch.where(cond, b1 - a0, b0 - a1)
    second1 = torch.where(cond, b0 - a1, b1 - a0)
    expected0 = torch.stack((first0, second0), dim=1).reshape(-1)
    expected1 = torch.stack((first1, second1), dim=1).reshape(-1)
    torch.testing.assert_close(out0.cpu(), expected0, atol=0, rtol=0)
    torch.testing.assert_close(out1.cpu(), expected1, atol=0, rtol=0)


@pytest.mark.parametrize("block,num_warps", [(128, 1), (512, 4)])
def test_map_elementwise_two_pack_runtime(block, num_warps):
    a = torch.arange(block, dtype=torch.int32, device="mps") - block // 2
    b = torch.arange(block, dtype=torch.int32, device="mps") % 17 - 8
    out0 = torch.empty(block, dtype=torch.int32, device="mps")
    out1 = torch.empty(block, dtype=torch.int32, device="mps")

    _map_elementwise_two_pack_kernel[(1,)](
        a, b, out0, out1, BLOCK=block, num_warps=num_warps
    )

    a0, a1 = a.cpu().reshape(-1, 2).unbind(dim=1)
    b0, b1 = b.cpu().reshape(-1, 2).unbind(dim=1)
    x00 = a0 * b1 + a1
    x01 = b0 - a0
    x10 = a1 + b0
    x11 = b1 - a1
    cond = x00 * x11 < x01 + x10
    expected0 = torch.where(cond, x01 - x10, x00 + x11)
    expected1 = torch.where(cond, x00 + x11, x01 - x10)
    other0 = torch.where(cond, x11 - x00, x10 - x01)
    other1 = torch.where(cond, x10 - x01, x11 - x00)
    torch.testing.assert_close(
        out0.cpu(), torch.stack((expected0, expected1), dim=1).reshape(-1), atol=0, rtol=0
    )
    torch.testing.assert_close(
        out1.cpu(), torch.stack((other0, other1), dim=1).reshape(-1), atol=0, rtol=0
    )


def test_map_elementwise_pack_rank2_multi_program_masked():
    programs, m, n, bm, bn = 3, 3, 30, 4, 32
    storage = programs * bm * bn + 16
    a = torch.arange(storage, dtype=torch.int32, device="mps") - 37
    b = torch.arange(storage, dtype=torch.int32, device="mps") % 19 - 9
    out0 = torch.full((storage,), -101, dtype=torch.int32, device="mps")
    out1 = torch.full((storage,), -101, dtype=torch.int32, device="mps")

    _map_elementwise_pack_rank2_kernel[(programs,)](
        a, b, out0, out1, m, n, BM=bm, BN=bn, num_warps=1
    )

    rows = torch.arange(bm)[None, :, None]
    cols = torch.arange(bn)[None, None, :]
    mask = (rows < m) & (cols < n)
    a0, a1 = a.cpu().reshape(-1, 2).unbind(dim=1)
    b0, b1 = b.cpu().reshape(-1, 2).unbind(dim=1)
    expected0 = torch.stack((a0 * b1 + a1, b0 - a0), dim=1).reshape(-1)
    expected1 = torch.stack((a1 + b0, b1 - a1), dim=1).reshape(-1)
    valid = mask.expand(programs, bm, bn).reshape(-1)
    valid = torch.cat((valid, torch.zeros(16, dtype=torch.bool)))
    sentinel = torch.full_like(expected0, -101)
    torch.testing.assert_close(
        out0.cpu(), torch.where(valid, expected0, sentinel), atol=0, rtol=0
    )
    torch.testing.assert_close(
        out1.cpu(), torch.where(valid, expected1, sentinel), atol=0, rtol=0
    )


def _check_map_pack1_recovery():
    block = 128
    a = torch.arange(block, dtype=torch.int32, device="mps")
    b = torch.arange(block, dtype=torch.int32, device="mps") * 2
    out0 = torch.empty(block, dtype=torch.int32, device="mps")
    out1 = torch.empty(block, dtype=torch.int32, device="mps")
    _map_elementwise_pack1_kernel[(1,)](
        a, b, out0, out1, MULTI=False, BLOCK=block
    )
    torch.testing.assert_close(out0.cpu(), (a * 3 + b).cpu(), atol=0, rtol=0)


def test_map_elementwise_pack_rejects_indivisible_elements(capfd):
    block = 128
    a = torch.ones(block, dtype=torch.int32, device="mps")
    b = torch.ones(block, dtype=torch.int32, device="mps")
    out0 = torch.empty_like(a)
    out1 = torch.empty_like(a)
    with pytest.raises(RuntimeError, match="convert-tritongpu-to-metal failed"):
        _map_elementwise_pack_kernel[(1,)](
            a, b, out0, out1, PACK=2, BLOCK=block, num_warps=4
        )
    assert "must divide the 1 unique elements per thread" in capfd.readouterr().err
    _check_map_pack1_recovery()


def test_map_elementwise_pack_rejects_rank3(capfd):
    out = torch.empty(128, dtype=torch.int32, device="mps")
    with pytest.raises(RuntimeError, match="convert-tritongpu-to-metal failed"):
        _map_elementwise_pack_rank3_kernel[(1,)](out)
    assert "requires a static rank-1/rank-2 blocked tensor" in capfd.readouterr().err
    _check_map_pack1_recovery()


def test_map_elementwise_pack_rejects_loop(capfd):
    block = 128
    a = torch.ones(block, dtype=torch.int32, device="mps")
    b = torch.ones(block, dtype=torch.int32, device="mps")
    out = torch.empty_like(a)
    with pytest.raises(RuntimeError, match="convert-tritongpu-to-metal failed"):
        _map_elementwise_pack_loop_kernel[(1,)](a, b, out, BLOCK=block)
    assert "inside a loop, nested scan, or conditional control flow" in capfd.readouterr().err
    _check_map_pack1_recovery()


def test_map_elementwise_pack_rejects_while(capfd):
    block = 128
    a = torch.ones(block, dtype=torch.int32, device="mps")
    b = torch.ones(block, dtype=torch.int32, device="mps")
    out = torch.empty_like(a)
    with pytest.raises(RuntimeError, match="convert-tritongpu-to-metal failed"):
        _map_elementwise_pack_while_kernel[(1,)](a, b, out, BLOCK=block)
    assert "inside a loop, nested scan, or conditional control flow" in capfd.readouterr().err
    _check_map_pack1_recovery()


def test_map_elementwise_pack_rejects_scratch_overflow(capfd):
    block = 16384
    a = torch.ones(block, dtype=torch.int32, device="mps")
    b = torch.ones(block, dtype=torch.int32, device="mps")
    out0 = torch.empty_like(a)
    out1 = torch.empty_like(a)
    with pytest.raises(RuntimeError, match="convert-tritongpu-to-metal failed"):
        _map_elementwise_pack_kernel[(1,)](
            a, b, out0, out1, PACK=4, BLOCK=block, num_warps=1
        )
    assert "32768-byte budget" in capfd.readouterr().err
    _check_map_pack1_recovery()


@pytest.mark.parametrize("N, R", [(1, 0), (31, 1), (1024, 2), (1025, 3), (2053, 10), (65539, 20)])
@pytest.mark.parametrize("dtype", [torch.int32, torch.uint32])
def test_rainbow_table_original_uint32_hash(N, R, dtype):
    """The original 1024-wide FNV loop must preserve all 32 bits."""
    import importlib.util
    from pathlib import Path

    import numpy as np

    if not torch.backends.mps.is_available():
        pytest.skip("Metal backend requires MPS")
    path = Path(__file__).parent / "fixtures" / "metal_leet" / "easy-rainbow_table.py"
    spec = importlib.util.spec_from_file_location("leet_rainbow_table", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    values = np.random.default_rng(123).integers(0, 2**32, size=N, dtype=np.uint32)
    edges = np.array([0, 1, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF], dtype=np.uint32)
    values[:min(N, len(edges))] = edges[:N]
    expected = values.copy()
    for _ in range(R):
        hashed = np.full(N, 2166136261, dtype=np.uint32)
        for byte_pos in range(4):
            hashed = (hashed ^ ((expected >> (8 * byte_pos)) & 255)) * np.uint32(16777619)
        expected = hashed
    storage = values.view(np.int32) if dtype == torch.int32 else values
    inp = torch.from_numpy(storage.copy()).to("mps")
    # A tail canary also catches stores from masked lanes of the last block.
    output = torch.full((N + 7,), 12345, dtype=dtype, device="mps")
    module.solve(inp, output, N, R)
    torch.mps.synchronize()
    result = output.cpu().numpy().view(np.uint32)
    np.testing.assert_array_equal(result[:N], expected)
    np.testing.assert_array_equal(result[N:], 12345)
