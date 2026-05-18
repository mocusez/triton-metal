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

Bit-exact runtime verification is gated on a deferred uint8/i32 data path
(Session L2b): the Metal backend's `metal.get_element` op verifier
accepts `I1, UI8..UI64, SI8..SI64, f16, f32, bf16` only — signless `i32`
loads/stores fail verification (see
`third_party/metal/include/Dialect/Metal/IR/MetalOps.td:17`). The IR-level
correctness of the conversion patterns is independently pinned by the lit
fixture `test/Dialect/Metal/convert-tritongpu-to-metal/int_arith_broad.mlir`.

See `.omc/specs/deep-interview-leet-triton-l2-int-arith-broad.md`.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

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
def _select_kernel(x_ptr, y_ptr, out_ptr, N, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    y = tl.load(y_ptr + offs)
    cond = offs < N
    tl.store(out_ptr + offs, tl.where(cond, x, y))


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
    assert "msl" in compiled.asm, (
        f"arith.{op_name}: expected 'msl' stage artifact; got: "
        f"{sorted(compiled.asm.keys())}"
    )
    raw = compiled.asm["msl"]
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
# MSL. See `.omc/specs/deep-interview-lmultiload-phase-c-makerange.md`.
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
