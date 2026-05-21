"""End-to-end: @triton.jit -> TTIR -> TTGIR -> MSL text.

Acceptance test for `.omc/specs/deep-interview-metal-jit-to-msl-text.md`
(AC.J1, AC.J2, AC.J3). Runs the Metal backend in-process and asserts
that the canonical unmasked vector_add lowers to MSL with the expected
substrings (same shape as the `vector_add_unmasked.mlir` lit fixture).

Skipped automatically when the metal backend's pybind module isn't
linked (e.g. a Linux build without the Metal plugin).
"""

from __future__ import annotations

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
        assert needle in msl, (
            f"MSL output missing required substring {needle!r}.\n"
            f"--- MSL ---\n{msl}\n"
        )
