"""End-to-end: @triton.jit with a runtime i32 scalar arg -> MSL text.

Acceptance test for `.omc/specs/deep-interview-metal-dynamic-scalar-args.md`
(AC.D4). Compiles the canonical masked vector_add (with `n_elements: i32`
as a true runtime arg, not constexpr) through the Metal backend in-process
and asserts that the resulting MSL has the scalar-arg wrapper in its
kernel signature plus the masked-load `if` shape.

Skipped automatically when the metal backend's pybind module isn't built.
"""

from __future__ import annotations

import pytest

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource


pytest.importorskip(
    "triton._C.libtriton.metal",
    reason="Metal backend pybind module not built into libtriton",
)


@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)


def test_dynamic_scalar_n_compiles_to_msl():
    signature = {
        "x_ptr": "*fp32",
        "y_ptr": "*fp32",
        "output_ptr": "*fp32",
        "n_elements": "i32",
        "BLOCK_SIZE": "constexpr",
    }
    src = ASTSource(
        fn=add_kernel,
        signature=signature,
        constexprs={"BLOCK_SIZE": 128},
    )
    target = GPUTarget(backend="metal", arch=80, warp_size=32)
    compiled = triton.compile(src, target=target, options={"num_warps": 4})

    raw = compiled.asm["metal"]
    msl = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    assert msl, "MSL output is empty"

    # Signature: three float buffer args + the wrapped i32 scalar arg.
    # `device float` for the pointer args; `device uint` (the wrapper
    # transport type for signless i32) for n_elements.
    for needle in (
        "kernel void add_kernel",
        "device float",
        "device uint",
        "thread_position_in_grid",
    ):
        assert needle in msl, (
            f"MSL output missing required substring {needle!r}.\n"
            f"--- MSL ---\n{msl}\n"
        )

    # Masked-load shape: a comparison against id.x, a guard `if (...)`
    # block, and a per-thread store. Post-Lmultiload-Phase-C the 1D
    # canonical short-circuit is gone, so the store index is the
    # arithmetic-explicit `pid*BLOCK + (id.x - pid*tpb)` form rather
    # than the collapsed `id.x`. See `.omc/specs/deep-interview-
    # lmultiload-phase-c-makerange.md`.
    assert "id.x <" in msl, f"MSL missing per-thread cmp.\n--- MSL ---\n{msl}\n"
    assert "if (" in msl, f"MSL missing guard if-block.\n--- MSL ---\n{msl}\n"
    assert "(id.x - (tgid.x * 128))" in msl, (
        f"MSL missing per-thread localTid term.\n--- MSL ---\n{msl}\n"
    )
    assert "(tgid.x * 128) + (id.x - (tgid.x * 128))" in msl, (
        f"MSL missing pid*BLOCK + localTid offset.\n--- MSL ---\n{msl}\n"
    )
