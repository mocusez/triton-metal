"""Compile a vector_add Triton kernel just far enough to dump TTGIR.

Avoids importing torch (so the stub Metal env doesn't need it). Uses
`triton.jit(...).warmup(...)` which runs make_ttir + make_ttgir without
launching anything.

This script is the AC3 driver for
`.omc/specs/deep-interview-triton-dev-env-ttgir.md`. Run via:

    TRITON_BACKENDS_IN_TREE=1 TRITON_KERNEL_DUMP=1 \
    TRITON_DUMP_DIR=/tmp/triton-dump TRITON_ALWAYS_COMPILE=1 \
    pixi run python python/test/unit/tools/dump_vector_add_ttgir.py
"""

from __future__ import annotations

import sys

import triton
import triton.language as tl


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
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


def main() -> int:
    # The signature matters more than the actual pointer values; we use
    # warmup() so no kernel launch is attempted. `*fp32` declares a pointer
    # to fp32 input.
    BLOCK_SIZE = 1024
    n_elements = 98432

    # Use the lower-level triton.compile API so we never need real
    # tensors. The signature dict maps positional kernel args (by name)
    # to type strings: "*fp32" for pointers, "i32" for n_elements; the
    # constexpr BLOCK_SIZE lives in the `constants` dict.
    from triton.compiler import ASTSource, CompiledKernel
    from triton.runtime import driver

    signature = {
        "x_ptr": "*fp32",
        "y_ptr": "*fp32",
        "output_ptr": "*fp32",
        "n_elements": "i32",
        "BLOCK_SIZE": "constexpr",
    }
    constants = {"BLOCK_SIZE": BLOCK_SIZE}

    src = ASTSource(fn=add_kernel, signature=signature, constexprs=constants)
    target = driver.active.get_current_target()
    compiled = triton.compile(src, target=target, options={"num_warps": 4})
    print(f"compile OK; ttgir present: {'ttgir' in compiled.asm}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
