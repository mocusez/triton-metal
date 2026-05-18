"""End-to-end: @triton.jit with BLOCK_SIZE > threads_per_block -> MSL text.

Acceptance test for `.omc/specs/deep-interview-metal-block-size-loop.md`
(AC.B4). Compiles the canonical Triton vector_add with BLOCK_SIZE=1024
(elem_per_thread=8 on 128 threads) and asserts that the resulting MSL
contains a per-thread loop, a per-iteration mask check, and the strided
index expression (`id.x + (iv * 128)`) that matches Triton's default
sizePerThread=[1] layout.

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


def test_block_size_larger_than_threads_compiles_to_msl():
    # BLOCK_SIZE=1024 with num_warps=4*warp_size=32 = 128 threads per block.
    # elem_per_thread = 1024 / 128 = 8. Triton picks sizePerThread=[1]
    # by default for this shape (strided per-thread layout).
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
        constexprs={"BLOCK_SIZE": 1024},
    )
    target = GPUTarget(backend="metal", arch=80, warp_size=32)
    compiled = triton.compile(src, target=target, options={"num_warps": 4})

    raw = compiled.asm["msl"]
    msl = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    assert msl, "MSL output is empty"

    for needle in (
        "kernel void add_kernel",
        "device float",
        "device uint32_t",
        "thread_position_in_grid",
        # Tile loop: 8 iterations covering 1024 elements over 128 threads.
        "< 8;",
        # Strided per-thread index (sizePerThread=[1]); the layout
        # repeats 8 times across the tensor extent.
        "(id.x + (",
        "* 128))",
        # Mask check against n_elements (wrapped scalar arg).
        "< v",
    ):
        assert needle in msl, (
            f"MSL output missing required substring {needle!r}.\n"
            f"--- MSL ---\n{msl}\n"
        )
