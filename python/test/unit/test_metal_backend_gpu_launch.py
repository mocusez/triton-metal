"""End-to-end: @triton.jit -> .metallib -> Metal GPU -> assert x+y.

Acceptance test for `.omc/specs/deep-interview-metal-gpu-launch.md`
(AC.G3-G4). Skipped automatically on non-Darwin or when the Metal
runtime callables aren't compiled into libtriton.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource


libmetal = pytest.importorskip(
    "triton._C.libtriton.metal",
    reason="Metal backend pybind module not built into libtriton",
)

if not hasattr(libmetal, "launch_kernel"):
    pytest.skip(
        "Metal runtime not compiled (non-Darwin build or Xcode CLT absent)",
        allow_module_level=True,
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


def test_metal_gpu_launch_vector_add():
    BLOCK_SIZE = 128
    N = 100  # < BLOCK_SIZE -> exercises the masked tail (28 threads off)

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
        constexprs={"BLOCK_SIZE": BLOCK_SIZE},
    )
    target = GPUTarget(backend="metal", arch=80, warp_size=32)
    compiled = triton.compile(src, target=target, options={"num_warps": 4})

    raw = compiled.asm["metal"]
    msl = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    assert msl, "MSL stage produced empty string"

    # MSL -> .metallib via xcrun.
    metallib = libmetal.compile_msl_to_metallib(msl)
    assert isinstance(metallib, (bytes, bytearray)) and len(metallib) > 0
    # macOS metallib magic: starts with `MTLB` in the FAT-format wrapper or
    # `metal` header; at minimum, file is non-trivially sized.
    assert len(metallib) > 64, "metallib too small to be a valid Metal library"

    # Allocate buffers. BLOCK_SIZE=128 stride; only N=100 are populated.
    # The masked tail's 28 threads write nothing.
    nbytes_f32 = BLOCK_SIZE * 4
    nbytes_i32 = 4
    x_buf = libmetal.alloc_buffer(nbytes_f32)
    y_buf = libmetal.alloc_buffer(nbytes_f32)
    out_buf = libmetal.alloc_buffer(nbytes_f32)
    n_buf = libmetal.alloc_buffer(nbytes_i32)
    try:
        rng = np.random.default_rng(0xC0FFEE)
        # Pad with zeros to fill BLOCK_SIZE; only first N are meaningful.
        x = np.zeros(BLOCK_SIZE, dtype=np.float32)
        y = np.zeros(BLOCK_SIZE, dtype=np.float32)
        x[:N] = rng.standard_normal(N, dtype=np.float32)
        y[:N] = rng.standard_normal(N, dtype=np.float32)
        # Pre-fill out with a sentinel so we can verify only the first N
        # were written. The masked-off tail should still see whatever was
        # in out_buf before launch.
        out_init = np.full(BLOCK_SIZE, np.nan, dtype=np.float32)

        libmetal.copy_h2d(x_buf, x.tobytes())
        libmetal.copy_h2d(y_buf, y.tobytes())
        libmetal.copy_h2d(out_buf, out_init.tobytes())
        libmetal.copy_h2d(
            n_buf, np.array([N], dtype=np.uint32).tobytes()
        )

        # Single threadgroup of 128 threads = num_warps(4) * warp_size(32).
        # grid=(1,1,1) since pid > 0 lowering isn't yet implemented.
        libmetal.launch_kernel(
            metallib,
            "add_kernel",
            [x_buf, y_buf, out_buf, n_buf],
            (1, 1, 1),
            (BLOCK_SIZE, 1, 1),
        )

        out_bytes = libmetal.copy_d2h(out_buf, nbytes_f32)
        out_gpu = np.frombuffer(out_bytes, dtype=np.float32)

        expected = x + y
        # Active threads (first N): bit-exact equivalent to x+y on CPU.
        np.testing.assert_array_equal(out_gpu[:N], expected[:N])
        # Masked-off threads (tail): out_buf preserved the NaN sentinel
        # because the masked store guard is `if (id.x < n_elements)`.
        assert np.all(np.isnan(out_gpu[N:])), (
            "Masked-off tail should have preserved its pre-launch value; "
            "got: {}".format(out_gpu[N:])
        )
    finally:
        libmetal.free_buffer(x_buf)
        libmetal.free_buffer(y_buf)
        libmetal.free_buffer(out_buf)
        libmetal.free_buffer(n_buf)
