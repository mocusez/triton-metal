"""End-to-end: @triton.jit -> MSL -> torch.mps.compile_shader -> assert x+y.

Acceptance test for the implementation notes
(AC.G3-G4). Ported from the legacy native-runtime path
(compile_msl_to_metallib + alloc_buffer + launch_kernel, removed) to the
MPS launch path. Exercises the masked tail: N=100 < BLOCK_SIZE=128, so the
28 masked-off threads must leave the output's pre-launch sentinel untouched.
"""

from __future__ import annotations

import pytest
import torch

import triton
import triton.language as tl

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="Metal backend requires an MPS-enabled PyTorch (Apple Silicon)",
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
    torch.manual_seed(0xC0FFEE)

    # Pad to BLOCK_SIZE; only the first N elements are meaningful.
    x = torch.zeros(BLOCK_SIZE, dtype=torch.float32, device="mps")
    y = torch.zeros(BLOCK_SIZE, dtype=torch.float32, device="mps")
    x[:N] = torch.randn(N, device="mps")
    y[:N] = torch.randn(N, device="mps")
    # NaN sentinel so we can verify the masked-off tail is left untouched
    # (the store guard is `if (offs < n_elements)`).
    out = torch.full((BLOCK_SIZE,), float("nan"), dtype=torch.float32, device="mps")

    # Single threadgroup of 128 threads = num_warps(4) * warp_size(32).
    add_kernel[(1, 1, 1)](x, y, out, N, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)
    torch.mps.synchronize()

    # Active threads (first N): bit-exact equivalent to x + y.
    torch.testing.assert_close(out[:N], (x + y)[:N], atol=0, rtol=0)
    # Masked-off tail: the store guard preserved the pre-launch NaN sentinel.
    assert torch.isnan(out[N:]).all(), (
        f"masked-off tail should keep its pre-launch value; got: {out[N:]}"
    )
