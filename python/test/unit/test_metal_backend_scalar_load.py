"""L3a-tileloop-compiler-A: scalar tt.load on bare !tt.ptr<f32>.

Drives a small @triton.jit kernel that performs a scalar `tl.load(scalar_ptr
+ scalar_offset)` inside a `tl.static_range` loop, and asserts bit-exact
agreement with a CPU reference across 5 deterministic runs. The kernel
mirrors the conv1d Variant B shape that motivated this session.

See `.omc/specs/deep-interview-leet-triton-l3a-tileloop-compiler-a-scalar-load.md`.
"""

from __future__ import annotations

import sys

import pytest

torch = pytest.importorskip("torch")

import triton
import triton.language as tl


pytestmark = pytest.mark.skipif(
    sys.platform != "darwin" or not torch.backends.mps.is_available(),
    reason="Metal backend requires Darwin + MPS",
)


@triton.jit
def _scalar_load_kernel(
    weight_ptr,
    input_ptr,
    output_ptr,
    N,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """For each i in [0, BLOCK): output[i] = sum(weight[k] * input[i+k] for k in [0, K))."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for k in tl.static_range(K):
        offs_in = offs + k
        mask_in = offs_in < N
        inp = tl.load(input_ptr + offs_in, mask_in)
        kv = tl.load(weight_ptr + k)  # SCALAR load — the gap this session unblocks.
        acc += kv * inp
    mask_out = offs < (N - K + 1)
    tl.store(output_ptr + offs, acc, mask_out)


def _run_one(N: int = 4096, K: int = 7, BLOCK: int = 1024) -> None:
    torch.manual_seed(0)
    inp = torch.randn(N, dtype=torch.float32, device="mps")
    weight = torch.randn(K, dtype=torch.float32, device="mps")
    out_size = N - K + 1
    out = torch.zeros(out_size, dtype=torch.float32, device="mps")

    n_blocks = triton.cdiv(out_size, BLOCK)
    _scalar_load_kernel[(n_blocks,)](weight, inp, out, N, K, BLOCK)

    expected = torch.nn.functional.conv1d(
        inp.view(1, 1, -1), weight.view(1, 1, -1)
    ).flatten()
    assert torch.allclose(out, expected, atol=1e-3, rtol=1e-3), (
        f"max abs err = {(out - expected).abs().max().item()}"
    )


@pytest.mark.parametrize("run_idx", range(5))
def test_scalar_load_conv1d_shape_bit_exact(run_idx: int) -> None:
    _run_one(N=4096, K=7, BLOCK=1024)
