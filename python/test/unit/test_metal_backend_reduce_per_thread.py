"""Per-thread-owned axis=1 sum reduce on the Metal backend (M*N >> tpb).

Exercises the rank-2 axis=1 `ReduceLowering` body on the conv1d-style layout
where the reduce axis is fully serial within each thread
(`threadsPerCTA[axis_dim] == 1`) and `M*N >> tpb`.

The L3a-tileloop-2 redesign makes the body self-contained: it walks back to
the producing `tt.load` and reduces each row directly from device memory into
a per-row threadgroup buffer (`rowBuf[M]`) hoisted above the tile loop. This
closed the former carry-forward gap (the old body gave one scalar per
(thread, tile-iv) pair and could not express the per-row gather), so the
multi-element-per-thread cases below now produce correct results.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import triton
import triton.language as tl

libmetal = pytest.importorskip(
    "triton._C.libtriton.metal",
    reason="Metal backend pybind module not built into libtriton",
)
if not torch.backends.mps.is_available():
    pytest.skip(
        "Metal backend requires an MPS-enabled PyTorch (Apple Silicon)",
        allow_module_level=True,
    )


@triton.jit
def reduce_per_thread_kernel(
    x_ptr,
    out_ptr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    addr = offs_m[:, None] * BLOCK_N + offs_n[None, :]
    x = tl.load(x_ptr + addr)
    s = tl.sum(x, axis=1)
    tl.store(out_ptr + offs_m, s)


@triton.jit
def conv1d_runtime_reduce_kernel(
    input_ptr,
    kernel_ptr,
    output_ptr,
    input_size,
    kernel_size,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    p0 = tl.program_id(0)
    offs_out = p0 * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for k in range(0, kernel_size, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        mask_k = offs_k < kernel_size
        kernel = tl.load(kernel_ptr + offs_k, mask_k)

        offs_i = offs_out[:, None] + offs_k[None, :]
        mask_i = offs_i < input_size
        values = tl.load(input_ptr + offs_i, mask_i)

        acc += tl.sum(kernel[None, :] * values, axis=1)

    mask_out = offs_out < (input_size - kernel_size + 1)
    tl.store(output_ptr + offs_out, acc, mask_out)


@pytest.mark.parametrize(
    "M, N",
    [
        (1024, 64),
        (512, 32),
        (256, 128),
    ],
)
def test_reduce_per_thread_owned_f32(M, N):
    torch.manual_seed(0xC0FFEE)
    x = torch.randn((M, N), dtype=torch.float32).contiguous()
    out = torch.zeros((M,), dtype=torch.float32).contiguous()
    reduce_per_thread_kernel[(1, 1, 1)](x, out, BLOCK_M=M, BLOCK_N=N)
    expected = torch.sum(x.cpu(), dim=1)
    torch.testing.assert_close(out, expected, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize(
    "input_size, kernel_size",
    [
        (1024, 7),
        (2053, 73),
        (4096, 129),
    ],
)
def test_conv1d_runtime_loop_masked_product_reduce(input_size, kernel_size):
    torch.manual_seed(input_size * 13 + kernel_size)
    x = torch.randn((input_size,), dtype=torch.float32, device="mps")
    kernel = torch.randn((kernel_size,), dtype=torch.float32, device="mps")
    out_size = input_size - kernel_size + 1
    out = torch.zeros((out_size,), dtype=torch.float32, device="mps")

    conv1d_runtime_reduce_kernel[(triton.cdiv(out_size, 1024),)](
        x,
        kernel,
        out,
        input_size,
        kernel_size,
        BLOCK_SIZE=1024,
        BLOCK_K=64,
        num_warps=4,
    )
    torch.mps.synchronize()

    expected = torch.nn.functional.conv1d(
        x.cpu().view(1, 1, -1), kernel.cpu().view(1, 1, -1)
    ).flatten()
    torch.testing.assert_close(out.cpu(), expected, atol=1e-3, rtol=1e-3)
