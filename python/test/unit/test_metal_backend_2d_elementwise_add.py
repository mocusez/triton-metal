"""Canonical 2D elementwise-add on the Metal backend.

Acceptance tests AC.P1 and AC.P2 for
`.omc/specs/deep-interview-metal-2d-maskedaccess-session2.md`. Exercises
the full canonical 2D Triton pattern:

    pid_m = tl.program_id(0); pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask   = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    addr   = offs_m[:, None] * stride + offs_n[None, :]
    tl.store(out + addr, tl.load(x + addr, mask=mask)
                       + tl.load(y + addr, mask=mask), mask=mask)

Both tests assert bit-exactness against torch.add over the active
(unmasked) region.
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
if not hasattr(libmetal, "launch_kernel_with_pipeline"):
    pytest.skip(
        "Metal runtime not compiled (non-Darwin build or Xcode CLT absent)",
        allow_module_level=True,
    )


def _cdiv(a, b):
    return (a + b - 1) // b


@triton.jit
def add_kernel_2d(
    x_ptr,
    y_ptr,
    out_ptr,
    M,
    N,
    stride_m,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    addr = offs_m[:, None] * stride_m + offs_n[None, :]
    x = tl.load(x_ptr + addr, mask=mask)
    y = tl.load(y_ptr + addr, mask=mask)
    tl.store(out_ptr + addr, x + y, mask=mask)


def test_2d_elementwise_add_clean():
    """AC.P1: (128,128) fp32, BLOCK=(16,16), 2D grid (8,8,1), bit-exact."""
    M = 128
    N = 128
    BLOCK_M = 16
    BLOCK_N = 16
    torch.manual_seed(0xC0FFEE)
    x = torch.rand((M, N), dtype=torch.float32).contiguous()
    y = torch.rand((M, N), dtype=torch.float32).contiguous()
    out = torch.zeros((M, N), dtype=torch.float32).contiguous()
    grid = (_cdiv(M, BLOCK_M), _cdiv(N, BLOCK_N), 1)
    add_kernel_2d[grid](x, y, out, M, N, N, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
    expected = x + y
    torch.testing.assert_close(out, expected, atol=0, rtol=0)


def test_2d_elementwise_add_masked():
    """AC.P2: (100,100) fp32 active, BLOCK=(16,16), grid (7,7,1), masked tail."""
    M = 100
    N = 100
    BLOCK_M = 16
    BLOCK_N = 16
    padM = _cdiv(M, BLOCK_M) * BLOCK_M  # 112
    padN = _cdiv(N, BLOCK_N) * BLOCK_N  # 112
    torch.manual_seed(0xC0FFEE)
    x = torch.rand((padM, padN), dtype=torch.float32).contiguous()
    y = torch.rand((padM, padN), dtype=torch.float32).contiguous()
    out = torch.zeros((padM, padN), dtype=torch.float32).contiguous()
    grid = (_cdiv(M, BLOCK_M), _cdiv(N, BLOCK_N), 1)
    add_kernel_2d[grid](x, y, out, M, N, padN, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
    expected = (x + y)[:M, :N]
    torch.testing.assert_close(out[:M, :N], expected, atol=0, rtol=0)
