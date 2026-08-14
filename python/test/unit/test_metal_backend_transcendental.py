"""Session L4: fp32 `tl.sqrt` and `tl.erf` runtime bit-close on MPS.

Asserts that `tl.sqrt(x)` and `tl.erf(x)` over a `tensor<BLOCK_SIZExf32>`
lower through `math.sqrt` / `math.erf` -> `metal.unary_exp` -> MSL
`metal::precise::sqrt(...)` / `metal::erf(...)` and produce element-wise
close agreement with the CPU reference `torch.sqrt` / `torch.erf` at
`atol=1e-3, rtol=1e-3` (the L1 envelope).

See `.omc/specs/deep-interview-leet-triton-l4-transcendentals.md`.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import triton  # noqa: E402
import triton.language as tl  # noqa: E402

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
def sqrt_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.sqrt(x)
    tl.store(output_ptr + offsets, y, mask=mask)


@triton.jit
def erf_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.erf(x)
    tl.store(output_ptr + offsets, y, mask=mask)


@pytest.mark.parametrize("BLOCK_SIZE", [256, 1024])
def test_sqrt_bit_close(BLOCK_SIZE):
    N = BLOCK_SIZE
    torch.manual_seed(0xC0FFEE)
    # Non-negative inputs so torch.sqrt is real-valued.
    x = torch.rand(N, dtype=torch.float32) + 1e-3
    out = torch.zeros(N, dtype=torch.float32)

    sqrt_kernel[(1, 1, 1)](x, out, N, BLOCK_SIZE=BLOCK_SIZE)

    expected = torch.sqrt(x)
    torch.testing.assert_close(out, expected, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("BLOCK_SIZE", [256, 1024])
def test_erf_bit_close(BLOCK_SIZE):
    N = BLOCK_SIZE
    torch.manual_seed(0xC0FFEE)
    x = torch.randn(N, dtype=torch.float32)
    out = torch.zeros(N, dtype=torch.float32)

    erf_kernel[(1, 1, 1)](x, out, N, BLOCK_SIZE=BLOCK_SIZE)

    expected = torch.erf(x)
    torch.testing.assert_close(out, expected, atol=1e-3, rtol=1e-3)


@triton.jit
def exp_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.exp(x)
    tl.store(output_ptr + offsets, y, mask=mask)


@triton.jit
def log_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=1.0)
    y = tl.log(x)
    tl.store(output_ptr + offsets, y, mask=mask)


@triton.jit
def rsqrt_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=1.0)
    y = tl.rsqrt(x)
    tl.store(output_ptr + offsets, y, mask=mask)


@triton.jit
def exp2_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.exp2(x)
    tl.store(output_ptr + offsets, y, mask=mask)


@triton.jit
def log2_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=1.0)
    y = tl.log2(x)
    tl.store(output_ptr + offsets, y, mask=mask)


@triton.jit
def abs_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.abs(x)
    tl.store(output_ptr + offsets, y, mask=mask)


@triton.jit
def sqrt_rn_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=1.0)
    y = tl.sqrt_rn(x)
    tl.store(output_ptr + offsets, y, mask=mask)


@triton.jit
def floor_runtime_scalar_kernel(output_ptr, value):
    center = tl.floor(value / 2).to(tl.int32)
    tl.store(output_ptr, center)


@pytest.mark.parametrize("BLOCK_SIZE", [256, 1024])
def test_exp_bit_close(BLOCK_SIZE):
    N = BLOCK_SIZE
    torch.manual_seed(0xC0FFEE)
    # Scale randn down to avoid fp32 overflow in exp.
    x = torch.randn(N, dtype=torch.float32) * 0.5
    out = torch.zeros(N, dtype=torch.float32)

    exp_kernel[(1, 1, 1)](x, out, N, BLOCK_SIZE=BLOCK_SIZE)

    expected = torch.exp(x)
    torch.testing.assert_close(out, expected, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("BLOCK_SIZE", [256, 1024])
def test_log_bit_close(BLOCK_SIZE):
    N = BLOCK_SIZE
    torch.manual_seed(0xC0FFEE)
    # Strictly positive inputs; offset away from 0 for numerical stability.
    x = torch.rand(N, dtype=torch.float32) + 0.1
    out = torch.zeros(N, dtype=torch.float32)

    log_kernel[(1, 1, 1)](x, out, N, BLOCK_SIZE=BLOCK_SIZE)

    expected = torch.log(x)
    torch.testing.assert_close(out, expected, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("BLOCK_SIZE", [256, 1024])
def test_rsqrt_bit_close(BLOCK_SIZE):
    N = BLOCK_SIZE
    torch.manual_seed(0xC0FFEE)
    # Strictly positive inputs; offset away from 0 for numerical stability.
    x = torch.rand(N, dtype=torch.float32) + 0.1
    out = torch.zeros(N, dtype=torch.float32)

    rsqrt_kernel[(1, 1, 1)](x, out, N, BLOCK_SIZE=BLOCK_SIZE)

    expected = torch.rsqrt(x)
    torch.testing.assert_close(out, expected, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("BLOCK_SIZE", [256, 1024])
def test_exp2_bit_close(BLOCK_SIZE):
    N = BLOCK_SIZE
    torch.manual_seed(0xC0FFEE)
    # Keep inputs small enough that exp2 stays well away from fp32 overflow.
    x = torch.randn(N, dtype=torch.float32) * 0.5
    out = torch.zeros(N, dtype=torch.float32)

    exp2_kernel[(1, 1, 1)](x, out, N, BLOCK_SIZE=BLOCK_SIZE)

    expected = torch.exp2(x)
    torch.testing.assert_close(out, expected, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("BLOCK_SIZE", [256, 1024])
def test_log2_bit_close(BLOCK_SIZE):
    N = BLOCK_SIZE
    torch.manual_seed(0xC0FFEE)
    # Strictly positive inputs; offset away from 0 for numerical stability.
    x = torch.rand(N, dtype=torch.float32) + 0.1
    out = torch.zeros(N, dtype=torch.float32)

    log2_kernel[(1, 1, 1)](x, out, N, BLOCK_SIZE=BLOCK_SIZE)

    expected = torch.log2(x)
    torch.testing.assert_close(out, expected, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("BLOCK_SIZE", [256, 1024])
def test_abs_bit_exact(BLOCK_SIZE):
    N = BLOCK_SIZE
    torch.manual_seed(0xC0FFEE)
    x = torch.randn(N, dtype=torch.float32)
    out = torch.zeros(N, dtype=torch.float32)

    abs_kernel[(1, 1, 1)](x, out, N, BLOCK_SIZE=BLOCK_SIZE)

    expected = torch.abs(x)
    torch.testing.assert_close(out, expected, atol=0, rtol=0)


@pytest.mark.parametrize("BLOCK_SIZE", [256, 1024])
def test_sqrt_rn_bit_close(BLOCK_SIZE):
    N = BLOCK_SIZE
    torch.manual_seed(0xC0FFEE)
    # Non-negative inputs so sqrt_rn is real-valued.
    x = torch.rand(N, dtype=torch.float32) + 1e-3
    out = torch.zeros(N, dtype=torch.float32)

    sqrt_rn_kernel[(1, 1, 1)](x, out, N, BLOCK_SIZE=BLOCK_SIZE)

    expected = torch.sqrt(x)
    torch.testing.assert_close(out, expected, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("value, expected", [(5, 2), (6, 3)])
def test_floor_runtime_scalar(value, expected):
    out = torch.zeros(1, dtype=torch.int32)

    floor_runtime_scalar_kernel[(1, 1, 1)](out, value)

    assert out.item() == expected


# Lmultiload Phase B audit canary: existing transcendental tests use only
# the canonical `offs = pid*BLOCK + arange` pattern. This canary adds a
# constant divergent offset so a regression of the per-thread index path
# would surface in this file. Pre-Phase-B the index ignored `+ K`.
@triton.jit
def _canary_sqrt_offset_kernel(x_ptr, output_ptr, n_elements,
                               K: tl.constexpr,
                               BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = (offs + K) < n_elements
    x = tl.load(x_ptr + offs + K, mask=mask, other=1.0)
    y = tl.sqrt(x)
    tl.store(output_ptr + offs, y, mask=offs < (n_elements - K))


def test_canary_sqrt_with_divergent_const_offset():
    BLOCK_SIZE = 128
    K = 7
    N = 256
    torch.manual_seed(0xC0FFEE)
    x = torch.rand(N, dtype=torch.float32) + 1e-3
    out = torch.zeros(N, dtype=torch.float32)
    _canary_sqrt_offset_kernel[(N // BLOCK_SIZE, 1, 1)](
        x, out, N, K=K, BLOCK_SIZE=BLOCK_SIZE
    )
    expected = torch.zeros(N, dtype=torch.float32)
    expected[: N - K] = torch.sqrt(x[K:])
    torch.testing.assert_close(out, expected, atol=1e-3, rtol=1e-3)
