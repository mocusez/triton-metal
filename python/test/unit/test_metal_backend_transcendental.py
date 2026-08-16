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


# --- triton.language.extra.libdevice -------------------------------------
#
# `MetalBackend.get_module_map()` redirects `triton.language.extra.libdevice`
# to `third_party/metal/language/metal/libdevice.py`. Before that it returned
# `{}`, so every libdevice call resolved to the generic bodies-only shim and
# failed with "cannot convert None of type NoneType to tensor".
#
# Each `__metal_*` symbol maps to an MSL intrinsic whose EXISTENCE was verified
# by compiling it through torch.mps.compile_shader — a wrong name compiles fine
# here and only fails when the shader is loaded, so these tests launch for real
# rather than stopping at MSL text.

from triton.language.extra import libdevice  # noqa: E402


@triton.jit
def _libdevice_unary_kernel(x_ptr, out_ptr, WHICH: tl.constexpr,
                            BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    v = tl.load(x_ptr + offs)
    if WHICH == 0:
        r = libdevice.asin(v)
    elif WHICH == 1:
        r = libdevice.acos(v)
    elif WHICH == 2:
        r = libdevice.atan(v)
    elif WHICH == 3:
        r = libdevice.sinh(v)
    elif WHICH == 4:
        r = libdevice.cosh(v)
    elif WHICH == 5:
        r = libdevice.tanh(v)
    elif WHICH == 6:
        r = libdevice.asinh(v)
    elif WHICH == 7:
        r = libdevice.atanh(v)
    elif WHICH == 8:
        r = libdevice.exp10(v)
    elif WHICH == 9:
        r = libdevice.log10(v)
    elif WHICH == 10:
        r = libdevice.rsqrt(v)
    else:
        r = libdevice.erf(v)
    tl.store(out_ptr + offs, r)


_LIBDEVICE_UNARY = [
    (0, torch.asin), (1, torch.acos), (2, torch.atan), (3, torch.sinh),
    (4, torch.cosh), (5, torch.tanh), (6, torch.asinh), (7, torch.atanh),
    (8, lambda t: torch.pow(torch.tensor(10.0), t)), (9, torch.log10),
    (10, torch.rsqrt), (11, torch.erf),
]


@pytest.mark.parametrize("which,reference", _LIBDEVICE_UNARY)
def test_libdevice_unary_f32(which, reference):
    torch.manual_seed(which)
    # (0.05, 0.95) keeps every function in-domain, asin/acos/atanh included.
    x = torch.rand(64, dtype=torch.float32) * 0.9 + 0.05
    out = torch.empty(64, dtype=torch.float32, device="mps")
    _libdevice_unary_kernel[(1,)](x.to("mps"), out, WHICH=which, BLOCK=64)
    torch.mps.synchronize()
    torch.testing.assert_close(out.cpu(), reference(x), atol=1e-5, rtol=1e-5)


@triton.jit
def _libdevice_binary_kernel(x_ptr, out_ptr, WHICH: tl.constexpr,
                             BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    v = tl.load(x_ptr + offs)
    if WHICH == 0:
        r = libdevice.pow(v, v)
    elif WHICH == 1:
        r = libdevice.atan2(v, v + 1.0)
    elif WHICH == 2:
        r = libdevice.copysign(v, -v)
    elif WHICH == 3:
        r = libdevice.fmod(v * 10.0, 3.0)
    else:
        r = libdevice.fma(v, v, v)
    tl.store(out_ptr + offs, r)


_LIBDEVICE_BINARY = [
    (0, lambda t: torch.pow(t, t)),
    (1, lambda t: torch.atan2(t, t + 1.0)),
    (2, lambda t: torch.copysign(t, -t)),
    (3, lambda t: torch.fmod(t * 10.0, 3.0)),
    (4, lambda t: t * t + t),
]


@pytest.mark.parametrize("which,reference", _LIBDEVICE_BINARY)
def test_libdevice_binary_f32(which, reference):
    torch.manual_seed(100 + which)
    x = torch.rand(64, dtype=torch.float32) * 0.9 + 0.05
    out = torch.empty(64, dtype=torch.float32, device="mps")
    _libdevice_binary_kernel[(1,)](x.to("mps"), out, WHICH=which, BLOCK=64)
    torch.mps.synchronize()
    torch.testing.assert_close(out.cpu(), reference(x), atol=1e-5, rtol=1e-5)


@triton.jit
def _libdevice_clz_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    tl.store(out_ptr + offs, libdevice.clz(tl.load(x_ptr + offs)))


def test_libdevice_clz_i32():
    """Integer intrinsics need the ui32 bridge: `Metal_Type` has no signless
    integer, so a signless i32 operand would fail the op verifier."""
    x = torch.tensor([1, 2, 255, 1 << 30] * 4, dtype=torch.int32)
    out = torch.empty(16, dtype=torch.int32, device="mps")
    _libdevice_clz_kernel[(1,)](x.to("mps"), out, BLOCK=16)
    torch.mps.synchronize()
    expected = torch.tensor([31, 30, 24, 1] * 4, dtype=torch.int32)
    torch.testing.assert_close(out.cpu(), expected, atol=0, rtol=0)


@triton.jit
def _libdevice_hypot_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    v = tl.load(x_ptr + offs)
    tl.store(out_ptr + offs, libdevice.hypot(v, v))


def test_libdevice_missing_function_is_a_clean_error():
    """MSL has no `hypot`, so it is deliberately absent from the Metal table.
    Using it must name the missing function, not emit a shader that fails to
    link (which is what a guessed-at symbol name would produce)."""
    x = torch.rand(16, dtype=torch.float32, device="mps")
    out = torch.empty(16, dtype=torch.float32, device="mps")
    with pytest.raises(Exception) as excinfo:
        _libdevice_hypot_kernel[(1,)](x, out, BLOCK=16)
    assert "hypot" in str(excinfo.value)
