"""Session L4: fp32 `tl.sqrt` and `tl.erf` runtime bit-close on MPS.

Asserts that `tl.sqrt(x)` and `tl.erf(x)` over a `tensor<BLOCK_SIZExf32>`
lower through `math.sqrt` / `math.erf` -> `metal.unary_exp` -> MSL
`metal::precise::sqrt(...)` / `metal::erf(...)` and produce element-wise
close agreement with the CPU reference `torch.sqrt` / `torch.erf` at
`atol=1e-3, rtol=1e-3` (the L1 envelope).

See the implementation notes.
"""

from __future__ import annotations

import math

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
def _fma_f32_kernel(X, Y, Z, O, N: tl.constexpr, BLOCK: tl.constexpr):
    if N == 1:
        x = tl.load(X).to(tl.float32)
        y = tl.load(Y).to(tl.float32)
        z = tl.load(Z).to(tl.float32)
        tl.store(O, tl.fma(x, y, z))
    else:
        offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(X + offset, offset < N, other=0).to(tl.float32)
        y = tl.load(Y + offset, offset < N, other=0).to(tl.float32)
        z = tl.load(Z + offset, offset < N, other=0).to(tl.float32)
        tl.store(O + offset, tl.fma(x, y, z), offset < N)


@pytest.mark.parametrize("n", [1, 259])
@pytest.mark.parametrize("num_warps", [1, 4])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_fma_f32_preserves_fusion_and_explicit_promotion(n, num_warps, dtype):
    # The first f32 triple yields -2**-46 when fused, but zero after a
    # separately rounded multiply. All triples have exact f64 intermediates.
    x = torch.tensor([1 + 2**-23, -2, 0.5, 4], dtype=dtype).repeat(n)[:n]
    y = torch.tensor([1 - 2**-23, 3, -4, 0.25], dtype=dtype).repeat(n)[:n]
    z = torch.tensor([-1, 1, 3, -2], dtype=dtype).repeat(n)[:n]
    expected = (x.double() * y.double() + z.double()).float()
    output = torch.full((n + 3,), 12345.0, device="mps")
    compiled = _fma_f32_kernel[(triton.cdiv(n, 256),)](
        x.to("mps"), y.to("mps"), z.to("mps"), output,
        N=n, BLOCK=256, num_warps=num_warps,
    )
    torch.mps.synchronize()
    torch.testing.assert_close(output[:n].cpu(), expected, rtol=0, atol=0)
    assert torch.all(output[n:].cpu() == 12345.0)
    msl = compiled.asm["metal"]
    if isinstance(msl, bytes):
        msl = msl.decode("utf-8")
    assert "metal::precise::fma" in msl


@triton.jit
def _unary_f32_kernel(X, O, N: tl.constexpr, OP: tl.constexpr, BLOCK: tl.constexpr):
    if N == 1:
        offset = 0
    else:
        offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offset, offset < N, other=1.0)
    if OP == "sqrt":
        y = tl.sqrt(x)
    elif OP == "erf":
        y = tl.erf(x)
    elif OP == "exp":
        y = tl.exp(x)
    elif OP == "exp2":
        y = tl.exp2(x)
    elif OP == "log":
        y = tl.log(x)
    elif OP == "log2":
        y = tl.log2(x)
    elif OP == "rsqrt":
        y = tl.rsqrt(x)
    elif OP == "sin":
        y = tl.sin(x)
    elif OP == "cos":
        y = tl.cos(x)
    tl.store(O + offset, y, offset < N)


@pytest.mark.parametrize("op", ["sqrt", "erf", "exp", "exp2", "log", "log2", "rsqrt", "sin", "cos"])
@pytest.mark.parametrize("n", [1, 259])
@pytest.mark.parametrize("num_warps", [1, 4])
def test_unary_math_f32_preserves_scalar_and_tensor_support(op, n, num_warps):
    x = torch.linspace(0.125, 4.0, n, dtype=torch.float32)
    expected = getattr(torch, op)(x)
    output = torch.full((n + 3,), 12345.0, device="mps")
    _unary_f32_kernel[(triton.cdiv(n, 256),)](
        x.to("mps"), output, N=n, OP=op, BLOCK=256, num_warps=num_warps,
    )
    torch.mps.synchronize()
    torch.testing.assert_close(output[:n].cpu(), expected, rtol=1e-5, atol=1e-6)
    assert torch.all(output[n:].cpu() == 12345.0)


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
def _libdevice_pow_integer_exponent_kernel(x_ptr, exp_ptr, even_ptr, odd_ptr,
                                           dynamic_ptr, float_ptr,
                                           BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    v = tl.load(x_ptr + offs)
    e = tl.load(exp_ptr + offs)
    tl.store(even_ptr + offs, libdevice.pow(v, 2))
    tl.store(odd_ptr + offs, libdevice.pow(v, 3))
    tl.store(dynamic_ptr + offs, libdevice.pow(v, e))
    tl.store(float_ptr + offs, libdevice.pow(v, v))


def test_libdevice_pow_f32_i32_preserves_integer_exponent_semantics():
    x = torch.tensor(
        [-3.0, -2.0, -1.5, -0.5, 0.0, 0.25, 1.5, 2.0,
         -0.0, -0.0, -1.0, -1.0, -1.0, 1.0, -float("inf"), float("nan")],
        dtype=torch.float32,
    )
    exponents = torch.tensor(
        [2, 3, -1, -2, 0, 1, -3, 4, -3, 3, 16777217, -2147483648,
         2147483647, -2147483648, 3, 0], dtype=torch.int32)
    even = torch.empty_like(x, device="mps")
    odd = torch.empty_like(x, device="mps")
    dynamic = torch.empty_like(x, device="mps")
    floating = torch.empty_like(x, device="mps")
    _libdevice_pow_integer_exponent_kernel[(1,)](
        x.to("mps"), exponents.to("mps"), even, odd, dynamic, floating,
        BLOCK=x.numel()
    )
    torch.mps.synchronize()
    torch.testing.assert_close(even.cpu(), x * x, atol=0, rtol=0, equal_nan=True)
    torch.testing.assert_close(odd.cpu(), x * x * x, atol=0, rtol=0, equal_nan=True)
    expected = torch.cat((torch.pow(x[:8], exponents[:8]), torch.tensor(
        [-float("inf"), -0.0, -1.0, 1.0, -1.0, 1.0, -float("inf"), 1.0])))
    torch.testing.assert_close(
        dynamic.cpu(),
        expected,
        atol=1e-6,
        rtol=1e-6,
    )

    assert torch.equal(torch.signbit(dynamic.cpu()), torch.signbit(expected))
    assert torch.equal(torch.signbit(odd.cpu()[8:10]), torch.tensor([True, True]))

    positive = x > 0
    torch.testing.assert_close(
        floating.cpu()[positive],
        torch.pow(x[positive], x[positive]),
        atol=1e-5,
        rtol=1e-5,
    )


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


# ---------------------------------------------------------------------------
# `tl.minimum` / `tl.maximum` NaN policy.
#
# `propagate_nan=tl.PropagateNan.ALL` is IEEE-754 maximum/minimum: a NaN
# operand makes the result NaN. The default `PropagateNan.NONE` is IEEE-754
# maxNum/minNum: a NaN operand is ignored and the other operand wins. MSL's
# `max`/`min`/`fmax`/`fmin` are all number-preferring, so the ALL policy needs
# its own spelling; before that existed both policies returned the number.
# ---------------------------------------------------------------------------

_NAN = float("nan")
_INF = float("inf")

# (lhs, rhs) pairs: NaN on the left, on the right, on both sides, against the
# infinities that bound the reductions, plus ordinary finite pairs and equal
# pairs. Signed zeros get their own bit-exact test below.
_MINMAX_PAIRS = [
    (_NAN, 2.0),
    (2.0, _NAN),
    (_NAN, _NAN),
    (_NAN, -_INF),
    (_INF, _NAN),
    (_NAN, 0.0),
    (1.0, 2.0),
    (2.0, 1.0),
    (-1.0, -2.0),
    (-2.0, -1.0),
    (-_INF, 2.0),
    (_INF, 2.0),
    (3.5, 3.5),
    (0.0, 0.0),
]


def _ieee_extremum(a, b, want_max):
    """IEEE-754 maximum/minimum: a NaN operand wins, -0.0 orders below +0.0."""
    if math.isnan(a):
        return a
    if math.isnan(b):
        return b
    if a == b:  # the only interesting case is the +0.0 / -0.0 pair
        negative = math.copysign(1.0, a) < 0.0
        return (b if negative else a) if want_max else (a if negative else b)
    return max(a, b) if want_max else min(a, b)


def _num_extremum(a, b, want_max):
    """IEEE-754 maxNum/minNum: a NaN operand is ignored (Triton's default)."""
    if math.isnan(a):
        return b
    if math.isnan(b):
        return a
    return max(a, b) if want_max else min(a, b)


def _minmax_inputs(n):
    pairs = [_MINMAX_PAIRS[i % len(_MINMAX_PAIRS)] for i in range(n)]
    lhs = torch.tensor([p[0] for p in pairs], dtype=torch.float32)
    rhs = torch.tensor([p[1] for p in pairs], dtype=torch.float32)
    return pairs, lhs, rhs


@triton.jit
def _minmax_nan_policy_kernel(X, Y, MAX_ALL, MIN_ALL, MAX_NUM, MIN_NUM, N,
                              BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask, other=0.0)
    y = tl.load(Y + offs, mask, other=0.0)
    tl.store(MAX_ALL + offs, tl.maximum(x, y, propagate_nan=tl.PropagateNan.ALL), mask)
    tl.store(MIN_ALL + offs, tl.minimum(x, y, propagate_nan=tl.PropagateNan.ALL), mask)
    tl.store(MAX_NUM + offs, tl.maximum(x, y), mask)
    tl.store(MIN_NUM + offs, tl.minimum(x, y), mask)


@pytest.mark.parametrize("n", [1, 14, 259])
@pytest.mark.parametrize("num_warps", [1, 4])
def test_minmax_propagate_nan_all_returns_nan(n, num_warps):
    """The ALL policy must carry a NaN operand into the result, from either
    side and from both. n=259 with BLOCK=128 also covers three programs and a
    masked tail."""
    pairs, lhs, rhs = _minmax_inputs(n)
    outs = [torch.full((n,), 12345.0, device="mps") for _ in range(4)]
    _minmax_nan_policy_kernel[(triton.cdiv(n, 128),)](
        lhs.to("mps"), rhs.to("mps"), *outs, n, BLOCK=128, num_warps=num_warps,
    )
    torch.mps.synchronize()
    max_all, min_all = outs[0].cpu(), outs[1].cpu()
    expected_max = torch.tensor([_ieee_extremum(a, b, True) for a, b in pairs])
    expected_min = torch.tensor([_ieee_extremum(a, b, False) for a, b in pairs])
    torch.testing.assert_close(max_all, expected_max, rtol=0, atol=0, equal_nan=True)
    torch.testing.assert_close(min_all, expected_min, rtol=0, atol=0, equal_nan=True)
    # State the defect directly: every pair holding a NaN must produce a NaN,
    # which is what an `assert_close` against a number-preferring result would
    # otherwise let through if the reference were computed the same wrong way.
    has_nan = torch.tensor([math.isnan(a) or math.isnan(b) for a, b in pairs])
    assert torch.equal(torch.isnan(max_all), has_nan)
    assert torch.equal(torch.isnan(min_all), has_nan)


@pytest.mark.parametrize("n", [1, 14, 259])
@pytest.mark.parametrize("num_warps", [1, 4])
def test_minmax_propagate_nan_none_prefers_the_number(n, num_warps):
    """Adjacent policy: the default must keep ignoring a NaN operand. This is
    what softmax and every clamp idiom in the backend rely on, so the ALL fix
    must not leak into it."""
    pairs, lhs, rhs = _minmax_inputs(n)
    outs = [torch.full((n,), 12345.0, device="mps") for _ in range(4)]
    _minmax_nan_policy_kernel[(triton.cdiv(n, 128),)](
        lhs.to("mps"), rhs.to("mps"), *outs, n, BLOCK=128, num_warps=num_warps,
    )
    torch.mps.synchronize()
    max_num, min_num = outs[2].cpu(), outs[3].cpu()
    expected_max = torch.tensor([_num_extremum(a, b, True) for a, b in pairs])
    expected_min = torch.tensor([_num_extremum(a, b, False) for a, b in pairs])
    torch.testing.assert_close(max_num, expected_max, rtol=0, atol=0, equal_nan=True)
    torch.testing.assert_close(min_num, expected_min, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("num_warps", [1, 4])
def test_minmax_propagate_nan_all_orders_signed_zero(num_warps):
    """IEEE-754 maximum/minimum order -0.0 below +0.0. The values compare equal,
    so this has to be checked on the bit pattern."""
    pairs = [(0.0, -0.0), (-0.0, 0.0), (0.0, 0.0), (-0.0, -0.0)]
    n = len(pairs)
    lhs = torch.tensor([p[0] for p in pairs], dtype=torch.float32)
    rhs = torch.tensor([p[1] for p in pairs], dtype=torch.float32)
    outs = [torch.full((n,), 12345.0, device="mps") for _ in range(4)]
    _minmax_nan_policy_kernel[(1,)](
        lhs.to("mps"), rhs.to("mps"), *outs, n, BLOCK=128, num_warps=num_warps,
    )
    torch.mps.synchronize()
    max_bits = outs[0].cpu().view(torch.int32)
    min_bits = outs[1].cpu().view(torch.int32)
    expected_max = torch.tensor(
        [_ieee_extremum(a, b, True) for a, b in pairs]).view(torch.int32)
    expected_min = torch.tensor(
        [_ieee_extremum(a, b, False) for a, b in pairs]).view(torch.int32)
    assert torch.equal(max_bits, expected_max), (max_bits, expected_max)
    assert torch.equal(min_bits, expected_min), (min_bits, expected_min)


@triton.jit
def _minmax_nan_policy_scalar_kernel(X, Y, MAX_ALL, MIN_ALL, MAX_NUM, MIN_NUM):
    x = tl.load(X)
    y = tl.load(Y)
    tl.store(MAX_ALL, tl.maximum(x, y, propagate_nan=tl.PropagateNan.ALL))
    tl.store(MIN_ALL, tl.minimum(x, y, propagate_nan=tl.PropagateNan.ALL))
    tl.store(MAX_NUM, tl.maximum(x, y))
    tl.store(MIN_NUM, tl.minimum(x, y))


@pytest.mark.parametrize("lhs,rhs", _MINMAX_PAIRS)
@pytest.mark.parametrize("num_warps", [1, 4])
def test_minmax_propagate_nan_all_scalar_form(lhs, rhs, num_warps):
    """The scalar form takes a different emitter path than the scalarized
    tensor form, so it needs its own coverage of both policies."""
    x = torch.tensor([lhs], dtype=torch.float32, device="mps")
    y = torch.tensor([rhs], dtype=torch.float32, device="mps")
    outs = [torch.full((1,), 12345.0, device="mps") for _ in range(4)]
    _minmax_nan_policy_scalar_kernel[(1,)](x, y, *outs, num_warps=num_warps)
    torch.mps.synchronize()
    got = [o.cpu().item() for o in outs]
    want = [
        _ieee_extremum(lhs, rhs, True),
        _ieee_extremum(lhs, rhs, False),
        _num_extremum(lhs, rhs, True),
        _num_extremum(lhs, rhs, False),
    ]
    for g, w, name in zip(got, want, ["max ALL", "min ALL", "max NONE", "min NONE"]):
        if math.isnan(w):
            assert math.isnan(g), f"{name}: expected NaN, got {g}"
        else:
            assert g == w, f"{name}: expected {w}, got {g}"


@triton.jit
def _reduce_max_all(a, b):
    return tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)


@triton.jit
def _reduce_min_all(a, b):
    return tl.minimum(a, b, propagate_nan=tl.PropagateNan.ALL)


@triton.jit
def _reduce_nan_policy_kernel(X, OMAX, OMIN, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(X + offs)
    tl.store(OMAX, tl.reduce(x, 0, _reduce_max_all))
    tl.store(OMIN, tl.reduce(x, 0, _reduce_min_all))


@pytest.mark.parametrize("nan_at", [None, 0, 5, -1])
@pytest.mark.parametrize("BLOCK", [32, 256])
@pytest.mark.parametrize("num_warps", [1, 4])
def test_reduce_minmax_propagate_nan_all_returns_nan(nan_at, BLOCK, num_warps):
    """Reduction consumer: a NaN anywhere in the tile must survive the fold,
    wherever it lands relative to the per-thread partials and the cross-lane
    butterfly. `nan_at=None` is the control that the fold still reduces."""
    x = torch.linspace(-3.0, 5.0, BLOCK, dtype=torch.float32)
    if nan_at is not None:
        x[nan_at] = _NAN
    omax = torch.full((1,), 12345.0, device="mps")
    omin = torch.full((1,), 12345.0, device="mps")
    _reduce_nan_policy_kernel[(1,)](
        x.to("mps"), omax, omin, BLOCK=BLOCK, num_warps=num_warps,
    )
    torch.mps.synchronize()
    got_max, got_min = omax.cpu().item(), omin.cpu().item()
    if nan_at is None:
        assert got_max == x.max().item()
        assert got_min == x.min().item()
    else:
        assert math.isnan(got_max), f"max fold dropped the NaN: {got_max}"
        assert math.isnan(got_min), f"min fold dropped the NaN: {got_min}"


@triton.jit
def _minmax_nan_policy_half_kernel(X, Y, MAX_ALL, MIN_ALL, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(X + offs)
    y = tl.load(Y + offs)
    tl.store(MAX_ALL + offs, tl.maximum(x, y, propagate_nan=tl.PropagateNan.ALL))
    tl.store(MIN_ALL + offs, tl.minimum(x, y, propagate_nan=tl.PropagateNan.ALL))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("num_warps", [1, 4])
def test_minmax_propagate_nan_all_half_widths(dtype, num_warps):
    """The NaN-propagating extremum is emitted as a template helper, so the
    narrow float widths have to instantiate it rather than fall back to the
    number-preferring builtin."""
    pairs = [(_NAN, 2.0), (2.0, _NAN), (_NAN, _NAN), (1.0, 4.0)] * 8
    n = len(pairs)
    lhs = torch.tensor([p[0] for p in pairs], dtype=dtype)
    rhs = torch.tensor([p[1] for p in pairs], dtype=dtype)
    max_all = torch.zeros(n, dtype=dtype, device="mps")
    min_all = torch.zeros(n, dtype=dtype, device="mps")
    _minmax_nan_policy_half_kernel[(1,)](
        lhs.to("mps"), rhs.to("mps"), max_all, min_all, BLOCK=n,
        num_warps=num_warps,
    )
    torch.mps.synchronize()
    got_max = max_all.float().cpu()
    got_min = min_all.float().cpu()
    has_nan = torch.tensor([math.isnan(a) or math.isnan(b) for a, b in pairs])
    assert torch.equal(torch.isnan(got_max), has_nan)
    assert torch.equal(torch.isnan(got_min), has_nan)
    finite = ~has_nan
    torch.testing.assert_close(got_max[finite], torch.full((finite.sum(),), 4.0))
    torch.testing.assert_close(got_min[finite], torch.full((finite.sum(),), 1.0))


@triton.jit
def _reduce_rank2_axis0_kernel(X, OMAX, OMIN, M: tl.constexpr, N: tl.constexpr):
    rows = tl.arange(0, M)[:, None]
    cols = tl.arange(0, N)[None, :]
    x = tl.load(X + rows * N + cols)
    tl.store(OMAX + tl.arange(0, N), tl.reduce(x, 0, _reduce_max_all))
    tl.store(OMIN + tl.arange(0, N), tl.reduce(x, 0, _reduce_min_all))


@pytest.mark.parametrize("num_warps", [1, 4])
def test_reduce_rank2_axis0_propagate_nan_all_returns_nan(num_warps):
    """The rank-2 axis-0 planner builds its own fold, separately from the
    rank-1 butterfly. A NaN must reach exactly the column that holds it and
    leave every other column bit-exact."""
    M, N = 8, 16
    x = torch.arange(M * N, dtype=torch.float32).reshape(M, N)
    x[3, 5] = _NAN
    out_max = torch.zeros(N, device="mps")
    out_min = torch.zeros(N, device="mps")
    _reduce_rank2_axis0_kernel[(1,)](
        x.to("mps"), out_max, out_min, M=M, N=N, num_warps=num_warps,
    )
    torch.mps.synchronize()
    got_max, got_min = out_max.cpu(), out_min.cpu()
    only_col5 = torch.zeros(N, dtype=torch.bool)
    only_col5[5] = True
    assert torch.equal(torch.isnan(got_max), only_col5)
    assert torch.equal(torch.isnan(got_min), only_col5)
    keep = ~only_col5
    torch.testing.assert_close(got_max[keep], x.amax(dim=0)[keep], rtol=0, atol=0)
    torch.testing.assert_close(got_min[keep], x.amin(dim=0)[keep], rtol=0, atol=0)


@triton.jit
def _reduce_rank3_kernel(X, O, A: tl.constexpr, B: tl.constexpr, C: tl.constexpr):
    a = tl.arange(0, A)[:, None, None]
    b = tl.arange(0, B)[None, :, None]
    c = tl.arange(0, C)[None, None, :]
    x = tl.load(X + a * (B * C) + b * C + c)
    r = tl.reduce(x, 1, _reduce_max_all)
    out_a = tl.arange(0, A)[:, None]
    out_c = tl.arange(0, C)[None, :]
    tl.store(O + out_a * C + out_c, r)


@pytest.mark.parametrize("num_warps", [1, 4])
def test_reduce_rank3_staged_propagate_nan_all_returns_nan(num_warps):
    """Rank > 2 takes the staged planner, which publishes the tile to
    threadgroup memory and walks the reduced axis. Its fold needs the same NaN
    policy as the rank-1 and rank-2 ones."""
    A, B, C = 4, 8, 4
    x = torch.arange(A * B * C, dtype=torch.float32).reshape(A, B, C)
    x[2, 3, 1] = _NAN
    out = torch.zeros(A * C, device="mps")
    _reduce_rank3_kernel[(1,)](
        x.to("mps"), out, A=A, B=B, C=C, num_warps=num_warps,
    )
    torch.mps.synchronize()
    got = out.cpu().reshape(A, C)
    only_cell = torch.zeros((A, C), dtype=torch.bool)
    only_cell[2, 1] = True
    assert torch.equal(torch.isnan(got), only_cell)
    keep = ~only_cell
    torch.testing.assert_close(got[keep], x.amax(dim=1)[keep], rtol=0, atol=0)


# ---------------------------------------------------------------------------
# `tl.clamp` float widths. MSL spells clamp as `fmin(fmax(x, lo), hi)`, which
# carries f32 and half natively; bf16 has no overload, so the emitter converts
# the promoted float result back. f64 keeps a named pre-conversion rejection.
# ---------------------------------------------------------------------------


@triton.jit
def _clamp_widths_kernel(X, LO, HI, OUT_NONE, OUT_ALL, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(X + offs)
    lo = tl.load(LO)
    hi = tl.load(HI)
    tl.store(OUT_NONE + offs, tl.clamp(x, lo, hi))
    tl.store(OUT_ALL + offs,
             tl.clamp(x, lo, hi, propagate_nan=tl.PropagateNan.ALL))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("num_warps", [1, 4])
def test_clamp_float_widths_and_nan_policy(dtype, num_warps):
    """f16 clamp used to be rejected outright; this is the width that actually
    reaches `metal.clampf` in f16. Triton widens a bf16 `tl.clamp` to f32 in the
    frontend, so the bf16 row here proves the user-visible result, not the
    narrow emitter path (that one is pinned by the metal-translate lit). Both
    NaN policies must hold at every width: ALL returns NaN for a NaN input,
    NONE lets the number-preferring min/max consume it."""
    # A power of two, as tl.arange requires. Covers both bounds exactly, the
    # two signed zeros, values outside on each side, both infinities and NaN.
    values = [-5.0, -2.0, -1.0, -0.5, -0.25, -0.0, 0.0, 0.25,
              0.5, 1.0, 2.0, 5.0, _NAN, _INF, -_INF, _NAN]
    n = len(values)
    x = torch.tensor(values, dtype=dtype)
    lo = torch.tensor([-1.0], dtype=dtype)
    hi = torch.tensor([1.0], dtype=dtype)
    out_none = torch.zeros(n, dtype=dtype, device="mps")
    out_all = torch.zeros(n, dtype=dtype, device="mps")
    _clamp_widths_kernel[(1,)](
        x.to("mps"), lo.to("mps"), hi.to("mps"), out_none, out_all,
        BLOCK=n, num_warps=num_warps,
    )
    torch.mps.synchronize()
    got_none = out_none.float().cpu()
    got_all = out_all.float().cpu()
    is_nan_in = torch.tensor([math.isnan(v) for v in values])

    # torch.clamp propagates NaN, so it is the oracle for the ALL policy.
    expected_all = torch.clamp(x.float(), -1.0, 1.0)
    torch.testing.assert_close(got_all, expected_all, rtol=0, atol=0,
                               equal_nan=True)
    assert torch.equal(torch.isnan(got_all), is_nan_in)

    # Triton leaves the NONE result for a NaN input unspecified beyond "not a
    # NaN", which is exactly what the upstream conformance test asserts.
    assert not torch.isnan(got_none).any()
    keep = ~is_nan_in
    torch.testing.assert_close(got_none[keep], expected_all[keep],
                               rtol=0, atol=0)
