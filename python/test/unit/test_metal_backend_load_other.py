"""Masked tt.load with constant and runtime-uniform `other` operands.

Asserts that `tl.load(ptr, mask=m, other=splat_const)` on the elementwise
path lowers correctly and produces bit-exact agreement with the CPU
reference `torch.where(mask, x, other_scalar)`. The masked-off tail must
carry the user-provided `other` value (not the v1 hardcoded zero).

Also covers a dynamic scalar expression (`other - 1`) broadcast by Triton as a
uniform `tt.splat`, including the signless-i32 to ui32 Metal storage bridge.

See `.omc/specs/deep-interview-leet-triton-l1-refine-and-ship.md`.
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
def masked_load_with_other_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    OTHER: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=OTHER)
    tl.store(output_ptr + offsets, x)


@pytest.mark.parametrize("BLOCK_SIZE", [128, 1024])
@pytest.mark.parametrize("OTHER", [0.0, 0.5, -3.25])
def test_masked_load_with_other_bit_exact(BLOCK_SIZE, OTHER):
    # Leave a non-empty masked tail so the `other` path is observed.
    N = BLOCK_SIZE - 24
    torch.manual_seed(0xC0FFEE)
    x = torch.rand(BLOCK_SIZE, dtype=torch.float32)
    out = torch.zeros(BLOCK_SIZE, dtype=torch.float32)

    masked_load_with_other_kernel[(1, 1, 1)](
        x, out, N, OTHER=OTHER, BLOCK_SIZE=BLOCK_SIZE
    )

    mask = torch.arange(BLOCK_SIZE) < N
    expected = torch.where(
        mask, x, torch.full_like(x, OTHER, dtype=torch.float32)
    )
    # bit-exact: the kernel just passes x through (no FP ops on the masked
    # path other than the splat-constant select).
    torch.testing.assert_close(out, expected, atol=0, rtol=0)


@triton.jit
def _masked_load_with_dynamic_other_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    other,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=other - 1)
    tl.store(output_ptr + offsets, x)


def test_masked_load_with_dynamic_i32_other():
    """A uniform runtime scalar may supply a masked load's `other` value."""
    BLOCK_SIZE = 1024
    N = 1000
    other = 7
    x_cpu = torch.arange(N, dtype=torch.int32)
    x = x_cpu.to("mps")
    out = torch.zeros(BLOCK_SIZE, dtype=torch.int32, device="mps")

    _masked_load_with_dynamic_other_kernel[(1,)](
        x, out, N, other, BLOCK_SIZE=BLOCK_SIZE
    )
    torch.mps.synchronize()

    expected = torch.full((BLOCK_SIZE,), other - 1, dtype=torch.int32)
    expected[:N] = x_cpu
    torch.testing.assert_close(out.cpu(), expected, atol=0, rtol=0)


@triton.jit
def _masked_uniform_block1_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    OTHER: tl.constexpr,
):
    """Force BLOCK=1 so the tensor address folds to a splat scalar pointer."""
    offset = tl.program_id(0) + tl.arange(0, 1)
    mask = offset < n_elements
    x = tl.load(x_ptr + offset, mask=mask, other=OTHER)
    tl.atomic_add(output_ptr, tl.sum(x, axis=0))


@pytest.mark.parametrize(
    "grid, n_elements, other",
    [(1, 1, 0.0), (2, 1, 0.5), (3, 3, -3.25)],
)
def test_masked_uniform_block1_load(grid, n_elements, other):
    """Regression for BLOCK=1 masked loads whose ptr is `tt.splat`."""
    x = torch.arange(1, n_elements + 1, dtype=torch.float32, device="mps")
    out = torch.zeros(1, dtype=torch.float32, device="mps")

    _masked_uniform_block1_kernel[(grid,)](
        x, out, n_elements, OTHER=other
    )
    torch.mps.synchronize()

    expected = x.cpu().sum().item() + (grid - n_elements) * other
    torch.testing.assert_close(out.cpu()[0].item(), expected, atol=0, rtol=0)


# Lmultiload Phase B audit canary: existing tests in this file only
# exercise the canonical `offs = pid*BLOCK + arange` shape. This canary
# adds a divergent constant offset (`offs + 1`) so any future regression
# of the offset-arithmetic path would surface here. Pre-Phase-B this
# test would have failed (the index ignored `+ 1`).
@triton.jit
def _canary_const_offset_kernel(x_ptr, output_ptr, n_elements,
                                K: tl.constexpr,
                                BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = (offs + K) < n_elements
    x = tl.load(x_ptr + offs + K, mask=mask, other=0.0)
    tl.store(output_ptr + offs, x, mask=offs < (n_elements - K))


def test_canary_divergent_const_offset():
    BLOCK_SIZE = 128
    K = 5
    N = 256
    torch.manual_seed(0xC0FFEE)
    x = torch.arange(N, dtype=torch.float32)
    out = torch.zeros(N, dtype=torch.float32)
    _canary_const_offset_kernel[(N // BLOCK_SIZE, 1, 1)](
        x, out, N, K=K, BLOCK_SIZE=BLOCK_SIZE
    )
    expected = torch.zeros(N, dtype=torch.float32)
    expected[: N - K] = x[K:]
    torch.testing.assert_close(out, expected, atol=0, rtol=0)
