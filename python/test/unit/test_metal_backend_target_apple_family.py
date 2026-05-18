"""Verify the Metal backend reports a real Apple-GPU family integer,
not the legacy CUDA SM_80 literal that shipped earlier.

Acceptance bar from `.omc/specs/deep-interview-metal-backend-cuda-fix.md`:
- backend == "metal", warp_size == 32 (hardware-invariant)
- On Darwin with a built libmetal probe: arch is a real Apple family int
- Otherwise: arch falls back to _APPLE_GPU_FAMILY_FALLBACK
- arch is never 80 on Darwin (regression gate against re-introducing the
  CUDA SM_80 literal).
"""
from __future__ import annotations

import sys

import pytest

from triton.backends.compiler import GPUTarget
from triton.backends.metal.driver import (
    MetalDriver,
    _APPLE_GPU_FAMILY_FALLBACK,
)

try:
    from triton._C.libtriton import metal as _libmetal
except Exception:  # pragma: no cover - libtriton always built in test envs
    _libmetal = None


def _target() -> GPUTarget:
    return MetalDriver().get_current_target()


def test_target_backend_and_warp_size():
    target = _target()
    assert isinstance(target, GPUTarget)
    assert target.backend == "metal"
    assert target.warp_size == 32


def test_arch_is_apple_family():
    target = _target()
    probe_available = (
        sys.platform == "darwin"
        and _libmetal is not None
        and hasattr(_libmetal, "get_apple_gpu_family")
    )
    if probe_available:
        # Apple families currently span 1..9; 20 is generous headroom.
        assert 1 <= target.arch <= 20, (
            f"Apple GPU family out of expected range: {target.arch}"
        )
    else:
        assert target.arch == _APPLE_GPU_FAMILY_FALLBACK


@pytest.mark.skipif(sys.platform != "darwin", reason="probe is Darwin-only")
def test_arch_is_not_cuda_sm_80():
    """Regression gate: catch any future revert to the literal arch=80."""
    assert _target().arch != 80, (
        "GPUTarget.arch == 80 looks like a revert to the CUDA SM_80 literal"
    )
