"""Verify the Metal backend reports an Apple-GPU family tag, not the legacy
CUDA SM_80 literal that shipped earlier.

Acceptance bar from the implementation notes:
- backend == "metal", warp_size == 32 (hardware-invariant)
- arch is the Apple-family tag (driver._APPLE_GPU_FAMILY), a stable integer in
  the Apple-family range. The live native probe was removed with the legacy
  runtime, so arch is now a constant cache-key tag (codegen is decoupled from
  it — see compiler._TTGPUIR_PARSER_STUB_ARCH).
- arch is never 80 (regression gate against re-introducing the CUDA SM_80
  literal).
"""
from __future__ import annotations

from triton.backends.compiler import GPUTarget
from triton.backends.metal.driver import MetalDriver, _APPLE_GPU_FAMILY


def _target() -> GPUTarget:
    return MetalDriver().get_current_target()


def test_target_backend_and_warp_size():
    target = _target()
    assert isinstance(target, GPUTarget)
    assert target.backend == "metal"
    assert target.warp_size == 32


def test_arch_is_apple_family():
    target = _target()
    assert target.arch == _APPLE_GPU_FAMILY
    # Apple families currently span 1..9; 20 is generous headroom.
    assert 1 <= target.arch <= 20, (
        f"Apple GPU family tag out of expected range: {target.arch}"
    )


def test_arch_is_not_cuda_sm_80():
    """Regression gate: catch any future revert to the literal arch=80."""
    assert _target().arch != 80, (
        "GPUTarget.arch == 80 looks like a revert to the CUDA SM_80 literal"
    )
