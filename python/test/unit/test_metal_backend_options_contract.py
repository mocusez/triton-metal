"""Compile-option contract for the Metal backend (Phase 1 of
metal-num-stages-pipelining-plan.md).

parse_options must no longer silently drop options the user deliberately set
that Metal cannot honor. Each unsupported-but-set option is surfaced (warned)
or rejected (ValueError); honored options and defaults stay quiet; the returned
MetalOptions is unchanged in value.

Pure-Python (no GPU / no rebuild): exercises parse_options directly, plus one
end-to-end triton.compile to confirm the wiring survives the full compile path.
Skipped when the Metal pybind module isn't linked.
"""

from __future__ import annotations

import warnings

import pytest

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget

pytest.importorskip(
    "triton._C.libtriton.metal",
    reason="Metal backend pybind module not built into libtriton",
)

import triton.backends.metal.compiler as metal_compiler  # noqa: E402
from triton.backends.metal.compiler import MetalBackend, MetalOptions  # noqa: E402


def _backend() -> MetalBackend:
    return MetalBackend(GPUTarget(backend="metal", arch=80, warp_size=32))


def _metal_warnings(records) -> list[str]:
    return [str(r.message) for r in records if "[triton-metal]" in str(r.message)]


@pytest.fixture(autouse=True)
def _reset_contract_dedup():
    # Warnings dedup once-per-field across the whole process; clear it so each
    # test observes the warning it expects.
    metal_compiler._OPT_CONTRACT_WARNED.clear()
    yield
    metal_compiler._OPT_CONTRACT_WARNED.clear()


# (field, non-default value) — one representative unhonorable value per WARN field.
_WARN_CASES = [
    ("num_stages", 3),
    ("num_ctas", 2),
    ("enable_fp_fusion", False),
    ("launch_cooperative_grid", True),
    ("maxnreg", 128),
    ("extern_libs", {"libdevice": "/x"}),
    ("max_num_imprecise_acc_default", 1),
    ("instrumentation_mode", "profile"),
]


@pytest.mark.parametrize("field,value", _WARN_CASES)
def test_unsupported_option_warns_but_is_accepted(field, value):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        opts = _backend().parse_options({field: value})
    metal = _metal_warnings(rec)
    assert len(metal) == 1, f"expected exactly one warning for {field}, got: {metal}"
    assert field in metal[0]
    # Behavior preserved: the value is still recorded on the options object.
    assert getattr(opts, field) == value


def test_honored_and_default_options_are_quiet():
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        # num_warps honored; num_stages/enable_fp_fusion at their defaults.
        opts = _backend().parse_options(
            {"num_warps": 8, "num_stages": 1, "enable_fp_fusion": True})
    assert _metal_warnings(rec) == []
    assert opts.num_warps == 8
    assert opts.num_stages == 1


def test_empty_container_option_is_not_a_warning():
    # extern_libs={} is "unset", not a request we cannot honor.
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        _backend().parse_options({"extern_libs": {}})
    assert _metal_warnings(rec) == []


def test_warning_is_deduplicated_across_compiles():
    b = _backend()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        b.parse_options({"num_stages": 3})
        b.parse_options({"num_stages": 5})  # second call: already warned
    assert len(_metal_warnings(rec)) == 1


def test_warp_size_non_32_is_rejected():
    with pytest.raises(ValueError, match="warp_size"):
        _backend().parse_options({"warp_size": 16})
    # 32 (the only valid value) is accepted silently.
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        opts = _backend().parse_options({"warp_size": 32})
    assert _metal_warnings(rec) == []
    assert opts.warp_size == 32


def test_unknown_key_dropped_by_default_raises_under_strict(monkeypatch):
    monkeypatch.delenv("TRITON_METAL_STRICT_OPTIONS", raising=False)
    # Default: unknown keys are silently dropped (no regression).
    opts = _backend().parse_options({"num_warps": 4, "not_a_real_option": 7})
    assert opts.num_warps == 4
    # Strict: unknown keys raise for debugging.
    monkeypatch.setenv("TRITON_METAL_STRICT_OPTIONS", "1")
    with pytest.raises(ValueError, match="Unknown compile option"):
        _backend().parse_options({"not_a_real_option": 7})


def test_contract_covers_every_option_field():
    # The import-time assertion already guards this; assert here too so the
    # intent is visible and a regression fails a test, not just an import.
    classified = (set(metal_compiler._OPT_WARN_UNSUPPORTED)
                  | set(metal_compiler._OPT_REJECT_UNSUPPORTED)
                  | metal_compiler._OPT_HONORED)
    assert classified == set(MetalOptions.__dataclass_fields__)


@triton.jit
def _add_kernel(x_ptr, y_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
    offs = tl.arange(0, BLOCK_SIZE)
    tl.store(out_ptr + offs, tl.load(x_ptr + offs) + tl.load(y_ptr + offs))


def test_end_to_end_num_stages_warns_but_compiles():
    from triton.compiler import ASTSource
    src = ASTSource(
        fn=_add_kernel,
        signature={"x_ptr": "*fp32", "y_ptr": "*fp32", "out_ptr": "*fp32",
                   "BLOCK_SIZE": "constexpr"},
        constexprs={"BLOCK_SIZE": 128},
    )
    target = GPUTarget(backend="metal", arch=80, warp_size=32)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        compiled = triton.compile(
            src, target=target, options={"num_warps": 4, "num_stages": 2})
    # The unsupported num_stages is surfaced, and compilation still succeeds.
    assert any("num_stages" in m for m in _metal_warnings(rec))
    assert "metal" in compiled.asm
