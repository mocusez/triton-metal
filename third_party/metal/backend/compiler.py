"""Metal compiler backend: @triton.jit -> TTIR -> TTGIR -> MSL text.

The pipeline stops at MSL string (no .metallib, no GPU dispatch). The
TTGIR -> Metal dialect -> MSL stage runs in-process via the pybind
bindings in `third_party/metal/triton_metal.cc`, exposed as
`triton._C.libtriton.metal.{load_dialects, ttgir_to_msl}`. See
the implementation notes.
"""

import os
import re
import warnings
from dataclasses import dataclass
from typing import Any, Dict
from types import ModuleType

from triton.backends.compiler import BaseBackend, GPUTarget
from triton._C.libtriton import ir, passes, metal as libmetal


@dataclass(frozen=True)
class MetalOptions:
    num_warps: int = 4
    num_ctas: int = 1
    num_stages: int = 1
    warp_size: int = 32
    extern_libs: dict = None
    debug: bool = False
    sanitize_overflow: bool = True
    # Uninitialized sentinel. The real Apple GPU family is supplied by
    # the driver via GPUTarget.arch (see driver._get_apple_gpu_family).
    # A `metal-0-32` cache key would mean MetalOptions was constructed
    # without a target context — fix the caller in that case.
    arch: int = 0
    # Required by Triton's compile loop / metadata even if unused at v1.
    # E4M3FN/E5M2 are accepted only by the exact software tt.dot_scaled path.
    # They are byte-backed in the Metal ABI and decoded to bf16 before
    # arithmetic.
    supported_fp8_dtypes: tuple = ("fp8e4nv", "fp8e5")
    deprecated_fp8_dot_operand_dtypes: tuple = ()
    default_dot_input_precision: str = "ieee"
    allowed_dot_input_precisions: tuple = ("ieee",)
    enable_fp_fusion: bool = True
    launch_cooperative_grid: bool = False
    max_num_imprecise_acc_default: int = 0
    maxnreg: int = None
    backend_name: str = "metal"
    instrumentation_mode: str = ""
    # Injected unconditionally by jit.py from knobs.compilation, so the field
    # has to exist even though Metal runs no sanitizer passes. It stays at the
    # default unless the user sets TRITON_FPSAN_HOMOMORPHIC_CASTS, which is
    # exactly when the contract layer below should speak up.
    fpsan_homomorphic_casts: bool = False
    ir_override: str = None

    def __post_init__(self):
        assert self.num_warps > 0 and (self.num_warps & (self.num_warps - 1)) == 0, (
            "num_warps must be a power of 2")

    def hash(self):
        # AC4 v6: include num_warps (and other fields that drive code
        # emission) so the on-disk cache key correctly distinguishes
        # nw=1/2/4 compiles of the same source. Pre-AC4 this returned a
        # constant stub, which made every multi-warp kernel collide with
        # the first nw=1 compile in the in-process cache.
        return (
            f"metal-stub-v4-nw{self.num_warps}-nctas{self.num_ctas}"
            f"-ns{self.num_stages}-ws{self.warp_size}-arch{self.arch}"
            f"-precfp{self.default_dot_input_precision}"
            f"-fpfuse{int(self.enable_fp_fusion)}"
            f"-saniof{int(self.sanitize_overflow)}"
        )


# ---------------------------------------------------------------------------
# Compile-option contract (Phase 1 of metal-num-stages-pipelining-plan.md).
#
# Historically parse_options silently filtered `opts` to MetalOptions fields and
# dropped everything else, INCLUDING options the user deliberately set that the
# Metal backend does not implement (num_stages, maxnreg, ...). A user passing
# `num_stages=3` believed they got software pipelining; they got a no-op. This
# layer makes every unsupported-but-set option loud instead of silent.
#
# Each MetalOptions field is classified below. The import-time assertion forces
# any newly-added field to be classified here too — no silent fall-through.
#
#   WARN   : setting a non-default value is accepted but ignored -> one warning.
#   REJECT : setting a non-default value is a hard error (ValueError).
#   HONORED: consumed by codegen / the frontend / core, or an internal field.
#
# "non-default" == "differs from the MetalOptions default", which is exactly the
# value that produces the backend's actual behavior. See jit.py:698
# (`backend.parse_options(kwargs)` receives ONLY user-set launch kwargs, so a
# warning fires precisely when the user explicitly asked for the behavior).
_OPT_WARN_UNSUPPORTED = {
    "num_stages": "Metal has no cp.async analog; a Phase 0 spike proved register "
                  "prefetch pipelining is a no-op on Apple MPS (0.95-1.02x). The "
                  "loop runs unpipelined; results are unchanged.",
    "num_ctas": "Metal has no multi-CTA cluster hardware; only num_ctas=1 works.",
    "enable_fp_fusion": "MSL controls fp contraction at its own compile stage; "
                        "disabling fusion from Triton is not implemented.",
    "launch_cooperative_grid": "Metal has no cooperative-grid launch.",
    "maxnreg": "Metal (MSL text) exposes no per-thread register cap.",
    "extern_libs": "The Metal backend loads no external libdevice; core libdevice "
                   "maps to MSL intrinsics (get_module_map returns {}).",
    "max_num_imprecise_acc_default": (
        "Metal has no hardware fp8 dot path; the exact E5M2 scaled-dot path is software-only."
    ),
    "instrumentation_mode": "Metal runs no instrumentation passes.",
    "fpsan_homomorphic_casts": "Metal runs no fp sanitizer, so there are no "
                               "homomorphic casts to instrument.",
}
_OPT_REJECT_UNSUPPORTED = {
    "warp_size": "Apple GPUs have a fixed SIMD width of 32; warp_size must be 32 "
                 "(codegen is decoupled from it — see driver._get_apple_gpu_family).",
}
# Consumed by codegen (num_warps, arch, dot precision), the frontend
# (sanitize_overflow), Triton core (debug, ir_override), or internal metadata.
_OPT_HONORED = {
    "num_warps", "arch", "debug", "sanitize_overflow", "default_dot_input_precision",
    "allowed_dot_input_precisions", "supported_fp8_dtypes",
    "deprecated_fp8_dot_operand_dtypes", "backend_name", "ir_override",
}
assert (set(_OPT_WARN_UNSUPPORTED) | set(_OPT_REJECT_UNSUPPORTED) | _OPT_HONORED) == \
    set(MetalOptions.__dataclass_fields__), (
        "Metal compile-option contract is out of sync with MetalOptions: "
        f"{set(MetalOptions.__dataclass_fields__).symmetric_difference(set(_OPT_WARN_UNSUPPORTED) | set(_OPT_REJECT_UNSUPPORTED) | _OPT_HONORED)}")

# Once-per-field dedup across compiles (autotuner calls parse_options per Config;
# the user should see each ignored-option warning once, not per config).
_OPT_CONTRACT_WARNED: set = set()


def _opt_value_is_set(val, default) -> bool:
    """True when `val` meaningfully deviates from `default`. None and empty
    containers count as "unset" so e.g. extern_libs={} does not warn."""
    if val == default:
        return False
    if val is None:
        return False
    try:
        if len(val) == 0:
            return False
    except TypeError:
        pass
    return True


def _enforce_option_contract(opts: dict) -> None:
    """Reject / warn on user-set options the Metal backend cannot honor.

    REJECT first (hard errors take priority over warnings). Unknown keys are
    dropped as before, unless TRITON_METAL_STRICT_OPTIONS=1, which raises on
    them for debugging."""
    fields = MetalOptions.__dataclass_fields__
    for name, reason in _OPT_REJECT_UNSUPPORTED.items():
        if name in opts and _opt_value_is_set(opts[name], fields[name].default):
            raise ValueError(
                f"[triton-metal] Unsupported compile option `{name}={opts[name]!r}`: "
                f"{reason}")
    for name, reason in _OPT_WARN_UNSUPPORTED.items():
        if name in _OPT_CONTRACT_WARNED:
            continue
        if name in opts and _opt_value_is_set(opts[name], fields[name].default):
            _OPT_CONTRACT_WARNED.add(name)
            warnings.warn(
                f"[triton-metal] Compile option `{name}={opts[name]!r}` is not "
                f"honored by the Metal backend and will be ignored: {reason}",
                stacklevel=3)
    if os.environ.get("TRITON_METAL_STRICT_OPTIONS", "0") == "1":
        unknown = sorted(set(opts) - set(fields))
        if unknown:
            raise ValueError(
                f"[triton-metal] Unknown compile option(s) {unknown} "
                "(TRITON_METAL_STRICT_OPTIONS=1).")


# Upstream TritonGPU passes only parse target strings prefixed with "cuda:"
# or "hip:". The post-prefix integer is parsed as a real SM compute
# capability by getNVIDIAComputeCapability (lib/Dialect/TritonGPU/Transforms/
# Utility.cpp:1083), and several passes branch on it. Apple GPUs are not
# Nvidia hardware; this stub exists purely to satisfy the upstream parser.
#
# Why 80 is acceptable HERE (and not universally safe):
#   The Metal `make_ttgir` pipeline (this file, see make_ttgir) only runs
#   add_convert_to_ttgpuir, add_coalesce, add_remove_layout_conversions,
#   add_optimize_thread_locality, canonicalizer, cse, symbol_dce. It does
#   NOT invoke add_accelerate_matmul (which has `if (computeCapability <
#   80)` at AccelerateMatmul.cpp:386 and would activate Ampere MMA codegen
#   for cuda:80) or add_prefetch (which dispatches via Prefetch.cpp:735 on
#   cuda: targets). The remaining passes consume capability transitively
#   through getNumWarpsPerCTA etc., where the only relevant gate is `>=90`
#   (Hopper-only, Utility.cpp:152), which 80 cleanly avoids.
#
# DELIBERATE INVARIANT: if the Metal pipeline ever adds matmul/prefetch
# passes — or any pass that branches on a capability boundary <= 80 — this
# stub value MUST be re-evaluated. Sweep with:
#   grep -nr "getNVIDIAComputeCapability\|computeCapability" \
#       lib/Dialect/TritonGPU/Transforms/
# and confirm every gate is either unreached from the Metal pipeline or
# selects the conservative branch at 80.
#
# Decoupled from GPUTarget.arch, which carries the Apple GPU family tag
# (driver._get_apple_gpu_family). Upstream parser sites:
# Prefetch.cpp:736, Utility.cpp:149, Utility.cpp:1097.
_TTGPUIR_PARSER_STUB_ARCH = 80
_TTGPUIR_PARSER_STUB_TRIPLE = f"cuda:{_TTGPUIR_PARSER_STUB_ARCH}"


class MetalBackend(BaseBackend):
    """Metal backend: TTIR -> TTGIR -> MSL text (terminal).

    MPS-only: the terminal artifact is the MSL text. The driver's load_binary
    feeds it to torch.mps.compile_shader, which compiles and dispatches on
    PyTorch MPS tensors (zero-copy). The legacy `.metallib` stage and its
    native xcrun runtime were removed.
    """

    binary_ext = "metal"

    @staticmethod
    def supports_target(target: GPUTarget) -> bool:
        return target.backend == "metal"

    def hash(self) -> str:
        return f"metal-{self.target.arch}-{self.target.warp_size}"

    def parse_options(self, opts: dict) -> Any:
        # Surface (warn) or reject user-set options the Metal backend cannot
        # honor, instead of silently dropping them. See _enforce_option_contract.
        _enforce_option_contract(opts)
        args = {k: v for k, v in opts.items() if k in MetalOptions.__dataclass_fields__}
        return MetalOptions(**args)

    def get_module_map(self) -> Dict[str, ModuleType]:
        # Redirect `triton.language.extra.libdevice` to the Metal table. The
        # generic shim in triton/language/extra/libdevice.py is bodies-only
        # (`...`), so without this every libdevice call returned None and failed
        # with "cannot convert None of type NoneType to tensor". See
        # third_party/metal/language/metal/libdevice.py for which functions
        # exist and why the rest deliberately do not.
        from triton.language.extra.metal import libdevice
        return {"triton.language.extra.libdevice": libdevice}

    def load_dialects(self, ctx):
        # Register the Metal dialect on Triton's MLIRContext so the
        # `convert-tritongpu-to-metal` pass and the MSL emitter can
        # produce / consume metal ops in-process.
        libmetal.load_dialects(ctx)

    @staticmethod
    def make_ttir(mod, metadata, options):
        # Mirror of the generic subset of nvidia/amd make_ttir
        # (no backend-specific passes).
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.common.add_inliner(pm)
        passes.ttir.add_rewrite_tensor_descriptor_to_pointer(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_combine(pm)
        passes.ttir.add_reorder_broadcast(pm)
        passes.common.add_cse(pm)
        passes.ttir.add_triton_licm(pm)
        passes.common.add_symbol_dce(pm)
        passes.ttir.add_loop_unroll(pm)
        pm.run(mod, "make_ttir")
        return mod

    @staticmethod
    def make_ttgir(mod, metadata, options):
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        # See _TTGPUIR_PARSER_STUB_TRIPLE for why this is a fixed "cuda:80"
        # rather than a Metal-native target string.
        passes.ttir.add_convert_to_ttgpuir(
            pm,
            _TTGPUIR_PARSER_STUB_TRIPLE,
            options.num_warps,
            options.warp_size,
            options.num_ctas,
        )
        passes.ttgpuir.add_coalesce(pm)
        # Preserve Coalesce's sizePerThread>1 layout across eligible rank-1
        # reduces, or promote an eligible sizePerThread=1 operand when
        # Coalesce did not. The pass has no target-attribute gate; placement
        # in this Metal-only pipeline is the gate.
        passes.ttgpuir.add_propagate_coalesced_layouts(pm)
        passes.ttgpuir.add_remove_layout_conversions(pm)
        passes.ttgpuir.add_optimize_thread_locality(pm)
        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)
        pm.run(mod, "make_ttgir")
        return mod

    @staticmethod
    def make_msl(mod, metadata, options):
        # Populate load-bearing metadata fields the standard Triton
        # launch path expects (CompiledKernel._init_handles +
        # CudaLauncher-shape attrs). We don't run shared/tmem allocation
        # passes, so these are zero.
        # Floor `shared` to 1: CUDA-style tutorials (e.g. 02-fused-softmax)
        # compute occupancy via `SIZE_SMEM // size_smem`, which zero-divides
        # when the kernel uses no threadgroup memory. We don't run a shared
        # allocator, so without the floor `shared` stays 0. See
        # the implementation notes AC5/R6/R6b.
        metadata["shared"] = max(metadata.get("shared", 0), 1)
        metadata.setdefault("tmem_size", 0)
        metadata.setdefault("global_scratch_size", 0)
        metadata.setdefault("global_scratch_align", 1)
        metadata.setdefault("profile_scratch_size", 0)
        metadata.setdefault("profile_scratch_align", 1)
        metadata.setdefault("launch_cooperative_grid", False)
        metadata.setdefault("launch_pdl", False)
        # (existing make_msl body below)
        # In-process TTGIR -> Metal dialect -> MSL text. The pybind in
        # third_party/metal/triton_metal.cc drives the conversion pass
        # and the ModuleTranslation emitter; we only forward the module.
        msl, threads_per_group, debug_messages = libmetal.ttgir_to_msl(mod)
        # `tl.device_print` / `tl.device_assert` write records into a trailing
        # buffer the launcher binds; these are the strings it formats them
        # with, and a non-empty list is also how the launcher knows the kernel
        # has that extra parameter at all.
        metadata["debug_messages"] = list(debug_messages)
        # A structurally scheduled kernel may report its exact threadgroup size
        # (attention, or a proven straight-line single-SIMD-group matrix body).
        # Record it separately rather than overwriting `num_warps`: that field
        # drove TTGIR codegen and remains part of the cache key.
        if threads_per_group:
            metadata["threads_per_group"] = threads_per_group
        # CompiledKernel.__init__ requires metadata["name"]; mirror how
        # nvidia's make_cubin pulls the entry-point name out of the
        # produced artifact.
        match = re.search(r"kernel\s+void\s+(\w+)", msl)
        if match is None:
            raise RuntimeError(
                "Metal backend: no `kernel void <name>` found in MSL output")
        metadata["name"] = match.group(1)
        return msl

    def add_stages(self, stages, options, language=None):
        stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
        stages["ttgir"] = lambda src, metadata: self.make_ttgir(src, metadata, options)
        # Terminal artifact is the MSL text; torch.mps.compile_shader (driver
        # load_binary) owns final compilation + dispatch. No metallib stage.
        stages["metal"] = lambda src, metadata: self.make_msl(src, metadata, options)

    def pack_metadata(self, metadata):
        return (metadata.num_warps, metadata.num_ctas, getattr(metadata, "shared", 0))

    def get_codegen_implementation(self, options):
        # Apple SIMD-group matrix multiply operates on 8x8 fp32 tiles
        # natively, so the minimum dot shape is (8, 8, 8). Required by the
        # Triton frontend's tl.dot shape gate. See
        # the implementation notes.
        return {
            "min_dot_size": lambda lhs_type, rhs_type: (8, 8, 8),
        }
