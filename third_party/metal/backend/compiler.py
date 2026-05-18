"""Metal compiler backend: @triton.jit -> TTIR -> TTGIR -> MSL text.

The pipeline stops at MSL string (no .metallib, no GPU dispatch). The
TTGIR -> Metal dialect -> MSL stage runs in-process via the pybind
bindings in `third_party/metal/triton_metal.cc`, exposed as
`triton._C.libtriton.metal.{load_dialects, ttgir_to_msl}`. See
`.omc/specs/deep-interview-metal-jit-to-msl-text.md`.
"""

import re
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
    supported_fp8_dtypes: tuple = ()
    deprecated_fp8_dot_operand_dtypes: tuple = ()
    default_dot_input_precision: str = "ieee"
    allowed_dot_input_precisions: tuple = ("ieee",)
    enable_fp_fusion: bool = True
    launch_cooperative_grid: bool = False
    max_num_imprecise_acc_default: int = 0
    maxnreg: int = None
    backend_name: str = "metal"
    instrumentation_mode: str = ""
    ir_override: str = None

    def __post_init__(self):
        assert self.num_warps > 0 and (self.num_warps & (self.num_warps - 1)) == 0, (
            "num_warps must be a power of 2")

    def hash(self):
        # Compact, version-bumpable identity for the options dataclass.
        return "metal-stub-v1"


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
# Decoupled from GPUTarget.arch, which carries the real Apple GPU family
# integer (1-9, via Runtime.mm:getAppleGpuFamily). Upstream parser sites:
# Prefetch.cpp:736, Utility.cpp:149, Utility.cpp:1097.
_TTGPUIR_PARSER_STUB_ARCH = 80
_TTGPUIR_PARSER_STUB_TRIPLE = f"cuda:{_TTGPUIR_PARSER_STUB_ARCH}"


class MetalBackend(BaseBackend):
    """Metal backend: TTIR -> TTGIR -> MSL text -> .metallib (terminal).

    Terminal artifact depends on platform: `.metallib` on Darwin (when
    the libmetal runtime is built in); otherwise the MSL text remains
    the terminal stage.
    """

    binary_ext = "metallib" if hasattr(libmetal, "compile_msl_to_metallib") else "msl"

    @staticmethod
    def supports_target(target: GPUTarget) -> bool:
        return target.backend == "metal"

    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)
        self.binary_ext = type(self).binary_ext

    def hash(self) -> str:
        return f"metal-{self.target.arch}-{self.target.warp_size}"

    def parse_options(self, opts: dict) -> Any:
        args = {k: v for k, v in opts.items() if k in MetalOptions.__dataclass_fields__}
        return MetalOptions(**args)

    def get_module_map(self) -> Dict[str, ModuleType]:
        # No backend-specific libdevice for v1. Triton's core libdevice is
        # already on the import path; not overriding it is the correct stub.
        return {}

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
        metadata.setdefault("shared", 0)
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
        msl = libmetal.ttgir_to_msl(mod)
        # CompiledKernel.__init__ requires metadata["name"]; mirror how
        # nvidia's make_cubin pulls the entry-point name out of the
        # produced artifact.
        match = re.search(r"kernel\s+void\s+(\w+)", msl)
        if match is None:
            raise RuntimeError(
                "Metal backend: no `kernel void <name>` found in MSL output")
        metadata["name"] = match.group(1)
        return msl

    @staticmethod
    def make_metallib(msl, metadata, options):
        # MSL -> .metallib via the Darwin-only pybind runtime
        # (libmetal.compile_msl_to_metallib invokes xcrun). On non-Darwin
        # this stage is absent so triton.compile stops at MSL.
        return libmetal.compile_msl_to_metallib(msl)

    def add_stages(self, stages, options, language=None):
        stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
        stages["ttgir"] = lambda src, metadata: self.make_ttgir(src, metadata, options)
        stages["msl"] = lambda src, metadata: self.make_msl(src, metadata, options)
        # The metallib stage requires the Darwin-only runtime layer in
        # libtriton (xcrun + Metal framework). On non-Darwin builds the
        # symbol is absent and the compile pipeline stops at MSL.
        if hasattr(libmetal, "compile_msl_to_metallib"):
            stages["metallib"] = lambda src, metadata: self.make_metallib(src, metadata, options)

    def pack_metadata(self, metadata):
        return (metadata.num_warps, metadata.num_ctas, getattr(metadata, "shared", 0))

    def get_codegen_implementation(self, options):
        # Apple SIMD-group matrix multiply operates on 8x8 fp32 tiles
        # natively, so the minimum dot shape is (8, 8, 8). Required by the
        # Triton frontend's tl.dot shape gate. See
        # `.omc/specs/deep-interview-metal-matmul-session5-pytest.md`.
        return {
            "min_dot_size": lambda lhs_type, rhs_type: (8, 8, 8),
        }
