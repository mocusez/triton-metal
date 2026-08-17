// Metal backend pybind init.
//
// `init_triton_metal` is invoked from python/src/main.cc as
// `init_triton_metal(m.def_submodule("metal"))`, so the bindings below
// land on the Python side as `triton._C.libtriton.metal.<symbol>`. The
// MetalBackend in python/triton/backends/metal/compiler.py drives the
// MSL stage through `load_dialects` + `ttgir_to_msl`.
//
// See `.omc/specs/deep-interview-metal-jit-to-msl-text.md` for the
// end-to-end story: `@triton.jit` -> TTIR -> TTGIR -> [this pybind] ->
// MSL text, in-process, no subprocess, no GPU dispatch.

#include "Conversion/TritonGPUToMetal/Passes.h"
#include "Dialect/Metal/IR/MetalDialect.h"
#include "Target/Metal/ModuleTranslation.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/raw_ostream.h"

#include <pybind11/pybind11.h>

#include <string>

namespace py = pybind11;

void init_triton_metal(py::module &&m) {
  // Register the Metal dialect on Triton's shared MLIRContext. Called
  // from MetalBackend.load_dialects(ctx) once per compile. Other
  // dialects the conversion produces (scf, arith) are already
  // registered by the core triton context bring-up.
  m.def("load_dialects", [](mlir::MLIRContext &context) {
    mlir::DialectRegistry registry;
    registry.insert<mlir::triton::metal::MetalDialect>();
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  });

  // Drive the in-process TTGIR -> Metal dialect -> MSL pipeline. Input
  // is the post-make_ttgir mlir::ModuleOp; output is the textual MSL
  // kernel. Mirrors what `triton-metal-opt --convert-tritongpu-to-metal`
  // piped into `triton-metal-translate --mlir-to-msl` produces in the
  // lit test harness.
  // Returns (msl, threads_per_group). `threads_per_group` is 0 when the launch
  // geometry should follow `num_warps` as usual, and 32 when every kernel in
  // the module is exactly one `metal.fused_attention`: that op's body runs on a
  // SINGLE warp (`ltid.x < 32`), so a source kernel written with the customary
  // `num_warps=4` would otherwise launch 128 threads and let 96 of them exit
  // immediately -- measured at ~12% on this backend, for nothing.
  m.def("ttgir_to_msl", [](mlir::ModuleOp &mod) -> py::tuple {
    mlir::PassManager pm(mod.getContext());
    pm.addPass(mlir::triton::metal::createConvertTritonGPUToMetalPass());
    if (mlir::failed(pm.run(mod))) {
      throw std::runtime_error(
          "Metal backend: convert-tritongpu-to-metal failed");
    }
    // Checked on the Metal IR, not by sniffing the MSL text: "this body is one
    // op" is a structural fact and should be read structurally.
    //
    // The test is that `metal.fused_attention` is the only op in the kernel
    // that DOES anything -- everything else has to be memory-effect-free. The
    // body is not literally one op: the matcher leaves behind the casts and
    // `metal.get_element`s that bridge its operands, and those are pure.
    int threadsPerGroup = 0;
    {
      bool anyKernel = false, allSingleWarp = true;
      mod.walk([&](mlir::triton::metal::KernelOp k) {
        anyKernel = true;
        int fused = 0, effectful = 0;
        for (mlir::Operation &o : k.getBodyRegion().front()) {
          if (mlir::isa<mlir::triton::metal::FusedAttentionOp>(o))
            ++fused;
          else if (!o.hasTrait<mlir::OpTrait::IsTerminator>() &&
                   !mlir::isMemoryEffectFree(&o))
            ++effectful;
        }
        if (fused != 1 || effectful != 0)
          allSingleWarp = false;
      });
      if (anyKernel && allSingleWarp)
        threadsPerGroup = 32;
    }

    // Debug-record message table. `tl.device_print` / `tl.device_assert` write
    // an INDEX into this table from the device; the launcher needs the strings
    // to format what it reads back, and its presence is also how the launcher
    // knows to bind the trailing debug buffer at all.
    py::list debugMessages;
    if (auto messages =
            mod->getAttrOfType<mlir::ArrayAttr>("metal.debug_messages"))
      for (mlir::Attribute attr : messages)
        debugMessages.append(
            py::str(mlir::cast<mlir::StringAttr>(attr).getValue().str()));

    std::string buffer;
    llvm::raw_string_ostream os(buffer);
    if (mlir::failed(
            mlir::triton::metal::ModuleTranslation::translateModule(mod, os))) {
      throw std::runtime_error(
          "Metal backend: MLIR -> MSL translation failed");
    }
    return py::make_tuple(buffer, threadsPerGroup, debugMessages);
  });

  // The Metal backend is MPS-only: kernels are launched via
  // torch.mps.compile_shader (zero-copy on PyTorch MPS tensors), so this
  // module exposes ONLY the compile path (load_dialects + ttgir_to_msl).
  // The legacy native runtime (MTLDevice/MTLBuffer alloc + copy + dispatch,
  // metallib compile) was removed — see the MetalLauncher MPS path in
  // backends/metal/driver.py.
}
