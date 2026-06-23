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
  m.def("ttgir_to_msl", [](mlir::ModuleOp &mod) -> std::string {
    mlir::PassManager pm(mod.getContext());
    pm.addPass(mlir::triton::metal::createConvertTritonGPUToMetalPass());
    if (mlir::failed(pm.run(mod))) {
      throw std::runtime_error(
          "Metal backend: convert-tritongpu-to-metal failed");
    }
    std::string buffer;
    llvm::raw_string_ostream os(buffer);
    if (mlir::failed(
            mlir::triton::metal::ModuleTranslation::translateModule(mod, os))) {
      throw std::runtime_error(
          "Metal backend: MLIR -> MSL translation failed");
    }
    return buffer;
  });

  // The Metal backend is MPS-only: kernels are launched via
  // torch.mps.compile_shader (zero-copy on PyTorch MPS tensors), so this
  // module exposes ONLY the compile path (load_dialects + ttgir_to_msl).
  // The legacy native runtime (MTLDevice/MTLBuffer alloc + copy + dispatch,
  // metallib compile) was removed — see the MetalLauncher MPS path in
  // backends/metal/driver.py.
}
