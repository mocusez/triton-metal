// Metal backend nanobind init.
//
// `init_triton_metal` is invoked from python/src/main.cc as
// `init_triton_metal(sub)` for `sub = m.def_submodule("metal")`, so the bindings below
// land on the Python side as `triton._C.libtriton.metal.<symbol>`. The
// MetalBackend in python/triton/backends/metal/compiler.py drives the
// MSL stage through `load_dialects` + `ttgir_to_msl`.
//
// See the implementation notes for the
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

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <string>

namespace py = nanobind;

void init_triton_metal(py::module_ &m) {
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
  // geometry should follow `num_warps` as usual. A kernel consisting of one
  // scheduled attention op reports the exact geometry chosen by the emitter.
  // Keeping this decision shared with ModuleTranslation prevents launch
  // metadata from drifting from the generated MSL.
  m.def("ttgir_to_msl", [](mlir::ModuleOp &mod) -> py::tuple {
    mlir::PassManager pm(mod.getContext());
    pm.addPass(mlir::triton::metal::createConvertTritonGPUToMetalPass());
    if (mlir::failed(pm.run(mod))) {
      throw std::runtime_error(
          "Metal backend: convert-tritongpu-to-metal failed");
    }
    // Checked on the Metal IR, not by sniffing the MSL text: "this body is one
    // op" and its schedule are structural facts and should be read
    // structurally.
    //
    // The scheduled attention op must be the only op in the kernel that DOES
    // anything -- everything else has to be memory-effect-free. The body is
    // not literally one op: whole-kernel matchers leave behind casts and
    // `metal.get_element`s that bridge operands, and those are pure.
    int threadsPerGroup = 0;
    {
      bool anyKernel = false, allScheduled = true;
      int commonThreadsPerGroup = 0;
      mod.walk([&](mlir::triton::metal::KernelOp k) {
        anyKernel = true;
        int scheduled = 0, effectful = 0, kernelThreads = 0;
        for (mlir::Operation &o : k.getBodyRegion().front()) {
          if (auto attention =
                  mlir::dyn_cast<mlir::triton::metal::FusedAttentionOp>(o)) {
            ++scheduled;
            kernelThreads =
                mlir::triton::metal::getFusedAttentionSchedule(attention)
                    .threadsPerGroup;
          } else if (auto attention = mlir::dyn_cast<
                         mlir::triton::metal::SoftmaxAttentionBackwardPreOp>(
                         o)) {
            ++scheduled;
            kernelThreads = mlir::triton::metal::
                getAttentionBackwardThreadsPerGroup(attention);
          } else if (auto attention = mlir::dyn_cast<
                         mlir::triton::metal::SoftmaxAttentionBackwardDqOp>(
                         o)) {
            ++scheduled;
            kernelThreads = mlir::triton::metal::
                getAttentionBackwardThreadsPerGroup(attention);
          } else if (auto attention = mlir::dyn_cast<
                         mlir::triton::metal::SoftmaxAttentionBackwardDkdvOp>(
                         o)) {
            ++scheduled;
            kernelThreads = mlir::triton::metal::
                getAttentionBackwardThreadsPerGroup(attention);
          } else if (!o.hasTrait<mlir::OpTrait::IsTerminator>() &&
                   !mlir::isMemoryEffectFree(&o))
            ++effectful;
        }
        if (scheduled != 1 || effectful != 0) {
          allScheduled = false;
          return;
        }
        if (commonThreadsPerGroup == 0)
          commonThreadsPerGroup = kernelThreads;
        else if (commonThreadsPerGroup != kernelThreads)
          allScheduled = false;
      });
      if (anyKernel && allScheduled)
        threadsPerGroup = commonThreadsPerGroup;
    }

    // Debug-record message table. `tl.device_print` / `tl.device_assert` write
    // an INDEX into this table from the device; the launcher needs the strings
    // to format what it reads back, and its presence is also how the launcher
    // knows to bind the trailing debug buffer at all.
    py::list debugMessages;
    if (auto messages =
            mod->getAttrOfType<mlir::ArrayAttr>("metal.debug_messages"))
      for (mlir::Attribute attr : messages) {
        // nanobind's `str` has no std::string ctor; pass (data, size) so an
        // embedded NUL in a user format string cannot truncate the message.
        llvm::StringRef s = mlir::cast<mlir::StringAttr>(attr).getValue();
        debugMessages.append(py::str(s.data(), s.size()));
      }

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
