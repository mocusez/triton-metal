//===--- Passes.td - TritonGPU → Metal pass definitions ----------*- tablegen -*-===//
//
// Scaffold-only per the implementation notes. The pass
// is registered but has an empty body in v1. Real conversion patterns to be
// added next.
//
//===----------------------------------------------------------------------===//

#ifndef TRITONGPUTOMETAL_PASSES
#define TRITONGPUTOMETAL_PASSES

include "mlir/Pass/PassBase.td"

def ConvertTritonGPUToMetal : Pass<"convert-tritongpu-to-metal", "mlir::ModuleOp"> {
  let summary = "Convert TritonGPU dialect IR into the metal dialect.";
  let description = [{
    Walks a TritonGPU (`ttg`) module produced by Triton's compiler and emits
    equivalent metal-dialect operations that the existing MSL emitter
    (`ModuleTranslation.cpp`) can lower to MSL.

    v1 scope (deferred to next session): handle the elemental subset needed
    for vector_add — `tt.func`, `tt.get_program_id`, `tt.make_range`,
    `tt.splat`, `tt.addptr`, `tt.load`, `arith.addf`, `tt.store`, plus the
    `#ttg.blocked<{sizePerThread=[1], threadsPerWarp=[32], warpsPerCTA=[4],
    order=[0]}>` distributed layout for `BLOCK_SIZE=1024`.

    Current state: scaffold only — pass registers itself but does nothing.
  }];

  let constructor = "mlir::triton::metal::createConvertTritonGPUToMetalPass()";

  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect",
    "mlir::memref::MemRefDialect",
    "mlir::scf::SCFDialect",
    "mlir::triton::metal::MetalDialect"
  ];
}

#endif // TRITONGPUTOMETAL_PASSES
