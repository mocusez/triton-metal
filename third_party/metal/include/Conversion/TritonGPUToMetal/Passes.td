//===--- Passes.td - TritonGPU → Metal pass definitions ----------*- tablegen -*-===//
//
// Pass definition for the TritonGPU-to-Metal conversion pipeline.
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

    The implementation covers the backend's validated elementwise, memory,
    control-flow, reduction/scan, layout-exchange, atomic, and dot envelopes.
    Unsupported shapes are rejected by preflight validation before full
    dialect conversion so callers receive a deterministic diagnostic.
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
