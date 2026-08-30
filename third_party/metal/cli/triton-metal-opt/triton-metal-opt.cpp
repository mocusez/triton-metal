//===--- triton-metal-opt.cpp ------------------------------------*- C++ -*-===//
//
// Narrow mlir-opt driver scoped to the metal dialect + its in-tree passes.
// Mirrors `reference/metal-dialect/metal-translate/metal-translate.cpp` shape
// but for mlir-opt. AC2 acceptance per
// the implementation notes.
//
//===----------------------------------------------------------------------===//

#include "Conversion/TritonGPUToMetal/Passes.h"
#include "Dialect/Metal/IR/MetalDialect.h"
#include "Dialect/Metal/Transforms/MetalPasses.h"

// Triton's own dialects, needed so AC3 fixtures can parse TTGIR.
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  mlir::triton::metal::registerMetalConversionPasses();
  mlir::triton::metal::registerTritonGPUToMetalPasses();

  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect,
                  mlir::cf::ControlFlowDialect,
                  mlir::func::FuncDialect,
                  mlir::LLVM::LLVMDialect,
                  mlir::linalg::LinalgDialect,
                  mlir::memref::MemRefDialect,
                  mlir::scf::SCFDialect,
                  mlir::triton::TritonDialect,
                  mlir::triton::gpu::TritonGPUDialect,
                  mlir::triton::metal::MetalDialect>();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "triton-metal-opt driver\n", registry));
}
