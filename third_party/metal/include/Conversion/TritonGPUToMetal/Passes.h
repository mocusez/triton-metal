//===--- Passes.h - TritonGPU → Metal pass declarations ---------*- C++ -*-===//
//
// Scaffold-only per the implementation notes. The real
// conversion patterns (handling `tt.func`, `tt.get_program_id`,
// `tt.make_range`, `tt.splat`, `tt.addptr`, `tt.load`, `arith.addf`,
// `tt.store`, and the `#ttg.blocked<>` distributed layout) land in a
// follow-up session.
//
//===----------------------------------------------------------------------===//

#ifndef TRITONGPUTOMETAL_PASSES_H
#define TRITONGPUTOMETAL_PASSES_H

// Include dialect headers BEFORE the generated Passes.h.inc expansion
// below: the GEN_PASS_DECL macros reference `mlir::memref::MemRefDialect`
// (etc.) by full name, so the namespaces must be visible at the point of
// generated-code expansion.
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace triton {
namespace metal {

#define GEN_PASS_DECL
#include "Conversion/TritonGPUToMetal/Passes.h.inc"

// Forward declaration is required because GEN_PASS_REGISTRATION below
// expands a `register*Pass()` helper that calls `createConvertTritonGPUToMetalPass()`.
// The real definition lives in lib/Conversion/TritonGPUToMetal/TritonGPUToMetal.cpp.
std::unique_ptr<mlir::Pass> createConvertTritonGPUToMetalPass();

#define GEN_PASS_REGISTRATION
#include "Conversion/TritonGPUToMetal/Passes.h.inc"

} // namespace metal
} // namespace triton
} // namespace mlir

#endif // TRITONGPUTOMETAL_PASSES_H
