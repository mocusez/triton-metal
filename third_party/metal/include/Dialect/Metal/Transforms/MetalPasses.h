//===--- MetalPasses.h - Metal passes ---------------------------*- C++ -*-===//
//
// This source file is part of the metal-dialect open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef METAL_METALPASSES_H
#define METAL_METALPASSES_H

#include "Dialect/Metal/IR/MetalDialect.h"
#include "Dialect/Metal/IR/MetalOps.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace triton { namespace metal {
#define GEN_PASS_DECL
#include "Dialect/Metal/Transforms/MetalPasses.h.inc"

#define GEN_PASS_REGISTRATION
#include "Dialect/Metal/Transforms/MetalPasses.h.inc"
} } // namespace metal, triton
} // namespace mlir

#endif // METAL_METALPASSES_H
