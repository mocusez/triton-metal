//===--- MetalDialect.cpp - Metal dialect ---------------------------------===//
//
// This source file is part of the metal-dialect open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "Dialect/Metal/IR/MetalDialect.h"
#include "Dialect/Metal/IR/MetalOps.h"
#include "Dialect/Metal/IR/MetalTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::triton::metal;

#include "Dialect/Metal/IR/MetalOpsDialect.cpp.inc"

void MetalDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Dialect/Metal/IR/MetalOps.cpp.inc"
      >();
  registerTypes();
}

mlir::Operation *MetalDialect::materializeConstant(mlir::OpBuilder &builder,
                                                   mlir::Attribute value,
                                                   mlir::Type type,
                                                   mlir::Location loc) {
  return mlir::triton::metal::ConstantOp::create(builder, loc, cast<TypedAttr>(value));
}
