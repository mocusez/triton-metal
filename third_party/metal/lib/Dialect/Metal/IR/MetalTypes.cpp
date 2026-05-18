//===- MetalTypes.cpp - Metal dialect types ---------------------*- C++ -*-===//
//
// This source file is part of the Metal open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "Dialect/Metal/IR/MetalTypes.h"
#include "Dialect/Metal/IR/MetalDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir::triton::metal;

#define GET_TYPEDEF_CLASSES
#include "Dialect/Metal/IR/MetalOpsTypes.cpp.inc"

void MetalDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "Dialect/Metal/IR/MetalOpsTypes.cpp.inc"
      >();
}

mlir::Type MetalMemRefType::parse(mlir::AsmParser &parser) {
  Type type;
  if (parser.parseLess())
    return Type();

  if (mlir::succeeded(parser.parseOptionalQuestion())) {
    if (parser.parseKeyword("x") || parser.parseType(type) ||
        parser.parseGreater())
      return Type();
    return MetalMemRefType::get(parser.getContext(), type, 0);
  }

  uint32_t size;
  if (parser.parseInteger(size) || parser.parseKeyword("x") ||
      parser.parseType(type) || parser.parseGreater())
    return Type();
  return MetalMemRefType::get(parser.getContext(), type, size);
}

void MetalMemRefType::print(mlir::AsmPrinter &printer) const {
  printer << "<";
  if (getSize() > 0)
    printer << getSize();
  else
    printer << "?";
  printer << " x " << getType() << ">";
}

mlir::Type MetalSimdgroupMatrixType::parse(mlir::AsmParser &parser) {
  uint32_t rows;
  uint32_t cols;
  Type elem;
  if (parser.parseLess() || parser.parseInteger(rows) ||
      parser.parseKeyword("x") || parser.parseInteger(cols) ||
      parser.parseKeyword("x") || parser.parseType(elem) ||
      parser.parseGreater())
    return Type();
  return MetalSimdgroupMatrixType::get(parser.getContext(), rows, cols, elem);
}

void MetalSimdgroupMatrixType::print(mlir::AsmPrinter &printer) const {
  printer << "<" << getRows() << " x " << getCols() << " x " << getElem() << ">";
}
