//===--- LinalgToMetal.h -----------------------------------------*- C++ -*-===//
//
// This source file is part of the metal-dialect open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef METAL_LINALGTOMETAL_H
#define METAL_LINALGTOMETAL_H

#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/ArrayRef.h"

#include <optional>

namespace mlir::triton::metal {

/// Computes the product of `shape` as a checked flat size for
/// MetalMemRefType (which stores size as `unsigned`). Returns std::nullopt
/// if any dim is dynamic (negative), if the product overflows uint64_t, or
/// if the product exceeds `unsigned`'s range. Shared between the wrapping
/// pass and the linalg→metal TypeConverter so both compute identical sizes.
std::optional<unsigned> safeFlatSize(llvm::ArrayRef<int64_t> shape);

/// Register MemRef -> MetalMemRef conversions on `typeConverter`. Standard
/// memrefs with fully static shape and element type in {f16, f32, bf16} are
/// converted to flat MetalMemRefType. All other cases return failure.
void populateLinalgToMetalTypeConversions(mlir::TypeConverter &typeConverter);

/// Register conversion patterns for the four supported linalg named ops:
/// softmax, reduce (sum/max/mean), matmul, matvec.
void populateLinalgToMetalConversionPatterns(
    mlir::TypeConverter &typeConverter, mlir::RewritePatternSet &patterns);

} // namespace mlir::triton::metal

#endif // METAL_LINALGTOMETAL_H
