//===--- MetalQuantizedHelpers.h - MLX-compatible quantized helpers --*-C++-*-===//
//
// This source file is part of the metal-dialect open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//
//
// Single source of truth for MLX-compatible packed-weight parameters shared by
// the verifier (`lib/metal/IR/MetalOps.cpp` QmvOp/QmmOp) and the MSL emitter
// (`lib/metal/Target/ModuleTranslation.cpp` translate(QmvOp)/translate(QmmOp)).
//
// Mirrors MLX 0.31.2 `quantized.h` `get_pack_factor` and `get_bytes_per_pack`
// byte-for-byte. See `docs/MLX_QUANTIZED_LAYOUT.md` for the locked layout.
//
//===----------------------------------------------------------------------===//

#ifndef METAL_METAL_QUANTIZED_HELPERS_H
#define METAL_METAL_QUANTIZED_HELPERS_H

#include "mlir/IR/Types.h"
#include <cstdint>

namespace mlir {
namespace triton { namespace metal {

// Number of weight values packed per container unit (one uint32 for {2,4,8};
// one byte-stream chunk of bytes_per_pack(bits) bytes for {3,5,6}).
inline int64_t packFactor(int64_t bits) {
  switch (bits) {
  case 2: return 16;
  case 4: return 8;
  case 8: return 4;
  case 3: case 5: case 6: return 8;
  default: return 0;
  }
}

// Bytes per packed group of pack_factor values. uint32-aligned (=4) for
// {2,4,8}; byte-stream-aligned ({3,5,6}) for the rest.
inline int64_t bytesPerPack(int64_t bits) {
  switch (bits) {
  case 2: case 4: case 8: return 4;
  case 3: return 3;
  case 5: return 5;
  case 6: return 6;
  default: return 0;
  }
}

inline bool isPow2(int64_t v) { return v > 0 && (v & (v - 1)) == 0; }

inline bool isValidBits(int64_t bits) {
  return bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6 ||
         bits == 8;
}

// Open-form `group_size` predicate (T1 from consensus plan).
inline bool isValidGroupSize(int64_t bits, int64_t groupSize, int64_t k) {
  if (groupSize < 32) return false;
  if (!isPow2(groupSize)) return false;
  if (groupSize % packFactor(bits) != 0) return false;
  if (k % groupSize != 0) return false;
  return true;
}

// `Wq` element type is bits-dependent: 32-bit int for {2,4,8}, 8-bit int for
// {3,5,6}. Either signedness accepted.
inline bool isWqElementTypeFor(mlir::Type t, int64_t bits) {
  auto intTy = llvm::dyn_cast<mlir::IntegerType>(t);
  if (!intTy) return false;
  switch (bits) {
  case 2: case 4: case 8:
    return intTy.getWidth() == 32;
  case 3: case 5: case 6:
    return intTy.getWidth() == 8;
  default:
    return false;
  }
}

// Human-readable form for verifier diagnostics.
inline const char *wqElementTypeName(int64_t bits) {
  switch (bits) {
  case 2: case 4: case 8: return "ui32/i32";
  case 3: case 5: case 6: return "ui8/i8";
  default: return "<invalid>";
  }
}

} } // namespace metal, triton
} // namespace mlir

#endif // METAL_METAL_QUANTIZED_HELPERS_H
