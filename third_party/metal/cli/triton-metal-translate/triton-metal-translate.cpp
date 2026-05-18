//===--- triton-metal-translate.cpp ------------------------------*- C++ -*-===//
//
// mlir-translate driver that registers the metal dialect + the ToMSL
// translation. AC2 acceptance per
// `.omc/specs/deep-interview-ac2-ac3-start.md`.
//
//===----------------------------------------------------------------------===//

#include "Dialect/Metal/IR/MetalDialect.h"

#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"

namespace mlir {
void registerToMSLTranslation();
} // namespace mlir

int main(int argc, char **argv) {
  // Narrowly register only the metal-dialect MSL translation. Triton's
  // vendored LLVM ships a partial set of translation libraries
  // (registerAllTranslations() pulls in symbols not linked into
  // libtriton's universe), so we deliberately stay minimal here.
  mlir::registerToMSLTranslation();

  return failed(mlir::mlirTranslateMain(
      argc, argv, "Triton Metal Translation Testing Tool"));
}
