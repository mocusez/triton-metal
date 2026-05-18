// RUN: triton-metal-translate --help | FileCheck %s
//
// In-process Metal translate registers only the MSL translation by
// design — `registerAllTranslations()` would pull in symbols not linked
// into libtriton's universe (deserialize-spirv, import-llvm,
// mlir-to-llvmir, serialize-spirv). Narrowed acceptance vs the
// upstream metal-translate.
// CHECK: --mlir-to-msl
