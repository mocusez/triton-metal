// RUN: triton-metal-opt %s --convert-metal-to-llvm | FileCheck %s

// AC5 positive lowering fixture: metal.print %mem, %n : memref<?xbf16>, i64
// must lower to llvm.call @_MetalPrintBF16(ptr, i64). The runtime function
// declaration is inserted at the top of the module before the user function.

// CHECK: llvm.func @_MetalPrintBF16(!llvm.ptr, i64)
// CHECK-LABEL: llvm.func @print_bf16
// CHECK: llvm.extractvalue {{.*}}[1]
// CHECK: llvm.call @_MetalPrintBF16

func.func @print_bf16(%mem: memref<?xbf16>, %n: i64) {
  metal.print %mem, %n : memref<?xbf16>, i64
  return
}
