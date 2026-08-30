// RUN: triton-metal-opt %s --convert-metal-to-llvm | FileCheck %s

// AC5 positive lowering fixture: metal.print %mem, %n : memref<?xf16>, i64
// must lower to llvm.call @_MetalPrintF16(ptr, i64). The runtime function
// declaration is inserted at the top of the module before the user function.

// CHECK: llvm.func @_MetalPrintF16(!llvm.ptr, i64)
// CHECK-LABEL: llvm.func @print_f16
// CHECK: llvm.extractvalue {{.*}}[1]
// CHECK: llvm.call @_MetalPrintF16

func.func @print_f16(%mem: memref<?xf16>, %n: i64) {
  metal.print %mem, %n : memref<?xf16>, i64
  return
}
