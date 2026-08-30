// RUN: triton-metal-opt --convert-funcs-to-metal-kernels %s | FileCheck %s

// AC-W2: func.func without `metal.kernel` attribute is left untouched.

func.func @k(%a: memref<4xf32>, %b: memref<4xf32>) {
  return
}

// CHECK-NOT:   metal.kernel
// CHECK:       func.func @k(
