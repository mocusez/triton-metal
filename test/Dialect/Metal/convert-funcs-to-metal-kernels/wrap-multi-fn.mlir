// RUN: triton-metal-opt --convert-funcs-to-metal-kernels %s | FileCheck %s

// AC-W3: only annotated funcs are wrapped; others survive untouched.

func.func @k_a(%a: memref<4xf32>, %b: memref<4xf32>) attributes {metal.kernel} {
  return
}

func.func @k_b(%a: memref<4xf32>, %b: memref<4xf32>) {
  return
}

// CHECK:       metal.module {
// CHECK:       metal.kernel k_a address_space_device [true, true]
// CHECK:       func.func @k_b(
