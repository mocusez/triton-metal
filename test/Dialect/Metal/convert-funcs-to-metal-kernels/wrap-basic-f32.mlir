// RUN: triton-metal-opt --convert-funcs-to-metal-kernels %s | FileCheck %s

// AC-W1: annotated func.func -> metal.module { metal.kernel { ... } }
// with all-device-writable address_space_device.

func.func @k(%a: memref<4xf32>, %b: memref<4xf32>) attributes {metal.kernel} {
  return
}

// CHECK:       metal.module {
// CHECK:       metal.kernel k address_space_device [true, true]
// CHECK:       ^bb0(%{{.*}}: !metal.memref<4 x f32>, %{{.*}}: !metal.memref<4 x f32>):
