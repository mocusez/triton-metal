// RUN: triton-metal-opt --convert-funcs-to-metal-kernels %s | FileCheck %s

// AC-W4: annotated func containing a linalg op is wrapped, with
// unrealized_conversion_cast bridges so the linalg op (which expects
// standard memref operands) still type-checks until
// convert-linalg-to-metal folds the casts away.

func.func @k(%a: memref<4x8xf32>, %b: memref<4x8xf32>) attributes {metal.kernel} {
  linalg.softmax dimension(1)
      ins(%a : memref<4x8xf32>)
      outs(%b : memref<4x8xf32>)
  return
}

// CHECK:       metal.module {
// CHECK:       metal.kernel k address_space_device [true, true]
// CHECK:       ^bb0(%[[A:.*]]: !metal.memref<32 x f32>, %[[B:.*]]: !metal.memref<32 x f32>):
// CHECK:       %[[CA:.*]] = builtin.unrealized_conversion_cast %[[A]] : !metal.memref<32 x f32> to memref<4x8xf32>
// CHECK:       %[[CB:.*]] = builtin.unrealized_conversion_cast %[[B]] : !metal.memref<32 x f32> to memref<4x8xf32>
// CHECK:       linalg.softmax
