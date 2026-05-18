// RUN: triton-metal-translate --mlir-to-msl %s | FileCheck %s

module {
  metal.module {
    metal.kernel mlx_elementwise address_space_device [false, false, false, true] {
    ^bb0(%arg0: !metal.memref<? x f32>, %arg1: !metal.memref<? x f32>, %arg2: !metal.memref<? x f32>, %arg3: !metal.memref<? x f32>):
      %id = metal.thread_id "x" : ui32
      %a = metal.get_element %arg0[%id] : (!metal.memref<? x f32>, ui32) -> f32
      %b = metal.get_element %arg1[%id] : (!metal.memref<? x f32>, ui32) -> f32
      %c = metal.get_element %arg2[%id] : (!metal.memref<? x f32>, ui32) -> f32
      %neg = metal.unary_exp %c, minusOp : (f32) -> f32
      %exp = metal.unary_exp %a, expOp : (f32) -> f32
      %sqrt = metal.unary_exp %b, sqrtOp : (f32) -> f32
      %mul = metal.binary_exp %sqrt, %neg, mulOp : (f32, f32) -> f32
      %out = metal.binary_exp %exp, %mul, addOp : (f32, f32) -> f32
      metal.store %out, %arg3[%id] : f32, !metal.memref<? x f32>, ui32
      metal.return
    }
    metal.module_end
  }
}

// CHECK: #include <metal_stdlib>
// CHECK: #include <metal_math>
// CHECK: using namespace metal;
// CHECK: kernel void mlx_elementwise(
// CHECK: constant float *v0 {{\[\[buffer\(0\)\]\]}},
// CHECK: constant float *v1 {{\[\[buffer\(1\)\]\]}},
// CHECK: constant float *v2 {{\[\[buffer\(2\)\]\]}},
// CHECK: device float *v3 {{\[\[buffer\(3\)\]\]}},
// CHECK: uint3 id {{\[\[thread_position_in_grid\]\]}})
// CHECK: v3[id.x] = (metal::precise::exp(v0[id.x])) + ((metal::precise::sqrt(v1[id.x])) * ((-v2[id.x])));
