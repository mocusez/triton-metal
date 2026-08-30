// RUN: triton-metal-translate --mlir-to-msl %s | FileCheck %s
//
// P0c translation coverage for direct MSL spellings. The FMA check requires a
// fused call and must not pass if the emitter splits it into multiply plus add.

module {
  metal.module {
    metal.kernel direct_math_ops address_space_device [false, false, false, true, true, true] {
    ^bb0(%a_buf: !metal.memref<? x f32>, %b_buf: !metal.memref<? x f32>, %c_buf: !metal.memref<? x f32>, %out: !metal.memref<? x f32>, %clamp_none_out: !metal.memref<? x f32>, %clamp_all_out: !metal.memref<? x f32>):
      %id = metal.thread_id "x" : ui32
      %a = metal.get_element %a_buf[%id] : (!metal.memref<? x f32>, ui32) -> f32
      %b = metal.get_element %b_buf[%id] : (!metal.memref<? x f32>, ui32) -> f32
      %c = metal.get_element %c_buf[%id] : (!metal.memref<? x f32>, ui32) -> f32
      %l2 = metal.unary_exp %a, log2Op : (f32) -> f32
      %abs = metal.unary_exp %b, absOp : (f32) -> f32
      %f = metal.fma %l2, %abs, %c : (f32, f32, f32) -> f32
      %clamp_none = metal.clampf %a, %b, %c, propagate_nan = false : (f32, f32, f32) -> f32
      %clamp_all = metal.clampf %a, %b, %c, propagate_nan = true : (f32, f32, f32) -> f32
      metal.store %f, %out[%id] : f32, !metal.memref<? x f32>, ui32
      metal.store %clamp_none, %clamp_none_out[%id] : f32, !metal.memref<? x f32>, ui32
      metal.store %clamp_all, %clamp_all_out[%id] : f32, !metal.memref<? x f32>, ui32
      metal.return
    }
    metal.module_end
  }
}

// CHECK: kernel void direct_math_ops
// CHECK: metal::precise::fma(metal::precise::log2(
// CHECK-SAME: metal::fabs(
// CHECK-NOT: *{{.*}}+
// CHECK: metal::fmin(metal::fmax(
// CHECK: metal::isnan(
// CHECK-SAME: ?
// CHECK-SAME: metal::fmin(metal::fmax(
