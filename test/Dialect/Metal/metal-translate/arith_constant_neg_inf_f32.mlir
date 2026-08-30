// RUN: triton-metal-translate --mlir-to-msl %s 2>&1 | FileCheck %s
//
// Exercises the FloatAttr → MSL macro path of `emitFloatLiteral` in
// `ModuleTranslation.cpp` via `metal.constant`. The MSL output must use
// `-INFINITY` / `INFINITY` / `NAN`; literal `-inf` / `inf` / `nan` are
// rejected by `xcrun metal`.
//
// IEEE-754 bit patterns: -inf=0xFF800000, +inf=0x7F800000, qNaN=0x7FC00000.

module {
  metal.module {
    metal.kernel inf_store address_space_device [true] {
    ^bb0(%arg0: !metal.memref<? x f32>):
      %i0 = metal.constant 0 : ui32
      %i1 = metal.constant 1 : ui32
      %i2 = metal.constant 2 : ui32
      %neg_inf = metal.constant 0xFF800000 : f32
      %pos_inf = metal.constant 0x7F800000 : f32
      %nan_val = metal.constant 0x7FC00000 : f32
      metal.store %neg_inf, %arg0[%i0] : f32, !metal.memref<? x f32>, ui32
      metal.store %pos_inf, %arg0[%i1] : f32, !metal.memref<? x f32>, ui32
      metal.store %nan_val, %arg0[%i2] : f32, !metal.memref<? x f32>, ui32
      metal.return
    }
    metal.module_end
  }
}

// Helper emits MSL macros, not `-inf` / `inf` / `nan` (which xcrun rejects).
// CHECK: kernel void inf_store
// CHECK-DAG: -INFINITY
// CHECK-DAG: INFINITY
// CHECK-DAG: NAN
// CHECK-NOT: -inf
// CHECK-NOT: nan
