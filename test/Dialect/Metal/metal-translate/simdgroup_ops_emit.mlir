// RUN: triton-metal-translate --mlir-to-msl %s 2>&1 | FileCheck %s
//
// iter-6 FA-derived: the three `metal.simdgroup_*` ops emit the MODERN
// Metal 17.5 MSL function calls (`simdgroup_load`,
// `simdgroup_multiply_accumulate`, `simdgroup_store`). The legacy
// `_matrix`-suffixed names (`simdgroup_load_matrix`,
// `simdgroup_matrix_multiply_accumulate`, `simdgroup_store_matrix`) are
// no longer accepted by the current MSL compiler. When both origin
// operands are literal `0` constants — as in this hand-written kernel —
// the emitter outputs the FA-precedent 3-arg form
// `simdgroup_load(dst, ptr, stride);` (declare-then-call). Non-zero
// origins fall back to 5-arg `(dst, ptr, stride, ulong2(col,row), false)`.

module {
  metal.module {
    metal.kernel simdgroup_smoke address_space_device [true, true, true] {
    ^bb0(%a_buf: !metal.memref<? x f32>, %b_buf: !metal.memref<? x f32>, %c_buf: !metal.memref<? x f32>):
      %row = metal.constant 0 : ui32
      %col = metal.constant 0 : ui32
      %stride = metal.constant 8 : ui32
      %a = metal.simdgroup_load %a_buf[%row, %col], %stride : (!metal.memref<? x f32>, ui32, ui32, ui32) -> !metal.simdgroup_matrix<8 x 8 x f32>
      %b = metal.simdgroup_load %b_buf[%row, %col], %stride : (!metal.memref<? x f32>, ui32, ui32, ui32) -> !metal.simdgroup_matrix<8 x 8 x f32>
      %c0 = metal.simdgroup_load %c_buf[%row, %col], %stride : (!metal.memref<? x f32>, ui32, ui32, ui32) -> !metal.simdgroup_matrix<8 x 8 x f32>
      %c = metal.simdgroup_multiply_accumulate %c0, %a, %b : (!metal.simdgroup_matrix<8 x 8 x f32>, !metal.simdgroup_matrix<8 x 8 x f32>, !metal.simdgroup_matrix<8 x 8 x f32>) -> !metal.simdgroup_matrix<8 x 8 x f32>
      metal.simdgroup_store %c, %c_buf[%row, %col], %stride : !metal.simdgroup_matrix<8 x 8 x f32>, !metal.memref<? x f32>, ui32, ui32, ui32
      metal.return
    }
  }
}

// CHECK: kernel void simdgroup_smoke
// CHECK: simdgroup_float8x8 v{{[0-9]+}};
// CHECK: simdgroup_load(v{{[0-9]+}},
// CHECK: simdgroup_float8x8 v{{[0-9]+}};
// CHECK: simdgroup_load(v{{[0-9]+}},
// CHECK: simdgroup_float8x8 v{{[0-9]+}};
// CHECK: simdgroup_load(v{{[0-9]+}},
// CHECK: simdgroup_float8x8 v{{[0-9]+}};
// CHECK: simdgroup_multiply_accumulate(v{{[0-9]+}},
// CHECK: simdgroup_store(
// CHECK: return
