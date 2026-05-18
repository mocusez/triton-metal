// RUN: triton-metal-translate --mlir-to-msl %s 2>&1 | FileCheck %s
//
// Scaffolding for Matmul Track Session 2: assert that the three new
// `metal.simdgroup_*` ops emit the canonical MSL function calls
// (`simdgroup_load_matrix`, `simdgroup_matrix_multiply_accumulate`,
// `simdgroup_store_matrix`). No conversion pattern uses these ops yet;
// the lit fixture round-trips a hand-written `metal.kernel` body.
// See `.omc/specs/deep-interview-metal-matmul-session2-simdgroup-scaffold.md`.

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
// CHECK: simdgroup_float8x8 {{.*}} = simdgroup_load_matrix
// CHECK: simdgroup_float8x8 {{.*}} = simdgroup_load_matrix
// CHECK: simdgroup_float8x8 {{.*}} = simdgroup_load_matrix
// CHECK: simdgroup_float8x8 {{.*}} = simdgroup_matrix_multiply_accumulate
// CHECK: simdgroup_store_matrix
// CHECK: return
