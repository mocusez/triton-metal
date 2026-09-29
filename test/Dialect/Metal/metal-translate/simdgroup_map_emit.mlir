// RUN: triton-metal-translate --mlir-to-msl %s 2>&1 | FileCheck %s
//
// `metal.simdgroup_map`: a matmul's value-only epilogue (here a leaky ReLU)
// applied to an accumulator matrix while it is still in registers. Each lane
// owns two elements of an 8x8 SIMD-group matrix, reached through
// `thread_elements()`; the body is emitted once per element, and the store
// that follows is the ordinary one, untouched.

module {
  metal.module {
    metal.kernel simdgroup_map_leaky address_space_device [true, true] {
    ^bb0(%a_buf: !metal.memref<? x f32>, %c_buf: !metal.memref<? x f32>):
      %row = metal.constant 0 : ui32
      %col = metal.constant 0 : ui32
      %stride = metal.constant 8 : ui32
      %acc = metal.simdgroup_load %a_buf[%row, %col], %stride : (!metal.memref<? x f32>, ui32, ui32, ui32) -> !metal.simdgroup_matrix<8 x 8 x f32>
      %mapped = metal.simdgroup_map %acc : !metal.simdgroup_matrix<8 x 8 x f32> -> !metal.simdgroup_matrix<8 x 8 x f32> {
      ^bb0(%x: f32):
        %zero = metal.constant 0x00000000 : f32
        %slope = metal.constant 0x3C23D70A : f32
        %ge = metal.binary_exp %x, %zero, geOp : (f32, f32) -> i1
        %scaled = metal.binary_exp %slope, %x, mulOp : (f32, f32) -> f32
        %y = arith.select %ge, %x, %scaled : f32
        metal.simdgroup_map_yield %y : f32
      }
      metal.simdgroup_store %mapped, %c_buf[%row, %col], %stride : !metal.simdgroup_matrix<8 x 8 x f32>, !metal.memref<? x f32>, ui32, ui32, ui32
      metal.return
    }
  }
}

// CHECK: kernel void simdgroup_map_leaky
// CHECK: simdgroup_float8x8 v[[ACC:[0-9]+]];
// CHECK: simdgroup_load(v[[ACC]],
// The result is a copy of the input, rewritten element by element.
// CHECK: simdgroup_float8x8 v[[OUT:[0-9]+]] = v[[ACC]];
// CHECK: for (ushort v[[E:[0-9]+]] = 0; v[[E]] < 2; ++v[[E]]) {
// CHECK: float v[[X:[0-9]+]] = v[[OUT]].thread_elements()[v[[E]]];
// CHECK: bool v{{[0-9]+}} = (v[[X]]) >= (v{{[0-9]+}});
// CHECK: v[[OUT]].thread_elements()[v[[E]]] = v{{[0-9]+}};
// CHECK: }
// The store reads the mapped matrix, not the accumulator.
// CHECK: simdgroup_store(v[[OUT]],
// CHECK: return
