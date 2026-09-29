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

    // `acc * alpha + bias[col]`: the body sees each element's global row and
    // column, and its captures (a runtime scalar, a buffer) are bound once.
    metal.kernel simdgroup_map_bias address_space_device [true, true, true, true] {
    ^bb0(%a_buf: !metal.memref<? x f32>, %c_buf: !metal.memref<? x f32>, %bias: !metal.memref<? x f32>, %alpha_buf: !metal.memref<? x f32>):
      %zero_u = metal.constant 0 : ui32
      %row = metal.constant 16 : ui32
      %col = metal.constant 8 : ui32
      %stride = metal.constant 8 : ui32
      %alpha = metal.get_element %alpha_buf[%zero_u] : (!metal.memref<? x f32>, ui32) -> f32
      %acc = metal.simdgroup_load %a_buf[%zero_u, %zero_u], %stride : (!metal.memref<? x f32>, ui32, ui32, ui32) -> !metal.simdgroup_matrix<8 x 8 x f32>
      %mapped = metal.simdgroup_map %acc origin [%row, %col] captures(%alpha, %bias : f32, !metal.memref<? x f32>) : !metal.simdgroup_matrix<8 x 8 x f32> -> !metal.simdgroup_matrix<8 x 8 x f32> {
      ^bb0(%x: f32, %r: si32, %c: si32, %s: f32, %b: !metal.memref<? x f32>):
        %cu = metal.cast %c : (si32) -> ui32
        %bv = metal.get_element %b[%cu] : (!metal.memref<? x f32>, ui32) -> f32
        %scaled = metal.binary_exp %x, %s, mulOp : (f32, f32) -> f32
        %y = metal.binary_exp %scaled, %bv, addOp : (f32, f32) -> f32
        metal.simdgroup_map_yield %y : f32
      }
      metal.simdgroup_store %mapped, %c_buf[%zero_u, %zero_u], %stride : !metal.simdgroup_matrix<8 x 8 x f32>, !metal.memref<? x f32>, ui32, ui32, ui32
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

// CHECK: kernel void simdgroup_map_bias
// CHECK: simdgroup_float8x8 v[[OUT2:[0-9]+]] = v{{[0-9]+}};
// Captures are bound once, before the element loop.
// CHECK: auto v[[S:[0-9]+]] = {{.*}};
// CHECK: auto v[[B:[0-9]+]] = v{{[0-9]+}};
// The lane's row, and its first column; the element index adds to the column.
// CHECK: ushort v[[L:[0-9]+]] = (ushort)(id.x & 31u);
// CHECK: int v[[R:[0-9]+]] = (int)({{.*}}16{{.*}}) + (((v[[L]] >> 2) & 4) + ((v[[L]] >> 1) & 3));
// CHECK: int v[[C0:[0-9]+]] = (int)({{.*}}8{{.*}}) + (((v[[L]] >> 2) & 2) * 2 + (v[[L]] & 1) * 2);
// CHECK: for (ushort v[[E2:[0-9]+]] = 0; v[[E2]] < 2; ++v[[E2]]) {
// CHECK: int v[[C:[0-9]+]] = v[[C0]] + v[[E2]];
// CHECK: v[[B]][
// CHECK: v[[OUT2]].thread_elements()[v[[E2]]] = v{{[0-9]+}};
// CHECK: simdgroup_store(v[[OUT2]],
