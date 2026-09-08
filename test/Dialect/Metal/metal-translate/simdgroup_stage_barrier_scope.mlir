// RUN: triton-metal-translate --mlir-to-msl %s 2>&1 | FileCheck %s
//
// The staged load exists because chaining `simdgroup_multiply_accumulate` on
// values loaded straight from `const device T*` silently drops output columns
// on Apple GPU family 9; the operands have to come from threadgroup memory.
// That staging is bracketed by fences, and their SCOPE is what this pins.
//
// When every staged load in a kernel carries a warp index, `_stage_shared` is
// declared `[num_warps][elems]` and each simdgroup writes and reads only its
// own slice. No other simdgroup can observe it, so a simdgroup-scoped fence is
// sufficient and a threadgroup-wide barrier would needlessly serialize the
// other simdgroups — measured at roughly a sixth of the runtime of a large
// matmul, whose inner loop is one 8x8 multiply-accumulate per staging pass.
//
// A SINGLE staged load without a warp index takes that back: it reads
// `_stage_shared[0]`, which every simdgroup shares, so all fences in that
// kernel must stay threadgroup-wide.

module attributes {"ttg.num-warps" = 4 : i32} {
  metal.module {
    // Every staged load is warp-indexed: simdgroup-scoped fences.
    metal.kernel all_per_warp address_space_device [true, true, true] {
    ^bb0(%a: !metal.memref<? x f32>, %b: !metal.memref<? x f32>, %c: !metal.memref<? x f32>):
      %z = metal.constant 0 : ui32
      %s = metal.constant 8 : ui32
      %w = metal.simdgroup_index : ui32
      %ta = metal.simdgroup_load_device_staged %a[%z, %z], %s warp [%w] : (!metal.memref<? x f32>, ui32, ui32, ui32, ui32) -> !metal.simdgroup_matrix<8 x 8 x f32>
      %tb = metal.simdgroup_load_device_staged %b[%z, %z], %s warp [%w] : (!metal.memref<? x f32>, ui32, ui32, ui32, ui32) -> !metal.simdgroup_matrix<8 x 8 x f32>
      %acc = metal.simdgroup_matrix_zero : !metal.simdgroup_matrix<8 x 8 x f32>
      %r = metal.simdgroup_multiply_accumulate %acc, %ta, %tb : (!metal.simdgroup_matrix<8 x 8 x f32>, !metal.simdgroup_matrix<8 x 8 x f32>, !metal.simdgroup_matrix<8 x 8 x f32>) -> !metal.simdgroup_matrix<8 x 8 x f32>
      metal.simdgroup_store %r, %c[%z, %z], %s : !metal.simdgroup_matrix<8 x 8 x f32>, !metal.memref<? x f32>, ui32, ui32, ui32
      metal.return
    }
    // One staged load WITHOUT a warp index shares slice zero across every
    // simdgroup, so the fences stay threadgroup-wide for the whole kernel.
    metal.kernel mixed_scope address_space_device [true, true, true] {
    ^bb0(%a: !metal.memref<? x f32>, %b: !metal.memref<? x f32>, %c: !metal.memref<? x f32>):
      %z = metal.constant 0 : ui32
      %s = metal.constant 8 : ui32
      %w = metal.simdgroup_index : ui32
      %ta = metal.simdgroup_load_device_staged %a[%z, %z], %s warp [%w] : (!metal.memref<? x f32>, ui32, ui32, ui32, ui32) -> !metal.simdgroup_matrix<8 x 8 x f32>
      %tb = metal.simdgroup_load_device_staged %b[%z, %z], %s : (!metal.memref<? x f32>, ui32, ui32, ui32) -> !metal.simdgroup_matrix<8 x 8 x f32>
      %acc = metal.simdgroup_matrix_zero : !metal.simdgroup_matrix<8 x 8 x f32>
      %r = metal.simdgroup_multiply_accumulate %acc, %ta, %tb : (!metal.simdgroup_matrix<8 x 8 x f32>, !metal.simdgroup_matrix<8 x 8 x f32>, !metal.simdgroup_matrix<8 x 8 x f32>) -> !metal.simdgroup_matrix<8 x 8 x f32>
      metal.simdgroup_store %r, %c[%z, %z], %s : !metal.simdgroup_matrix<8 x 8 x f32>, !metal.memref<? x f32>, ui32, ui32, ui32
      metal.return
    }
  }
}

// CHECK-LABEL: kernel void all_per_warp
// CHECK: threadgroup float _stage_shared[4][64]
// CHECK: simdgroup_barrier(mem_flags::mem_threadgroup)
// CHECK: simdgroup_load(v{{[0-9]+}}, &_stage_shared[
// CHECK-NOT: threadgroup_barrier

// CHECK-LABEL: kernel void mixed_scope
// CHECK: threadgroup_barrier(mem_flags::mem_threadgroup)
// CHECK-NOT: simdgroup_barrier
