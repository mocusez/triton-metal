// L1d2d cell D5: B1=non-identity, B2=barrier-absent, B3=cross-warp (64 threads).
// 8x8 transpose: idx = (lid%8)*8 + lid/8.
// Historical translation-only diagnostic: the cross-lane read is unsynchronized.
// Do not use numerical dispatches as a correctness oracle; D7 is synchronized.

module {
  metal.module {
    metal.kernel cell_D5 address_space_device [true, true] {
    ^bb0(%in: !metal.memref<? x f32>, %out: !metal.memref<? x f32>):
      %c8  = metal.constant 8 : ui32
      %lid = metal.thread_id "x" : ui32
      %buf = metal.threadgroup_alloca : !metal.memref<64 x f32>

      %v = metal.get_element %in[%lid] : (!metal.memref<? x f32>, ui32) -> f32
      metal.tg_store_indexed %buf[%lid], %v : !metal.memref<64 x f32>, ui32, f32

      // B2 = absent: SIMD execution does not supply shared-memory synchronization.

      // B1 = non-identity: idx = (lid % 8) * 8 + (lid / 8)
      %col = metal.binary_exp %lid, %c8, remOp : (ui32, ui32) -> ui32
      %row = metal.binary_exp %lid, %c8, divOp : (ui32, ui32) -> ui32
      %col_x_8 = metal.binary_exp %col, %c8, mulOp : (ui32, ui32) -> ui32
      %idx = metal.binary_exp %col_x_8, %row, addOp : (ui32, ui32) -> ui32

      %w = metal.tg_load_indexed %buf[%idx] : (!metal.memref<64 x f32>, ui32) -> f32
      metal.barrier
      metal.store %w, %out[%lid] : f32, !metal.memref<? x f32>, ui32
      metal.return
    }
    metal.module_end
  }
}
