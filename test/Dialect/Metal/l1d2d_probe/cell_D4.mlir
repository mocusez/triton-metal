// L1d2d cell D4: B1=non-identity, B2=barrier-absent, B3=in-warp (16 threads).
// 4x4 transpose: idx = (lid%4)*4 + lid/4.
// Expected: ? (axis disambiguation).

module {
  metal.module {
    metal.kernel cell_D4 address_space_device [true, true] {
    ^bb0(%in: !metal.memref<? x f32>, %out: !metal.memref<? x f32>):
      %c4  = metal.constant 4 : ui32
      %lid = metal.thread_id "x" : ui32
      %buf = metal.threadgroup_alloca : !metal.memref<16 x f32>

      %v = metal.get_element %in[%lid] : (!metal.memref<? x f32>, ui32) -> f32
      metal.tg_store_indexed %buf[%lid], %v : !metal.memref<16 x f32>, ui32, f32

      // B2 = absent

      // B1 = non-identity: idx = (lid % 4) * 4 + (lid / 4)
      %col = metal.binary_exp %lid, %c4, remOp : (ui32, ui32) -> ui32
      %row = metal.binary_exp %lid, %c4, divOp : (ui32, ui32) -> ui32
      %col_x_4 = metal.binary_exp %col, %c4, mulOp : (ui32, ui32) -> ui32
      %idx = metal.binary_exp %col_x_4, %row, addOp : (ui32, ui32) -> ui32

      %w = metal.tg_load_indexed %buf[%idx] : (!metal.memref<16 x f32>, ui32) -> f32
      metal.barrier
      metal.store %w, %out[%lid] : f32, !metal.memref<? x f32>, ui32
      metal.return
    }
    metal.module_end
  }
}
