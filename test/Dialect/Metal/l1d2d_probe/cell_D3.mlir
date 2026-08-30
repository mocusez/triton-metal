// L1d2d cell D3: B1=identity, B2=barrier-present, B3=cross-warp (64 threads).
// Expected: PASS (control).

module {
  metal.module {
    metal.kernel cell_D3 address_space_device [true, true] {
    ^bb0(%in: !metal.memref<? x f32>, %out: !metal.memref<? x f32>):
      %lid = metal.thread_id "x" : ui32
      %buf = metal.threadgroup_alloca : !metal.memref<64 x f32>

      %v = metal.get_element %in[%lid] : (!metal.memref<? x f32>, ui32) -> f32
      metal.tg_store_indexed %buf[%lid], %v : !metal.memref<64 x f32>, ui32, f32

      metal.barrier

      // B1 = identity
      %w = metal.tg_load_indexed %buf[%lid] : (!metal.memref<64 x f32>, ui32) -> f32
      metal.barrier
      metal.store %w, %out[%lid] : f32, !metal.memref<? x f32>, ui32
      metal.return
    }
    metal.module_end
  }
}
