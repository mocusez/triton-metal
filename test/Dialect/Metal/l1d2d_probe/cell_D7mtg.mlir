// L1d2d cell D7+multi-tg+mask: full L1d2c Phase A C6 MSL shape reproduction.
// Hand-crafted to match the L1d2 8x8 nw=2 anchor:
//   - Multi-threadgroup grid (2 threadgroups of 64 threads).
//   - Local index = id.x - tgid.x*64 (the actual Phase A C6 expression).
//   - Non-identity transpose: tg[(lid%8)*8 + lid/8].
//   - Trailing barrier-then-if(mask){devstore} downstream of tg_load.
// Dispatch: grid=(128,1,1), threadgroup=(64,1,1) -> 2 threadgroups.

module {
  metal.module {
    metal.kernel cell_D7mtg address_space_device [true, true] {
    ^bb0(%in: !metal.memref<? x f32>, %out: !metal.memref<? x f32>):
      %c8   = metal.constant 8 : ui32
      %c64  = metal.constant 64 : ui32
      %c128 = metal.constant 128 : ui32
      %gid = metal.thread_id "x" : ui32
      %tgid = metal.threadgroup_id "x" : ui32
      %tg_off = metal.binary_exp %tgid, %c64, mulOp : (ui32, ui32) -> ui32
      // lid = id.x - tgid.x*64  (local index inside this threadgroup)
      %lid = metal.binary_exp %gid, %tg_off, subOp : (ui32, ui32) -> ui32

      %buf = metal.threadgroup_alloca : !metal.memref<64 x f32>

      // tg_store: buf[lid] = in[gid]
      %v = metal.get_element %in[%gid] : (!metal.memref<? x f32>, ui32) -> f32
      metal.tg_store_indexed %buf[%lid], %v : !metal.memref<64 x f32>, ui32, f32

      metal.barrier

      // Non-identity 8x8 transpose: idx = (lid%8)*8 + lid/8
      %col = metal.binary_exp %lid, %c8, remOp : (ui32, ui32) -> ui32
      %row = metal.binary_exp %lid, %c8, divOp : (ui32, ui32) -> ui32
      %col_x_8 = metal.binary_exp %col, %c8, mulOp : (ui32, ui32) -> ui32
      %idx = metal.binary_exp %col_x_8, %row, addOp : (ui32, ui32) -> ui32

      %w = metal.tg_load_indexed %buf[%idx] : (!metal.memref<64 x f32>, ui32) -> f32
      metal.barrier

      // Runtime-but-uniformly-true mask: gid < 128.
      %mask = metal.binary_exp %gid, %c128, ltOp : (ui32, ui32) -> i1
      scf.if %mask {
        metal.store %w, %out[%gid] : f32, !metal.memref<? x f32>, ui32
      }
      metal.return
    }
    metal.module_end
  }
}
