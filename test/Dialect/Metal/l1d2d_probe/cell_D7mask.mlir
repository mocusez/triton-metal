// L1d2d cell D7+mask: D7 axes + downstream masked-store wrap.
// Hand-crafted reproduction of L1d2c Phase A's C6 MSL shape:
//   tg_store; barrier; let v3=tg_load[transpose_idx]; barrier; if(mask){ out[lid]=v3; }
// The mask is `lid < 64`, runtime-but-uniformly-true at threadgroup_size=64.
// This extension probe tests whether L1d2's failure requires the `if(mask)`
// wrap downstream of the trailing barrier (Phase A's RCA hypothesis).

module {
  metal.module {
    metal.kernel cell_D7mask address_space_device [true, true] {
    ^bb0(%in: !metal.memref<? x f32>, %out: !metal.memref<? x f32>):
      %c8   = metal.constant 8 : ui32
      %c64  = metal.constant 64 : ui32
      %lid = metal.thread_id "x" : ui32
      %buf = metal.threadgroup_alloca : !metal.memref<64 x f32>

      %v = metal.get_element %in[%lid] : (!metal.memref<? x f32>, ui32) -> f32
      metal.tg_store_indexed %buf[%lid], %v : !metal.memref<64 x f32>, ui32, f32

      metal.barrier

      // Non-identity 8x8 transpose: idx = (lid%8)*8 + lid/8
      %col = metal.binary_exp %lid, %c8, remOp : (ui32, ui32) -> ui32
      %row = metal.binary_exp %lid, %c8, divOp : (ui32, ui32) -> ui32
      %col_x_8 = metal.binary_exp %col, %c8, mulOp : (ui32, ui32) -> ui32
      %idx = metal.binary_exp %col_x_8, %row, addOp : (ui32, ui32) -> ui32

      %w = metal.tg_load_indexed %buf[%idx] : (!metal.memref<64 x f32>, ui32) -> f32
      metal.barrier

      // Runtime-but-uniformly-true mask (lid < 64 always true at threadgroup=64).
      // Avoids constant folding while exercising the `if(mask){devstore}` shape.
      %mask = metal.binary_exp %lid, %c64, ltOp : (ui32, ui32) -> i1
      scf.if %mask {
        metal.store %w, %out[%lid] : f32, !metal.memref<? x f32>, ui32
      }
      metal.return
    }
    metal.module_end
  }
}
