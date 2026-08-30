// L1d2d cell D7+scratch: D7 + L1d2c Phase B scratch-RMW + masked devstore.
// Reproduces the actual current L1d2c Phase A C6 MSL shape (post-Phase-B
// scratch-sentinel + select-on-value regression-guard rewrite):
//   tg_store; barrier; let v9 = tg_load[transpose]; barrier;
//   let v10 = scratch[lid]; scratch[lid] = mask ? v9 : v10;
//   if(mask){ out[gid] = v9; }

module {
  metal.module {
    metal.kernel cell_D7scratch address_space_device [true, true] {
    ^bb0(%in: !metal.memref<? x f32>, %out: !metal.memref<? x f32>):
      %c8   = metal.constant 8 : ui32
      %c64  = metal.constant 64 : ui32
      %c128 = metal.constant 128 : ui32
      %gid  = metal.thread_id "x" : ui32
      %tgid = metal.threadgroup_id "x" : ui32
      %tg_off = metal.binary_exp %tgid, %c64, mulOp : (ui32, ui32) -> ui32
      %lid = metal.binary_exp %gid, %tg_off, subOp : (ui32, ui32) -> ui32

      // Two threadgroup buffers: v6 = scratch, v8 = transpose stage.
      %scratch  = metal.threadgroup_alloca : !metal.memref<64 x f32>
      %stage    = metal.threadgroup_alloca : !metal.memref<64 x f32>

      // Stage[lid] = in[gid]
      %v = metal.get_element %in[%gid] : (!metal.memref<? x f32>, ui32) -> f32
      metal.tg_store_indexed %stage[%lid], %v : !metal.memref<64 x f32>, ui32, f32

      metal.barrier

      // v9 = stage[transpose_idx]
      %col = metal.binary_exp %lid, %c8, remOp : (ui32, ui32) -> ui32
      %row = metal.binary_exp %lid, %c8, divOp : (ui32, ui32) -> ui32
      %col_x_8 = metal.binary_exp %col, %c8, mulOp : (ui32, ui32) -> ui32
      %idx = metal.binary_exp %col_x_8, %row, addOp : (ui32, ui32) -> ui32
      %v9 = metal.tg_load_indexed %stage[%idx] : (!metal.memref<64 x f32>, ui32) -> f32

      metal.barrier

      // Scratch RMW: v10 = scratch[lid]; scratch[lid] = mask ? v9 : v10.
      // We don't have arith.select in metal-dialect; use scf.if to model the
      // select-on-value pre-pass shape conservatively (Phase B uses
      // metal.cond_op or similar; we approximate with scf.if since that's
      // what the actual MSL ternary expands from in convert-to-metal).
      // Actually Phase B emits a `?:` ternary directly. We can mimic via
      // scf.if with yield since metal dialect lacks ternary. The behavior
      // is identical for an always-true mask.
      %v10 = metal.tg_load_indexed %scratch[%lid] : (!metal.memref<64 x f32>, ui32) -> f32
      %mask = metal.binary_exp %gid, %c128, ltOp : (ui32, ui32) -> i1
      %sel = scf.if %mask -> (f32) {
        scf.yield %v9 : f32
      } else {
        scf.yield %v10 : f32
      }
      metal.tg_store_indexed %scratch[%lid], %sel : !metal.memref<64 x f32>, ui32, f32

      // if(mask){ out[gid] = v9 }
      scf.if %mask {
        metal.store %v9, %out[%gid] : f32, !metal.memref<? x f32>, ui32
      }
      metal.return
    }
    metal.module_end
  }
}
