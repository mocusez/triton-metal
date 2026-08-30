// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | FileCheck %s --check-prefix=METAL
// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | triton-metal-translate --mlir-to-msl | FileCheck %s --check-prefix=MSL
//
// Lmultiload Phase C — AC.M2.
//
// Verifies that `MakeRangeLowering` for a 1D tensor emits a non-zero
// per-thread expression (the threadgroup-local thread ID) rather than
// the prior `arith.constant 0` placeholder. The key indicators:
//   - METAL IR contains `arith.subi` of `metal.thread_id "x"` and
//     `metal.threadgroup_id "x" * threadsPerBlock` (the localTid term).
//   - MSL emits `(id.x - (tgid.x * <tpb>))` somewhere in the store
//     index expression.
//
// See the implementation notes.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  // No `pid * BLOCK` term: the only per-thread contribution is the
  // arange-derived localTid. Pre-Phase-C this would lower to
  // `output_ptr[0] = ...` (constant-zero arange). Post-Phase-C the store
  // index is the localTid value.
  tt.func public @arange_only(%output_ptr: !tt.ptr<f32>) {
    %offsets = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %o_splat = tt.splat %output_ptr : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %o_addr = tt.addptr %o_splat, %offsets : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    %c0 = arith.constant dense<0.0> : tensor<128xf32, #blocked>
    tt.store %o_addr, %c0 : tensor<128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// METAL: metal.kernel arange_only
// METAL: metal.thread_id "x"
// METAL: metal.threadgroup_id "x"
// METAL: arith.muli
// METAL: arith.subi
// METAL: metal.store

// MSL: kernel void arange_only(
// MSL: thread_position_in_grid
// MSL: threadgroup_position_in_grid
// MSL: (id.x - (tgid.x * 128))
