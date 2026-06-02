// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | FileCheck %s
//
// Tutorial-02 (fused softmax) lit fixture: tt.get_num_programs<axis=0>
// must lower to metal.threadgroups_per_grid "x" with a ui32 -> i32 cast bridge.
// Mirrors the one-condition-per-file precedent set by
// rank1_reduce_addf_block32.mlir, rank1_reduce_addf_block256.mlir,
// rank1_reduce_maxf_block32.mlir. axis=1/axis=2 fixtures are deferred until
// a tutorial exercises them.
// See .omc/plans/tutorial02-fused-softmax-fix-consensus.md AC9/AC9a.

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @get_num_programs_axis0(%out_ptr: !tt.ptr<f32>) {
    %n = tt.get_num_programs x : i32
    %f = arith.uitofp %n : i32 to f32
    tt.store %out_ptr, %f : !tt.ptr<f32>
    tt.return
  }
}
// CHECK-LABEL: get_num_programs_axis0
// CHECK: metal.threadgroups_per_grid "x"
// CHECK: builtin.unrealized_conversion_cast {{.*}} : ui32 to i32
