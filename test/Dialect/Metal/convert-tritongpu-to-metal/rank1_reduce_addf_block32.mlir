// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | FileCheck %s
//
// Phase B lit fixture: rank-1 reduce, arith.addf combine, BLOCK=32 < tpb=256.
// Verifies B2.1 (tail identity-fill) + B2.4 (butterfly with metal.binary_exp addOp).
// See .omc/plans/option-beta-spt-load-lowering.md Phase B, steps B2.1/B2.4/B5.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @rank1_reduce_addf_block32(%x_ptr: !tt.ptr<f32>) {
    %x = arith.constant dense<1.0> : tensor<32xf32, #blocked>
    %r = "tt.reduce"(%x) ({
    ^bb0(%a: f32, %b: f32):
      %s = arith.addf %a, %b : f32
      tt.reduce.return %s : f32
    }) {axis = 0 : i32} : (tensor<32xf32, #blocked>) -> f32
    tt.return
  }
}
// CHECK-LABEL: metal.kernel rank1_reduce_addf_block32
// B2.1: tail predicate — threads with tid >= BLOCK get identity (0.0) written.
// CHECK: metal.thread_id "x"
// CHECK: arith.cmpi ult, {{.*}}, %c32_i32
// CHECK: arith.constant 0.000000e+00 : f32
// CHECK: arith.select
// B2.1: threadgroup buffer size == tpb (256), not BLOCK (32).
// CHECK: metal.threadgroup_alloca : !metal.memref<256 x f32>
// CHECK: metal.store
// CHECK: metal.barrier
// B2.4: butterfly with metal.binary_exp addOp — at least one stage.
// CHECK: metal.binary_exp {{.*}}, {{.*}}, addOp
// B2.4: final broadcast read from slot 0.
// CHECK: metal.get_element
// CHECK: metal.return
