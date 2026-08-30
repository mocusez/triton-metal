// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | FileCheck %s
//
// Phase B lit fixture: rank-1 reduce, arith.addf combine, BLOCK=256 == tpb=256.
// Verifies B2.2 (no tail — all threads carry valid data, direct write) +
// B2.4 (butterfly with metal.binary_exp addOp).
// See the implementation notes Phase B, steps B2.2/B2.4/B5.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @rank1_reduce_addf_block256(%x_ptr: !tt.ptr<f32>) {
    %x = arith.constant dense<1.0> : tensor<256xf32, #blocked>
    %r = "tt.reduce"(%x) ({
    ^bb0(%a: f32, %b: f32):
      %s = arith.addf %a, %b : f32
      tt.reduce.return %s : f32
    }) {axis = 0 : i32} : (tensor<256xf32, #blocked>) -> f32
    tt.return
  }
}
// CHECK-LABEL: metal.kernel rank1_reduce_addf_block256
// B2.2: BLOCK == tpb — no tail identity-fill, no arith.select for padding.
// CHECK-NOT: arith.select
// B2.2: threadgroup buffer size == tpb (256).
// CHECK: metal.threadgroup_alloca : !metal.memref<256 x f32>
// CHECK: metal.store
// CHECK: metal.barrier
// B2.4: butterfly with metal.binary_exp addOp.
// CHECK: metal.binary_exp {{.*}}, {{.*}}, addOp
// B2.4: final broadcast read from slot 0.
// CHECK: metal.get_element {{.*}}[{{.*}}]
// CHECK: metal.return
