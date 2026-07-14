// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | FileCheck %s
//
// Rank-1 reduce, arith.minsi combine (tl.min on i32), BLOCK=32 < tpb=256.
// Mirror of rank1_reduce_maxf_block32.mlir for the min path:
//   - combine dispatch minsi → metal.binary_exp minOp
//   - identity is INT32_MAX (2147483647), not INT32_MIN (that's max's identity)
//   - si32 accumulator so MSL emits `min(int32_t(a), int32_t(b))` (signed).

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @rank1_reduce_minsi_block32(%x_ptr: !tt.ptr<i32>) {
    %x = arith.constant dense<5> : tensor<32xi32, #blocked>
    %r = "tt.reduce"(%x) ({
    ^bb0(%a: i32, %b: i32):
      %s = arith.minsi %a, %b : i32
      tt.reduce.return %s : i32
    }) {axis = 0 : i32} : (tensor<32xi32, #blocked>) -> i32
    tt.return
  }
}
// CHECK-LABEL: metal.kernel rank1_reduce_minsi_block32
// tail predicate for BLOCK=32 < tpb=256.
// CHECK: metal.thread_id "x"
// CHECK: arith.cmpi ult, {{.*}}, %c32_i32
// min identity is INT32_MAX.
// CHECK: arith.constant 2147483647 : i32
// CHECK: arith.select
// CHECK: metal.barrier
// butterfly uses minOp (not addOp/maxOp).
// CHECK: metal.binary_exp {{.*}}, {{.*}}, minOp
// CHECK: metal.return
