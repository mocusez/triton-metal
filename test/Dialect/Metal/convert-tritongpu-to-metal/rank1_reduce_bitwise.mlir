// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | FileCheck %s
//
// Bitwise integer reduce combines (arith.andi / ori / xori). `tl.xor_sum`
// lowers to the xori form; and/or come from hand-written tl.reduce combines.
// All three ride the same ui32 butterfly as arith.addi, so the only new pieces
// are the identity (~0 for and, 0 for or/xor) and the bitwise metal.binary_exp
// operator — the pre-existing andOp/orOp are the LOGICAL `&&`/`||` and produce
// i1, which is why bitAndOp/bitOrOp/bitXorOp had to be added.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [8, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @rank1_reduce_xori_block32(%x_ptr: !tt.ptr<i32>) {
    %x = arith.constant dense<3> : tensor<32xi32, #blocked>
    %r = "tt.reduce"(%x) ({
    ^bb0(%a: i32, %b: i32):
      %s = arith.xori %a, %b : i32
      tt.reduce.return %s : i32
    }) {axis = 0 : i32} : (tensor<32xi32, #blocked>) -> i32
    tt.return
  }

  tt.func public @rank1_reduce_ori_block32(%x_ptr: !tt.ptr<i32>) {
    %x = arith.constant dense<3> : tensor<32xi32, #blocked>
    %r = "tt.reduce"(%x) ({
    ^bb0(%a: i32, %b: i32):
      %s = arith.ori %a, %b : i32
      tt.reduce.return %s : i32
    }) {axis = 0 : i32} : (tensor<32xi32, #blocked>) -> i32
    tt.return
  }

  tt.func public @rank1_reduce_andi_block32(%x_ptr: !tt.ptr<i32>) {
    %x = arith.constant dense<3> : tensor<32xi32, #blocked>
    %r = "tt.reduce"(%x) ({
    ^bb0(%a: i32, %b: i32):
      %s = arith.andi %a, %b : i32
      tt.reduce.return %s : i32
    }) {axis = 0 : i32} : (tensor<32xi32, #blocked>) -> i32
    tt.return
  }
}
// CHECK-LABEL: metal.kernel rank1_reduce_xori_block32
// XOR identity is 0, so the BLOCK < tpb tail contributes nothing.
// CHECK: arith.constant 0 : i32
// CHECK: metal.threadgroup_alloca : !metal.memref<256 x ui32>
// CHECK: metal.barrier
// CHECK: metal.binary_exp {{.*}}, {{.*}}, bitXorOp
// CHECK: metal.return

// CHECK-LABEL: metal.kernel rank1_reduce_ori_block32
// OR shares XOR's zero identity.
// CHECK: arith.constant 0 : i32
// CHECK: metal.binary_exp {{.*}}, {{.*}}, bitOrOp
// CHECK: metal.return

// CHECK-LABEL: metal.kernel rank1_reduce_andi_block32
// AND's identity is all-ones; a 0 identity would zero every result.
// CHECK: arith.constant -1 : i32
// CHECK: metal.binary_exp {{.*}}, {{.*}}, bitAndOp
// CHECK: metal.return
