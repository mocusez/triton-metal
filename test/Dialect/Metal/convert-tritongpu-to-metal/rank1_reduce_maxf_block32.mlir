// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | FileCheck %s
//
// Phase B lit fixture: rank-1 reduce, arith.maxnumf combine, BLOCK=32 < tpb=256.
// Verifies B2.1 (exact infinity tail identity-fill), B2.4 (butterfly with
// metal.binary_exp maxOp/minOp), and B3 (combine dispatch).
// See .omc/plans/option-beta-spt-load-lowering.md Phase B, steps B2.1/B3/B5.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @rank1_reduce_maxf_block32(%x_ptr: !tt.ptr<f32>) {
    %x = arith.constant dense<1.0> : tensor<32xf32, #blocked>
    %r = "tt.reduce"(%x) ({
    ^bb0(%a: f32, %b: f32):
      %s = arith.maxnumf %a, %b : f32
      tt.reduce.return %s : f32
    }) {axis = 0 : i32} : (tensor<32xf32, #blocked>) -> f32
    tt.return
  }

  tt.func public @rank1_reduce_minf_block32(%x_ptr: !tt.ptr<f32>) {
    %x = arith.constant dense<1.0> : tensor<32xf32, #blocked>
    %r = "tt.reduce"(%x) ({
    ^bb0(%a: f32, %b: f32):
      %s = arith.minnumf %a, %b : f32
      tt.reduce.return %s : f32
    }) {axis = 0 : i32} : (tensor<32xf32, #blocked>) -> f32
    tt.return
  }

  // Canonical reducer emitted by tl.argmax(..., tie_break_left=True): carry
  // both the winning f32 value and its logical i32 index.
  tt.func public @rank1_argmax_f32_block32(%x_ptr: !tt.ptr<f32>) {
    %x = arith.constant dense<1.0> : tensor<32xf32, #blocked>
    %idx = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #blocked>
    %r:2 = "tt.reduce"(%x, %idx) ({
    ^bb0(%lhs_value: f32, %lhs_index: i32, %rhs_value: f32, %rhs_index: i32):
      %equal = arith.cmpf oeq, %lhs_value, %rhs_value : f32
      %lower_index = arith.cmpi slt, %lhs_index, %rhs_index : i32
      %equal_and_lower = arith.andi %equal, %lower_index : i1
      %greater = arith.cmpf ogt, %lhs_value, %rhs_value : f32
      %take_left = arith.ori %greater, %equal_and_lower : i1
      %value = arith.select %take_left, %lhs_value, %rhs_value : f32
      %index = arith.select %take_left, %lhs_index, %rhs_index : i32
      tt.reduce.return %value, %index : f32, i32
    }) {axis = 0 : i32} : (tensor<32xf32, #blocked>, tensor<32xi32, #blocked>) -> (f32, i32)
    tt.return
  }
}
// CHECK-LABEL: metal.kernel rank1_reduce_maxf_block32
// B2.1: tail predicate for BLOCK=32 < tpb=256.
// CHECK: metal.thread_id "x"
// CHECK: arith.cmpi ult, {{.*}}, %c32_i32
// B3: max identity is exact -infinity, so all--inf input remains -inf.
// CHECK: arith.constant 0xFF800000 : f32
// CHECK: arith.select
// B2.1: threadgroup buffer size == tpb (256).
// CHECK: metal.threadgroup_alloca : !metal.memref<256 x f32>
// CHECK: metal.store
// CHECK: metal.barrier
// B2.4 + B3: butterfly uses maxOp (not addOp).
// CHECK: metal.binary_exp {{.*}}, {{.*}}, maxOp
// B2.4: final broadcast read from slot 0.
// CHECK: metal.get_element
// CHECK: metal.return

// CHECK-LABEL: metal.kernel rank1_reduce_minf_block32
// Min uses the exact opposite identity and the scalar min emitter.
// CHECK: arith.constant 0x7F800000 : f32
// CHECK: arith.select
// CHECK: metal.threadgroup_alloca : !metal.memref<256 x f32>
// CHECK: metal.binary_exp {{.*}}, {{.*}}, minOp
// CHECK: metal.return

// CHECK-LABEL: metal.kernel rank1_argmax_f32_block32
// Tail lanes cannot beat a real -inf input: their paired index is INT32_MAX.
// CHECK: arith.constant 0xFF800000 : f32
// CHECK: arith.constant 2147483647 : i32
// CHECK: arith.select
// CHECK: arith.select
// The pair reduction owns one scratch buffer per carried value.
// CHECK: metal.threadgroup_alloca : !metal.memref<256 x f32>
// CHECK: metal.threadgroup_alloca : !metal.memref<256 x ui32>
// CHECK: metal.store
// CHECK: metal.store
// Pairwise winner selection preserves Triton's lower-index tie break.
// CHECK: arith.cmpf ogt
// CHECK: arith.cmpf oeq
// CHECK: arith.cmpi slt
// CHECK: arith.andi
// CHECK: arith.ori
// CHECK: arith.select
// CHECK: arith.select
// CHECK: metal.get_element
// CHECK: metal.get_element
// CHECK: metal.return
