// RUN: triton-metal-opt --convert-tritongpu-to-metal --split-input-file %s | FileCheck %s
//
// Session L3a positive fixture: in-envelope `tt.reduce` on `tensor<MxNxT>`
// (axis=1, combine ∈ {arith.addf, arith.addi}, T ∈ {f32, i32}) lowers to
// the sequential row-scan body via the new `ReduceLowering` pattern:
//   threadgroup_alloca M*N → store @tid → barrier → row=tid/N → unrolled
//   `get_element + add` chain over j=0..N-1.
// See `.omc/specs/deep-interview-leet-triton-l3a-reduce-body-axis1.md`.

// -----
// f32 / arith.addf, shape <8x16xf32>, axis=1, expected buffer size 128.
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#slice1 = #ttg.slice<{dim = 1, parent = #blocked}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reduce_sum_axis1_f32(%x_ptr: !tt.ptr<f32>) {
    %x = arith.constant dense<0.0> : tensor<8x16xf32, #blocked>
    %r = "tt.reduce"(%x) ({
    ^bb0(%a: f32, %b: f32):
      %s = arith.addf %a, %b : f32
      tt.reduce.return %s : f32
    }) {axis = 1 : i32} : (tensor<8x16xf32, #blocked>) -> tensor<8xf32, #slice1>
    tt.return
  }
}
// CHECK-LABEL: metal.kernel reduce_sum_axis1_f32
// CHECK: metal.threadgroup_alloca {{.*}} !metal.memref<128 x f32>
// CHECK: metal.thread_id "x"
// L3 budget: row = tid/N and rowBase = row*chunk_size (== row*N for the
// in-budget / single-pass case) are hoisted ABOVE the populate barrier
// because the chunked branch shares this math across chunks. The L3a
// single-pass body is preserved in behavior; only the relative order of
// the row-math vs. populate-barrier shifted.
// CHECK: arith.divui
// CHECK: arith.muli
// CHECK: metal.store
// CHECK: metal.barrier
// CHECK: metal.get_element
// CHECK: metal.return

// -----
// i32 / arith.addi, shape <8x16xi32>, axis=1.
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#slice1 = #ttg.slice<{dim = 1, parent = #blocked}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reduce_sum_axis1_i32(%x_ptr: !tt.ptr<i32>) {
    %x = arith.constant dense<0> : tensor<8x16xi32, #blocked>
    %r = "tt.reduce"(%x) ({
    ^bb0(%a: i32, %b: i32):
      %s = arith.addi %a, %b : i32
      tt.reduce.return %s : i32
    }) {axis = 1 : i32} : (tensor<8x16xi32, #blocked>) -> tensor<8xi32, #slice1>
    tt.return
  }
}
// CHECK-LABEL: metal.kernel reduce_sum_axis1_i32
// i32 reduce stores via ui32 (Metal_Type rejects signless i32; see
// ReduceLowering for the cast bridge).
// CHECK: metal.threadgroup_alloca {{.*}} !metal.memref<128 x ui32>
// CHECK: metal.thread_id "x"
// L3 budget: row math hoisted above the populate barrier (see f32 case
// above for the rationale).
// CHECK: arith.divui
// CHECK: arith.muli
// CHECK: metal.store
// CHECK: metal.barrier
// CHECK: metal.get_element
// CHECK: metal.return
