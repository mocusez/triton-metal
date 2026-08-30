// RUN: triton-metal-opt --convert-tritongpu-to-metal --split-input-file %s | FileCheck %s
//
// Deterministic guard for the reduce scratch-buffer barrier.
//
// `lowerRank1Reduce` allocates ONE threadgroup buffer and, when the reduce sits
// inside an scf.for, reuses that same static allocation on every trip. The
// butterfly's trailing barriers order accesses *within* a single reduce, but
// nothing separated iteration t's broadcast read of buf[0] from iteration t+1's
// write to buf[tid] — a loop-carried write-after-read race. The fix is a
// barrier immediately before the initial buf[tid] store.
//
// This is checked here rather than only at runtime because the race is
// timing-dependent: it needs two or more SIMD-groups (tpb > 32) to drift apart,
// so the companion pytest
// (python/test/unit/test_metal_backend_reduce_in_loop_multiwarp.py) detects a
// regression only probabilistically. These CHECK-NEXT lines pin the emission
// order exactly and fail immediately if the barrier is dropped.

// -----
// BLOCK == tpb == 64 (num_warps=2): the direct per-thread path.
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [2], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @rank1_reduce_leading_barrier_block_eq_tpb(%x_ptr: !tt.ptr<f32>) {
    %x = arith.constant dense<1.0> : tensor<64xf32, #blocked>
    %r = "tt.reduce"(%x) ({
    ^bb0(%a: f32, %b: f32):
      %s = arith.addf %a, %b : f32
      tt.reduce.return %s : f32
    }) {axis = 0 : i32} : (tensor<64xf32, #blocked>) -> f32
    tt.return
  }
}
// CHECK-LABEL: metal.kernel rank1_reduce_leading_barrier_block_eq_tpb
// The barrier must sit between the alloca and the first store into it.
// CHECK: metal.threadgroup_alloca : !metal.memref<64 x f32>
// CHECK-NEXT: metal.barrier
// CHECK-NEXT: metal.store {{.*}} : f32, !metal.memref<64 x f32>, ui32
// CHECK-NEXT: metal.barrier
// Butterfly still follows.
// CHECK: metal.binary_exp {{.*}}, {{.*}}, addOp

// -----
// BLOCK (32) < tpb (64): tail identity-fill path, same buffer discipline.
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [2], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @rank1_reduce_leading_barrier_block_lt_tpb(%x_ptr: !tt.ptr<f32>) {
    %x = arith.constant dense<1.0> : tensor<32xf32, #blocked1>
    %r = "tt.reduce"(%x) ({
    ^bb0(%a: f32, %b: f32):
      %s = arith.addf %a, %b : f32
      tt.reduce.return %s : f32
    }) {axis = 0 : i32} : (tensor<32xf32, #blocked1>) -> f32
    tt.return
  }
}
// CHECK-LABEL: metal.kernel rank1_reduce_leading_barrier_block_lt_tpb
// CHECK: metal.threadgroup_alloca : !metal.memref<64 x f32>
// CHECK-NEXT: metal.barrier
// CHECK-NEXT: metal.store {{.*}} : f32, !metal.memref<64 x f32>, ui32
// CHECK-NEXT: metal.barrier
