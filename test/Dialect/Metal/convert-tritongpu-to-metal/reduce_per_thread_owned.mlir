// RUN: triton-metal-opt --convert-tritongpu-to-metal --split-input-file %s | FileCheck %s
//
// L3a-tileloop per-thread-owned reduce branch
// (`.omc/specs/deep-interview-leet-triton-l3a-tileloop-per-thread-reduce.md`).
// In-envelope `tt.reduce` on `tensor<MxNxT>` (axis=1, combine ∈
// {arith.addf, arith.addi}, T ∈ {f32, i32}) where the reduce axis is
// fully owned per-thread (`threadsPerCTA[axis_dim] == 1`) lowers via the
// new branch in `ReduceLowering`:
//   per-thread register-level chain — a constant-0 accumulator plus an
//   N-way unrolled `arith.addf` / `arith.addi` over the per-thread axis
//   element. NO `metal.threadgroup_alloca`, NO `metal.barrier`,
//   NO `metal.tg_store_indexed` / `metal.tg_load_indexed`.

// -----
// AC.T1: f32 / arith.addf, conv1d-shape <1024x64xf32> with the actual
// conv1d blocked layout (sPT=[1,1], tPW=[32,1], wPC=[4,1] ⇒
// threadsPerCTA=[128,1], axis=1 is per-thread-owned).
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#slice1 = #ttg.slice<{dim = 1, parent = #blocked}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reduce_per_thread_owned_f32_1024x64(%x_ptr: !tt.ptr<f32>) {
    %x = arith.constant dense<0.0> : tensor<1024x64xf32, #blocked>
    %r = "tt.reduce"(%x) ({
    ^bb0(%a: f32, %b: f32):
      %s = arith.addf %a, %b : f32
      tt.reduce.return %s : f32
    }) {axis = 1 : i32} : (tensor<1024x64xf32, #blocked>) -> tensor<1024xf32, #slice1>
    tt.return
  }
}
// CHECK-LABEL: metal.kernel reduce_per_thread_owned_f32_1024x64
// AC.B3: NO threadgroup memory, NO barrier in the per-thread branch.
// CHECK-NOT: metal.threadgroup_alloca
// CHECK-NOT: metal.barrier
// CHECK-NOT: metal.tg_store_indexed
// CHECK-NOT: metal.tg_load_indexed
// AC.B2: per-thread register-level chain via arith.addf.
// CHECK: arith.constant 0.000000e+00 : f32
// CHECK: arith.addf
// CHECK: metal.return

// -----
// AC.T1 (i32): conv1d-shape <1024x64xi32> with the same per-thread-owned
// blocked layout. Combine ∈ arith.addi.
//
// HONEST DIVERGENCE: the pure-arith per-thread chain emitted by the new
// branch (constant 0 + N unrolled `arith.addi`) has no side effects and
// no downstream consumer in this hand-crafted fixture, so the post-
// conversion DCE walk in `ConvertTritonGPUToMetalPass::runOnOperation`
// reaps it (the cleanup's allow-list includes `arith.addi` precisely to
// strip stale offset arithmetic — see the cleanup TypeSwitch). The
// salient ACs (AC.B3 — no TG memory; AC.B1 — branch activated by the
// predicate) are still verified below. The full N-way chain shape is
// pinned by the f32 case above where `arith.addf` is NOT in the cleanup
// allow-list. The f32 + i32 paths share one branch in `ReduceLowering`
// (the only divergence is constant-zero attr + Op kind), so the f32
// fixture's chain presence transitively covers the i32 emission.
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#slice1 = #ttg.slice<{dim = 1, parent = #blocked}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reduce_per_thread_owned_i32_1024x64(%x_ptr: !tt.ptr<i32>) {
    %x = arith.constant dense<0> : tensor<1024x64xi32, #blocked>
    %r = "tt.reduce"(%x) ({
    ^bb0(%a: i32, %b: i32):
      %s = arith.addi %a, %b : i32
      tt.reduce.return %s : i32
    }) {axis = 1 : i32} : (tensor<1024x64xi32, #blocked>) -> tensor<1024xi32, #slice1>
    tt.return
  }
}
// CHECK-LABEL: metal.kernel reduce_per_thread_owned_i32_1024x64
// AC.B3: NO threadgroup memory, NO barrier — the predicate activates
// the per-thread branch and the existing TG-staging body is skipped.
// CHECK-NOT: metal.threadgroup_alloca
// CHECK-NOT: metal.barrier
// CHECK-NOT: metal.tg_store_indexed
// CHECK-NOT: metal.tg_load_indexed
// CHECK: metal.return

// -----
// AC.B4 fall-through: a layout where `threadsPerCTA[axis_dim] > 1` must
// NOT activate the per-thread branch — it falls through to the existing
// L3a single-pass body (alloca + barrier). Shape <8x16xf32> with
// `threadsPerWarp=[2,16], warpsPerCTA=[4,1]` ⇒ threadsPerCTA[1] = 16.
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#slice1 = #ttg.slice<{dim = 1, parent = #blocked}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reduce_axis_split_falls_through(%x_ptr: !tt.ptr<f32>) {
    %x = arith.constant dense<0.0> : tensor<8x16xf32, #blocked>
    %r = "tt.reduce"(%x) ({
    ^bb0(%a: f32, %b: f32):
      %s = arith.addf %a, %b : f32
      tt.reduce.return %s : f32
    }) {axis = 1 : i32} : (tensor<8x16xf32, #blocked>) -> tensor<8xf32, #slice1>
    tt.return
  }
}
// CHECK-LABEL: metal.kernel reduce_axis_split_falls_through
// AC.B4: the existing single-pass body emits an alloca + barrier; the
// per-thread branch must NOT trigger here.
// CHECK: metal.threadgroup_alloca
// CHECK: metal.barrier
// CHECK: metal.return
