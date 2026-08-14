// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | FileCheck %s
//
// Wall 13 regression-guard: masked rank-1 tt.store lowering via the
// existing MaskedStoreLowering (TritonGPUToMetal.cpp:3077) + Phase-B
// scratch sentinel pre-pass.
//
// Architect F3: the lit RUN line drives the COMPOSITE
// `--convert-tritongpu-to-metal` pass so `preprocessMaskedStoreSentinels`
// runs internally. The `metal.threadgroup_alloca` CHECK below is the
// sentinel proof — if the pre-pass did not fire, that op is absent and the
// next-step `MaskedStoreLowering` would emit
// `"no Phase-B scratch sentinel registered"`.
//
// Critic D7: blocked encoding pinned to threadsPerWarp=32 × warpsPerCTA=8
// → tpb = 256, matching the alloca size below.
// See .omc/specs/deep-interview-tutorial02-walls-9-to-13.md AC8.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @masked_store_rank1(%out_ptr: !tt.ptr<f32>, %val_ptr: !tt.ptr<f32>, %n_cols: i32) {
    %offsets = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %n_splat = tt.splat %n_cols : i32 -> tensor<1024xi32, #blocked>
    %mask = arith.cmpi slt, %offsets, %n_splat : tensor<1024xi32, #blocked>
    %v_splat = tt.splat %val_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %v_addr = tt.addptr %v_splat, %offsets : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    %vals = tt.load %v_addr, %mask : tensor<1024x!tt.ptr<f32>, #blocked>
    %o_splat = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %o_addr = tt.addptr %o_splat, %offsets : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    tt.store %o_addr, %vals, %mask : tensor<1024x!tt.ptr<f32>, #blocked>
    tt.return
  }

  // BLOCK=1 folds the all-true mask and the tensor addptr, leaving a uniform
  // tensor pointer. This is the K=1 output-store shape in MoE top-k gating.
  tt.func public @singleton_store_uniform_ptr(%out_ptr: !tt.ptr<f32>, %row: i32) {
    %scalar_ptr = tt.addptr %out_ptr, %row : !tt.ptr<f32>, i32
    %ptr = tt.splat %scalar_ptr : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    %value = arith.constant dense<1.0> : tensor<1xf32, #blocked>
    tt.store %ptr, %value : tensor<1x!tt.ptr<f32>, #blocked>
    tt.return
  }

  // A dynamic mask preserves the masked tt.store form after canonicalization.
  // The tensor has one logical element, so only lane 0 may perform the scalar
  // store when the user predicate is true.
  tt.func public @singleton_masked_store_uniform_ptr(%out_ptr: !tt.ptr<f32>, %row: i32, %limit: i32, %value: f32) {
    %scalar_ptr = tt.addptr %out_ptr, %row : !tt.ptr<f32>, i32
    %ptr = tt.splat %scalar_ptr : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    %row_tensor = tt.splat %row : i32 -> tensor<1xi32, #blocked>
    %limit_tensor = tt.splat %limit : i32 -> tensor<1xi32, #blocked>
    %mask = arith.cmpi slt, %row_tensor, %limit_tensor : tensor<1xi32, #blocked>
    %value_tensor = tt.splat %value : f32 -> tensor<1xf32, #blocked>
    tt.store %ptr, %value_tensor, %mask : tensor<1x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// CHECK-LABEL: metal.kernel masked_store_rank1
// tpb derivation: threadsPerWarp[0]=32 × warpsPerCTA[0]=8 = 256.
// Sentinel proof: the Phase-B pre-pass hoisted a threadgroup scratch alloca.
// CHECK: metal.threadgroup_alloca
// Phase-B unconditional rewrite emits arith.select between val and old, then
// an unconditional metal.store (no scf.if around the store).
// CHECK: arith.select
// CHECK: metal.store
// CHECK: metal.return

// CHECK-LABEL: metal.kernel singleton_store_uniform_ptr
// One local lane owns the scalarized singleton tensor store.
// CHECK: metal.thread_id "x"
// CHECK: metal.threadgroup_id "x"
// CHECK: arith.cmpi eq
// CHECK: scf.if
// CHECK: metal.store
// CHECK: metal.return

// CHECK-LABEL: metal.kernel singleton_masked_store_uniform_ptr
// CHECK: arith.cmpi slt
// CHECK: metal.thread_id "x"
// CHECK: metal.threadgroup_id "x"
// CHECK: arith.cmpi eq
// CHECK: arith.andi
// CHECK: scf.if
// CHECK: metal.store
// CHECK: metal.return
