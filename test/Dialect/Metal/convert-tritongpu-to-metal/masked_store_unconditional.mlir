// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | FileCheck %s
//
// L1d2c Phase B canary (MLIR-stage, post-`convert-tritongpu-to-metal`):
// pins the new threadgroup-scratch routing for every masked `tt.store`.
//
// Spec: the implementation notes
//
// AC.L1 — checks:
//   * per-kernel `metal.threadgroup_alloca<tpb × T>` hoisted at function
//     entry (AC.S1, AC.S3).
//   * `arith.select` present at the masked-store site, selecting the
//     value to write into the scratch slot (AC.F1).
//   * `metal.tg_store_indexed` is unconditional (AC.F2 partial).
//   * `metal.threadgroup_alloca` size = threadsPerBlock × sizeof(T)
//     (here tpb = 32 × 4 = 128, T = f32 → `!metal.memref<128 x f32>`).
//
// HONEST DIVERGENCE from the spec sketch §"Goal" (per spec §"Reporting
// expectations" item 6): the spec called for the DEVICE store to also
// be unconditional via select-on-address. That is not directly
// realisable under the current `metal.store` op signature (single
// memref operand, cannot `arith.select` between device and threadgroup
// memrefs). The pragmatic Phase B emission keeps an `scf.if` around the
// device store so masked-off lanes never touch device memory — which
// preserves pre-Phase-B correctness for multi-program / OOB-mask cases
// (`test_metal_backend_multiload.py::test_pattern_A_multi_program` would
// otherwise regress). The Apple Metal lane-aliasing miscompile that
// motivated Phase B was empirically reproduced even with a fully
// unconditional device store + RMW value select, so removing the `scf.if`
// here does NOT fix the runtime bug. Phase B ships the threadgroup-
// scratch sentinel + value select per AC.F1, AC.S1–S3; the underlying
// Apple compiler defect persists at the `tg_load_indexed` codegen level
// and is out of `MaskedStoreLowering`'s scope.
//
// Bug-class background:
//   the implementation notes §"Bug
//   class characterization (AC.D4)".

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @masked_store_unconditional(%x_ptr: !tt.ptr<f32>, %out_ptr: !tt.ptr<f32>) {
    %c128_i32 = arith.constant 128 : i32
    %c100_i32 = arith.constant 100 : i32
    %pid = tt.get_program_id x : i32
    %block_start = arith.muli %pid, %c128_i32 : i32
    %offsets = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %bs_splat = tt.splat %block_start : i32 -> tensor<128xi32, #blocked>
    %abs_off = arith.addi %bs_splat, %offsets : tensor<128xi32, #blocked>
    %n_splat = tt.splat %c100_i32 : i32 -> tensor<128xi32, #blocked>
    %mask = arith.cmpi slt, %abs_off, %n_splat : tensor<128xi32, #blocked>
    %x_splat = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %x_addr = tt.addptr %x_splat, %abs_off : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    %x_val = tt.load %x_addr, %mask : tensor<128x!tt.ptr<f32>, #blocked>
    %o_splat = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %o_addr = tt.addptr %o_splat, %abs_off : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    tt.store %o_addr, %x_val, %mask : tensor<128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// CHECK: metal.module
// CHECK: metal.kernel masked_store_unconditional
//
// AC.S1 / AC.S3: per-kernel threadgroup scratch sentinel hoisted at
// function entry (size = threadsPerBlock × sizeof(f32) = 128 × 4 B).
// CHECK: metal.threadgroup_alloca : !metal.memref<128 x f32>
//
// AC.F1: `arith.select` present at the masked-store site, plus an
// unconditional `metal.tg_store_indexed` that lands the lane's value
// into the threadgroup scratch.
// CHECK: arith.select
// CHECK: metal.tg_store_indexed
//
// Device store: still wrapped in an `scf.if` (honest divergence —
// see file header).
// CHECK: scf.if
// CHECK: metal.store
// CHECK: metal.return
