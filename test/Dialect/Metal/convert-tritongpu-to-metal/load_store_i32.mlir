// RUN: triton-metal-opt --convert-tritongpu-to-metal %s --split-input-file | FileCheck %s
//
// L2b: masked tt.load / tt.store on tensor<Nx!tt.ptr<i32>>.
// Unlike i8 (which is in Metal_Type and flows as signless i8 end-to-end),
// signless i32 is NOT in Metal_Type, so the backend routes i32 storage
// through ui32 (memref element + threadgroup scratch) and bridges back to
// signless i32 for downstream arith via builtin.unrealized_conversion_cast.
// This validates that
//   * the !tt.ptr<i32> kernel arg becomes a !metal.memref<? x ui32>,
//   * MaskedLoadLowering's dtype gate accepts the width-32 storage type,
//     get_element returns ui32, the default-`other` zero is a ui32 constant,
//     and the scf.if result is bridged ui32 -> i32,
//   * MaskedStoreLowering / preprocessMaskedStoreSentinels emit a ui32-typed
//     threadgroup-scratch alloca + scratch store + guarded device store, with
//     the incoming signless-i32 value bridged i32 -> ui32.
// See `.omc/specs/l2b-i32-tt-load-store-extension.md` (PROBE RESULT).

#blocked = #ttg.blocked<{sizePerThread = [16], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @masked_load_i32(%buf: !tt.ptr<i32>, %N: i32) -> tensor<4096xi32, #blocked> {
    %r = tt.make_range {end = 4096 : i32, start = 0 : i32} : tensor<4096xi32, #blocked>
    %N_sp = tt.splat %N : i32 -> tensor<4096xi32, #blocked>
    %mask = arith.cmpi slt, %r, %N_sp : tensor<4096xi32, #blocked>
    %p_sp = tt.splat %buf : !tt.ptr<i32> -> tensor<4096x!tt.ptr<i32>, #blocked>
    %addr = tt.addptr %p_sp, %r : tensor<4096x!tt.ptr<i32>, #blocked>, tensor<4096xi32, #blocked>
    %v = tt.load %addr, %mask : tensor<4096x!tt.ptr<i32>, #blocked>
    tt.return %v : tensor<4096xi32, #blocked>
  }
}

// CHECK: metal.kernel masked_load_i32
// The i32 data buffer arg is routed through ui32 storage.
// CHECK: ^bb0(%{{.*}}: !metal.memref<? x ui32>
// CHECK: scf.if {{.*}} -> (ui32)
// CHECK:   metal.get_element {{.*}} -> ui32
// CHECK: } else {
// CHECK:   metal.constant 0 : ui32

// -----

#blocked = #ttg.blocked<{sizePerThread = [16], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @masked_store_i32(%buf: !tt.ptr<i32>, %N: i32, %v: tensor<4096xi32, #blocked>) {
    %r = tt.make_range {end = 4096 : i32, start = 0 : i32} : tensor<4096xi32, #blocked>
    %N_sp = tt.splat %N : i32 -> tensor<4096xi32, #blocked>
    %mask = arith.cmpi slt, %r, %N_sp : tensor<4096xi32, #blocked>
    %p_sp = tt.splat %buf : !tt.ptr<i32> -> tensor<4096x!tt.ptr<i32>, #blocked>
    %addr = tt.addptr %p_sp, %r : tensor<4096x!tt.ptr<i32>, #blocked>, tensor<4096xi32, #blocked>
    tt.store %addr, %v, %mask : tensor<4096x!tt.ptr<i32>, #blocked>
    tt.return
  }
}

// CHECK: metal.kernel masked_store_i32
// Phase-B scratch sentinel is ui32-typed (i32 routed through ui32 storage).
// CHECK: metal.threadgroup_alloca : !metal.memref<{{[0-9]+}} x ui32>
// Incoming signless-i32 value bridged to ui32 storage.
// CHECK: builtin.unrealized_conversion_cast {{.*}} : i32 to ui32
// CHECK: metal.tg_load_indexed {{.*}} -> ui32
// CHECK: arith.select {{.*}} : ui32
// CHECK: metal.tg_store_indexed {{.*}} x ui32
// CHECK: scf.if
// CHECK:   metal.store {{.*}} ui32
