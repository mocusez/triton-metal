// RUN: triton-metal-opt --convert-tritongpu-to-metal %s --split-input-file | FileCheck %s
//
// L2b: masked tt.load / tt.store on tensor<Nx!tt.ptr<i8>>.
// Validates that
//   * MaskedLoadLowering's dtype gate accepts signless i8,
//   * the default-`other` zero is emitted as an i8 IntegerAttr (no
//     getFloatAttr path), and
//   * MaskedStoreLowering / preprocessMaskedStoreSentinels emit
//     i8-typed threadgroup-scratch alloca + scratch store + guarded
//     device store unchanged from the f32 baseline.
// See the implementation notes.

#blocked = #ttg.blocked<{sizePerThread = [16], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @masked_load_i8(%img: !tt.ptr<i8>, %N: i32) -> tensor<4096xi8, #blocked> {
    %r = tt.make_range {end = 4096 : i32, start = 0 : i32} : tensor<4096xi32, #blocked>
    %N_sp = tt.splat %N : i32 -> tensor<4096xi32, #blocked>
    %mask = arith.cmpi slt, %r, %N_sp : tensor<4096xi32, #blocked>
    %p_sp = tt.splat %img : !tt.ptr<i8> -> tensor<4096x!tt.ptr<i8>, #blocked>
    %addr = tt.addptr %p_sp, %r : tensor<4096x!tt.ptr<i8>, #blocked>, tensor<4096xi32, #blocked>
    %v = tt.load %addr, %mask : tensor<4096x!tt.ptr<i8>, #blocked>
    tt.return %v : tensor<4096xi8, #blocked>
  }
}

// CHECK: metal.kernel masked_load_i8
// CHECK: scf.if {{.*}} -> (i8)
// CHECK:   metal.get_element {{.*}} -> i8
// CHECK: } else {
// CHECK:   metal.constant 0 : i8

// -----

#blocked = #ttg.blocked<{sizePerThread = [16], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @masked_store_i8(%img: !tt.ptr<i8>, %N: i32, %v: tensor<4096xi8, #blocked>) {
    %r = tt.make_range {end = 4096 : i32, start = 0 : i32} : tensor<4096xi32, #blocked>
    %N_sp = tt.splat %N : i32 -> tensor<4096xi32, #blocked>
    %mask = arith.cmpi slt, %r, %N_sp : tensor<4096xi32, #blocked>
    %p_sp = tt.splat %img : !tt.ptr<i8> -> tensor<4096x!tt.ptr<i8>, #blocked>
    %addr = tt.addptr %p_sp, %r : tensor<4096x!tt.ptr<i8>, #blocked>, tensor<4096xi32, #blocked>
    tt.store %addr, %v, %mask : tensor<4096x!tt.ptr<i8>, #blocked>
    tt.return
  }
}

// CHECK: metal.kernel masked_store_i8
// Phase-B scratch sentinel is i8-typed.
// CHECK: metal.threadgroup_alloca : !metal.memref<{{[0-9]+}} x i8>
// CHECK: metal.tg_load_indexed {{.*}} -> i8
// CHECK: arith.select {{.*}} : i8
// CHECK: metal.tg_store_indexed {{.*}} x i8
// CHECK: scf.if
// CHECK:   metal.store {{.*}} i8
