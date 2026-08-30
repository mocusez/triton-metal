// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | FileCheck %s --check-prefix=METAL
// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | triton-metal-translate --mlir-to-msl | FileCheck %s --check-prefix=MSL
//
// The full conversion-to-MSL path is checked here. The obsolete pointer-cast
// bridge that once blocked `triton-metal-translate` no longer survives.
//
// 2D masked-load regression lock for the AC4-v6 follow-up.
// Shape `tensor<16x256xf32>` with `sizePerThread=[1,4]`, `warpsPerCTA=[4,2]`
// — the exact TTGIR the Triton frontend emits for the 2D test in
// `python/test/unit/test_metal_backend_vector_add_matrix.py` parametrization
// `2d_16x256`. Without the AC4-v6 DCE narrowing applied in
// the implementation notes Step 3 (the
// `if (op->getNumRegions() > 0) return;` guard), unused `tt.reduce` /
// region-bearing chains would be reaped before `ReduceLowering` ran. This
// fixture's 2D masked load doesn't directly exercise that DCE path, but its
// presence in the suite guards against future regressions that re-enable
// the over-aggressive DCE.

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 2], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @add_2d_masked(%x_ptr: !tt.ptr<f32>, %out_ptr: !tt.ptr<f32>, %M: i32, %N: i32) {
    %rm = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %rm2d = tt.expand_dims %rm {axis = 1 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<16x1xi32, #blocked>
    %M_sp = tt.splat %M : i32 -> tensor<16x1xi32, #blocked>
    %mask_m = arith.cmpi slt, %rm2d, %M_sp : tensor<16x1xi32, #blocked>
    %rn = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %rn2d = tt.expand_dims %rn {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x256xi32, #blocked>
    %N_sp = tt.splat %N : i32 -> tensor<1x256xi32, #blocked>
    %mask_n = arith.cmpi slt, %rn2d, %N_sp : tensor<1x256xi32, #blocked>
    %mask_mb = tt.broadcast %mask_m : tensor<16x1xi1, #blocked> -> tensor<16x256xi1, #blocked>
    %mask_nb = tt.broadcast %mask_n : tensor<1x256xi1, #blocked> -> tensor<16x256xi1, #blocked>
    %mask = arith.andi %mask_mb, %mask_nb : tensor<16x256xi1, #blocked>
    %N_off = tt.splat %N : i32 -> tensor<16x1xi32, #blocked>
    %rm_st = arith.muli %rm2d, %N_off : tensor<16x1xi32, #blocked>
    %rm_b = tt.broadcast %rm_st : tensor<16x1xi32, #blocked> -> tensor<16x256xi32, #blocked>
    %rn_b = tt.broadcast %rn2d : tensor<1x256xi32, #blocked> -> tensor<16x256xi32, #blocked>
    %offs = arith.addi %rm_b, %rn_b : tensor<16x256xi32, #blocked>
    %x_sp = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<16x256x!tt.ptr<f32>, #blocked>
    %x_addr = tt.addptr %x_sp, %offs : tensor<16x256x!tt.ptr<f32>, #blocked>, tensor<16x256xi32, #blocked>
    %x_val = tt.load %x_addr, %mask : tensor<16x256x!tt.ptr<f32>, #blocked>
    %o_sp = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<16x256x!tt.ptr<f32>, #blocked>
    %o_addr = tt.addptr %o_sp, %offs : tensor<16x256x!tt.ptr<f32>, #blocked>, tensor<16x256xi32, #blocked>
    tt.store %o_addr, %x_val, %mask : tensor<16x256x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// METAL: metal.module
// METAL: metal.kernel add_2d_masked
// METAL: arith.cmpi
// METAL: arith.andi
// METAL: scf.if
// METAL: metal.get_element
// METAL: metal.return
// MSL-LABEL: kernel void add_2d_masked(
// MSL: device float *
// MSL: if (
// MSL: = {{.*}}[{{.*}}];
