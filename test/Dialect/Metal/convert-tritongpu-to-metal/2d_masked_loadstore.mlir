// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | FileCheck %s
//
// 2D masked load + masked store: the mask is the canonical Triton 2D
// elementwise pattern `(offs_m[:,None] < M) & (offs_n[None,:] < N)`,
// which lowers to `arith.andi(arith.cmpi(.., splat M),
// arith.cmpi(.., splat N))` in TTGIR. The TritonGPUToMetal pass must
// match this 2D AND-reduced mask and emit per-axis cmpi (row vs col)
// ANDed together as the per-thread scf.if condition.
// See `.omc/specs/deep-interview-metal-2d-maskedaccess-emitperiterindex.md`.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @masked_2d(%x_ptr: !tt.ptr<f32>, %out_ptr: !tt.ptr<f32>, %M: i32, %N: i32) {
    // offs_m[:, None]
    %rm = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %rm2d = tt.expand_dims %rm {axis = 1 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<16x1xi32, #blocked>
    %M_sp1 = tt.splat %M : i32 -> tensor<16x1xi32, #blocked>
    %mask_m = arith.cmpi slt, %rm2d, %M_sp1 : tensor<16x1xi32, #blocked>
    %mask_m_b = tt.broadcast %mask_m : tensor<16x1xi1, #blocked> -> tensor<16x16xi1, #blocked>

    // offs_n[None, :]
    %rn = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %rn2d = tt.expand_dims %rn {axis = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x16xi32, #blocked>
    %N_sp1 = tt.splat %N : i32 -> tensor<1x16xi32, #blocked>
    %mask_n = arith.cmpi slt, %rn2d, %N_sp1 : tensor<1x16xi32, #blocked>
    %mask_n_b = tt.broadcast %mask_n : tensor<1x16xi1, #blocked> -> tensor<16x16xi1, #blocked>

    // 2D AND-reduced mask
    %mask = arith.andi %mask_m_b, %mask_n_b : tensor<16x16xi1, #blocked>

    // flat offsets: offs_m * N + offs_n (use a dummy splat 1 stride to keep IR simple)
    %one_sp = tt.splat %M : i32 -> tensor<16x1xi32, #blocked>
    %rm_strided = arith.muli %rm2d, %one_sp : tensor<16x1xi32, #blocked>
    %rm_b = tt.broadcast %rm_strided : tensor<16x1xi32, #blocked> -> tensor<16x16xi32, #blocked>
    %rn_b = tt.broadcast %rn2d : tensor<1x16xi32, #blocked> -> tensor<16x16xi32, #blocked>
    %flat_off = arith.addi %rm_b, %rn_b : tensor<16x16xi32, #blocked>

    // Masked load + masked store
    %x_splat = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>, #blocked>
    %x_addr = tt.addptr %x_splat, %flat_off : tensor<16x16x!tt.ptr<f32>, #blocked>, tensor<16x16xi32, #blocked>
    %x_val = tt.load %x_addr, %mask : tensor<16x16x!tt.ptr<f32>, #blocked>
    %o_splat = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>, #blocked>
    %o_addr = tt.addptr %o_splat, %flat_off : tensor<16x16x!tt.ptr<f32>, #blocked>, tensor<16x16xi32, #blocked>
    tt.store %o_addr, %x_val, %mask : tensor<16x16x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// CHECK: metal.module
// CHECK: metal.kernel masked_2d
// The mask must AND two per-axis cmpi (row check + col check).
// CHECK: arith.cmpi
// CHECK: arith.cmpi
// CHECK: arith.andi
// CHECK: scf.if
// CHECK: metal.return
