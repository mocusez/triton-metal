// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | FileCheck %s
// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | triton-metal-translate --mlir-to-msl | FileCheck %s --check-prefix=MSL
//
// AC2 fixture for the masked-tail epilogue lowering. The kernel below is
// a single-dot (K_TILES=1) matmul that follows the canonical Triton 2D
// matmul pointer shape, with a `tt.store` that carries the canonical local
// remaining-count `(arange_m < M) & (arange_n < N)` mask. The store pointer
// itself carries the program-grid origin, so the lowering must normalize the
// partial bounds to `origin + remaining` before emitting global gi/gj checks.
//
// Expected post-pass behavior:
//   1. `metal.simdgroup_multiply_accumulate` is emitted for the single dot.
//   2. `metal.simdgroup_store` is emitted with the optional `partial`
//      operand list `[%M, %N]` (mask extents extracted by the
//      `extractMaskExtents` walker).
//   3. No residual `tt.dot`, `tt.store`, or `ttg.convert_layout` (other
//      than ones the conversion pass leaves for unrelated ops).
//
// Plan: the implementation notes.
//
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @dot_partial_tile_masked(
      %a_ptr: !tt.ptr<f32>, %b_ptr: !tt.ptr<f32>, %c_ptr: !tt.ptr<f32>,
      %M: i32, %N: i32,
      %stride_am: i32, %stride_bk: i32, %stride_cm: i32) {
    %c8_i32 = arith.constant 8 : i32
    %pid_m = tt.get_program_id x : i32
    %pid_n = tt.get_program_id y : i32
    %offs_m_base = arith.muli %pid_m, %c8_i32 : i32
    %offs_n_base = arith.muli %pid_n, %c8_i32 : i32
    %r0 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %r1 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %offs_m_sp = tt.splat %offs_m_base : i32 -> tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %offs_m_full = arith.addi %offs_m_sp, %r0 : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %offs_n_sp = tt.splat %offs_n_base : i32 -> tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %offs_n_full = arith.addi %offs_n_sp, %r1 : tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked}>>

    // A pointers (8x8 tile).
    %a_row = tt.expand_dims %offs_m_full {axis = 1 : i32} : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<8x1xi32, #blocked>
    %a_stride = tt.splat %stride_am : i32 -> tensor<8x1xi32, #blocked>
    %a_row_off = arith.muli %a_row, %a_stride : tensor<8x1xi32, #blocked>
    %a_base_sp = tt.splat %a_ptr : !tt.ptr<f32> -> tensor<8x1x!tt.ptr<f32>, #blocked>
    %a_base = tt.addptr %a_base_sp, %a_row_off : tensor<8x1x!tt.ptr<f32>, #blocked>, tensor<8x1xi32, #blocked>
    %a_col = tt.expand_dims %r1 {axis = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x8xi32, #blocked>
    %a_base_bc = tt.broadcast %a_base : tensor<8x1x!tt.ptr<f32>, #blocked> -> tensor<8x8x!tt.ptr<f32>, #blocked>
    %a_col_bc = tt.broadcast %a_col : tensor<1x8xi32, #blocked> -> tensor<8x8xi32, #blocked>
    %a_ptrs = tt.addptr %a_base_bc, %a_col_bc : tensor<8x8x!tt.ptr<f32>, #blocked>, tensor<8x8xi32, #blocked>
    %a_blk = tt.load %a_ptrs : tensor<8x8x!tt.ptr<f32>, #blocked>

    // B pointers (8x8 tile).
    %b_row = tt.expand_dims %r0 {axis = 1 : i32} : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<8x1xi32, #blocked>
    %b_stride = tt.splat %stride_bk : i32 -> tensor<8x1xi32, #blocked>
    %b_row_off = arith.muli %b_row, %b_stride : tensor<8x1xi32, #blocked>
    %b_base_sp = tt.splat %b_ptr : !tt.ptr<f32> -> tensor<8x1x!tt.ptr<f32>, #blocked>
    %b_base = tt.addptr %b_base_sp, %b_row_off : tensor<8x1x!tt.ptr<f32>, #blocked>, tensor<8x1xi32, #blocked>
    %b_col = tt.expand_dims %offs_n_full {axis = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x8xi32, #blocked>
    %b_base_bc = tt.broadcast %b_base : tensor<8x1x!tt.ptr<f32>, #blocked> -> tensor<8x8x!tt.ptr<f32>, #blocked>
    %b_col_bc = tt.broadcast %b_col : tensor<1x8xi32, #blocked> -> tensor<8x8xi32, #blocked>
    %b_ptrs = tt.addptr %b_base_bc, %b_col_bc : tensor<8x8x!tt.ptr<f32>, #blocked>, tensor<8x8xi32, #blocked>
    %b_blk = tt.load %b_ptrs : tensor<8x8x!tt.ptr<f32>, #blocked>

    // Dot.
    %acc_init = arith.constant dense<0.000000e+00> : tensor<8x8xf32, #blocked>
    %a_dot = ttg.convert_layout %a_blk : tensor<8x8xf32, #blocked> -> tensor<8x8xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    %b_dot = ttg.convert_layout %b_blk : tensor<8x8xf32, #blocked> -> tensor<8x8xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
    %acc = tt.dot %a_dot, %b_dot, %acc_init : tensor<8x8xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<8x8xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<8x8xf32, #blocked>

    // C pointers + canonical local remaining-count mask.
    %c_stride = tt.splat %stride_cm : i32 -> tensor<8x1xi32, #blocked>
    %c_row_off = arith.muli %a_row, %c_stride : tensor<8x1xi32, #blocked>
    %c_base_sp = tt.splat %c_ptr : !tt.ptr<f32> -> tensor<8x1x!tt.ptr<f32>, #blocked>
    %c_base = tt.addptr %c_base_sp, %c_row_off : tensor<8x1x!tt.ptr<f32>, #blocked>, tensor<8x1xi32, #blocked>
    %c_base_bc = tt.broadcast %c_base : tensor<8x1x!tt.ptr<f32>, #blocked> -> tensor<8x8x!tt.ptr<f32>, #blocked>
    %c_ptrs = tt.addptr %c_base_bc, %b_col_bc : tensor<8x8x!tt.ptr<f32>, #blocked>, tensor<8x8xi32, #blocked>
    // mask = (arange_m[:, None] < M) & (arange_n[None, :] < N), matching a
    // tile-local remaining-count mask. The exact aranges also flow into the
    // global store pointer cone, which lets the matcher distinguish these
    // local coordinates from `%offs_m_full` / `%offs_n_full` global indices.
    %offs_m_2d = tt.expand_dims %r0 {axis = 1 : i32} : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<8x1xi32, #blocked>
    %m_splat_2d = tt.splat %M : i32 -> tensor<8x1xi32, #blocked>
    %m_cmp_2d = arith.cmpi slt, %offs_m_2d, %m_splat_2d : tensor<8x1xi32, #blocked>
    %m_cmp_2d_bc = tt.broadcast %m_cmp_2d : tensor<8x1xi1, #blocked> -> tensor<8x8xi1, #blocked>
    %offs_n_2d = tt.expand_dims %r1 {axis = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x8xi32, #blocked>
    %n_splat_2d = tt.splat %N : i32 -> tensor<1x8xi32, #blocked>
    %n_cmp_2d = arith.cmpi slt, %offs_n_2d, %n_splat_2d : tensor<1x8xi32, #blocked>
    %n_cmp_2d_bc = tt.broadcast %n_cmp_2d : tensor<1x8xi1, #blocked> -> tensor<8x8xi1, #blocked>
    %mask = arith.andi %m_cmp_2d_bc, %n_cmp_2d_bc : tensor<8x8xi1, #blocked>
    tt.store %c_ptrs, %acc, %mask : tensor<8x8x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// CHECK-LABEL: metal.kernel dot_partial_tile_masked
// CHECK-NOT: tt.dot
// CHECK-NOT: tt.store
// CHECK: metal.simdgroup_multiply_accumulate
// CHECK: metal.simdgroup_store
// CHECK-SAME: partial
// CHECK: metal.return

// Static canonical dot local bounds must be normalized to global origins.
// MSL: uint gi =
// MSL: uint gj =
// MSL: if (gi <
// MSL-SAME: +
// MSL: && gj <
// MSL-SAME: +
