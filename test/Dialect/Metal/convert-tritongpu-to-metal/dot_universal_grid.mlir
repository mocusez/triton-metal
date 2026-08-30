// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | FileCheck %s
//
// Universal-matmul AC6 fixture (the implementation notes).
// Same canonical-3-iter-arg matmul shape as
// `dot_canonical_preempt_3iterarg.mlir` (K_TILES=2), extended to K_TILES=4
// to verify that the rewriter:
//   1. unrolls all K iterations into a flat chain of
//      `metal.simdgroup_multiply_accumulate` ops (one per K-tile);
//   2. emits a single `metal.simdgroup_matrix_zero` for the acc init
//      (no spurious load from the C buffer);
//   3. produces no residual `scf.for` or `tt.dot` after the conversion.
//
// Acceptance: post-pass MLIR must contain exactly `K_TILES` =
// `K / BLOCK_K` = 4 `metal.simdgroup_multiply_accumulate` ops.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @dot_universal_kloop4(%a_ptr: !tt.ptr<f32>, %b_ptr: !tt.ptr<f32>, %c_ptr: !tt.ptr<f32>, %stride_am: i32, %stride_bk: i32, %stride_cm: i32) {
    %acc = arith.constant dense<0.000000e+00> : tensor<8x8xf32, #blocked>
    %c1_i32 = arith.constant 1 : i32
    %c4_i32 = arith.constant 4 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<8> : tensor<8x8xi32, #blocked>
    %c8_i32 = arith.constant 8 : i32
    %pid_m = tt.get_program_id x : i32
    %pid_n = tt.get_program_id y : i32
    %offs_m = arith.muli %pid_m, %c8_i32 : i32
    %r0 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %r1 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %offs_m_sp = tt.splat %offs_m : i32 -> tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %offs_m_full = arith.addi %offs_m_sp, %r0 : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %offs_n = arith.muli %pid_n, %c8_i32 : i32
    %offs_n_sp = tt.splat %offs_n : i32 -> tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %offs_n_full = arith.addi %offs_n_sp, %r1 : tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %a_row = tt.expand_dims %offs_m_full {axis = 1 : i32} : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<8x1xi32, #blocked>
    %a_stride = tt.splat %stride_am : i32 -> tensor<8x1xi32, #blocked>
    %a_row_off = arith.muli %a_row, %a_stride : tensor<8x1xi32, #blocked>
    %a_base_sp = tt.splat %a_ptr : !tt.ptr<f32> -> tensor<8x1x!tt.ptr<f32>, #blocked>
    %a_base = tt.addptr %a_base_sp, %a_row_off : tensor<8x1x!tt.ptr<f32>, #blocked>, tensor<8x1xi32, #blocked>
    %a_col = tt.expand_dims %r1 {axis = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x8xi32, #blocked>
    %a_base_bc = tt.broadcast %a_base : tensor<8x1x!tt.ptr<f32>, #blocked> -> tensor<8x8x!tt.ptr<f32>, #blocked>
    %a_col_bc = tt.broadcast %a_col : tensor<1x8xi32, #blocked> -> tensor<8x8xi32, #blocked>
    %a_ptrs = tt.addptr %a_base_bc, %a_col_bc : tensor<8x8x!tt.ptr<f32>, #blocked>, tensor<8x8xi32, #blocked>
    %b_row = tt.expand_dims %r0 {axis = 1 : i32} : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<8x1xi32, #blocked>
    %b_stride = tt.splat %stride_bk : i32 -> tensor<8x1xi32, #blocked>
    %b_row_off = arith.muli %b_row, %b_stride : tensor<8x1xi32, #blocked>
    %b_base_sp = tt.splat %b_ptr : !tt.ptr<f32> -> tensor<8x1x!tt.ptr<f32>, #blocked>
    %b_base = tt.addptr %b_base_sp, %b_row_off : tensor<8x1x!tt.ptr<f32>, #blocked>, tensor<8x1xi32, #blocked>
    %b_col = tt.expand_dims %offs_n_full {axis = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x8xi32, #blocked>
    %b_base_bc = tt.broadcast %b_base : tensor<8x1x!tt.ptr<f32>, #blocked> -> tensor<8x8x!tt.ptr<f32>, #blocked>
    %b_col_bc = tt.broadcast %b_col : tensor<1x8xi32, #blocked> -> tensor<8x8xi32, #blocked>
    %b_ptrs = tt.addptr %b_base_bc, %b_col_bc : tensor<8x8x!tt.ptr<f32>, #blocked>, tensor<8x8xi32, #blocked>
    %b_bump_s = arith.muli %stride_bk, %c8_i32 : i32
    %b_bump = tt.splat %b_bump_s : i32 -> tensor<8x8xi32, #blocked>
    %loop:3 = scf.for %_ = %c0_i32 to %c4_i32 step %c1_i32 iter_args(%a_p = %a_ptrs, %b_p = %b_ptrs, %acc_iter = %acc) -> (tensor<8x8x!tt.ptr<f32>, #blocked>, tensor<8x8x!tt.ptr<f32>, #blocked>, tensor<8x8xf32, #blocked>)  : i32 {
      %a_blk = tt.load %a_p : tensor<8x8x!tt.ptr<f32>, #blocked>
      %b_blk = tt.load %b_p : tensor<8x8x!tt.ptr<f32>, #blocked>
      %a_dot = ttg.convert_layout %a_blk : tensor<8x8xf32, #blocked> -> tensor<8x8xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
      %b_dot = ttg.convert_layout %b_blk : tensor<8x8xf32, #blocked> -> tensor<8x8xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
      %acc_next = tt.dot %a_dot, %b_dot, %acc_iter : tensor<8x8xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<8x8xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<8x8xf32, #blocked>
      %a_p_next = tt.addptr %a_p, %cst : tensor<8x8x!tt.ptr<f32>, #blocked>, tensor<8x8xi32, #blocked>
      %b_p_next = tt.addptr %b_p, %b_bump : tensor<8x8x!tt.ptr<f32>, #blocked>, tensor<8x8xi32, #blocked>
      scf.yield %a_p_next, %b_p_next, %acc_next : tensor<8x8x!tt.ptr<f32>, #blocked>, tensor<8x8x!tt.ptr<f32>, #blocked>, tensor<8x8xf32, #blocked>
    }
    %c_stride = tt.splat %stride_cm : i32 -> tensor<8x1xi32, #blocked>
    %c_row_off = arith.muli %a_row, %c_stride : tensor<8x1xi32, #blocked>
    %c_base_sp = tt.splat %c_ptr : !tt.ptr<f32> -> tensor<8x1x!tt.ptr<f32>, #blocked>
    %c_base = tt.addptr %c_base_sp, %c_row_off : tensor<8x1x!tt.ptr<f32>, #blocked>, tensor<8x1xi32, #blocked>
    %c_base_bc = tt.broadcast %c_base : tensor<8x1x!tt.ptr<f32>, #blocked> -> tensor<8x8x!tt.ptr<f32>, #blocked>
    %c_ptrs = tt.addptr %c_base_bc, %b_col_bc : tensor<8x8x!tt.ptr<f32>, #blocked>, tensor<8x8xi32, #blocked>
    tt.store %c_ptrs, %loop#2 : tensor<8x8x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// CHECK-LABEL: metal.kernel dot_universal_kloop4
// CHECK-NOT: ttg.convert_layout
// CHECK-NOT: tt.dot
// CHECK-NOT: scf.for
// CHECK: metal.simdgroup_matrix_zero
// CHECK-COUNT-4: metal.simdgroup_multiply_accumulate
// CHECK: metal.simdgroup_store
// CHECK: metal.return
