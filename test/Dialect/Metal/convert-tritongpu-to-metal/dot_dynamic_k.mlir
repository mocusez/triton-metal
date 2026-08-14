// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | FileCheck %s
// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | triton-metal-translate --mlir-to-msl | FileCheck %s --check-prefix=MSL
//
// W2a fixture (`metal-lora-linear-fix-plan.md`). Canonical 3-iter-arg matmul
// (same shape as `dot_universal_grid.mlir`) but with a RUNTIME K-loop trip
// count: `scf.for %k = 0 to %K step 8` where `%K` is a function argument, not
// a compile-time constant. The static unroller (`tryUnrollCanonical3IterArgDot`)
// bails on the non-constant upper bound; `tryRuntimeKLoopCanonicalDot` handles
// it by emitting a fresh `scf.for` stepping the K axis by 8 (one simdgroup 8x8
// subtile per iteration) whose single iter_arg is the `simdgroup_matrix`
// accumulator.
//
// Acceptance (the inverse of the unrolled fixture): the post-pass MLIR must
//   1. drop every `ttg.convert_layout` and `tt.dot`;
//   2. PRESERVE an `scf.for` (the runtime K-loop is NOT unrolled);
//   3. contain exactly ONE `metal.simdgroup_multiply_accumulate`, inside the
//      loop, whose accumulator is the loop iter_arg;
//   4. init the accumulator once with `metal.simdgroup_matrix_zero` and store
//      the loop result once.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @dot_dynamic_k(%a_ptr: !tt.ptr<f32>, %b_ptr: !tt.ptr<f32>, %c_ptr: !tt.ptr<f32>, %K: i32, %M: i32, %N: i32, %stride_am: i32, %stride_bk: i32, %stride_cm: i32) {
    %acc = arith.constant dense<0.000000e+00> : tensor<8x8xf32, #blocked>
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
    // Runtime K-loop: upper bound is the %K argument, step == BK == 8.
    %loop:3 = scf.for %k = %c0_i32 to %K step %c8_i32 iter_args(%a_p = %a_ptrs, %b_p = %b_ptrs, %acc_iter = %acc) -> (tensor<8x8x!tt.ptr<f32>, #blocked>, tensor<8x8x!tt.ptr<f32>, #blocked>, tensor<8x8xf32, #blocked>)  : i32 {
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
    // Local remaining-count mask. The store pointer already carries the
    // program-grid origin, so comparing these aranges against %M/%N is in tile
    // coordinates. The matcher must normalize each bound to origin+remaining
    // before the MSL epilogue compares it with global gi/gj.
    %offs_m_2d = tt.expand_dims %r0 {axis = 1 : i32} : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<8x1xi32, #blocked>
    %m_splat_2d = tt.splat %M : i32 -> tensor<8x1xi32, #blocked>
    %m_cmp_2d = arith.cmpi slt, %offs_m_2d, %m_splat_2d : tensor<8x1xi32, #blocked>
    %m_cmp_2d_bc = tt.broadcast %m_cmp_2d : tensor<8x1xi1, #blocked> -> tensor<8x8xi1, #blocked>
    %offs_n_2d = tt.expand_dims %r1 {axis = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x8xi32, #blocked>
    %n_splat_2d = tt.splat %N : i32 -> tensor<1x8xi32, #blocked>
    %n_cmp_2d = arith.cmpi slt, %offs_n_2d, %n_splat_2d : tensor<1x8xi32, #blocked>
    %n_cmp_2d_bc = tt.broadcast %n_cmp_2d : tensor<1x8xi1, #blocked> -> tensor<8x8xi1, #blocked>
    %store_mask = arith.andi %m_cmp_2d_bc, %n_cmp_2d_bc : tensor<8x8xi1, #blocked>
    tt.store %c_ptrs, %loop#2, %store_mask : tensor<8x8x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// CHECK-LABEL: metal.kernel dot_dynamic_k
// CHECK-NOT: ttg.convert_layout
// CHECK-NOT: tt.dot
// CHECK: metal.simdgroup_matrix_zero
// The runtime K-loop is preserved (not statically unrolled).
// CHECK: scf.for
// CHECK: metal.simdgroup_multiply_accumulate
// CHECK: metal.simdgroup_store
// CHECK-SAME: partial
// CHECK: metal.return

// The local M/N bounds are normalized to global origins. This must not regress
// to `gi < M && gj < N`, which drops every nonzero program tile.
// MSL: uint gi =
// MSL: uint gj =
// MSL: if (gi <
// MSL-SAME: +
// MSL: && gj <
// MSL-SAME: +
