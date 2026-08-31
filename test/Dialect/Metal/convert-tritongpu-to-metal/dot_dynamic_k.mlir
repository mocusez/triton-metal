// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | FileCheck %s
// RUN: sed -e 's/%other_zero = arith.constant dense<0.000000e+00>/%other_zero = arith.constant dense<1.000000e+00>/' %s | not triton-metal-opt --convert-tritongpu-to-metal 2>&1 | FileCheck %s --check-prefix=NONZERO
// RUN: sed -e 's/store_mask = arith.andi %m_ok_bc, %n_ok_bc/store_mask = arith.andi %m_ok_bc, %m_ok_bc/' %s | not triton-metal-opt --convert-tritongpu-to-metal 2>&1 | FileCheck %s --check-prefix=NONRECT
// RUN: sed -e 's/a_mask = arith.andi %a_m_bc, %k_bc2/a_mask = arith.andi %a_m_bc, %a_m_bc/' %s | not triton-metal-opt --convert-tritongpu-to-metal 2>&1 | FileCheck %s --check-prefix=BADINPUT
// RUN: sed -e 's/w_mask = arith.andi %w_n_bc, %k_bc2/w_mask = arith.andi %w_n_bc, %w_n_bc/' %s | not triton-metal-opt --convert-tritongpu-to-metal 2>&1 | FileCheck %s --check-prefix=BADINPUT
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

#blocked64 = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked64x32 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked32x64 = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [2, 16], warpsPerCTA = [1, 4], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: metal.kernel dyn_k_masked_multitile_transB
  tt.func public @dyn_k_masked_multitile_transB(%a_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %w_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %c_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %b_off: i32, %M: i32, %N: i32, %K: i32, %stride_am: i32 {tt.divisibility = 16 : i32}, %stride_wn: i32 {tt.divisibility = 16 : i32}, %stride_cm: i32) attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %c64_i32 = arith.constant 64 : i32
    %acc = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #blocked64>
    %bump = arith.constant dense<32> : tensor<64x32xi32, #blocked64x32>
    %other_zero = arith.constant dense<0.000000e+00> : tensor<64x32xf32, #blocked64x32>
    %pid_m = tt.get_program_id x : i32
    %pid_n = tt.get_program_id y : i32
    %origin_m = arith.muli %pid_m, %c64_i32 : i32
    %origin_n = arith.muli %pid_n, %c64_i32 : i32
    %rm = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked64x32}>>
    %rn = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked64x32}>>
    %rk = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked64x32}>>
    %origin_m_s = tt.splat %origin_m : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked64x32}>>
    %origin_n_s = tt.splat %origin_n : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked64x32}>>
    %offs_m = arith.addi %origin_m_s, %rm : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked64x32}>>
    %offs_n = arith.addi %origin_n_s, %rn : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked64x32}>>
    %a_row = tt.expand_dims %offs_m {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked64x32}>> -> tensor<64x1xi32, #blocked64x32>
    %a_stride = tt.splat %stride_am : i32 -> tensor<64x1xi32, #blocked64x32>
    %a_row_off = arith.muli %a_row, %a_stride : tensor<64x1xi32, #blocked64x32>
    %a_base_s = tt.splat %a_ptr : !tt.ptr<f32> -> tensor<64x1x!tt.ptr<f32>, #blocked64x32>
    %a_base = tt.addptr %a_base_s, %a_row_off : tensor<64x1x!tt.ptr<f32>, #blocked64x32>, tensor<64x1xi32, #blocked64x32>
    %k_2d = tt.expand_dims %rk {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked64x32}>> -> tensor<1x32xi32, #blocked64x32>
    %a_base_bc = tt.broadcast %a_base : tensor<64x1x!tt.ptr<f32>, #blocked64x32> -> tensor<64x32x!tt.ptr<f32>, #blocked64x32>
    %k_bc = tt.broadcast %k_2d : tensor<1x32xi32, #blocked64x32> -> tensor<64x32xi32, #blocked64x32>
    %a_ptrs = tt.addptr %a_base_bc, %k_bc : tensor<64x32x!tt.ptr<f32>, #blocked64x32>, tensor<64x32xi32, #blocked64x32>
    %w_row = tt.expand_dims %offs_n {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked64x32}>> -> tensor<64x1xi32, #blocked64x32>
    %w_stride = tt.splat %stride_wn : i32 -> tensor<64x1xi32, #blocked64x32>
    %w_row_off = arith.muli %w_row, %w_stride : tensor<64x1xi32, #blocked64x32>
    %w_offset_ptr = tt.addptr %w_ptr, %b_off : !tt.ptr<f32>, i32
    %w_base_s = tt.splat %w_offset_ptr : !tt.ptr<f32> -> tensor<64x1x!tt.ptr<f32>, #blocked64x32>
    %w_base = tt.addptr %w_base_s, %w_row_off : tensor<64x1x!tt.ptr<f32>, #blocked64x32>, tensor<64x1xi32, #blocked64x32>
    %w_base_bc = tt.broadcast %w_base : tensor<64x1x!tt.ptr<f32>, #blocked64x32> -> tensor<64x32x!tt.ptr<f32>, #blocked64x32>
    %w_ptrs = tt.addptr %w_base_bc, %k_bc : tensor<64x32x!tt.ptr<f32>, #blocked64x32>, tensor<64x32xi32, #blocked64x32>
    %loop:3 = scf.for %k0 = %c0_i32 to %K step %c32_i32 iter_args(%a_p = %a_ptrs, %w_p = %w_ptrs, %acc_iter = %acc) -> (tensor<64x32x!tt.ptr<f32>, #blocked64x32>, tensor<64x32x!tt.ptr<f32>, #blocked64x32>, tensor<64x64xf32, #blocked64>)  : i32 {
      %k_rem = arith.subi %K, %k0 : i32
      %k_rem_s = tt.splat %k_rem : i32 -> tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked64x32}>>
      %k_mask = arith.cmpi slt, %rk, %k_rem_s : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked64x32}>>
      %m_s = tt.splat %M : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked64x32}>>
      %n_s = tt.splat %N : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked64x32}>>
      %m_mask = arith.cmpi slt, %offs_m, %m_s : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked64x32}>>
      %n_mask = arith.cmpi slt, %offs_n, %n_s : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked64x32}>>
      %a_m = tt.expand_dims %m_mask {axis = 1 : i32} : tensor<64xi1, #ttg.slice<{dim = 1, parent = #blocked64x32}>> -> tensor<64x1xi1, #blocked64x32>
      %w_n = tt.expand_dims %n_mask {axis = 1 : i32} : tensor<64xi1, #ttg.slice<{dim = 1, parent = #blocked64x32}>> -> tensor<64x1xi1, #blocked64x32>
      %k_1x32 = tt.expand_dims %k_mask {axis = 0 : i32} : tensor<32xi1, #ttg.slice<{dim = 0, parent = #blocked64x32}>> -> tensor<1x32xi1, #blocked64x32>
      %a_m_bc = tt.broadcast %a_m : tensor<64x1xi1, #blocked64x32> -> tensor<64x32xi1, #blocked64x32>
      %w_n_bc = tt.broadcast %w_n : tensor<64x1xi1, #blocked64x32> -> tensor<64x32xi1, #blocked64x32>
      %k_bc2 = tt.broadcast %k_1x32 : tensor<1x32xi1, #blocked64x32> -> tensor<64x32xi1, #blocked64x32>
      %a_mask = arith.andi %a_m_bc, %k_bc2 : tensor<64x32xi1, #blocked64x32>
      %w_mask = arith.andi %w_n_bc, %k_bc2 : tensor<64x32xi1, #blocked64x32>
      %a = tt.load %a_p, %a_mask, %other_zero : tensor<64x32x!tt.ptr<f32>, #blocked64x32>
      %w = tt.load %w_p, %w_mask, %other_zero : tensor<64x32x!tt.ptr<f32>, #blocked64x32>
      %wt = tt.trans %w {order = array<i32: 1, 0>} : tensor<64x32xf32, #blocked64x32> -> tensor<32x64xf32, #blocked32x64>
      %a_dot = ttg.convert_layout %a : tensor<64x32xf32, #blocked64x32> -> tensor<64x32xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked64}>>
      %w_dot = ttg.convert_layout %wt : tensor<32x64xf32, #blocked32x64> -> tensor<32x64xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked64}>>
      %acc_next = tt.dot %a_dot, %w_dot, %acc_iter : tensor<64x32xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked64}>> * tensor<32x64xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked64}>> -> tensor<64x64xf32, #blocked64>
      %a_next = tt.addptr %a_p, %bump : tensor<64x32x!tt.ptr<f32>, #blocked64x32>, tensor<64x32xi32, #blocked64x32>
      %w_next = tt.addptr %w_p, %bump : tensor<64x32x!tt.ptr<f32>, #blocked64x32>, tensor<64x32xi32, #blocked64x32>
      scf.yield %a_next, %w_next, %acc_next : tensor<64x32x!tt.ptr<f32>, #blocked64x32>, tensor<64x32x!tt.ptr<f32>, #blocked64x32>, tensor<64x64xf32, #blocked64>
    }
    %rm_c = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked64}>>
    %rn_c = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked64}>>
    %om_c_s = tt.splat %origin_m : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked64}>>
    %on_c_s = tt.splat %origin_n : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked64}>>
    %offs_m_c = arith.addi %om_c_s, %rm_c : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked64}>>
    %offs_n_c = arith.addi %on_c_s, %rn_c : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked64}>>
    %c_row = tt.expand_dims %offs_m_c {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked64}>> -> tensor<64x1xi32, #blocked64>
    %c_stride = tt.splat %stride_cm : i32 -> tensor<64x1xi32, #blocked64>
    %c_row_off = arith.muli %c_row, %c_stride : tensor<64x1xi32, #blocked64>
    %c_base_s = tt.splat %c_ptr : !tt.ptr<f32> -> tensor<64x1x!tt.ptr<f32>, #blocked64>
    %c_base = tt.addptr %c_base_s, %c_row_off : tensor<64x1x!tt.ptr<f32>, #blocked64>, tensor<64x1xi32, #blocked64>
    %c_col = tt.expand_dims %offs_n_c {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked64}>> -> tensor<1x64xi32, #blocked64>
    %c_base_bc = tt.broadcast %c_base : tensor<64x1x!tt.ptr<f32>, #blocked64> -> tensor<64x64x!tt.ptr<f32>, #blocked64>
    %c_col_bc = tt.broadcast %c_col : tensor<1x64xi32, #blocked64> -> tensor<64x64xi32, #blocked64>
    %c_ptrs = tt.addptr %c_base_bc, %c_col_bc : tensor<64x64x!tt.ptr<f32>, #blocked64>, tensor<64x64xi32, #blocked64>
    %M_s_c = tt.splat %M : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked64}>>
    %N_s_c = tt.splat %N : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked64}>>
    %m_ok = arith.cmpi slt, %offs_m_c, %M_s_c : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked64}>>
    %n_ok = arith.cmpi slt, %offs_n_c, %N_s_c : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked64}>>
    %m_ok_2d = tt.expand_dims %m_ok {axis = 1 : i32} : tensor<64xi1, #ttg.slice<{dim = 1, parent = #blocked64}>> -> tensor<64x1xi1, #blocked64>
    %n_ok_2d = tt.expand_dims %n_ok {axis = 0 : i32} : tensor<64xi1, #ttg.slice<{dim = 0, parent = #blocked64}>> -> tensor<1x64xi1, #blocked64>
    %m_ok_bc = tt.broadcast %m_ok_2d : tensor<64x1xi1, #blocked64> -> tensor<64x64xi1, #blocked64>
    %n_ok_bc = tt.broadcast %n_ok_2d : tensor<1x64xi1, #blocked64> -> tensor<64x64xi1, #blocked64>
    %store_mask = arith.andi %m_ok_bc, %n_ok_bc : tensor<64x64xi1, #blocked64>
    tt.store %c_ptrs, %loop#2, %store_mask : tensor<64x64x!tt.ptr<f32>, #blocked64>
    tt.return
  }
}

// CHECK: metal.simdgroup_index
// CHECK-COUNT-8: metal.simdgroup_load_device_staged_masked
// CHECK-COUNT-16: metal.simdgroup_multiply_accumulate
// CHECK-COUNT-16: metal.simdgroup_store
// CHECK-SAME: partial
// CHECK-SAME: warp
// CHECK-NOT: tt.dot
// CHECK-NOT: tt.trans
// CHECK-NOT: ttg.convert_layout
// NONZERO: masked dot load requires numeric zero `other`
// NONRECT: masked multi-tile canonical dot requires a rectangular output mask
// BADINPUT: rectangular input masks tied to the loaded pointer coordinates
