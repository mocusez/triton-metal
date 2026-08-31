// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | FileCheck %s
// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | FileCheck %s --check-prefix=STATIC
// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | FileCheck %s --check-prefix=STATIC-LOAD
// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | FileCheck %s --check-prefix=STATIC-MMA
// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | FileCheck %s --check-prefix=STATIC-STORE
// RUN: sed -e 's/warpsPerCTA = \[4, 1\]/warpsPerCTA = [128, 1]/g' -e 's/"ttg.num-warps" = 4 : i32/"ttg.num-warps" = 128 : i32/g' %s | not triton-metal-opt --convert-tritongpu-to-metal 2>&1 | FileCheck %s --check-prefix=REJECT
//
// AC4 v6 multi-warp 64×64 dot at num_warps=4. The conversion pass
// matches `tryUnrollCanonical3IterArgDot` and emits a per-warp
// triple-unrolled simdgroup-matrix body partitioned by
// `simdgroup_index_in_threadgroup`. factorWarps(4, 8, 8) picks
// (warpsM, warpsN) = (2, 2), so each warp owns mPerWarp=4 × nPerWarp=4
// output tiles × K_TILES=4 inner k-iterations.
//
// AC4-S1 (a/b/c/d) predicate counts:
//   - exactly 1 metal.simdgroup_index
//   - exactly 1 arith.divui %widx, %c2  (warpM)
//   - exactly 1 arith.remui %widx, %c2  (warpN)
//   - exactly 64 metal.simdgroup_multiply_accumulate (=256/num_warps)
//   - every metal.simdgroup_load_device_staged carries `warp[…]`
//   - per-warp output stores via metal.simdgroup_store
//
// These checks cover both multi-warp coordination and partial-tile handling.

#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: metal.kernel k
  tt.func public @k(%a_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %b_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %c_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %stride_am: i32 {tt.divisibility = 16 : i32}, %stride_bk: i32 {tt.divisibility = 16 : i32}, %stride_cm: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %acc = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #blocked>
    %c1_i32 = arith.constant 1 : i32
    %c4_i32 = arith.constant 4 : i32
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c8_i32 = arith.constant 8 : i32
    %cst = arith.constant dense<8> : tensor<64x8xi32, #blocked1>
    %pid_m = tt.get_program_id x : i32
    %pid_n = tt.get_program_id y : i32
    %offs_m = arith.muli %pid_m, %c64_i32 : i32
    %offs_m_0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %offs_m_1 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %offs_m_2 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
    %offs_m_3 = tt.splat %offs_m : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %offs_m_4 = tt.splat %offs_m : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %offs_m_5 = arith.addi %offs_m_3, %offs_m_0 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %offs_m_6 = arith.addi %offs_m_4, %offs_m_1 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %offs_n = arith.muli %pid_n, %c64_i32 : i32
    %offs_n_7 = tt.splat %offs_n : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
    %offs_n_8 = arith.addi %offs_n_7, %offs_m_2 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
    %a_ptrs = tt.expand_dims %offs_m_5 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<64x1xi32, #blocked1>
    %a_ptrs_9 = tt.expand_dims %offs_m_6 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<64x1xi32, #blocked2>
    %a_ptrs_10 = tt.splat %stride_am : i32 -> tensor<64x1xi32, #blocked1>
    %a_ptrs_11 = arith.muli %a_ptrs, %a_ptrs_10 : tensor<64x1xi32, #blocked1>
    %a_ptrs_12 = tt.splat %a_ptr : !tt.ptr<f32> -> tensor<64x1x!tt.ptr<f32>, #blocked1>
    %a_ptrs_13 = tt.addptr %a_ptrs_12, %a_ptrs_11 : tensor<64x1x!tt.ptr<f32>, #blocked1>, tensor<64x1xi32, #blocked1>
    %a_ptrs_14 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %a_ptrs_15 = tt.expand_dims %a_ptrs_14 {axis = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x8xi32, #blocked1>
    %a_ptrs_16 = tt.broadcast %a_ptrs_13 : tensor<64x1x!tt.ptr<f32>, #blocked1> -> tensor<64x8x!tt.ptr<f32>, #blocked1>
    %a_ptrs_17 = tt.broadcast %a_ptrs_15 : tensor<1x8xi32, #blocked1> -> tensor<64x8xi32, #blocked1>
    %a_ptrs_18 = tt.addptr %a_ptrs_16, %a_ptrs_17 : tensor<64x8x!tt.ptr<f32>, #blocked1>, tensor<64x8xi32, #blocked1>
    %b_ptrs = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %b_ptrs_19 = tt.expand_dims %b_ptrs {axis = 1 : i32} : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<8x1xi32, #blocked2>
    %b_ptrs_20 = tt.splat %stride_bk : i32 -> tensor<8x1xi32, #blocked2>
    %b_ptrs_21 = arith.muli %b_ptrs_19, %b_ptrs_20 : tensor<8x1xi32, #blocked2>
    %b_ptrs_22 = tt.splat %b_ptr : !tt.ptr<f32> -> tensor<8x1x!tt.ptr<f32>, #blocked2>
    %b_ptrs_23 = tt.addptr %b_ptrs_22, %b_ptrs_21 : tensor<8x1x!tt.ptr<f32>, #blocked2>, tensor<8x1xi32, #blocked2>
    %b_ptrs_24 = tt.expand_dims %offs_n_8 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x64xi32, #blocked2>
    %b_ptrs_25 = tt.broadcast %b_ptrs_23 : tensor<8x1x!tt.ptr<f32>, #blocked2> -> tensor<8x64x!tt.ptr<f32>, #blocked2>
    %b_ptrs_26 = tt.broadcast %b_ptrs_24 : tensor<1x64xi32, #blocked2> -> tensor<8x64xi32, #blocked2>
    %b_ptrs_27 = tt.addptr %b_ptrs_25, %b_ptrs_26 : tensor<8x64x!tt.ptr<f32>, #blocked2>, tensor<8x64xi32, #blocked2>
    %b_ptrs_28 = arith.muli %stride_bk, %c8_i32 : i32
    %b_ptrs_29 = tt.splat %b_ptrs_28 : i32 -> tensor<8x64xi32, #blocked2>
    %acc_30:3 = scf.for %_ = %c0_i32 to %c4_i32 step %c1_i32 iter_args(%acc_37 = %acc, %a_ptrs_38 = %a_ptrs_18, %b_ptrs_39 = %b_ptrs_27) -> (tensor<64x64xf32, #blocked>, tensor<64x8x!tt.ptr<f32>, #blocked1>, tensor<8x64x!tt.ptr<f32>, #blocked2>)  : i32 {
      %a = tt.load %a_ptrs_38 : tensor<64x8x!tt.ptr<f32>, #blocked1>
      %b = tt.load %b_ptrs_39 : tensor<8x64x!tt.ptr<f32>, #blocked2>
      %a_40 = ttg.convert_layout %a : tensor<64x8xf32, #blocked1> -> tensor<64x8xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
      %b_41 = ttg.convert_layout %b : tensor<8x64xf32, #blocked2> -> tensor<8x64xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
      %acc_42 = tt.dot %a_40, %b_41, %acc_37 : tensor<64x8xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<8x64xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<64x64xf32, #blocked>
      %a_ptrs_43 = tt.addptr %a_ptrs_38, %cst : tensor<64x8x!tt.ptr<f32>, #blocked1>, tensor<64x8xi32, #blocked1>
      %b_ptrs_44 = tt.addptr %b_ptrs_39, %b_ptrs_29 : tensor<8x64x!tt.ptr<f32>, #blocked2>, tensor<8x64xi32, #blocked2>
      scf.yield %acc_42, %a_ptrs_43, %b_ptrs_44 : tensor<64x64xf32, #blocked>, tensor<64x8x!tt.ptr<f32>, #blocked1>, tensor<8x64x!tt.ptr<f32>, #blocked2>
    }
    %c_ptrs = tt.splat %stride_cm : i32 -> tensor<64x1xi32, #blocked2>
    %c_ptrs_31 = arith.muli %a_ptrs_9, %c_ptrs : tensor<64x1xi32, #blocked2>
    %c_ptrs_32 = tt.splat %c_ptr : !tt.ptr<f32> -> tensor<64x1x!tt.ptr<f32>, #blocked2>
    %c_ptrs_33 = tt.addptr %c_ptrs_32, %c_ptrs_31 : tensor<64x1x!tt.ptr<f32>, #blocked2>, tensor<64x1xi32, #blocked2>
    %c_ptrs_34 = tt.broadcast %c_ptrs_33 : tensor<64x1x!tt.ptr<f32>, #blocked2> -> tensor<64x64x!tt.ptr<f32>, #blocked2>
    %c_ptrs_35 = tt.broadcast %b_ptrs_24 : tensor<1x64xi32, #blocked2> -> tensor<64x64xi32, #blocked2>
    %c_ptrs_36 = tt.addptr %c_ptrs_34, %c_ptrs_35 : tensor<64x64x!tt.ptr<f32>, #blocked2>, tensor<64x64xi32, #blocked2>
    %0 = ttg.convert_layout %acc_30#0 : tensor<64x64xf32, #blocked> -> tensor<64x64xf32, #blocked2>
    tt.store %c_ptrs_36, %0 : tensor<64x64x!tt.ptr<f32>, #blocked2>
    tt.return
  }
}

// CHECK: metal.simdgroup_index
// CHECK-NOT: metal.simdgroup_index
// CHECK-COUNT-64: metal.simdgroup_multiply_accumulate
// CHECK-NOT: metal.simdgroup_multiply_accumulate
// CHECK: metal.return
// REJECT: canonical multi-tile dot has no valid warp partition

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  // STATIC-LABEL: metal.kernel k_masked_static
  tt.func public @k_masked_static(%a_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %b_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %c_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %M: i32, %N: i32, %stride_am: i32 {tt.divisibility = 16 : i32}, %stride_bk: i32 {tt.divisibility = 16 : i32}, %stride_cm: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c0 = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #blocked>
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %c64_i32 = arith.constant 64 : i32
    %a_bump = arith.constant dense<8> : tensor<64x8xi32, #blocked1>
    %b_bump = arith.constant dense<8> : tensor<8x64xi32, #blocked2>
    %a_zero = arith.constant dense<0.000000e+00> : tensor<64x8xf32, #blocked1>
    %b_zero = arith.constant dense<0.000000e+00> : tensor<8x64xf32, #blocked2>
    %pid_m = tt.get_program_id x : i32
    %pid_n = tt.get_program_id y : i32
    %origin_m = arith.muli %pid_m, %c64_i32 : i32
    %origin_n = arith.muli %pid_n, %c64_i32 : i32
    %rm_a = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %rk_a = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %rk_b = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %rn_b = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
    %om_s_a = tt.splat %origin_m : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %on_s_b = tt.splat %origin_n : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
    %offs_m_a = arith.addi %om_s_a, %rm_a : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %offs_n_b = arith.addi %on_s_b, %rn_b : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
    %a_row = tt.expand_dims %offs_m_a {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<64x1xi32, #blocked1>
    %a_stride = tt.splat %stride_am : i32 -> tensor<64x1xi32, #blocked1>
    %a_row_off = arith.muli %a_row, %a_stride : tensor<64x1xi32, #blocked1>
    %a_base_s = tt.splat %a_ptr : !tt.ptr<f32> -> tensor<64x1x!tt.ptr<f32>, #blocked1>
    %a_base = tt.addptr %a_base_s, %a_row_off : tensor<64x1x!tt.ptr<f32>, #blocked1>, tensor<64x1xi32, #blocked1>
    %a_k = tt.expand_dims %rk_a {axis = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x8xi32, #blocked1>
    %a_base_bc = tt.broadcast %a_base : tensor<64x1x!tt.ptr<f32>, #blocked1> -> tensor<64x8x!tt.ptr<f32>, #blocked1>
    %a_k_bc = tt.broadcast %a_k : tensor<1x8xi32, #blocked1> -> tensor<64x8xi32, #blocked1>
    %a_ptrs = tt.addptr %a_base_bc, %a_k_bc : tensor<64x8x!tt.ptr<f32>, #blocked1>, tensor<64x8xi32, #blocked1>
    %b_row = tt.expand_dims %rk_b {axis = 1 : i32} : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<8x1xi32, #blocked2>
    %b_stride = tt.splat %stride_bk : i32 -> tensor<8x1xi32, #blocked2>
    %b_row_off = arith.muli %b_row, %b_stride : tensor<8x1xi32, #blocked2>
    %b_base_s = tt.splat %b_ptr : !tt.ptr<f32> -> tensor<8x1x!tt.ptr<f32>, #blocked2>
    %b_base = tt.addptr %b_base_s, %b_row_off : tensor<8x1x!tt.ptr<f32>, #blocked2>, tensor<8x1xi32, #blocked2>
    %b_col = tt.expand_dims %offs_n_b {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x64xi32, #blocked2>
    %b_base_bc = tt.broadcast %b_base : tensor<8x1x!tt.ptr<f32>, #blocked2> -> tensor<8x64x!tt.ptr<f32>, #blocked2>
    %b_col_bc = tt.broadcast %b_col : tensor<1x64xi32, #blocked2> -> tensor<8x64xi32, #blocked2>
    %b_ptrs = tt.addptr %b_base_bc, %b_col_bc : tensor<8x64x!tt.ptr<f32>, #blocked2>, tensor<8x64xi32, #blocked2>
    %m_s = tt.splat %M : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %n_s = tt.splat %N : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
    %m_ok = arith.cmpi slt, %offs_m_a, %m_s : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %n_ok = arith.cmpi slt, %offs_n_b, %n_s : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
    %m_ok_2d = tt.expand_dims %m_ok {axis = 1 : i32} : tensor<64xi1, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<64x1xi1, #blocked1>
    %n_ok_2d = tt.expand_dims %n_ok {axis = 0 : i32} : tensor<64xi1, #ttg.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x64xi1, #blocked2>
    %k_s_a = tt.splat %c8_i32 : i32 -> tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %k_s_b = tt.splat %c8_i32 : i32 -> tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %k_ok_a = arith.cmpi slt, %rk_a, %k_s_a : tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %k_ok_b = arith.cmpi slt, %rk_b, %k_s_b : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %k_ok_a_2d = tt.expand_dims %k_ok_a {axis = 0 : i32} : tensor<8xi1, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x8xi1, #blocked1>
    %k_ok_b_2d = tt.expand_dims %k_ok_b {axis = 1 : i32} : tensor<8xi1, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<8x1xi1, #blocked2>
    %a_m_bc = tt.broadcast %m_ok_2d : tensor<64x1xi1, #blocked1> -> tensor<64x8xi1, #blocked1>
    %a_k_mask_bc = tt.broadcast %k_ok_a_2d : tensor<1x8xi1, #blocked1> -> tensor<64x8xi1, #blocked1>
    %b_k_bc = tt.broadcast %k_ok_b_2d : tensor<8x1xi1, #blocked2> -> tensor<8x64xi1, #blocked2>
    %b_n_bc = tt.broadcast %n_ok_2d : tensor<1x64xi1, #blocked2> -> tensor<8x64xi1, #blocked2>
    %a_mask = arith.andi %a_m_bc, %a_k_mask_bc : tensor<64x8xi1, #blocked1>
    %b_mask = arith.andi %b_k_bc, %b_n_bc : tensor<8x64xi1, #blocked2>
    %loop:3 = scf.for %k0 = %c0_i32 to %c8_i32 step %c8_i32 iter_args(%a_p = %a_ptrs, %b_p = %b_ptrs, %acc_iter = %c0) -> (tensor<64x8x!tt.ptr<f32>, #blocked1>, tensor<8x64x!tt.ptr<f32>, #blocked2>, tensor<64x64xf32, #blocked>) : i32 {
      %a = tt.load %a_p, %a_mask, %a_zero : tensor<64x8x!tt.ptr<f32>, #blocked1>
      %b = tt.load %b_p, %b_mask, %b_zero : tensor<8x64x!tt.ptr<f32>, #blocked2>
      %a_dot = ttg.convert_layout %a : tensor<64x8xf32, #blocked1> -> tensor<64x8xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
      %b_dot = ttg.convert_layout %b : tensor<8x64xf32, #blocked2> -> tensor<8x64xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
      %acc_next = tt.dot %a_dot, %b_dot, %acc_iter : tensor<64x8xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<8x64xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<64x64xf32, #blocked>
      %a_next = tt.addptr %a_p, %a_bump : tensor<64x8x!tt.ptr<f32>, #blocked1>, tensor<64x8xi32, #blocked1>
      %b_next = tt.addptr %b_p, %b_bump : tensor<8x64x!tt.ptr<f32>, #blocked2>, tensor<8x64xi32, #blocked2>
      scf.yield %a_next, %b_next, %acc_next : tensor<64x8x!tt.ptr<f32>, #blocked1>, tensor<8x64x!tt.ptr<f32>, #blocked2>, tensor<64x64xf32, #blocked>
    }
    %rm_c = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %rn_c = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
    %om_c_s = tt.splat %origin_m : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %offs_m_c = arith.addi %om_c_s, %rm_c : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %c_row = tt.expand_dims %offs_m_c {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<64x1xi32, #blocked2>
    %c_stride = tt.splat %stride_cm : i32 -> tensor<64x1xi32, #blocked2>
    %c_row_off = arith.muli %c_row, %c_stride : tensor<64x1xi32, #blocked2>
    %c_base_s = tt.splat %c_ptr : !tt.ptr<f32> -> tensor<64x1x!tt.ptr<f32>, #blocked2>
    %c_base = tt.addptr %c_base_s, %c_row_off : tensor<64x1x!tt.ptr<f32>, #blocked2>, tensor<64x1xi32, #blocked2>
    %c_col = tt.expand_dims %rn_c {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x64xi32, #blocked2>
    %c_col_origin = tt.expand_dims %offs_n_b {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x64xi32, #blocked2>
    %c_base_bc = tt.broadcast %c_base : tensor<64x1x!tt.ptr<f32>, #blocked2> -> tensor<64x64x!tt.ptr<f32>, #blocked2>
    %c_col_bc = tt.broadcast %c_col_origin : tensor<1x64xi32, #blocked2> -> tensor<64x64xi32, #blocked2>
    %c_ptrs = tt.addptr %c_base_bc, %c_col_bc : tensor<64x64x!tt.ptr<f32>, #blocked2>, tensor<64x64xi32, #blocked2>
    %m_s_c = tt.splat %M : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %m_ok_c = arith.cmpi slt, %offs_m_c, %m_s_c : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %m_ok_c_2d = tt.expand_dims %m_ok_c {axis = 1 : i32} : tensor<64xi1, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<64x1xi1, #blocked2>
    %m_ok_c_bc = tt.broadcast %m_ok_c_2d : tensor<64x1xi1, #blocked2> -> tensor<64x64xi1, #blocked2>
    %n_ok_c_bc = tt.broadcast %n_ok_2d : tensor<1x64xi1, #blocked2> -> tensor<64x64xi1, #blocked2>
    %store_mask = arith.andi %m_ok_c_bc, %n_ok_c_bc : tensor<64x64xi1, #blocked2>
    %acc_store = ttg.convert_layout %loop#2 : tensor<64x64xf32, #blocked> -> tensor<64x64xf32, #blocked2>
    tt.store %c_ptrs, %acc_store, %store_mask : tensor<64x64x!tt.ptr<f32>, #blocked2>
    tt.return
  }
}

// STATIC: metal.simdgroup_index
// STATIC-NOT: tt.dot
// STATIC-LOAD-LABEL: metal.kernel k_masked_static
// STATIC-LOAD-COUNT-32: metal.simdgroup_load_device_staged_masked
// STATIC-MMA-LABEL: metal.kernel k_masked_static
// STATIC-MMA-COUNT-16: metal.simdgroup_multiply_accumulate
// STATIC-STORE-LABEL: metal.kernel k_masked_static
// STATIC-STORE-COUNT-16: metal.simdgroup_store{{.*}}partial{{.*}}warp
