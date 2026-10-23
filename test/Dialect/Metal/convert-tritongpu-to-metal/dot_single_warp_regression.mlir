// RUN: triton-metal-opt --split-input-file --convert-tritongpu-to-metal %s | FileCheck %s
//
// AC4-S6b: single-warp regression — when ttg.num-warps=1, the AC4 matcher
// must NOT emit any multi-warp scaffolding (no metal.simdgroup_index,
// no arith.divui/remui, no `warp [...]` operand on staged loads). The
// shared stage buffer reduces to a flat `[64]` shape (no per-warp slice).
//
// At 64×64 with num_warps=1, factorWarps is skipped (multiWarp=false) and
// the matcher emits 8×8×4 = 256 metal.simdgroup_multiply_accumulate ops
// for a single warp owning all output tiles.

#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [2, 16], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [16, 2], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: metal.kernel k
  // CHECK-NOT: metal.simdgroup_index
  // CHECK-NOT: arith.divui
  // CHECK-NOT: arith.remui
  // CHECK-NOT: warp [
  // CHECK-COUNT-256: metal.simdgroup_multiply_accumulate
  // CHECK-NOT: metal.simdgroup_multiply_accumulate
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

// -----

// The output broadcast changes the row cone from #blocked2 to #blocked3
// after the dot. Its scalar pid * BM origin has no layout to remap: cloning
// that scalar beside the store used to put its definition after the matmul
// operations that consume it. Keep the original scalar for both A and C.
// CHECK-LABEL: metal.kernel dot_store_broadcast_preserves_scalar_origin
// CHECK: %[[ROW_PID:.*]] = metal.threadgroup_id "x"
// CHECK: %[[ROW_I32:.*]] = builtin.unrealized_conversion_cast %[[ROW_PID]] : ui32 to i32
// CHECK: %[[ROW:.*]] = arith.muli %[[ROW_I32]], {{.*}} : i32
// CHECK: %[[COL_PID:.*]] = metal.threadgroup_id "y"
// CHECK: %[[COL_I32:.*]] = builtin.unrealized_conversion_cast %[[COL_PID]] : ui32 to i32
// CHECK: %[[COL:.*]] = arith.muli %[[COL_I32]], {{.*}} : i32
// CHECK: builtin.unrealized_conversion_cast %[[ROW]] : i32 to ui32
// CHECK: builtin.unrealized_conversion_cast %[[COL]] : i32 to ui32
// CHECK: builtin.unrealized_conversion_cast %[[ROW]] : i32 to ui32
// CHECK: builtin.unrealized_conversion_cast %[[COL]] : i32 to ui32
// CHECK: metal.simdgroup_load_device_staged
// CHECK: metal.simdgroup_multiply_accumulate
// CHECK: metal.simdgroup_store
// CHECK-NOT: metal.threadgroup_id
// CHECK-NOT: tt.dot
// CHECK: metal.return

#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @dot_store_broadcast_preserves_scalar_origin(%a_ptr: !tt.ptr<f32>, %b_ptr: !tt.ptr<f32>, %c_ptr: !tt.ptr<f32>, %N: i32, %K: i32) {
    %c = arith.constant dense<0.000000e+00> : tensor<8x512xf32, #blocked>
    %offs_n = arith.constant 512 : i32
    %offs_m = arith.constant 8 : i32
    %offs_m_0 = tt.get_program_id x : i32
    %offs_m_1 = arith.muli %offs_m_0, %offs_m : i32
    %offs_m_2 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %offs_m_3 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %offs_m_4 = tt.splat %offs_m_1 : i32 -> tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %offs_m_5 = tt.splat %offs_m_1 : i32 -> tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %offs_m_6 = arith.addi %offs_m_4, %offs_m_2 : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %offs_m_7 = arith.addi %offs_m_5, %offs_m_3 : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %offs_n_8 = tt.get_program_id y : i32
    %offs_n_9 = arith.muli %offs_n_8, %offs_n : i32
    %offs_n_10 = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32, #ttg.slice<{dim = 0, parent = #blocked3}>>
    %offs_n_11 = tt.splat %offs_n_9 : i32 -> tensor<512xi32, #ttg.slice<{dim = 0, parent = #blocked3}>>
    %offs_n_12 = arith.addi %offs_n_11, %offs_n_10 : tensor<512xi32, #ttg.slice<{dim = 0, parent = #blocked3}>>
    %a = tt.expand_dims %offs_m_6 {axis = 1 : i32} : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<8x1xi32, #blocked1>
    %a_13 = tt.expand_dims %offs_m_7 {axis = 1 : i32} : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<8x1xi32, #blocked2>
    %a_14 = tt.splat %K : i32 -> tensor<8x1xi32, #blocked1>
    %a_15 = arith.muli %a, %a_14 : tensor<8x1xi32, #blocked1>
    %a_16 = tt.splat %a_ptr : !tt.ptr<f32> -> tensor<8x1x!tt.ptr<f32>, #blocked1>
    %a_17 = tt.addptr %a_16, %a_15 : tensor<8x1x!tt.ptr<f32>, #blocked1>, tensor<8x1xi32, #blocked1>
    %a_18 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %a_19 = tt.expand_dims %a_18 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x32xi32, #blocked1>
    %a_20 = tt.broadcast %a_17 : tensor<8x1x!tt.ptr<f32>, #blocked1> -> tensor<8x32x!tt.ptr<f32>, #blocked1>
    %a_21 = tt.broadcast %a_19 : tensor<1x32xi32, #blocked1> -> tensor<8x32xi32, #blocked1>
    %a_22 = tt.addptr %a_20, %a_21 : tensor<8x32x!tt.ptr<f32>, #blocked1>, tensor<8x32xi32, #blocked1>
    %a_23 = tt.load %a_22 : tensor<8x32x!tt.ptr<f32>, #blocked1>
    %b = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %b_24 = tt.expand_dims %b {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<32x1xi32, #blocked2>
    %b_25 = tt.splat %N : i32 -> tensor<32x1xi32, #blocked2>
    %b_26 = arith.muli %b_24, %b_25 : tensor<32x1xi32, #blocked2>
    %b_27 = tt.splat %b_ptr : !tt.ptr<f32> -> tensor<32x1x!tt.ptr<f32>, #blocked2>
    %b_28 = tt.addptr %b_27, %b_26 : tensor<32x1x!tt.ptr<f32>, #blocked2>, tensor<32x1xi32, #blocked2>
    %b_29 = tt.expand_dims %offs_n_12 {axis = 0 : i32} : tensor<512xi32, #ttg.slice<{dim = 0, parent = #blocked3}>> -> tensor<1x512xi32, #blocked3>
    %b_30 = tt.broadcast %b_28 : tensor<32x1x!tt.ptr<f32>, #blocked2> -> tensor<32x512x!tt.ptr<f32>, #blocked3>
    %b_31 = tt.broadcast %b_29 : tensor<1x512xi32, #blocked3> -> tensor<32x512xi32, #blocked3>
    %b_32 = tt.addptr %b_30, %b_31 : tensor<32x512x!tt.ptr<f32>, #blocked3>, tensor<32x512xi32, #blocked3>
    %b_33 = tt.load %b_32 : tensor<32x512x!tt.ptr<f32>, #blocked3>
    %a_34 = ttg.convert_layout %a_23 : tensor<8x32xf32, #blocked1> -> tensor<8x32xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    %b_35 = ttg.convert_layout %b_33 : tensor<32x512xf32, #blocked3> -> tensor<32x512xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
    %c_36 = tt.dot %a_34, %b_35, %c : tensor<8x32xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<32x512xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<8x512xf32, #blocked>
    %0 = tt.splat %N : i32 -> tensor<8x1xi32, #blocked2>
    %1 = arith.muli %a_13, %0 : tensor<8x1xi32, #blocked2>
    %2 = tt.splat %c_ptr : !tt.ptr<f32> -> tensor<8x1x!tt.ptr<f32>, #blocked2>
    %3 = tt.addptr %2, %1 : tensor<8x1x!tt.ptr<f32>, #blocked2>, tensor<8x1xi32, #blocked2>
    %4 = tt.broadcast %3 : tensor<8x1x!tt.ptr<f32>, #blocked2> -> tensor<8x512x!tt.ptr<f32>, #blocked3>
    %5 = tt.broadcast %b_29 : tensor<1x512xi32, #blocked3> -> tensor<8x512xi32, #blocked3>
    %6 = tt.addptr %4, %5 : tensor<8x512x!tt.ptr<f32>, #blocked3>, tensor<8x512xi32, #blocked3>
    %7 = ttg.convert_layout %c_36 : tensor<8x512xf32, #blocked> -> tensor<8x512xf32, #blocked3>
    tt.store %6, %7 : tensor<8x512x!tt.ptr<f32>, #blocked3>
    tt.return
  }
}
