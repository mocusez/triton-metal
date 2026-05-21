// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | FileCheck %s
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
// See `.omc/plans/ac4-multiwarp.md` and `.omc/research/ac4-probe.md`.

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
