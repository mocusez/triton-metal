// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | FileCheck %s
//
// W1 fixture (metal-lora-linear-fix-plan.md). Transposed-B runtime-K matmul
// `tl.dot(a, tl.trans(w))` with w kept as [N,K]. The preempt peels the
// cvt(trans(load)) chain, and the runtime-K matcher folds the transpose into
// the B simdgroup staged load (the `transposed` attr swaps the staging index).
//
// Acceptance: no tt.dot / tt.trans / convert_layout survive; the A staged load
// is plain and the B staged load carries {transposed}; the runtime scf.for and
// its simdgroup_matrix accumulator are preserved.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [8, 4], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [2, 1], threadsPerWarp = [4, 8], warpsPerCTA = [1, 1], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @dyn_k_transB(%a_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %w_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %c_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %K: i32 {tt.divisibility = 16 : i32}, %stride_am: i32 {tt.divisibility = 16 : i32}, %stride_wn: i32 {tt.divisibility = 16 : i32}, %stride_cm: i32) attributes {noinline = false} {
    %c8_i32 = arith.constant 8 : i32
    %acc = arith.constant dense<0.000000e+00> : tensor<8x8xf32, #blocked>
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<8> : tensor<8x8xi32, #blocked1>
    %pid_m = tt.get_program_id x : i32
    %pid_n = tt.get_program_id y : i32
    %offs_m = arith.muli %pid_m, %c8_i32 : i32
    %offs_m_0 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %offs_m_1 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %offs_m_2 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %offs_m_3 = tt.splat %offs_m : i32 -> tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %offs_m_4 = tt.splat %offs_m : i32 -> tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %offs_m_5 = arith.addi %offs_m_3, %offs_m_0 : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %offs_m_6 = arith.addi %offs_m_4, %offs_m_1 : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %offs_n = arith.muli %pid_n, %c8_i32 : i32
    %offs_n_7 = tt.splat %offs_n : i32 -> tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %offs_n_8 = tt.splat %offs_n : i32 -> tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %offs_n_9 = arith.addi %offs_n_7, %offs_m_0 : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %offs_n_10 = arith.addi %offs_n_8, %offs_m_2 : tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %a_ptrs = tt.expand_dims %offs_m_5 {axis = 1 : i32} : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<8x1xi32, #blocked1>
    %a_ptrs_11 = tt.expand_dims %offs_m_6 {axis = 1 : i32} : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<8x1xi32, #blocked>
    %a_ptrs_12 = tt.splat %stride_am : i32 -> tensor<8x1xi32, #blocked1>
    %a_ptrs_13 = arith.muli %a_ptrs, %a_ptrs_12 : tensor<8x1xi32, #blocked1>
    %a_ptrs_14 = tt.splat %a_ptr : !tt.ptr<f32> -> tensor<8x1x!tt.ptr<f32>, #blocked1>
    %a_ptrs_15 = tt.addptr %a_ptrs_14, %a_ptrs_13 : tensor<8x1x!tt.ptr<f32>, #blocked1>, tensor<8x1xi32, #blocked1>
    %a_ptrs_16 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %a_ptrs_17 = tt.expand_dims %a_ptrs_16 {axis = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x8xi32, #blocked1>
    %a_ptrs_18 = tt.broadcast %a_ptrs_15 : tensor<8x1x!tt.ptr<f32>, #blocked1> -> tensor<8x8x!tt.ptr<f32>, #blocked1>
    %a_ptrs_19 = tt.broadcast %a_ptrs_17 : tensor<1x8xi32, #blocked1> -> tensor<8x8xi32, #blocked1>
    %a_ptrs_20 = tt.addptr %a_ptrs_18, %a_ptrs_19 : tensor<8x8x!tt.ptr<f32>, #blocked1>, tensor<8x8xi32, #blocked1>
    %w_ptrs = tt.expand_dims %offs_n_9 {axis = 1 : i32} : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<8x1xi32, #blocked1>
    %w_ptrs_21 = tt.splat %stride_wn : i32 -> tensor<8x1xi32, #blocked1>
    %w_ptrs_22 = arith.muli %w_ptrs, %w_ptrs_21 : tensor<8x1xi32, #blocked1>
    %w_ptrs_23 = tt.splat %w_ptr : !tt.ptr<f32> -> tensor<8x1x!tt.ptr<f32>, #blocked1>
    %w_ptrs_24 = tt.addptr %w_ptrs_23, %w_ptrs_22 : tensor<8x1x!tt.ptr<f32>, #blocked1>, tensor<8x1xi32, #blocked1>
    %w_ptrs_25 = tt.broadcast %w_ptrs_24 : tensor<8x1x!tt.ptr<f32>, #blocked1> -> tensor<8x8x!tt.ptr<f32>, #blocked1>
    %w_ptrs_26 = tt.addptr %w_ptrs_25, %a_ptrs_19 : tensor<8x8x!tt.ptr<f32>, #blocked1>, tensor<8x8xi32, #blocked1>
    %acc_27:3 = scf.for %_k = %c0_i32 to %K step %c8_i32 iter_args(%acc_35 = %acc, %a_ptrs_36 = %a_ptrs_20, %w_ptrs_37 = %w_ptrs_26) -> (tensor<8x8xf32, #blocked>, tensor<8x8x!tt.ptr<f32>, #blocked1>, tensor<8x8x!tt.ptr<f32>, #blocked1>)  : i32 {
      %a = tt.load %a_ptrs_36 : tensor<8x8x!tt.ptr<f32>, #blocked1>
      %w = tt.load %w_ptrs_37 : tensor<8x8x!tt.ptr<f32>, #blocked1>
      %acc_38 = tt.trans %w {order = array<i32: 1, 0>} : tensor<8x8xf32, #blocked1> -> tensor<8x8xf32, #blocked2>
      %a_39 = ttg.convert_layout %a : tensor<8x8xf32, #blocked1> -> tensor<8x8xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
      %acc_40 = ttg.convert_layout %acc_38 : tensor<8x8xf32, #blocked2> -> tensor<8x8xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
      %acc_41 = tt.dot %a_39, %acc_40, %acc_35 : tensor<8x8xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<8x8xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<8x8xf32, #blocked>
      %a_ptrs_42 = tt.addptr %a_ptrs_36, %cst : tensor<8x8x!tt.ptr<f32>, #blocked1>, tensor<8x8xi32, #blocked1>
      %w_ptrs_43 = tt.addptr %w_ptrs_37, %cst : tensor<8x8x!tt.ptr<f32>, #blocked1>, tensor<8x8xi32, #blocked1>
      scf.yield %acc_41, %a_ptrs_42, %w_ptrs_43 : tensor<8x8xf32, #blocked>, tensor<8x8x!tt.ptr<f32>, #blocked1>, tensor<8x8x!tt.ptr<f32>, #blocked1>
    }
    %c_ptrs = tt.splat %stride_cm : i32 -> tensor<8x1xi32, #blocked>
    %c_ptrs_28 = arith.muli %a_ptrs_11, %c_ptrs : tensor<8x1xi32, #blocked>
    %c_ptrs_29 = tt.splat %c_ptr : !tt.ptr<f32> -> tensor<8x1x!tt.ptr<f32>, #blocked>
    %c_ptrs_30 = tt.addptr %c_ptrs_29, %c_ptrs_28 : tensor<8x1x!tt.ptr<f32>, #blocked>, tensor<8x1xi32, #blocked>
    %c_ptrs_31 = tt.expand_dims %offs_n_10 {axis = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x8xi32, #blocked>
    %c_ptrs_32 = tt.broadcast %c_ptrs_30 : tensor<8x1x!tt.ptr<f32>, #blocked> -> tensor<8x8x!tt.ptr<f32>, #blocked>
    %c_ptrs_33 = tt.broadcast %c_ptrs_31 : tensor<1x8xi32, #blocked> -> tensor<8x8xi32, #blocked>
    %c_ptrs_34 = tt.addptr %c_ptrs_32, %c_ptrs_33 : tensor<8x8x!tt.ptr<f32>, #blocked>, tensor<8x8xi32, #blocked>
    tt.store %c_ptrs_34, %acc_27#0 : tensor<8x8x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// CHECK-LABEL: metal.kernel dyn_k_transB
// CHECK-NOT: tt.dot
// CHECK-NOT: tt.trans
// CHECK-NOT: convert_layout
// CHECK: metal.simdgroup_matrix_zero
// CHECK: scf.for
// The B operand staged load is transposed; the A one is not.
// CHECK: metal.simdgroup_load_device_staged
// CHECK: metal.simdgroup_load_device_staged {{.*}}{transposed}
// CHECK: metal.simdgroup_multiply_accumulate
// CHECK: metal.simdgroup_store
