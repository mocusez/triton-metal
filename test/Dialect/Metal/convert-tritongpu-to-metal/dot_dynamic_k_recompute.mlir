// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | FileCheck %s
//
// W2b fixture (metal-lora-linear-fix-plan.md). Recompute-from-IV runtime-K
// matmul (the shape medium-lora_linear.py uses): the loop carries only the
// accumulator and rebuilds addresses from the induction variable each
// iteration (offs_k = k + arange), with a transposed B operand. Handled by
// tryRuntimeKLoopRecomputeDot: origins/strides are pulled from the loads,
// and a fresh scf.for stepping K by 8 carries the simdgroup_matrix
// accumulator.
//
// Acceptance: no tt.dot/tt.trans/convert_layout survive; a runtime scf.for is
// preserved with a simdgroup_matrix accumulator; the B load is {transposed}.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [8, 4], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [2, 1], threadsPerWarp = [4, 8], warpsPerCTA = [1, 1], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @recompute_single(%x_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %w_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %c_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %K: i32 {tt.divisibility = 16 : i32}, %sxm: i32 {tt.divisibility = 16 : i32}, %swn: i32 {tt.divisibility = 16 : i32}, %scm: i32) attributes {noinline = false} {
    %acc = arith.constant dense<0.000000e+00> : tensor<8x8xf32, #blocked>
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %pid_m = tt.get_program_id x : i32
    %pid_n = tt.get_program_id y : i32
    %offs_m = arith.muli %pid_m, %c8_i32 : i32
    %offs_m_0 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %offs_m_1 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %offs_m_2 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %offs_m_3 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %offs_m_4 = tt.splat %offs_m : i32 -> tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %offs_m_5 = tt.splat %offs_m : i32 -> tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %offs_m_6 = arith.addi %offs_m_4, %offs_m_0 : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %offs_m_7 = arith.addi %offs_m_5, %offs_m_1 : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %offs_n = arith.muli %pid_n, %c8_i32 : i32
    %offs_n_8 = tt.splat %offs_n : i32 -> tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %offs_n_9 = tt.splat %offs_n : i32 -> tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %offs_n_10 = arith.addi %offs_n_8, %offs_m_0 : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %offs_n_11 = arith.addi %offs_n_9, %offs_m_3 : tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %x = tt.expand_dims %offs_m_6 {axis = 1 : i32} : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<8x1xi32, #blocked1>
    %x_12 = tt.splat %sxm : i32 -> tensor<8x1xi32, #blocked1>
    %x_13 = arith.muli %x, %x_12 : tensor<8x1xi32, #blocked1>
    %x_14 = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<8x1x!tt.ptr<f32>, #blocked1>
    %x_15 = tt.addptr %x_14, %x_13 : tensor<8x1x!tt.ptr<f32>, #blocked1>, tensor<8x1xi32, #blocked1>
    %x_16 = tt.broadcast %x_15 : tensor<8x1x!tt.ptr<f32>, #blocked1> -> tensor<8x8x!tt.ptr<f32>, #blocked1>
    %w = tt.expand_dims %offs_n_10 {axis = 1 : i32} : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<8x1xi32, #blocked1>
    %w_17 = tt.splat %swn : i32 -> tensor<8x1xi32, #blocked1>
    %w_18 = arith.muli %w, %w_17 : tensor<8x1xi32, #blocked1>
    %w_19 = tt.splat %w_ptr : !tt.ptr<f32> -> tensor<8x1x!tt.ptr<f32>, #blocked1>
    %w_20 = tt.addptr %w_19, %w_18 : tensor<8x1x!tt.ptr<f32>, #blocked1>, tensor<8x1xi32, #blocked1>
    %w_21 = tt.broadcast %w_20 : tensor<8x1x!tt.ptr<f32>, #blocked1> -> tensor<8x8x!tt.ptr<f32>, #blocked1>
    %acc_22 = scf.for %k = %c0_i32 to %K step %c8_i32 iter_args(%acc_23 = %acc) -> (tensor<8x8xf32, #blocked>)  : i32 {
      %offs_k = tt.splat %k : i32 -> tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
      %offs_k_24 = arith.addi %offs_k, %offs_m_2 : tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
      %x_25 = tt.expand_dims %offs_k_24 {axis = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x8xi32, #blocked1>
      %x_26 = tt.broadcast %x_25 : tensor<1x8xi32, #blocked1> -> tensor<8x8xi32, #blocked1>
      %x_27 = tt.addptr %x_16, %x_26 : tensor<8x8x!tt.ptr<f32>, #blocked1>, tensor<8x8xi32, #blocked1>
      %x_28 = tt.load %x_27 : tensor<8x8x!tt.ptr<f32>, #blocked1>
      %w_29 = tt.addptr %w_21, %x_26 : tensor<8x8x!tt.ptr<f32>, #blocked1>, tensor<8x8xi32, #blocked1>
      %w_30 = tt.load %w_29 : tensor<8x8x!tt.ptr<f32>, #blocked1>
      %acc_31 = tt.trans %w_30 {order = array<i32: 1, 0>} : tensor<8x8xf32, #blocked1> -> tensor<8x8xf32, #blocked2>
      %x_32 = ttg.convert_layout %x_28 : tensor<8x8xf32, #blocked1> -> tensor<8x8xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
      %acc_33 = ttg.convert_layout %acc_31 : tensor<8x8xf32, #blocked2> -> tensor<8x8xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
      %acc_34 = tt.dot %x_32, %acc_33, %acc_23 : tensor<8x8xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<8x8xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<8x8xf32, #blocked>
      scf.yield %acc_34 : tensor<8x8xf32, #blocked>
    }
    %0 = tt.expand_dims %offs_m_7 {axis = 1 : i32} : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<8x1xi32, #blocked>
    %1 = tt.splat %scm : i32 -> tensor<8x1xi32, #blocked>
    %2 = arith.muli %0, %1 : tensor<8x1xi32, #blocked>
    %3 = tt.splat %c_ptr : !tt.ptr<f32> -> tensor<8x1x!tt.ptr<f32>, #blocked>
    %4 = tt.addptr %3, %2 : tensor<8x1x!tt.ptr<f32>, #blocked>, tensor<8x1xi32, #blocked>
    %5 = tt.expand_dims %offs_n_11 {axis = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x8xi32, #blocked>
    %6 = tt.broadcast %4 : tensor<8x1x!tt.ptr<f32>, #blocked> -> tensor<8x8x!tt.ptr<f32>, #blocked>
    %7 = tt.broadcast %5 : tensor<1x8xi32, #blocked> -> tensor<8x8xi32, #blocked>
    %8 = tt.addptr %6, %7 : tensor<8x8x!tt.ptr<f32>, #blocked>, tensor<8x8xi32, #blocked>
    tt.store %8, %acc_22 : tensor<8x8x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
// CHECK-LABEL: metal.kernel recompute_single
// CHECK-NOT: tt.dot
// CHECK-NOT: tt.trans
// CHECK-NOT: convert_layout
// CHECK: metal.simdgroup_matrix_zero
// CHECK: scf.for
// CHECK: metal.simdgroup_load_device_staged
// CHECK: metal.simdgroup_load_device_staged {{.*}}{transposed}
// CHECK: metal.simdgroup_multiply_accumulate
// CHECK: metal.simdgroup_store
