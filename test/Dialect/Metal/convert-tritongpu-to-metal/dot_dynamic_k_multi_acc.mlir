// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | FileCheck %s
//
// W2b fixture (metal-lora-linear-fix-plan.md). Multi-accumulator recompute-
// from-IV runtime-K loop: two dots share the A operand x and carry two
// accumulators, each transposed-B, stored after the loop (LoRA inner loop).
// tryRuntimeKLoopRecomputeMultiDot emits ONE fresh scf.for with two
// simdgroup_matrix accumulators and a single shared staged A load per iter.
//
// Acceptance: no tt.dot/tt.trans/convert_layout; two simdgroup_matrix_zero
// inits; a runtime scf.for; two multiply_accumulate; two stores.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [8, 4], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [2, 1], threadsPerWarp = [4, 8], warpsPerCTA = [1, 1], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @two_matmul(%x_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %w_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %a_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %o0_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %o1_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %K: i32 {tt.divisibility = 16 : i32}, %sxm: i32 {tt.divisibility = 16 : i32}, %swn: i32 {tt.divisibility = 16 : i32}, %sar: i32 {tt.divisibility = 16 : i32}, %s0m: i32, %s1m: i32) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<8x8xf32, #blocked>
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %pid_m = tt.get_program_id x : i32
    %offs_m = arith.muli %pid_m, %c8_i32 : i32
    %offs_m_0 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %offs_m_1 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %offs_m_2 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %offs_m_3 = tt.splat %offs_m : i32 -> tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %offs_m_4 = tt.splat %offs_m : i32 -> tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %offs_m_5 = arith.addi %offs_m_3, %offs_m_0 : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %offs_m_6 = arith.addi %offs_m_4, %offs_m_1 : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %x = tt.expand_dims %offs_m_5 {axis = 1 : i32} : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<8x1xi32, #blocked1>
    %x_7 = tt.splat %sxm : i32 -> tensor<8x1xi32, #blocked1>
    %x_8 = arith.muli %x, %x_7 : tensor<8x1xi32, #blocked1>
    %x_9 = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<8x1x!tt.ptr<f32>, #blocked1>
    %x_10 = tt.addptr %x_9, %x_8 : tensor<8x1x!tt.ptr<f32>, #blocked1>, tensor<8x1xi32, #blocked1>
    %x_11 = tt.broadcast %x_10 : tensor<8x1x!tt.ptr<f32>, #blocked1> -> tensor<8x8x!tt.ptr<f32>, #blocked1>
    %w = tt.expand_dims %offs_m_0 {axis = 1 : i32} : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<8x1xi32, #blocked1>
    %w_12 = tt.splat %swn : i32 -> tensor<8x1xi32, #blocked1>
    %w_13 = arith.muli %w, %w_12 : tensor<8x1xi32, #blocked1>
    %w_14 = tt.splat %w_ptr : !tt.ptr<f32> -> tensor<8x1x!tt.ptr<f32>, #blocked1>
    %w_15 = tt.addptr %w_14, %w_13 : tensor<8x1x!tt.ptr<f32>, #blocked1>, tensor<8x1xi32, #blocked1>
    %w_16 = tt.broadcast %w_15 : tensor<8x1x!tt.ptr<f32>, #blocked1> -> tensor<8x8x!tt.ptr<f32>, #blocked1>
    %a = tt.splat %sar : i32 -> tensor<8x1xi32, #blocked1>
    %a_17 = arith.muli %w, %a : tensor<8x1xi32, #blocked1>
    %a_18 = tt.splat %a_ptr : !tt.ptr<f32> -> tensor<8x1x!tt.ptr<f32>, #blocked1>
    %a_19 = tt.addptr %a_18, %a_17 : tensor<8x1x!tt.ptr<f32>, #blocked1>, tensor<8x1xi32, #blocked1>
    %a_20 = tt.broadcast %a_19 : tensor<8x1x!tt.ptr<f32>, #blocked1> -> tensor<8x8x!tt.ptr<f32>, #blocked1>
    %acc1:2 = scf.for %k = %c0_i32 to %K step %c8_i32 iter_args(%acc0 = %cst, %acc1_21 = %cst) -> (tensor<8x8xf32, #blocked>, tensor<8x8xf32, #blocked>)  : i32 {
      %offs_k = tt.splat %k : i32 -> tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
      %offs_k_22 = arith.addi %offs_k, %offs_m_2 : tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
      %x_23 = tt.expand_dims %offs_k_22 {axis = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x8xi32, #blocked1>
      %x_24 = tt.broadcast %x_23 : tensor<1x8xi32, #blocked1> -> tensor<8x8xi32, #blocked1>
      %x_25 = tt.addptr %x_11, %x_24 : tensor<8x8x!tt.ptr<f32>, #blocked1>, tensor<8x8xi32, #blocked1>
      %x_26 = tt.load %x_25 : tensor<8x8x!tt.ptr<f32>, #blocked1>
      %w_27 = tt.addptr %w_16, %x_24 : tensor<8x8x!tt.ptr<f32>, #blocked1>, tensor<8x8xi32, #blocked1>
      %w_28 = tt.load %w_27 : tensor<8x8x!tt.ptr<f32>, #blocked1>
      %a_29 = tt.addptr %a_20, %x_24 : tensor<8x8x!tt.ptr<f32>, #blocked1>, tensor<8x8xi32, #blocked1>
      %a_30 = tt.load %a_29 : tensor<8x8x!tt.ptr<f32>, #blocked1>
      %acc0_31 = tt.trans %w_28 {order = array<i32: 1, 0>} : tensor<8x8xf32, #blocked1> -> tensor<8x8xf32, #blocked2>
      %x_32 = ttg.convert_layout %x_26 : tensor<8x8xf32, #blocked1> -> tensor<8x8xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
      %acc0_33 = ttg.convert_layout %acc0_31 : tensor<8x8xf32, #blocked2> -> tensor<8x8xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
      %acc0_34 = tt.dot %x_32, %acc0_33, %acc0 : tensor<8x8xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<8x8xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<8x8xf32, #blocked>
      %acc1_35 = tt.trans %a_30 {order = array<i32: 1, 0>} : tensor<8x8xf32, #blocked1> -> tensor<8x8xf32, #blocked2>
      %acc1_36 = ttg.convert_layout %acc1_35 : tensor<8x8xf32, #blocked2> -> tensor<8x8xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
      %acc1_37 = tt.dot %x_32, %acc1_36, %acc1_21 : tensor<8x8xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<8x8xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<8x8xf32, #blocked>
      scf.yield %acc0_34, %acc1_37 : tensor<8x8xf32, #blocked>, tensor<8x8xf32, #blocked>
    }
    %0 = tt.expand_dims %offs_m_6 {axis = 1 : i32} : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<8x1xi32, #blocked>
    %1 = tt.splat %s0m : i32 -> tensor<8x1xi32, #blocked>
    %2 = arith.muli %0, %1 : tensor<8x1xi32, #blocked>
    %3 = tt.splat %o0_ptr : !tt.ptr<f32> -> tensor<8x1x!tt.ptr<f32>, #blocked>
    %4 = tt.addptr %3, %2 : tensor<8x1x!tt.ptr<f32>, #blocked>, tensor<8x1xi32, #blocked>
    %5 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %6 = tt.expand_dims %5 {axis = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x8xi32, #blocked>
    %7 = tt.broadcast %4 : tensor<8x1x!tt.ptr<f32>, #blocked> -> tensor<8x8x!tt.ptr<f32>, #blocked>
    %8 = tt.broadcast %6 : tensor<1x8xi32, #blocked> -> tensor<8x8xi32, #blocked>
    %9 = tt.addptr %7, %8 : tensor<8x8x!tt.ptr<f32>, #blocked>, tensor<8x8xi32, #blocked>
    tt.store %9, %acc1#0 : tensor<8x8x!tt.ptr<f32>, #blocked>
    %10 = tt.splat %s1m : i32 -> tensor<8x1xi32, #blocked>
    %11 = arith.muli %0, %10 : tensor<8x1xi32, #blocked>
    %12 = tt.splat %o1_ptr : !tt.ptr<f32> -> tensor<8x1x!tt.ptr<f32>, #blocked>
    %13 = tt.addptr %12, %11 : tensor<8x1x!tt.ptr<f32>, #blocked>, tensor<8x1xi32, #blocked>
    %14 = tt.broadcast %13 : tensor<8x1x!tt.ptr<f32>, #blocked> -> tensor<8x8x!tt.ptr<f32>, #blocked>
    %15 = tt.addptr %14, %8 : tensor<8x8x!tt.ptr<f32>, #blocked>, tensor<8x8xi32, #blocked>
    tt.store %15, %acc1#1 : tensor<8x8x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
// CHECK-LABEL: metal.kernel two_matmul
// CHECK-NOT: tt.dot
// CHECK-NOT: tt.trans
// CHECK-NOT: convert_layout
// CHECK-COUNT-2: metal.simdgroup_matrix_zero
// CHECK: scf.for
// CHECK-COUNT-2: metal.simdgroup_multiply_accumulate
// CHECK-COUNT-2: metal.simdgroup_store
