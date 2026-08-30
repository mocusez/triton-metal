// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | FileCheck %s
//
// W2c fixture (metal-lora-linear-fix-plan.md). Fully-fused LoRA: a two-
// accumulator recompute-from-IV loop plus the post-loop epilogue
// acc0 += scale * tl.dot(acc1, tl.trans(b)). tryFusedLoRAEpilogue folds the
// scale-and-add into one metal.simdgroup_fused_store(acc0, acc1*trans(b), scale).
//
// Acceptance: no tt.dot/tt.trans/convert_layout/arith.mulf/arith.addf survive;
// a runtime scf.for with two accumulators; three multiply_accumulate (two in
// the loop, one for dot #3); a simdgroup_fused_store; no plain simdgroup_store.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [8, 4], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [2, 1], threadsPerWarp = [4, 8], warpsPerCTA = [1, 1], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @fused_lora(%x_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %w_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %a_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %b_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %o_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %K: i32 {tt.divisibility = 16 : i32}, %scale: f32, %sxm: i32 {tt.divisibility = 16 : i32}, %swn: i32 {tt.divisibility = 16 : i32}, %sar: i32 {tt.divisibility = 16 : i32}, %sbn: i32, %som: i32) attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<8x8xf32, #blocked>
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
    %a = tt.expand_dims %offs_m_0 {axis = 1 : i32} : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<8x1xi32, #blocked1>
    %a_22 = tt.splat %sar : i32 -> tensor<8x1xi32, #blocked1>
    %a_23 = arith.muli %a, %a_22 : tensor<8x1xi32, #blocked1>
    %a_24 = tt.splat %a_ptr : !tt.ptr<f32> -> tensor<8x1x!tt.ptr<f32>, #blocked1>
    %a_25 = tt.addptr %a_24, %a_23 : tensor<8x1x!tt.ptr<f32>, #blocked1>, tensor<8x1xi32, #blocked1>
    %a_26 = tt.broadcast %a_25 : tensor<8x1x!tt.ptr<f32>, #blocked1> -> tensor<8x8x!tt.ptr<f32>, #blocked1>
    %acc1:2 = scf.for %k = %c0_i32 to %K step %c8_i32 iter_args(%acc0_41 = %cst, %acc1_42 = %cst) -> (tensor<8x8xf32, #blocked>, tensor<8x8xf32, #blocked>)  : i32 {
      %offs_k = tt.splat %k : i32 -> tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
      %offs_k_43 = arith.addi %offs_k, %offs_m_2 : tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
      %x_44 = tt.expand_dims %offs_k_43 {axis = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x8xi32, #blocked1>
      %x_45 = tt.broadcast %x_44 : tensor<1x8xi32, #blocked1> -> tensor<8x8xi32, #blocked1>
      %x_46 = tt.addptr %x_16, %x_45 : tensor<8x8x!tt.ptr<f32>, #blocked1>, tensor<8x8xi32, #blocked1>
      %x_47 = tt.load %x_46 : tensor<8x8x!tt.ptr<f32>, #blocked1>
      %w_48 = tt.addptr %w_21, %x_45 : tensor<8x8x!tt.ptr<f32>, #blocked1>, tensor<8x8xi32, #blocked1>
      %w_49 = tt.load %w_48 : tensor<8x8x!tt.ptr<f32>, #blocked1>
      %a_50 = tt.addptr %a_26, %x_45 : tensor<8x8x!tt.ptr<f32>, #blocked1>, tensor<8x8xi32, #blocked1>
      %a_51 = tt.load %a_50 : tensor<8x8x!tt.ptr<f32>, #blocked1>
      %acc0_52 = tt.trans %w_49 {order = array<i32: 1, 0>} : tensor<8x8xf32, #blocked1> -> tensor<8x8xf32, #blocked2>
      %x_53 = ttg.convert_layout %x_47 : tensor<8x8xf32, #blocked1> -> tensor<8x8xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
      %acc0_54 = ttg.convert_layout %acc0_52 : tensor<8x8xf32, #blocked2> -> tensor<8x8xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
      %acc0_55 = tt.dot %x_53, %acc0_54, %acc0_41 : tensor<8x8xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<8x8xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<8x8xf32, #blocked>
      %acc1_56 = tt.trans %a_51 {order = array<i32: 1, 0>} : tensor<8x8xf32, #blocked1> -> tensor<8x8xf32, #blocked2>
      %acc1_57 = ttg.convert_layout %acc1_56 : tensor<8x8xf32, #blocked2> -> tensor<8x8xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
      %acc1_58 = tt.dot %x_53, %acc1_57, %acc1_42 : tensor<8x8xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<8x8xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<8x8xf32, #blocked>
      scf.yield %acc0_55, %acc1_58 : tensor<8x8xf32, #blocked>, tensor<8x8xf32, #blocked>
    }
    %b = tt.splat %sbn : i32 -> tensor<8x1xi32, #blocked1>
    %b_27 = arith.muli %w, %b : tensor<8x1xi32, #blocked1>
    %b_28 = tt.splat %b_ptr : !tt.ptr<f32> -> tensor<8x1x!tt.ptr<f32>, #blocked1>
    %b_29 = tt.addptr %b_28, %b_27 : tensor<8x1x!tt.ptr<f32>, #blocked1>, tensor<8x1xi32, #blocked1>
    %b_30 = tt.expand_dims %offs_m_2 {axis = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x8xi32, #blocked1>
    %b_31 = tt.broadcast %b_29 : tensor<8x1x!tt.ptr<f32>, #blocked1> -> tensor<8x8x!tt.ptr<f32>, #blocked1>
    %b_32 = tt.broadcast %b_30 : tensor<1x8xi32, #blocked1> -> tensor<8x8xi32, #blocked1>
    %b_33 = tt.addptr %b_31, %b_32 : tensor<8x8x!tt.ptr<f32>, #blocked1>, tensor<8x8xi32, #blocked1>
    %b_34 = tt.load %b_33 : tensor<8x8x!tt.ptr<f32>, #blocked1>
    %acc0 = tt.trans %b_34 {order = array<i32: 1, 0>} : tensor<8x8xf32, #blocked1> -> tensor<8x8xf32, #blocked2>
    %acc1_35 = ttg.convert_layout %acc1#1 : tensor<8x8xf32, #blocked> -> tensor<8x8xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    %acc0_36 = ttg.convert_layout %acc0 : tensor<8x8xf32, #blocked2> -> tensor<8x8xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
    %acc0_37 = tt.dot %acc1_35, %acc0_36, %cst : tensor<8x8xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<8x8xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<8x8xf32, #blocked>
    %acc0_38 = tt.splat %scale : f32 -> tensor<8x8xf32, #blocked>
    %acc0_39 = arith.mulf %acc0_38, %acc0_37 : tensor<8x8xf32, #blocked>
    %acc0_40 = arith.addf %acc1#0, %acc0_39 : tensor<8x8xf32, #blocked>
    %0 = tt.expand_dims %offs_m_7 {axis = 1 : i32} : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<8x1xi32, #blocked>
    %1 = tt.splat %som : i32 -> tensor<8x1xi32, #blocked>
    %2 = arith.muli %0, %1 : tensor<8x1xi32, #blocked>
    %3 = tt.splat %o_ptr : !tt.ptr<f32> -> tensor<8x1x!tt.ptr<f32>, #blocked>
    %4 = tt.addptr %3, %2 : tensor<8x1x!tt.ptr<f32>, #blocked>, tensor<8x1xi32, #blocked>
    %5 = tt.expand_dims %offs_n_11 {axis = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x8xi32, #blocked>
    %6 = tt.broadcast %4 : tensor<8x1x!tt.ptr<f32>, #blocked> -> tensor<8x8x!tt.ptr<f32>, #blocked>
    %7 = tt.broadcast %5 : tensor<1x8xi32, #blocked> -> tensor<8x8xi32, #blocked>
    %8 = tt.addptr %6, %7 : tensor<8x8x!tt.ptr<f32>, #blocked>, tensor<8x8xi32, #blocked>
    tt.store %8, %acc0_40 : tensor<8x8x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
// CHECK-LABEL: metal.kernel fused_lora
// CHECK-NOT: tt.dot
// CHECK-NOT: tt.trans
// CHECK-NOT: convert_layout
// CHECK-NOT: arith.mulf
// CHECK-NOT: arith.addf
// CHECK: scf.for
// CHECK-COUNT-3: metal.simdgroup_multiply_accumulate
// CHECK: metal.simdgroup_fused_store
