// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | FileCheck %s
//
// Mask fixture (metal-lora-linear-fix-plan.md). Fully-fused LoRA with every
// load masked ((offs_row<ROW)&(offs_k<K), other=0.0) and a masked store.
// Masked loads lower to metal.simdgroup_load_device_staged_masked (out-of-
// bounds -> 0); the store folds into metal.simdgroup_fused_store with
// partial extents.
//
// Acceptance: no tt.dot/tt.trans/convert_layout; masked staged loads present;
// a fused store; a runtime scf.for.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 1], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @fused_lora_masked(%x_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %w_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %a_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %b_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %o_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %M: i32, %N: i32, %K: i32, %R: i32, %scale: f32, %sxm: i32, %swn: i32, %sar: i32, %sbn: i32, %som: i32) attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<8x8xf32, #blocked>
    %c8_i32 = arith.constant 8 : i32
    %pid_m = tt.get_program_id x : i32
    %pid_n = tt.get_program_id y : i32
    %offs_m = arith.muli %pid_m, %c8_i32 : i32
    %offs_m_0 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %offs_m_1 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %offs_m_2 = tt.splat %offs_m : i32 -> tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %offs_m_3 = arith.addi %offs_m_2, %offs_m_0 : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %offs_n = arith.muli %pid_n, %c8_i32 : i32
    %offs_n_4 = tt.splat %offs_n : i32 -> tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %offs_n_5 = tt.splat %offs_n : i32 -> tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %offs_n_6 = arith.addi %offs_n_4, %offs_m_0 : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %offs_n_7 = arith.addi %offs_n_5, %offs_m_1 : tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %mx = tt.expand_dims %offs_m_3 {axis = 1 : i32} : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<8x1xi32, #blocked>
    %mx_8 = tt.splat %M : i32 -> tensor<8x1xi32, #blocked>
    %mx_9 = arith.cmpi slt, %mx, %mx_8 : tensor<8x1xi32, #blocked>
    %mx_10 = tt.splat %K : i32 -> tensor<1x8xi32, #blocked>
    %mx_11 = tt.broadcast %mx_9 : tensor<8x1xi1, #blocked> -> tensor<8x8xi1, #blocked>
    %mw = tt.expand_dims %offs_n_6 {axis = 1 : i32} : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<8x1xi32, #blocked>
    %mw_12 = tt.splat %N : i32 -> tensor<8x1xi32, #blocked>
    %mw_13 = arith.cmpi slt, %mw, %mw_12 : tensor<8x1xi32, #blocked>
    %mw_14 = tt.broadcast %mw_13 : tensor<8x1xi1, #blocked> -> tensor<8x8xi1, #blocked>
    %ma = tt.expand_dims %offs_m_0 {axis = 1 : i32} : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<8x1xi32, #blocked>
    %ma_15 = tt.splat %R : i32 -> tensor<8x1xi32, #blocked>
    %ma_16 = arith.cmpi slt, %ma, %ma_15 : tensor<8x1xi32, #blocked>
    %ma_17 = tt.broadcast %ma_16 : tensor<8x1xi1, #blocked> -> tensor<8x8xi1, #blocked>
    %x = tt.splat %sxm : i32 -> tensor<8x1xi32, #blocked>
    %x_18 = arith.muli %mx, %x : tensor<8x1xi32, #blocked>
    %x_19 = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<8x1x!tt.ptr<f32>, #blocked>
    %x_20 = tt.addptr %x_19, %x_18 : tensor<8x1x!tt.ptr<f32>, #blocked>, tensor<8x1xi32, #blocked>
    %x_21 = tt.broadcast %x_20 : tensor<8x1x!tt.ptr<f32>, #blocked> -> tensor<8x8x!tt.ptr<f32>, #blocked>
    %w = tt.splat %swn : i32 -> tensor<8x1xi32, #blocked>
    %w_22 = arith.muli %mw, %w : tensor<8x1xi32, #blocked>
    %w_23 = tt.splat %w_ptr : !tt.ptr<f32> -> tensor<8x1x!tt.ptr<f32>, #blocked>
    %w_24 = tt.addptr %w_23, %w_22 : tensor<8x1x!tt.ptr<f32>, #blocked>, tensor<8x1xi32, #blocked>
    %w_25 = tt.broadcast %w_24 : tensor<8x1x!tt.ptr<f32>, #blocked> -> tensor<8x8x!tt.ptr<f32>, #blocked>
    %a = tt.splat %sar : i32 -> tensor<8x1xi32, #blocked>
    %a_26 = arith.muli %ma, %a : tensor<8x1xi32, #blocked>
    %a_27 = tt.splat %a_ptr : !tt.ptr<f32> -> tensor<8x1x!tt.ptr<f32>, #blocked>
    %a_28 = tt.addptr %a_27, %a_26 : tensor<8x1x!tt.ptr<f32>, #blocked>, tensor<8x1xi32, #blocked>
    %a_29 = tt.broadcast %a_28 : tensor<8x1x!tt.ptr<f32>, #blocked> -> tensor<8x8x!tt.ptr<f32>, #blocked>
    %acc1:2 = scf.for %k = %c0_i32 to %K step %c8_i32 iter_args(%acc0_51 = %cst, %acc1_52 = %cst) -> (tensor<8x8xf32, #blocked>, tensor<8x8xf32, #blocked>)  : i32 {
      %offs_k = tt.splat %k : i32 -> tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %offs_k_53 = arith.addi %offs_k, %offs_m_1 : tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %mx_54 = tt.expand_dims %offs_k_53 {axis = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x8xi32, #blocked>
      %mx_55 = arith.cmpi slt, %mx_54, %mx_10 : tensor<1x8xi32, #blocked>
      %mx_56 = tt.broadcast %mx_55 : tensor<1x8xi1, #blocked> -> tensor<8x8xi1, #blocked>
      %mx_57 = arith.andi %mx_11, %mx_56 : tensor<8x8xi1, #blocked>
      %mw_58 = arith.andi %mw_14, %mx_56 : tensor<8x8xi1, #blocked>
      %ma_59 = arith.andi %ma_17, %mx_56 : tensor<8x8xi1, #blocked>
      %x_60 = tt.broadcast %mx_54 : tensor<1x8xi32, #blocked> -> tensor<8x8xi32, #blocked>
      %x_61 = tt.addptr %x_21, %x_60 : tensor<8x8x!tt.ptr<f32>, #blocked>, tensor<8x8xi32, #blocked>
      %x_62 = tt.load %x_61, %mx_57, %cst : tensor<8x8x!tt.ptr<f32>, #blocked>
      %w_63 = tt.addptr %w_25, %x_60 : tensor<8x8x!tt.ptr<f32>, #blocked>, tensor<8x8xi32, #blocked>
      %w_64 = tt.load %w_63, %mw_58, %cst : tensor<8x8x!tt.ptr<f32>, #blocked>
      %a_65 = tt.addptr %a_29, %x_60 : tensor<8x8x!tt.ptr<f32>, #blocked>, tensor<8x8xi32, #blocked>
      %a_66 = tt.load %a_65, %ma_59, %cst : tensor<8x8x!tt.ptr<f32>, #blocked>
      %acc0_67 = tt.trans %w_64 {order = array<i32: 1, 0>} : tensor<8x8xf32, #blocked> -> tensor<8x8xf32, #blocked1>
      %x_68 = ttg.convert_layout %x_62 : tensor<8x8xf32, #blocked> -> tensor<8x8xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
      %acc0_69 = ttg.convert_layout %acc0_67 : tensor<8x8xf32, #blocked1> -> tensor<8x8xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
      %acc0_70 = tt.dot %x_68, %acc0_69, %acc0_51 : tensor<8x8xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<8x8xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<8x8xf32, #blocked>
      %acc1_71 = tt.trans %a_66 {order = array<i32: 1, 0>} : tensor<8x8xf32, #blocked> -> tensor<8x8xf32, #blocked1>
      %acc1_72 = ttg.convert_layout %acc1_71 : tensor<8x8xf32, #blocked1> -> tensor<8x8xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
      %acc1_73 = tt.dot %x_68, %acc1_72, %acc1_52 : tensor<8x8xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<8x8xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<8x8xf32, #blocked>
      scf.yield %acc0_70, %acc1_73 : tensor<8x8xf32, #blocked>, tensor<8x8xf32, #blocked>
    }
    %mb = tt.expand_dims %offs_m_1 {axis = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x8xi32, #blocked>
    %mb_30 = tt.splat %R : i32 -> tensor<1x8xi32, #blocked>
    %mb_31 = arith.cmpi slt, %mb, %mb_30 : tensor<1x8xi32, #blocked>
    %mb_32 = tt.broadcast %mb_31 : tensor<1x8xi1, #blocked> -> tensor<8x8xi1, #blocked>
    %mb_33 = arith.andi %mw_14, %mb_32 : tensor<8x8xi1, #blocked>
    %b = tt.splat %sbn : i32 -> tensor<8x1xi32, #blocked>
    %b_34 = arith.muli %mw, %b : tensor<8x1xi32, #blocked>
    %b_35 = tt.splat %b_ptr : !tt.ptr<f32> -> tensor<8x1x!tt.ptr<f32>, #blocked>
    %b_36 = tt.addptr %b_35, %b_34 : tensor<8x1x!tt.ptr<f32>, #blocked>, tensor<8x1xi32, #blocked>
    %b_37 = tt.broadcast %b_36 : tensor<8x1x!tt.ptr<f32>, #blocked> -> tensor<8x8x!tt.ptr<f32>, #blocked>
    %b_38 = tt.broadcast %mb : tensor<1x8xi32, #blocked> -> tensor<8x8xi32, #blocked>
    %b_39 = tt.addptr %b_37, %b_38 : tensor<8x8x!tt.ptr<f32>, #blocked>, tensor<8x8xi32, #blocked>
    %b_40 = tt.load %b_39, %mb_33, %cst : tensor<8x8x!tt.ptr<f32>, #blocked>
    %acc0 = tt.trans %b_40 {order = array<i32: 1, 0>} : tensor<8x8xf32, #blocked> -> tensor<8x8xf32, #blocked1>
    %acc1_41 = ttg.convert_layout %acc1#1 : tensor<8x8xf32, #blocked> -> tensor<8x8xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    %acc0_42 = ttg.convert_layout %acc0 : tensor<8x8xf32, #blocked1> -> tensor<8x8xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
    %acc0_43 = tt.dot %acc1_41, %acc0_42, %cst : tensor<8x8xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<8x8xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<8x8xf32, #blocked>
    %acc0_44 = tt.splat %scale : f32 -> tensor<8x8xf32, #blocked>
    %acc0_45 = arith.mulf %acc0_44, %acc0_43 : tensor<8x8xf32, #blocked>
    %acc0_46 = arith.addf %acc1#0, %acc0_45 : tensor<8x8xf32, #blocked>
    %my = tt.expand_dims %offs_n_7 {axis = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x8xi32, #blocked>
    %my_47 = tt.splat %N : i32 -> tensor<1x8xi32, #blocked>
    %my_48 = arith.cmpi slt, %my, %my_47 : tensor<1x8xi32, #blocked>
    %my_49 = tt.broadcast %my_48 : tensor<1x8xi1, #blocked> -> tensor<8x8xi1, #blocked>
    %my_50 = arith.andi %mx_11, %my_49 : tensor<8x8xi1, #blocked>
    %0 = tt.splat %som : i32 -> tensor<8x1xi32, #blocked>
    %1 = arith.muli %mx, %0 : tensor<8x1xi32, #blocked>
    %2 = tt.splat %o_ptr : !tt.ptr<f32> -> tensor<8x1x!tt.ptr<f32>, #blocked>
    %3 = tt.addptr %2, %1 : tensor<8x1x!tt.ptr<f32>, #blocked>, tensor<8x1xi32, #blocked>
    %4 = tt.broadcast %3 : tensor<8x1x!tt.ptr<f32>, #blocked> -> tensor<8x8x!tt.ptr<f32>, #blocked>
    %5 = tt.broadcast %my : tensor<1x8xi32, #blocked> -> tensor<8x8xi32, #blocked>
    %6 = tt.addptr %4, %5 : tensor<8x8x!tt.ptr<f32>, #blocked>, tensor<8x8xi32, #blocked>
    tt.store %6, %acc0_46, %my_50 : tensor<8x8x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
// CHECK-LABEL: metal.kernel fused_lora_masked
// CHECK-NOT: tt.dot
// CHECK-NOT: tt.trans
// CHECK-NOT: convert_layout
// CHECK: scf.for
// CHECK: metal.simdgroup_load_device_staged_masked
// CHECK: metal.simdgroup_fused_store
