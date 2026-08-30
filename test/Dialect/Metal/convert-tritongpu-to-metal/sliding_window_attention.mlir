// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | FileCheck %s
//
// The sliding-window self-attention loop from
// leet-triton/hard-sliding_window_self_attention.py is recognized and collapsed
// into one metal.fused_attention. The band mask is not an operand and not a
// mode: it is ORDINARY ARITHMETIC INSIDE THE SCORE REGION, which is the whole
// point of the op -- a new masking scheme costs a cone-table entry, not a new
// op with a new matcher and a new hand-printed body.
//
// Compared with the multi-head kernel in flash_attention.mlir it differs in six
// independent ways, each of which the matcher had to learn:
//
//   1. band mask   `|offset_m - offset_n| <= window_size`
//                  (arith.subi -> math.absi -> arith.cmpi sle) -> region
//   2. no heads    1-D grid, no `pid1 * d_head` column offset, so `heads` is
//                  ABSENT and d_head == d_model == the `d` kernel arg
//   3. loop form   `range(0, cdiv(N, BLOCK_N))` with step 1 and key offset
//                  `iv * BLOCK_N + arange`, not `range(0, N, BLOCK_N)`
//   4. scale       `dot / splat(sqrt(d))`, not `dot * splat(1/sqrt(d_head))`
//   5. addressing  two-level `addptr(broadcast(addptr(base, row*d)), col)`
//   6. algebra     `select`-to-zero numerator chain and `mul`+`add` /
//                  dot-with-rescaled-C accumulation instead of `math.fma`
//
// M and N are separate kernel args here, so the op takes both.
//
// CHECK: metal.kernel attention
// CHECK: metal.fused_attention
// CHECK-SAME: bd = 16
// CHECK-SAME: bm = 16
// CHECK-SAME: bn = 16
// CHECK-SAME: norm = 1
// No `heads` operand: this kernel has no head dimension.
// CHECK-NOT: heads
//
// The band mask has to REACH THE REGION. Checking only that the kernel
// collapsed would pass just as happily on a body that dropped the mask and
// computed full attention -- which is exactly what the predecessor op did, at
// max abs err 0.95-2.4, for six commits before anyone noticed.
// `abs(row - key)` has no unary form in the dialect, so it lowers to
// `select(d < 0, 0 - d, d)`, and the masked-out logit is -inf (0xFF800000).
// CHECK: ^bb0(%[[SCORE:.*]]: f32, %[[ROW:.*]]: si32, %[[KEY:.*]]: si32, %{{.*}}: si32
// CHECK: metal.unary_exp %{{.*}}, sqrtOp
// CHECK: metal.binary_exp %[[SCORE]], %{{.*}}, divOp
// CHECK: %[[DIFF:.*]] = metal.binary_exp %[[ROW]], %[[KEY]], subOp
// CHECK: metal.binary_exp %{{.*}}, %[[DIFF]], subOp
// CHECK: metal.binary_exp %[[DIFF]], %{{.*}}, ltOp
// CHECK: %[[ABS:.*]] = arith.select
// CHECK: metal.binary_exp %[[ABS]], %{{.*}}, leOp
// CHECK: metal.constant 0xFF800000
// CHECK: arith.select
// CHECK: metal.score_yield
//
// The key range comes from the op's second region, which reproduces the
// source's `range(0, cdiv(N, BLOCK_N))` as `cdiv(n, 16) * 16` rather than
// substituting `n` for it -- the bound is REPRODUCED, not reasoned about.
// CHECK: ^bb0(%[[BLK:.*]]: si32, %{{.*}}: si32, %{{.*}}: si32, %[[N:.*]]: si32
// CHECK: metal.key_bounds_yield
//
// The whole loop / dots / softmax reduces / band mask / convert_layouts are gone.
// CHECK-NOT: tt.dot
// CHECK-NOT: tt.reduce
// CHECK-NOT: math.absi
// CHECK-NOT: ttg.convert_layout
// CHECK-NOT: scf.for
//
#blocked = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [2, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1]}>
#loc = loc("/Users/mocus/Code/triton/leet-triton/hard-sliding_window_self_attention.py":6:1)
#loc1 = loc(unknown)
#loc35 = loc("/Users/mocus/Code/triton/leet-triton/hard-sliding_window_self_attention.py":37:29)
#loc46 = loc("/Users/mocus/Code/triton/leet-triton/hard-sliding_window_self_attention.py":40:19)
#loc58 = loc("Q"(#loc))
#loc59 = loc("K"(#loc))
#loc60 = loc("V"(#loc))
#loc61 = loc("output"(#loc))
#loc62 = loc("M"(#loc))
#loc63 = loc("N"(#loc))
#loc64 = loc("d"(#loc))
#loc65 = loc("window_size"(#loc))
#loc98 = loc("ma_now"(#loc35))
#loc107 = loc("sum_now"(#loc46))
#loc116 = loc(callsite(#loc1 at #loc98))
#loc118 = loc(callsite(#loc1 at #loc107))
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @attention(%Q: !tt.ptr<f32> {tt.divisibility = 16 : i32} loc("Q"(#loc)), %K: !tt.ptr<f32> {tt.divisibility = 16 : i32} loc("K"(#loc)), %V: !tt.ptr<f32> {tt.divisibility = 16 : i32} loc("V"(#loc)), %output: !tt.ptr<f32> {tt.divisibility = 16 : i32} loc("output"(#loc)), %M: i32 {tt.divisibility = 16 : i32} loc("M"(#loc)), %N: i32 {tt.divisibility = 16 : i32} loc("N"(#loc)), %d: i32 {tt.divisibility = 16 : i32} loc("d"(#loc)), %window_size: i32 loc("window_size"(#loc))) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #blocked> loc(#loc1)
    %c16_i32 = arith.constant 16 : i32 loc(#loc1)
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #blocked1> loc(#loc1)
    %c15_i32 = arith.constant 15 : i32 loc(#loc66)
    %c1_i32 = arith.constant 1 : i32 loc(#loc3)
    %c0_i32 = arith.constant 0 : i32 loc(#loc3)
    %cst_1 = arith.constant dense<-1.000000e+02> : tensor<16x16xf32, #blocked1> loc(#loc1)
    %sum = arith.constant dense<0.000000e+00> : tensor<16xf32, #ttg.slice<{dim = 1, parent = #blocked1}>> loc(#loc67)
    %ma = arith.constant dense<0xFF800000> : tensor<16xf32, #ttg.slice<{dim = 1, parent = #blocked1}>> loc(#loc68)
    %pid = tt.get_program_id x : i32 loc(#loc69)
    %offset_m = arith.muli %pid, %c16_i32 : i32 loc(#loc70)
    %offset_m_2 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>> loc(#loc71)
    %offset_m_3 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> loc(#loc71)
    %offset_m_4 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked}>> loc(#loc71)
    %offset_m_5 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #blocked2> loc(#loc71)
    %offset_m_6 = tt.splat %offset_m : i32 -> tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>> loc(#loc70)
    %offset_m_7 = tt.splat %offset_m : i32 -> tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> loc(#loc70)
    %offset_m_8 = arith.addi %offset_m_6, %offset_m_2 : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>> loc(#loc70)
    %offset_m_9 = arith.addi %offset_m_7, %offset_m_3 : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> loc(#loc70)
    %mask_m = tt.splat %M : i32 -> tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>> loc(#loc72)
    %mask_m_10 = tt.splat %M : i32 -> tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> loc(#loc72)
    %mask_m_11 = arith.cmpi slt, %offset_m_8, %mask_m : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>> loc(#loc72)
    %mask_m_12 = arith.cmpi slt, %offset_m_9, %mask_m_10 : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> loc(#loc72)
    %mask_d = tt.splat %d : i32 -> tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked}>> loc(#loc73)
    %mask_d_13 = arith.cmpi slt, %offset_m_4, %mask_d : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked}>> loc(#loc73)
    %vals_q = tt.expand_dims %offset_m_8 {axis = 1 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<16x1xi32, #blocked> loc(#loc74)
    %vals_q_14 = tt.expand_dims %offset_m_9 {axis = 1 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<16x1xi32, #blocked1> loc(#loc74)
    %vals_q_15 = tt.splat %d : i32 -> tensor<16x1xi32, #blocked> loc(#loc74)
    %vals_q_16 = arith.muli %vals_q, %vals_q_15 : tensor<16x1xi32, #blocked> loc(#loc74)
    %vals_q_17 = tt.splat %Q : !tt.ptr<f32> -> tensor<16x1x!tt.ptr<f32>, #blocked> loc(#loc75)
    %vals_q_18 = tt.addptr %vals_q_17, %vals_q_16 : tensor<16x1x!tt.ptr<f32>, #blocked>, tensor<16x1xi32, #blocked> loc(#loc75)
    %vals_q_19 = tt.expand_dims %offset_m_4 {axis = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x16xi32, #blocked> loc(#loc76)
    %vals_q_20 = tt.broadcast %vals_q_18 : tensor<16x1x!tt.ptr<f32>, #blocked> -> tensor<16x16x!tt.ptr<f32>, #blocked> loc(#loc75)
    %vals_q_21 = tt.broadcast %vals_q_19 : tensor<1x16xi32, #blocked> -> tensor<16x16xi32, #blocked> loc(#loc75)
    %vals_q_22 = tt.addptr %vals_q_20, %vals_q_21 : tensor<16x16x!tt.ptr<f32>, #blocked>, tensor<16x16xi32, #blocked> loc(#loc75)
    %vals_q_23 = tt.expand_dims %mask_m_11 {axis = 1 : i32} : tensor<16xi1, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<16x1xi1, #blocked> loc(#loc77)
    %vals_q_24 = tt.expand_dims %mask_m_12 {axis = 1 : i32} : tensor<16xi1, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<16x1xi1, #blocked1> loc(#loc77)
    %vals_q_25 = tt.expand_dims %mask_d_13 {axis = 0 : i32} : tensor<16xi1, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x16xi1, #blocked> loc(#loc78)
    %vals_q_26 = tt.broadcast %vals_q_23 : tensor<16x1xi1, #blocked> -> tensor<16x16xi1, #blocked> loc(#loc77)
    %vals_q_27 = tt.broadcast %vals_q_24 : tensor<16x1xi1, #blocked1> -> tensor<16x16xi1, #blocked1> loc(#loc77)
    %vals_q_28 = tt.broadcast %vals_q_25 : tensor<1x16xi1, #blocked> -> tensor<16x16xi1, #blocked> loc(#loc77)
    %vals_q_29 = arith.andi %vals_q_26, %vals_q_28 : tensor<16x16xi1, #blocked> loc(#loc77)
    %vals_q_30 = tt.load %vals_q_22, %vals_q_29, %cst : tensor<16x16x!tt.ptr<f32>, #blocked> loc(#loc79)
    %scale = arith.sitofp %d : i32 to f32 loc(#loc80)
    %scale_31 = math.sqrt %scale : f32 loc(#loc81)
    %0 = arith.addi %N, %c15_i32 : i32 loc(#loc82)
    %1 = arith.divsi %0, %c16_i32 : i32 loc(#loc83)
    %mask_n = tt.splat %N : i32 -> tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>> loc(#loc84)
    %mask_n_32 = tt.splat %N : i32 -> tensor<16xi32, #blocked2> loc(#loc84)
    %vals_k = tt.splat %K : !tt.ptr<f32> -> tensor<16x1x!tt.ptr<f32>, #blocked> loc(#loc85)
    %vals_qk = tt.splat %scale_31 : f32 -> tensor<16x16xf32, #blocked1> loc(#loc86)
    %mask = tt.broadcast %vals_q_14 : tensor<16x1xi32, #blocked1> -> tensor<16x16xi32, #blocked1> loc(#loc87)
    %mask_33 = tt.splat %window_size : i32 -> tensor<16x16xi32, #blocked1> loc(#loc88)
    %vals_v = tt.splat %V : !tt.ptr<f32> -> tensor<16x1x!tt.ptr<f32>, #blocked> loc(#loc89)
    %sum_34:3 = scf.for %step = %c0_i32 to %1 step %c1_i32 iter_args(%out_vals = %cst_0, %ma_35 = %ma, %sum_36 = %sum) -> (tensor<16x16xf32, #blocked1>, tensor<16xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<16xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>)  : i32 {
      %offset_n = arith.muli %step, %c16_i32 : i32 loc(#loc91)
      %offset_n_37 = tt.splat %offset_n : i32 -> tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>> loc(#loc91)
      %offset_n_38 = tt.splat %offset_n : i32 -> tensor<16xi32, #blocked2> loc(#loc91)
      %offset_n_39 = arith.addi %offset_n_37, %offset_m_2 : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>> loc(#loc91)
      %offset_n_40 = arith.addi %offset_n_38, %offset_m_5 : tensor<16xi32, #blocked2> loc(#loc91)
      %mask_n_41 = arith.cmpi slt, %offset_n_39, %mask_n : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>> loc(#loc84)
      %mask_n_42 = arith.cmpi slt, %offset_n_40, %mask_n_32 : tensor<16xi32, #blocked2> loc(#loc84)
      %vals_k_43 = tt.expand_dims %offset_n_39 {axis = 1 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<16x1xi32, #blocked> loc(#loc92)
      %vals_k_44 = arith.muli %vals_k_43, %vals_q_15 : tensor<16x1xi32, #blocked> loc(#loc92)
      %vals_k_45 = tt.addptr %vals_k, %vals_k_44 : tensor<16x1x!tt.ptr<f32>, #blocked>, tensor<16x1xi32, #blocked> loc(#loc85)
      %vals_k_46 = tt.broadcast %vals_k_45 : tensor<16x1x!tt.ptr<f32>, #blocked> -> tensor<16x16x!tt.ptr<f32>, #blocked> loc(#loc85)
      %vals_k_47 = tt.addptr %vals_k_46, %vals_q_21 : tensor<16x16x!tt.ptr<f32>, #blocked>, tensor<16x16xi32, #blocked> loc(#loc85)
      %vals_k_48 = tt.expand_dims %mask_n_41 {axis = 1 : i32} : tensor<16xi1, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<16x1xi1, #blocked> loc(#loc93)
      %vals_k_49 = tt.broadcast %vals_k_48 : tensor<16x1xi1, #blocked> -> tensor<16x16xi1, #blocked> loc(#loc93)
      %vals_k_50 = arith.andi %vals_k_49, %vals_q_28 : tensor<16x16xi1, #blocked> loc(#loc93)
      %vals_k_51 = tt.load %vals_k_47, %vals_k_50, %cst : tensor<16x16x!tt.ptr<f32>, #blocked> loc(#loc94)
      %vals_qk_52 = tt.trans %vals_k_51 {order = array<i32: 1, 0>} : tensor<16x16xf32, #blocked> -> tensor<16x16xf32, #blocked3> loc(#loc95)
      %vals_q_53 = ttg.convert_layout %vals_q_30 : tensor<16x16xf32, #blocked> -> tensor<16x16xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked1}>> loc(#loc79)
      %vals_qk_54 = ttg.convert_layout %vals_qk_52 : tensor<16x16xf32, #blocked3> -> tensor<16x16xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked1}>> loc(#loc86)
      %vals_qk_55 = tt.dot %vals_q_53, %vals_qk_54, %cst_0 : tensor<16x16xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked1}>> * tensor<16x16xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked1}>> -> tensor<16x16xf32, #blocked1> loc(#loc86)
      %vals_qk_56 = arith.divf %vals_qk_55, %vals_qk : tensor<16x16xf32, #blocked1> loc(#loc86)
      %mask_57 = ttg.convert_layout %offset_n_40 : tensor<16xi32, #blocked2> -> tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked4}>> loc(#loc96)
      %mask_58 = tt.expand_dims %mask_57 {axis = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked4}>> -> tensor<1x16xi32, #blocked4> loc(#loc96)
      %mask_59 = ttg.convert_layout %mask_58 : tensor<1x16xi32, #blocked4> -> tensor<1x16xi32, #blocked1> loc(#loc87)
      %mask_60 = tt.broadcast %mask_59 : tensor<1x16xi32, #blocked1> -> tensor<16x16xi32, #blocked1> loc(#loc87)
      %mask_61 = arith.subi %mask, %mask_60 : tensor<16x16xi32, #blocked1> loc(#loc87)
      %mask_62 = math.absi %mask_61 : tensor<16x16xi32, #blocked1> loc(#loc88)
      %mask_63 = arith.cmpi sle, %mask_62, %mask_33 : tensor<16x16xi32, #blocked1> loc(#loc88)
      %vals_qk_ma = arith.select %mask_63, %vals_qk_56, %cst_1 : tensor<16x16xi1, #blocked1>, tensor<16x16xf32, #blocked1> loc(#loc97)
      %ma_now = "tt.reduce"(%vals_qk_ma) <{axis = 1 : i32}> ({
      ^bb0(%ma_now_89: f32 loc(callsite(#loc1 at #loc98)), %ma_now_90: f32 loc(callsite(#loc1 at #loc98))):
        %ma_now_91 = arith.maxnumf %ma_now_89, %ma_now_90 : f32 loc(#loc120)
        tt.reduce.return %ma_now_91 : f32 loc(#loc115)
      }) : (tensor<16x16xf32, #blocked1>) -> tensor<16xf32, #ttg.slice<{dim = 1, parent = #blocked1}>> loc(#loc115)
      %ma_now_64 = arith.maxnumf %ma_now, %ma_35 : tensor<16xf32, #ttg.slice<{dim = 1, parent = #blocked1}>> loc(#loc99)
      %vals_exp = ttg.convert_layout %mask_n_42 : tensor<16xi1, #blocked2> -> tensor<16xi1, #ttg.slice<{dim = 0, parent = #blocked4}>> loc(#loc100)
      %vals_exp_65 = tt.expand_dims %vals_exp {axis = 0 : i32} : tensor<16xi1, #ttg.slice<{dim = 0, parent = #blocked4}>> -> tensor<1x16xi1, #blocked4> loc(#loc100)
      %vals_exp_66 = ttg.convert_layout %vals_exp_65 : tensor<1x16xi1, #blocked4> -> tensor<1x16xi1, #blocked1> loc(#loc101)
      %vals_exp_67 = tt.broadcast %vals_exp_66 : tensor<1x16xi1, #blocked1> -> tensor<16x16xi1, #blocked1> loc(#loc101)
      %vals_exp_68 = arith.andi %vals_q_27, %vals_exp_67 : tensor<16x16xi1, #blocked1> loc(#loc101)
      %vals_exp_69 = tt.expand_dims %ma_now_64 {axis = 1 : i32} : tensor<16xf32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<16x1xf32, #blocked1> loc(#loc102)
      %vals_exp_70 = tt.broadcast %vals_exp_69 : tensor<16x1xf32, #blocked1> -> tensor<16x16xf32, #blocked1> loc(#loc103)
      %vals_exp_71 = arith.subf %vals_qk_56, %vals_exp_70 : tensor<16x16xf32, #blocked1> loc(#loc103)
      %vals_exp_72 = math.exp %vals_exp_71 : tensor<16x16xf32, #blocked1> loc(#loc104)
      %vals_exp_73 = arith.select %vals_exp_68, %vals_exp_72, %cst_0 : tensor<16x16xi1, #blocked1>, tensor<16x16xf32, #blocked1> loc(#loc105)
      %vals_exp_74 = arith.select %mask_63, %vals_exp_73, %cst_0 : tensor<16x16xi1, #blocked1>, tensor<16x16xf32, #blocked1> loc(#loc106)
      %sum_now = "tt.reduce"(%vals_exp_74) <{axis = 1 : i32}> ({
      ^bb0(%sum_now_89: f32 loc(callsite(#loc1 at #loc107)), %sum_now_90: f32 loc(callsite(#loc1 at #loc107))):
        %sum_now_91 = arith.addf %sum_now_89, %sum_now_90 : f32 loc(#loc121)
        tt.reduce.return %sum_now_91 : f32 loc(#loc117)
      }) : (tensor<16x16xf32, #blocked1>) -> tensor<16xf32, #ttg.slice<{dim = 1, parent = #blocked1}>> loc(#loc117)
      %vals_v_75 = tt.addptr %vals_v, %vals_k_44 : tensor<16x1x!tt.ptr<f32>, #blocked>, tensor<16x1xi32, #blocked> loc(#loc89)
      %vals_v_76 = tt.broadcast %vals_v_75 : tensor<16x1x!tt.ptr<f32>, #blocked> -> tensor<16x16x!tt.ptr<f32>, #blocked> loc(#loc89)
      %vals_v_77 = tt.addptr %vals_v_76, %vals_q_21 : tensor<16x16x!tt.ptr<f32>, #blocked>, tensor<16x16xi32, #blocked> loc(#loc89)
      %vals_v_78 = tt.load %vals_v_77, %vals_k_50, %cst : tensor<16x16x!tt.ptr<f32>, #blocked> loc(#loc108)
      %out_vals_79 = arith.subf %ma_35, %ma_now_64 : tensor<16xf32, #ttg.slice<{dim = 1, parent = #blocked1}>> loc(#loc109)
      %out_vals_80 = math.exp %out_vals_79 : tensor<16xf32, #ttg.slice<{dim = 1, parent = #blocked1}>> loc(#loc110)
      %out_vals_81 = tt.expand_dims %out_vals_80 {axis = 1 : i32} : tensor<16xf32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<16x1xf32, #blocked1> loc(#loc110)
      %out_vals_82 = tt.broadcast %out_vals_81 : tensor<16x1xf32, #blocked1> -> tensor<16x16xf32, #blocked1> loc(#loc111)
      %out_vals_83 = arith.mulf %out_vals, %out_vals_82 : tensor<16x16xf32, #blocked1> loc(#loc111)
      %vals_exp_84 = ttg.convert_layout %vals_exp_74 : tensor<16x16xf32, #blocked1> -> tensor<16x16xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked1}>> loc(#loc106)
      %vals_v_85 = ttg.convert_layout %vals_v_78 : tensor<16x16xf32, #blocked> -> tensor<16x16xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked1}>> loc(#loc108)
      %out_vals_86 = tt.dot %vals_exp_84, %vals_v_85, %out_vals_83 : tensor<16x16xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked1}>> * tensor<16x16xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked1}>> -> tensor<16x16xf32, #blocked1> loc(#loc112)
      %sum_87 = arith.mulf %sum_36, %out_vals_80 : tensor<16xf32, #ttg.slice<{dim = 1, parent = #blocked1}>> loc(#loc113)
      %sum_88 = arith.addf %sum_87, %sum_now : tensor<16xf32, #ttg.slice<{dim = 1, parent = #blocked1}>> loc(#loc113)
      scf.yield %out_vals_86, %ma_now_64, %sum_88 : tensor<16x16xf32, #blocked1>, tensor<16xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<16xf32, #ttg.slice<{dim = 1, parent = #blocked1}>> loc(#loc3)
    } loc(#loc119)
    %2 = tt.splat %output : !tt.ptr<f32> -> tensor<16x1x!tt.ptr<f32>, #blocked> loc(#loc54)
    %3 = tt.addptr %2, %vals_q_16 : tensor<16x1x!tt.ptr<f32>, #blocked>, tensor<16x1xi32, #blocked> loc(#loc54)
    %4 = tt.broadcast %3 : tensor<16x1x!tt.ptr<f32>, #blocked> -> tensor<16x16x!tt.ptr<f32>, #blocked> loc(#loc54)
    %5 = tt.addptr %4, %vals_q_21 : tensor<16x16x!tt.ptr<f32>, #blocked>, tensor<16x16xi32, #blocked> loc(#loc54)
    %6 = tt.expand_dims %sum_34#2 {axis = 1 : i32} : tensor<16xf32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<16x1xf32, #blocked1> loc(#loc55)
    %7 = tt.broadcast %6 : tensor<16x1xf32, #blocked1> -> tensor<16x16xf32, #blocked1> loc(#loc56)
    %8 = arith.divf %sum_34#0, %7 : tensor<16x16xf32, #blocked1> loc(#loc56)
    %9 = ttg.convert_layout %8 : tensor<16x16xf32, #blocked1> -> tensor<16x16xf32, #blocked> loc(#loc57)
    tt.store %5, %9, %vals_q_29 : tensor<16x16x!tt.ptr<f32>, #blocked> loc(#loc57)
    tt.return loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc2 = loc("/Users/mocus/Code/triton/leet-triton/hard-sliding_window_self_attention.py":26:26)
#loc3 = loc("/Users/mocus/Code/triton/leet-triton/hard-sliding_window_self_attention.py":26:5)
#loc4 = loc("/Users/mocus/Code/triton/leet-triton/hard-sliding_window_self_attention.py":24:11)
#loc5 = loc("/Users/mocus/Code/triton/leet-triton/hard-sliding_window_self_attention.py":23:10)
#loc6 = loc("/Users/mocus/Code/triton/leet-triton/hard-sliding_window_self_attention.py":14:11)
#loc7 = loc("/Users/mocus/Code/triton/leet-triton/hard-sliding_window_self_attention.py":15:16)
#loc8 = loc("/Users/mocus/Code/triton/leet-triton/hard-sliding_window_self_attention.py":15:32)
#loc9 = loc("/Users/mocus/Code/triton/leet-triton/hard-sliding_window_self_attention.py":17:14)
#loc10 = loc("/Users/mocus/Code/triton/leet-triton/hard-sliding_window_self_attention.py":18:14)
#loc11 = loc("/Users/mocus/Code/triton/leet-triton/hard-sliding_window_self_attention.py":19:26)
#loc12 = loc("/Users/mocus/Code/triton/leet-triton/hard-sliding_window_self_attention.py":19:22)
#loc13 = loc("/Users/mocus/Code/triton/leet-triton/hard-sliding_window_self_attention.py":19:50)
#loc14 = loc("/Users/mocus/Code/triton/leet-triton/hard-sliding_window_self_attention.py":19:69)
#loc15 = loc("/Users/mocus/Code/triton/leet-triton/hard-sliding_window_self_attention.py":19:87)
#loc16 = loc("/Users/mocus/Code/triton/leet-triton/hard-sliding_window_self_attention.py":19:14)
#loc17 = loc("/Users/mocus/Code/triton/leet-triton/hard-sliding_window_self_attention.py":20:21)
#loc18 = loc("/Users/mocus/Code/triton/leet-triton/hard-sliding_window_self_attention.py":20:13)
#loc19 = loc("/Users/mocus/Code/triton/python/triton/language/standard.py":43:13)
#loc20 = loc("/Users/mocus/Code/triton/python/triton/language/standard.py":43:12)
#loc21 = loc("/Users/mocus/Code/triton/leet-triton/hard-sliding_window_self_attention.py":28:18)
#loc22 = loc("/Users/mocus/Code/triton/leet-triton/hard-sliding_window_self_attention.py":30:26)
#loc23 = loc("/Users/mocus/Code/triton/leet-triton/hard-sliding_window_self_attention.py":32:19)
#loc24 = loc("/Users/mocus/Code/triton/leet-triton/hard-sliding_window_self_attention.py":34:23)
#loc25 = loc("/Users/mocus/Code/triton/leet-triton/hard-sliding_window_self_attention.py":34:16)
#loc26 = loc("/Users/mocus/Code/triton/leet-triton/hard-sliding_window_self_attention.py":42:26)
#loc27 = loc("/Users/mocus/Code/triton/leet-triton/hard-sliding_window_self_attention.py":27:20)
#loc28 = loc("/Users/mocus/Code/triton/leet-triton/hard-sliding_window_self_attention.py":30:30)
#loc29 = loc("/Users/mocus/Code/triton/leet-triton/hard-sliding_window_self_attention.py":30:73)
#loc30 = loc("/Users/mocus/Code/triton/leet-triton/hard-sliding_window_self_attention.py":30:18)
#loc31 = loc("/Users/mocus/Code/triton/leet-triton/hard-sliding_window_self_attention.py":32:34)
#loc32 = loc("/Users/mocus/Code/triton/leet-triton/hard-sliding_window_self_attention.py":34:43)
#loc33 = loc("/Users/mocus/Code/triton/leet-triton/hard-sliding_window_self_attention.py":36:22)
#loc34 = loc("/Users/mocus/Code/triton/python/triton/language/standard.py":191:16)
#loc36 = loc("/Users/mocus/Code/triton/python/triton/language/standard.py":170:12)
#loc37 = loc("/Users/mocus/Code/triton/leet-triton/hard-sliding_window_self_attention.py":37:18)
#loc38 = loc("/Users/mocus/Code/triton/leet-triton/hard-sliding_window_self_attention.py":38:47)
#loc39 = loc("/Users/mocus/Code/triton/leet-triton/hard-sliding_window_self_attention.py":38:29)
#loc40 = loc("/Users/mocus/Code/triton/leet-triton/hard-sliding_window_self_attention.py":38:81)
#loc41 = loc("/Users/mocus/Code/triton/leet-triton/hard-sliding_window_self_attention.py":38:71)
#loc42 = loc("/Users/mocus/Code/triton/leet-triton/hard-sliding_window_self_attention.py":38:64)
#loc43 = loc("/Users/mocus/Code/triton/leet-triton/hard-sliding_window_self_attention.py":38:20)
#loc44 = loc("/Users/mocus/Code/triton/leet-triton/hard-sliding_window_self_attention.py":39:20)
#loc45 = loc("/Users/mocus/Code/triton/python/triton/language/standard.py":293:12)
#loc47 = loc("/Users/mocus/Code/triton/python/triton/language/standard.py":263:12)
#loc48 = loc("/Users/mocus/Code/triton/leet-triton/hard-sliding_window_self_attention.py":42:18)
#loc49 = loc("/Users/mocus/Code/triton/leet-triton/hard-sliding_window_self_attention.py":44:38)
#loc50 = loc("/Users/mocus/Code/triton/leet-triton/hard-sliding_window_self_attention.py":44:31)
#loc51 = loc("/Users/mocus/Code/triton/leet-triton/hard-sliding_window_self_attention.py":44:20)
#loc52 = loc("/Users/mocus/Code/triton/leet-triton/hard-sliding_window_self_attention.py":44:62)
#loc53 = loc("/Users/mocus/Code/triton/leet-triton/hard-sliding_window_self_attention.py":46:15)
#loc54 = loc("/Users/mocus/Code/triton/leet-triton/hard-sliding_window_self_attention.py":49:14)
#loc55 = loc("/Users/mocus/Code/triton/leet-triton/hard-sliding_window_self_attention.py":49:77)
#loc56 = loc("/Users/mocus/Code/triton/leet-triton/hard-sliding_window_self_attention.py":49:66)
#loc57 = loc("/Users/mocus/Code/triton/leet-triton/hard-sliding_window_self_attention.py":49:5)
#loc66 = loc(callsite(#loc1 at #loc2))
#loc67 = loc("sum"(#loc4))
#loc68 = loc("ma"(#loc5))
#loc69 = loc("pid"(#loc6))
#loc70 = loc("offset_m"(#loc7))
#loc71 = loc("offset_m"(#loc8))
#loc72 = loc("mask_m"(#loc9))
#loc73 = loc("mask_d"(#loc10))
#loc74 = loc("vals_q"(#loc11))
#loc75 = loc("vals_q"(#loc12))
#loc76 = loc("vals_q"(#loc13))
#loc77 = loc("vals_q"(#loc14))
#loc78 = loc("vals_q"(#loc15))
#loc79 = loc("vals_q"(#loc16))
#loc80 = loc("scale"(#loc17))
#loc81 = loc("scale"(#loc18))
#loc82 = loc(callsite(#loc19 at #loc2))
#loc83 = loc(callsite(#loc20 at #loc2))
#loc84 = loc("mask_n"(#loc21))
#loc85 = loc("vals_k"(#loc22))
#loc86 = loc("vals_qk"(#loc23))
#loc87 = loc("mask"(#loc24))
#loc88 = loc("mask"(#loc25))
#loc89 = loc("vals_v"(#loc26))
#loc90 = loc("out_vals"(#loc3))
#loc91 = loc("offset_n"(#loc27))
#loc92 = loc("vals_k"(#loc28))
#loc93 = loc("vals_k"(#loc29))
#loc94 = loc("vals_k"(#loc30))
#loc95 = loc("vals_qk"(#loc31))
#loc96 = loc("mask"(#loc32))
#loc97 = loc("vals_qk_ma"(#loc33))
#loc99 = loc("ma_now"(#loc37))
#loc100 = loc("vals_exp"(#loc38))
#loc101 = loc("vals_exp"(#loc39))
#loc102 = loc("vals_exp"(#loc40))
#loc103 = loc("vals_exp"(#loc41))
#loc104 = loc("vals_exp"(#loc42))
#loc105 = loc("vals_exp"(#loc43))
#loc106 = loc("vals_exp"(#loc44))
#loc108 = loc("vals_v"(#loc48))
#loc109 = loc("out_vals"(#loc49))
#loc110 = loc("out_vals"(#loc50))
#loc111 = loc("out_vals"(#loc51))
#loc112 = loc("out_vals"(#loc52))
#loc113 = loc("sum"(#loc53))
#loc114 = loc("ma"(#loc90))
#loc115 = loc(callsite(#loc34 at #loc98))
#loc117 = loc(callsite(#loc45 at #loc107))
#loc119 = loc("sum"(#loc114))
#loc120 = loc(callsite(#loc36 at #loc115))
#loc121 = loc(callsite(#loc47 at #loc117))
