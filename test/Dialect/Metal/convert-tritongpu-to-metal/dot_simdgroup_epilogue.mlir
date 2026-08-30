// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | FileCheck %s
//
// SIMD-group GEMM fast path (default): the zero-accumulator f32 dot + masked
// `alpha*out + beta*C` epilogue is lowered by ScalarDotLowering's SIMD-group
// branch. The warp-cooperative cone runs once (guarded by `iv == 0`): masked
// staged loads of A/B feed `simdgroup_multiply_accumulate` over the K axis, the
// 8x8 tile is `simdgroup_store`d into a threadgroup scratch buffer, and every
// thread then reloads its element (`tg_load_indexed`) for the epilogue. Ragged
// M/N/K are handled by the masked staged loads' extents.
//
// CHECK-LABEL: metal.kernel
// CHECK: metal.threadgroup_alloca
// CHECK: metal.simdgroup_index
// CHECK: metal.simdgroup_matrix_zero
// CHECK: metal.simdgroup_load_device_staged_masked
// CHECK: metal.simdgroup_multiply_accumulate
// CHECK: metal.simdgroup_store
// CHECK: metal.barrier
// CHECK: metal.tg_load_indexed
// No dot / convert_layout survives to the metal dialect.
// CHECK-NOT: tt.dot
// CHECK-NOT: ttg.convert_layout
//
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @gemm16(%a: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %b: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %c: !tt.ptr<f16> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c16_i32 = arith.constant 16 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #blocked>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #blocked1>
    %out = arith.constant dense<2.000000e+00> : tensor<16x16xf32, #blocked>
    %cst_1 = arith.constant dense<16> : tensor<1x16xi32, #blocked1>
    %cst_2 = arith.constant dense<16> : tensor<16x1xi32, #blocked1>
    %bx = tt.get_program_id x : i32
    %by = tt.get_program_id y : i32
    %row = arith.muli %by, %c16_i32 : i32
    %col = arith.muli %bx, %c16_i32 : i32
    %out_3 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %out_4 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %out_5 = tt.expand_dims %out_3 {axis = 1 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<16x1xi32, #blocked1>
    %out_6 = tt.expand_dims %out_4 {axis = 1 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<16x1xi32, #blocked>
    %out_7 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %out_8 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %out_9 = tt.expand_dims %out_7 {axis = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x16xi32, #blocked1>
    %out_10 = tt.expand_dims %out_8 {axis = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x16xi32, #blocked>
    %out_11 = tt.broadcast %out_6 : tensor<16x1xi32, #blocked> -> tensor<16x16xi32, #blocked>
    %out_12 = tt.broadcast %out_9 : tensor<1x16xi32, #blocked1> -> tensor<16x16xi32, #blocked1>
    %out_13 = tt.broadcast %out_10 : tensor<1x16xi32, #blocked> -> tensor<16x16xi32, #blocked>
    %out_14 = arith.muli %out_11, %out_13 : tensor<16x16xi32, #blocked>
    %out_15 = arith.sitofp %out_14 : tensor<16x16xi32, #blocked> to tensor<16x16xf32, #blocked>
    %out_16 = arith.mulf %out_15, %cst : tensor<16x16xf32, #blocked>
    %ay = tt.splat %row : i32 -> tensor<16x1xi32, #blocked1>
    %ay_17 = arith.addi %out_5, %ay : tensor<16x1xi32, #blocked1>
    %bxo = tt.splat %col : i32 -> tensor<1x16xi32, #blocked1>
    %bxo_18 = arith.addi %out_9, %bxo : tensor<1x16xi32, #blocked1>
    %aym = arith.cmpi slt, %ay_17, %cst_2 : tensor<16x1xi32, #blocked1>
    %bxm = arith.cmpi slt, %bxo_18, %cst_1 : tensor<1x16xi32, #blocked1>
    %ad = arith.cmpi slt, %out_9, %cst_1 : tensor<1x16xi32, #blocked1>
    %ad_19 = tt.broadcast %ad : tensor<1x16xi1, #blocked1> -> tensor<16x16xi1, #blocked1>
    %ad_20 = tt.broadcast %aym : tensor<16x1xi1, #blocked1> -> tensor<16x16xi1, #blocked1>
    %ad_21 = arith.andi %ad_19, %ad_20 : tensor<16x16xi1, #blocked1>
    %ad_22 = arith.muli %ay_17, %cst_2 : tensor<16x1xi32, #blocked1>
    %ad_23 = tt.splat %a : !tt.ptr<f16> -> tensor<16x1x!tt.ptr<f16>, #blocked1>
    %ad_24 = tt.addptr %ad_23, %ad_22 : tensor<16x1x!tt.ptr<f16>, #blocked1>, tensor<16x1xi32, #blocked1>
    %ad_25 = tt.broadcast %ad_24 : tensor<16x1x!tt.ptr<f16>, #blocked1> -> tensor<16x16x!tt.ptr<f16>, #blocked1>
    %ad_26 = tt.addptr %ad_25, %out_12 : tensor<16x16x!tt.ptr<f16>, #blocked1>, tensor<16x16xi32, #blocked1>
    %ad_27 = tt.load %ad_26, %ad_21, %cst_0 : tensor<16x16x!tt.ptr<f16>, #blocked1>
    %bd = arith.cmpi slt, %out_5, %cst_2 : tensor<16x1xi32, #blocked1>
    %bd_28 = tt.broadcast %bxm : tensor<1x16xi1, #blocked1> -> tensor<16x16xi1, #blocked1>
    %bd_29 = tt.broadcast %bd : tensor<16x1xi1, #blocked1> -> tensor<16x16xi1, #blocked1>
    %bd_30 = arith.andi %bd_28, %bd_29 : tensor<16x16xi1, #blocked1>
    %bd_31 = arith.muli %out_5, %cst_2 : tensor<16x1xi32, #blocked1>
    %bd_32 = tt.splat %b : !tt.ptr<f16> -> tensor<16x1x!tt.ptr<f16>, #blocked1>
    %bd_33 = tt.addptr %bd_32, %bd_31 : tensor<16x1x!tt.ptr<f16>, #blocked1>, tensor<16x1xi32, #blocked1>
    %bd_34 = tt.broadcast %bd_33 : tensor<16x1x!tt.ptr<f16>, #blocked1> -> tensor<16x16x!tt.ptr<f16>, #blocked1>
    %bd_35 = tt.broadcast %bxo_18 : tensor<1x16xi32, #blocked1> -> tensor<16x16xi32, #blocked1>
    %bd_36 = tt.addptr %bd_34, %bd_35 : tensor<16x16x!tt.ptr<f16>, #blocked1>, tensor<16x16xi32, #blocked1>
    %bd_37 = tt.load %bd_36, %bd_30, %cst_0 : tensor<16x16x!tt.ptr<f16>, #blocked1>
    %out_38 = arith.extf %ad_27 : tensor<16x16xf16, #blocked1> to tensor<16x16xf32, #blocked1>
    %out_39 = arith.extf %bd_37 : tensor<16x16xf16, #blocked1> to tensor<16x16xf32, #blocked1>
    %out_40 = ttg.convert_layout %out_38 : tensor<16x16xf32, #blocked1> -> tensor<16x16xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    %out_41 = ttg.convert_layout %out_39 : tensor<16x16xf32, #blocked1> -> tensor<16x16xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
    %out_42 = tt.dot %out_40, %out_41, %out_16 : tensor<16x16xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<16x16xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<16x16xf32, #blocked>
    %co = tt.splat %c : !tt.ptr<f16> -> tensor<16x1x!tt.ptr<f16>, #blocked1>
    %co_43 = tt.addptr %co, %ad_22 : tensor<16x1x!tt.ptr<f16>, #blocked1>, tensor<16x1xi32, #blocked1>
    %co_44 = tt.broadcast %co_43 : tensor<16x1x!tt.ptr<f16>, #blocked1> -> tensor<16x16x!tt.ptr<f16>, #blocked1>
    %co_45 = tt.addptr %co_44, %bd_35 : tensor<16x16x!tt.ptr<f16>, #blocked1>, tensor<16x16xi32, #blocked1>
    %cm = arith.andi %ad_20, %bd_28 : tensor<16x16xi1, #blocked1>
    %cd = tt.load %co_45, %cm, %cst_0 : tensor<16x16x!tt.ptr<f16>, #blocked1>
    %out_46 = arith.mulf %out_42, %out : tensor<16x16xf32, #blocked>
    %out_47 = ttg.convert_layout %out_46 : tensor<16x16xf32, #blocked> -> tensor<16x16xf32, #blocked1>
    %out_48 = arith.extf %cd : tensor<16x16xf16, #blocked1> to tensor<16x16xf32, #blocked1>
    %out_49 = arith.addf %out_47, %out_48 : tensor<16x16xf32, #blocked1>
    %0 = arith.truncf %out_49 : tensor<16x16xf32, #blocked1> to tensor<16x16xf16, #blocked1>
    tt.store %co_45, %0, %cm : tensor<16x16x!tt.ptr<f16>, #blocked1>
    tt.return
  }
}
