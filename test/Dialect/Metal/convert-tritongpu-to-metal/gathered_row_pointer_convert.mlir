// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | FileCheck %s
//
// The embedding gather `emb + ids[:, None] * D + cols[None, :]` with as many
// rows as threads (128 at four warps). Triton bridges the row pointers with a
// `ttg.convert_layout` of a POINTER tile; the convert exchanged the addresses as
// integers, the masked load then had no buffer to read, declined inside the
// conversion, and the rollback moved back a block `FuncOpLowering` had already
// freed -- this file crashed triton-metal-opt on every run. The pass now
// converts the row offsets instead, so the load reads the embedding buffer.
// CHECK-LABEL: metal.kernel gather_copy
// CHECK: metal.get_element %arg1[
// CHECK: metal.store
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @gather_copy(%tid_ptr: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %emb_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %o_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %N: i32, %D: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %x = arith.constant dense<0.000000e+00> : tensor<128x32xf32, #blocked>
    %ids = arith.constant dense<0> : tensor<128xi32, #blocked1>
    %rows = arith.constant 128 : i32
    %rows_0 = tt.get_program_id x : i32
    %rows_1 = arith.muli %rows_0, %rows : i32
    %rows_2 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %rows_3 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked1>
    %rows_4 = tt.splat %rows_1 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %rows_5 = tt.splat %rows_1 : i32 -> tensor<128xi32, #blocked1>
    %rows_6 = arith.addi %rows_4, %rows_2 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %rows_7 = arith.addi %rows_5, %rows_3 : tensor<128xi32, #blocked1>
    %cols = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %rm = tt.splat %N : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %rm_8 = tt.splat %N : i32 -> tensor<128xi32, #blocked1>
    %rm_9 = arith.cmpi slt, %rows_6, %rm : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %rm_10 = arith.cmpi slt, %rows_7, %rm_8 : tensor<128xi32, #blocked1>
    %m2 = tt.expand_dims %rm_9 {axis = 1 : i32} : tensor<128xi1, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi1, #blocked>
    %m2_11 = tt.splat %D : i32 -> tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %m2_12 = arith.cmpi slt, %cols, %m2_11 : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %m2_13 = tt.expand_dims %m2_12 {axis = 0 : i32} : tensor<32xi1, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi1, #blocked>
    %m2_14 = tt.broadcast %m2 : tensor<128x1xi1, #blocked> -> tensor<128x32xi1, #blocked>
    %m2_15 = tt.broadcast %m2_13 : tensor<1x32xi1, #blocked> -> tensor<128x32xi1, #blocked>
    %m2_16 = arith.andi %m2_14, %m2_15 : tensor<128x32xi1, #blocked>
    %ids_17 = tt.splat %tid_ptr : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>, #blocked1>
    %ids_18 = tt.addptr %ids_17, %rows_7 : tensor<128x!tt.ptr<i32>, #blocked1>, tensor<128xi32, #blocked1>
    %ids_19 = tt.load %ids_18, %rm_10, %ids : tensor<128x!tt.ptr<i32>, #blocked1>
    %x_20 = ttg.convert_layout %ids_19 : tensor<128xi32, #blocked1> -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %ids_21 = arith.extsi %x_20 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> to tensor<128xi64, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %x_22 = ttg.convert_layout %ids_21 : tensor<128xi64, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<128xi64, #ttg.slice<{dim = 1, parent = #blocked3}>>
    %x_23 = tt.expand_dims %x_22 {axis = 1 : i32} : tensor<128xi64, #ttg.slice<{dim = 1, parent = #blocked3}>> -> tensor<128x1xi64, #blocked3>
    %x_24 = arith.extsi %D : i32 to i64
    %x_25 = tt.splat %x_24 : i64 -> tensor<128x1xi64, #blocked3>
    %x_26 = arith.muli %x_23, %x_25 : tensor<128x1xi64, #blocked3>
    %x_27 = tt.splat %emb_ptr : !tt.ptr<f32> -> tensor<128x1x!tt.ptr<f32>, #blocked3>
    %x_28 = tt.addptr %x_27, %x_26 : tensor<128x1x!tt.ptr<f32>, #blocked3>, tensor<128x1xi64, #blocked3>
    %x_29 = tt.expand_dims %cols {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked>
    %x_30 = ttg.convert_layout %x_28 : tensor<128x1x!tt.ptr<f32>, #blocked3> -> tensor<128x1x!tt.ptr<f32>, #blocked>
    %x_31 = tt.broadcast %x_30 : tensor<128x1x!tt.ptr<f32>, #blocked> -> tensor<128x32x!tt.ptr<f32>, #blocked>
    %x_32 = tt.broadcast %x_29 : tensor<1x32xi32, #blocked> -> tensor<128x32xi32, #blocked>
    %x_33 = tt.addptr %x_31, %x_32 : tensor<128x32x!tt.ptr<f32>, #blocked>, tensor<128x32xi32, #blocked>
    %x_34 = tt.load %x_33, %m2_16, %x : tensor<128x32x!tt.ptr<f32>, #blocked>
    %0 = tt.expand_dims %rows_6 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
    %1 = tt.splat %D : i32 -> tensor<128x1xi32, #blocked>
    %2 = arith.muli %0, %1 : tensor<128x1xi32, #blocked>
    %3 = tt.splat %o_ptr : !tt.ptr<f32> -> tensor<128x1x!tt.ptr<f32>, #blocked>
    %4 = tt.addptr %3, %2 : tensor<128x1x!tt.ptr<f32>, #blocked>, tensor<128x1xi32, #blocked>
    %5 = tt.broadcast %4 : tensor<128x1x!tt.ptr<f32>, #blocked> -> tensor<128x32x!tt.ptr<f32>, #blocked>
    %6 = tt.addptr %5, %x_32 : tensor<128x32x!tt.ptr<f32>, #blocked>, tensor<128x32xi32, #blocked>
    tt.store %6, %x_34, %m2_16 : tensor<128x32x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
