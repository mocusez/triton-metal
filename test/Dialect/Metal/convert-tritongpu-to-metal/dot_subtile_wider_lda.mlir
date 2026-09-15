// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | FileCheck %s
// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | triton-metal-translate --mlir-to-msl | FileCheck %s --check-prefix=MSL
//
// A standalone 16x16x8 dot whose operands are SUB-TILES of a wider buffer:
// every leading dimension is 64 while the tile is 8 wide in K and 16 wide in N.
//
// Two separate guesses used to stand in for facts this IR carries.
//   1. The reduction length defaulted to `stride_a`. That equals K only when
//      A's rows sit exactly K apart, so this shape reduced over 64 instead of
//      8 and walked off the end of BOTH operands.
//   2. The row stride was searched for as a `tt.splat` of an i32 kernel
//      argument only, so a COMPILE-TIME stride found nothing and fell back to
//      the hardcoded 8.
// Both are out-of-range READS, so the wrong answer was stable for a given
// launch order and exact in a fresh process -- it looked like a race.
//
// The reduction bound below must be the tile's own K and the A row stride must
// be the constant 64. Revert either fix and the corresponding MSL line stops
// matching.

#blocked = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [4, 8], warpsPerCTA = [2, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [4, 8], warpsPerCTA = [2, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [8, 4], warpsPerCTA = [2, 1], order = [1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [2, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @dot_subtile_wider_lda(%A: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %B: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %C: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #blocked>
    %b = arith.constant dense<64> : tensor<8x1xi32, #blocked1>
    %cst_0 = arith.constant dense<64> : tensor<16x1xi32, #blocked2>
    %cst_1 = arith.constant dense<64> : tensor<16x1xi32, #blocked3>
    %m = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %m_2 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked3}>>
    %m_3 = tt.expand_dims %m {axis = 1 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<16x1xi32, #blocked2>
    %m_4 = tt.expand_dims %m_2 {axis = 1 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked3}>> -> tensor<16x1xi32, #blocked3>
    %n = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %n_5 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked3}>>
    %n_6 = tt.expand_dims %n {axis = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x16xi32, #blocked1>
    %n_7 = tt.expand_dims %n_5 {axis = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked3}>> -> tensor<1x16xi32, #blocked3>
    %a = arith.muli %m_3, %cst_0 : tensor<16x1xi32, #blocked2>
    %a_8 = arith.muli %m_4, %cst_1 : tensor<16x1xi32, #blocked3>
    %a_9 = tt.splat %A : !tt.ptr<f32> -> tensor<16x1x!tt.ptr<f32>, #blocked2>
    %a_10 = tt.addptr %a_9, %a : tensor<16x1x!tt.ptr<f32>, #blocked2>, tensor<16x1xi32, #blocked2>
    %a_11 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
    %a_12 = tt.expand_dims %a_11 {axis = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x8xi32, #blocked2>
    %a_13 = tt.broadcast %a_10 : tensor<16x1x!tt.ptr<f32>, #blocked2> -> tensor<16x8x!tt.ptr<f32>, #blocked2>
    %a_14 = tt.broadcast %a_12 : tensor<1x8xi32, #blocked2> -> tensor<16x8xi32, #blocked2>
    %a_15 = tt.addptr %a_13, %a_14 : tensor<16x8x!tt.ptr<f32>, #blocked2>, tensor<16x8xi32, #blocked2>
    %a_16 = tt.load %a_15 : tensor<16x8x!tt.ptr<f32>, #blocked2>
    %b_17 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %b_18 = tt.expand_dims %b_17 {axis = 1 : i32} : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<8x1xi32, #blocked1>
    %b_19 = arith.muli %b_18, %b : tensor<8x1xi32, #blocked1>
    %b_20 = tt.splat %B : !tt.ptr<f32> -> tensor<8x1x!tt.ptr<f32>, #blocked1>
    %b_21 = tt.addptr %b_20, %b_19 : tensor<8x1x!tt.ptr<f32>, #blocked1>, tensor<8x1xi32, #blocked1>
    %b_22 = tt.broadcast %b_21 : tensor<8x1x!tt.ptr<f32>, #blocked1> -> tensor<8x16x!tt.ptr<f32>, #blocked1>
    %b_23 = tt.broadcast %n_6 : tensor<1x16xi32, #blocked1> -> tensor<8x16xi32, #blocked1>
    %b_24 = tt.addptr %b_22, %b_23 : tensor<8x16x!tt.ptr<f32>, #blocked1>, tensor<8x16xi32, #blocked1>
    %b_25 = tt.load %b_24 : tensor<8x16x!tt.ptr<f32>, #blocked1>
    %0 = tt.splat %C : !tt.ptr<f32> -> tensor<16x1x!tt.ptr<f32>, #blocked3>
    %1 = tt.addptr %0, %a_8 : tensor<16x1x!tt.ptr<f32>, #blocked3>, tensor<16x1xi32, #blocked3>
    %2 = tt.broadcast %1 : tensor<16x1x!tt.ptr<f32>, #blocked3> -> tensor<16x16x!tt.ptr<f32>, #blocked3>
    %3 = tt.broadcast %n_7 : tensor<1x16xi32, #blocked3> -> tensor<16x16xi32, #blocked3>
    %4 = tt.addptr %2, %3 : tensor<16x16x!tt.ptr<f32>, #blocked3>, tensor<16x16xi32, #blocked3>
    %a_26 = ttg.convert_layout %a_16 : tensor<16x8xf32, #blocked2> -> tensor<16x8xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    %b_27 = ttg.convert_layout %b_25 : tensor<8x16xf32, #blocked1> -> tensor<8x16xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
    %5 = tt.dot %a_26, %b_27, %cst : tensor<16x8xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<8x16xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<16x16xf32, #blocked>
    %6 = ttg.convert_layout %5 : tensor<16x16xf32, #blocked> -> tensor<16x16xf32, #blocked3>
    tt.store %4, %6 : tensor<16x16x!tt.ptr<f32>, #blocked3>
    tt.return
  }
}

// CHECK: metal.kernel dot_subtile_wider_lda
// CHECK: metal.simdgroup_multiply_accumulate

// The K loop runs over the TILE's K (8), not over the leading dimension.
// MSL: for (int [[K:v[0-9]+]] = 0; [[K]] < 8; [[K]] += 8)
// A is staged with the real row stride 64 and masked at the tile's K.
// MSL: gj < ((0 + 8) + 0)) ? v0[gi * 64 + gj]
