// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | FileCheck %s
// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | triton-metal-translate --mlir-to-msl | FileCheck %s --check-prefix=MSL
//
// A direct, loop-free 16x16x32 dot whose OUTPUT tile needs a second register
// band (E == 2 elements per thread). The SIMD-group path publishes the whole
// result tile into threadgroup scratch and every thread reads its own element
// back by ABSOLUTE element number, so the scratch read and the store address
// must be the same function of (thread, iteration).
//
// Triton emits the dot into #blocked (sizePerThread = [1, 1]) and converts to
// #blocked1 (sizePerThread = [1, 2]) for the store. Both are 16x16, so
// `findLargestRank2Tile` -- which answers by element COUNT -- used to hand the
// range that builds the store address the WRONG one of the two: the scratch was
// read at `tid * 2 + iv` while the address was decomposed from `tid + iv * 128`.
// Two bijections over one tile; elementwise code cancels the difference and only
// this scratch exposes it. Every E == 2 cell was wrong, every E == 1 cell exact.
//
// Delete `retileFromRangeParent`'s use in MakeRangeLowering and the MSL below
// stops matching: the store index reverts to the `+ (v3 * 128)` form while the
// scratch read keeps `* 2) + v3`.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @dot_multi_register_band(%A: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %B: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %C: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #blocked>
    %cst_0 = arith.constant dense<16> : tensor<16x1xi32, #blocked1>
    %b = arith.constant dense<16> : tensor<32x1xi32, #blocked2>
    %a = arith.constant dense<32> : tensor<16x1xi32, #blocked3>
    %m = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked3}>>
    %m_1 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %m_2 = tt.expand_dims %m {axis = 1 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked3}>> -> tensor<16x1xi32, #blocked3>
    %m_3 = tt.expand_dims %m_1 {axis = 1 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<16x1xi32, #blocked1>
    %n = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
    %n_4 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %n_5 = tt.expand_dims %n {axis = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x16xi32, #blocked2>
    %n_6 = tt.expand_dims %n_4 {axis = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x16xi32, #blocked1>
    %ka = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked3}>>
    %ka_7 = tt.expand_dims %ka {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked3}>> -> tensor<1x32xi32, #blocked3>
    %kb = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %kb_8 = tt.expand_dims %kb {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<32x1xi32, #blocked2>
    %a_9 = arith.muli %m_2, %a : tensor<16x1xi32, #blocked3>
    %a_10 = tt.splat %A : !tt.ptr<f32> -> tensor<16x1x!tt.ptr<f32>, #blocked3>
    %a_11 = tt.addptr %a_10, %a_9 : tensor<16x1x!tt.ptr<f32>, #blocked3>, tensor<16x1xi32, #blocked3>
    %a_12 = tt.broadcast %a_11 : tensor<16x1x!tt.ptr<f32>, #blocked3> -> tensor<16x32x!tt.ptr<f32>, #blocked3>
    %a_13 = tt.broadcast %ka_7 : tensor<1x32xi32, #blocked3> -> tensor<16x32xi32, #blocked3>
    %a_14 = tt.addptr %a_12, %a_13 : tensor<16x32x!tt.ptr<f32>, #blocked3>, tensor<16x32xi32, #blocked3>
    %a_15 = tt.load %a_14 : tensor<16x32x!tt.ptr<f32>, #blocked3>
    %b_16 = arith.muli %kb_8, %b : tensor<32x1xi32, #blocked2>
    %b_17 = tt.splat %B : !tt.ptr<f32> -> tensor<32x1x!tt.ptr<f32>, #blocked2>
    %b_18 = tt.addptr %b_17, %b_16 : tensor<32x1x!tt.ptr<f32>, #blocked2>, tensor<32x1xi32, #blocked2>
    %b_19 = tt.broadcast %b_18 : tensor<32x1x!tt.ptr<f32>, #blocked2> -> tensor<32x16x!tt.ptr<f32>, #blocked2>
    %b_20 = tt.broadcast %n_5 : tensor<1x16xi32, #blocked2> -> tensor<32x16xi32, #blocked2>
    %b_21 = tt.addptr %b_19, %b_20 : tensor<32x16x!tt.ptr<f32>, #blocked2>, tensor<32x16xi32, #blocked2>
    %b_22 = tt.load %b_21 : tensor<32x16x!tt.ptr<f32>, #blocked2>
    %0 = arith.muli %m_3, %cst_0 : tensor<16x1xi32, #blocked1>
    %1 = tt.splat %C : !tt.ptr<f32> -> tensor<16x1x!tt.ptr<f32>, #blocked1>
    %2 = tt.addptr %1, %0 : tensor<16x1x!tt.ptr<f32>, #blocked1>, tensor<16x1xi32, #blocked1>
    %3 = tt.broadcast %2 : tensor<16x1x!tt.ptr<f32>, #blocked1> -> tensor<16x16x!tt.ptr<f32>, #blocked1>
    %4 = tt.broadcast %n_6 : tensor<1x16xi32, #blocked1> -> tensor<16x16xi32, #blocked1>
    %5 = tt.addptr %3, %4 : tensor<16x16x!tt.ptr<f32>, #blocked1>, tensor<16x16xi32, #blocked1>
    %a_23 = ttg.convert_layout %a_15 : tensor<16x32xf32, #blocked3> -> tensor<16x32xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    %b_24 = ttg.convert_layout %b_22 : tensor<32x16xf32, #blocked2> -> tensor<32x16xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
    %6 = tt.dot %a_23, %b_24, %cst : tensor<16x32xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<32x16xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<16x16xf32, #blocked>
    %7 = ttg.convert_layout %6 : tensor<16x16xf32, #blocked> -> tensor<16x16xf32, #blocked1>
    tt.store %5, %7 : tensor<16x16x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}

// The tile loop runs twice: two register bands over one 16x16 tile.
// CHECK: metal.kernel dot_multi_register_band
// CHECK: scf.for
// The scratch read's index is the thread-major form for this layout.
// CHECK: metal.tg_load_indexed

// Both the scratch read and the store address are the SAME expression.
// MSL: threadgroup float [[SCRATCH:v[0-9]+]][256];
// MSL: [[VAL:v[0-9]+]] = [[SCRATCH]][(((id.x - (tgid.x * 128)) * 2) + [[IV:v[0-9]+]])];
// MSL-NEXT: v2[((((int32_t)((((id.x - (tgid.x * 128)) * 2) + [[IV]])) / (int32_t)(16)) * 16) + ((int32_t)((((id.x - (tgid.x * 128)) * 2) + [[IV]])) % (int32_t)(16)))] = [[VAL]];
