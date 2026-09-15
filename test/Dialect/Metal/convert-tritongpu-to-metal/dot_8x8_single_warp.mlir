// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | FileCheck %s
// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | triton-metal-translate --mlir-to-msl | FileCheck %s --check-prefix=MSL
//
// A direct 8x8x8 dot at ONE warp. 32 threads cannot cover a 64-element tile
// one element each, so Triton gives the dot a `#blocked` with
// `sizePerThread = [1, 1]` on a 4x8 thread grid, the store a `#blocked1` with
// `[1, 2]` on an 8x4 one, and a `ttg.convert_layout` between them.
//
// `preprocessDotCvtChains` peels the C-side conversion only for the LOOP chain
// (`dot -> yield -> for -> cvt -> store`), so the standalone one reached
// `rewriteSingleDot` with a `ttg.convert_layout` where it expected a
// `tt.store`, declined, and the whole shape fell out of every tier's envelope:
// refused BY NAME at one warp while working at two and four.
//
// The conversion has no consequence here — the tile is written by
// `simdgroup_store` with explicit (row, column, stride) origins from the
// STORE's own pointer, never through a per-thread element index — so peeling
// it is sound. What this fixture pins is that the shape is CLAIMED.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [8, 4], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @dot_8x8_single_warp(%A: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %B: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %C: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<8x8xf32, #blocked>
    %cst_0 = arith.constant dense<8> : tensor<8x1xi32, #blocked1>
    %m = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %m_1 = tt.expand_dims %m {axis = 1 : i32} : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<8x1xi32, #blocked1>
    %n = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %n_2 = tt.expand_dims %n {axis = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x8xi32, #blocked1>
    %a = arith.muli %m_1, %cst_0 : tensor<8x1xi32, #blocked1>
    %a_3 = tt.splat %A : !tt.ptr<f32> -> tensor<8x1x!tt.ptr<f32>, #blocked1>
    %a_4 = tt.addptr %a_3, %a : tensor<8x1x!tt.ptr<f32>, #blocked1>, tensor<8x1xi32, #blocked1>
    %a_5 = tt.broadcast %a_4 : tensor<8x1x!tt.ptr<f32>, #blocked1> -> tensor<8x8x!tt.ptr<f32>, #blocked1>
    %a_6 = tt.broadcast %n_2 : tensor<1x8xi32, #blocked1> -> tensor<8x8xi32, #blocked1>
    %a_7 = tt.addptr %a_5, %a_6 : tensor<8x8x!tt.ptr<f32>, #blocked1>, tensor<8x8xi32, #blocked1>
    %a_8 = tt.load %a_7 : tensor<8x8x!tt.ptr<f32>, #blocked1>
    %b = tt.splat %B : !tt.ptr<f32> -> tensor<8x1x!tt.ptr<f32>, #blocked1>
    %b_9 = tt.addptr %b, %a : tensor<8x1x!tt.ptr<f32>, #blocked1>, tensor<8x1xi32, #blocked1>
    %b_10 = tt.broadcast %b_9 : tensor<8x1x!tt.ptr<f32>, #blocked1> -> tensor<8x8x!tt.ptr<f32>, #blocked1>
    %b_11 = tt.addptr %b_10, %a_6 : tensor<8x8x!tt.ptr<f32>, #blocked1>, tensor<8x8xi32, #blocked1>
    %b_12 = tt.load %b_11 : tensor<8x8x!tt.ptr<f32>, #blocked1>
    %0 = tt.splat %C : !tt.ptr<f32> -> tensor<8x1x!tt.ptr<f32>, #blocked1>
    %1 = tt.addptr %0, %a : tensor<8x1x!tt.ptr<f32>, #blocked1>, tensor<8x1xi32, #blocked1>
    %2 = tt.broadcast %1 : tensor<8x1x!tt.ptr<f32>, #blocked1> -> tensor<8x8x!tt.ptr<f32>, #blocked1>
    %3 = tt.addptr %2, %a_6 : tensor<8x8x!tt.ptr<f32>, #blocked1>, tensor<8x8xi32, #blocked1>
    %a_13 = ttg.convert_layout %a_8 : tensor<8x8xf32, #blocked1> -> tensor<8x8xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    %b_14 = ttg.convert_layout %b_12 : tensor<8x8xf32, #blocked1> -> tensor<8x8xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
    %4 = tt.dot %a_13, %b_14, %cst : tensor<8x8xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<8x8xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<8x8xf32, #blocked>
    %5 = ttg.convert_layout %4 : tensor<8x8xf32, #blocked> -> tensor<8x8xf32, #blocked1>
    tt.store %3, %5 : tensor<8x8x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}

// No `tt.dot` and no `ttg.convert_layout` survive: the shape is claimed, not
// refused.
// CHECK: metal.kernel dot_8x8_single_warp
// CHECK-NOT: tt.dot
// CHECK-NOT: ttg.convert_layout
// CHECK: metal.simdgroup_multiply_accumulate
// CHECK: metal.simdgroup_store

// One 8x8 accumulate, no K loop, and the result written straight to C.
// MSL: simdgroup_float8x8 [[ACC:v[0-9]+]](0.0f);
// MSL-NEXT: simdgroup_multiply_accumulate([[ACC]], {{v[0-9]+}}, {{v[0-9]+}}, [[ACC]]);
// MSL-NEXT: simdgroup_store([[ACC]], v2, 8);
