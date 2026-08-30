// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | FileCheck %s
//
// Two rank-2 tiles of DIFFERENT shapes in one kernel, which is what `tl.trans`
// and a shape-changing rank-2 `tl.reshape` produce. This module is the verbatim
// TTGIR of
//
//     i = tl.arange(0, 16); j = tl.arange(0, 8)
//     v = tl.load(x + i[:, None] * 8 + j[None, :])          # a [16, 8] tile
//     tl.store(o + j[:, None] * 16 + i[None, :], tl.trans(v))  # an [8, 16] tile
//
// `MakeRangeLowering` used to decompose EVERY rank-2 index range against one
// module-wide tile (`findLargestRank2Tile`), which here is the output-reaching
// [8, 16] one. The load's row index then came out as `flat / 16` instead of
// `flat / 8` and the kernel read the wrong elements — silently. Each range now
// resolves the tile its own broadcast cone feeds, so the two divisors differ.
//
// Pinning the DIVISORS is the point: 8 is the load tile's row length, 16 the
// store tile's. Seeing only one of them back would be the bug.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @trans_two_tiles(%x: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %o: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<8> : tensor<16x1xi32, #blocked>
    %cst_0 = arith.constant dense<16> : tensor<8x1xi32, #blocked1>
    // Store cone: an [8, 16] tile, row length 16.
    %0 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<8x1xi32, #blocked1>
    %2 = arith.muli %1, %cst_0 : tensor<8x1xi32, #blocked1>
    %3 = tt.splat %o : !tt.ptr<f32> -> tensor<8x1x!tt.ptr<f32>, #blocked1>
    %4 = tt.addptr %3, %2 : tensor<8x1x!tt.ptr<f32>, #blocked1>, tensor<8x1xi32, #blocked1>
    %5 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %6 = tt.expand_dims %5 {axis = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x16xi32, #blocked1>
    %7 = tt.broadcast %4 : tensor<8x1x!tt.ptr<f32>, #blocked1> -> tensor<8x16x!tt.ptr<f32>, #blocked1>
    %8 = tt.broadcast %6 : tensor<1x16xi32, #blocked1> -> tensor<8x16xi32, #blocked1>
    %9 = tt.addptr %7, %8 : tensor<8x16x!tt.ptr<f32>, #blocked1>, tensor<8x16xi32, #blocked1>
    // Load cone: a [16, 8] tile, row length 8.
    %10 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %11 = tt.expand_dims %10 {axis = 1 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<16x1xi32, #blocked>
    %12 = arith.muli %11, %cst : tensor<16x1xi32, #blocked>
    %13 = tt.splat %x : !tt.ptr<f32> -> tensor<16x1x!tt.ptr<f32>, #blocked>
    %14 = tt.addptr %13, %12 : tensor<16x1x!tt.ptr<f32>, #blocked>, tensor<16x1xi32, #blocked>
    %15 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %16 = tt.expand_dims %15 {axis = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x8xi32, #blocked>
    %17 = tt.broadcast %14 : tensor<16x1x!tt.ptr<f32>, #blocked> -> tensor<16x8x!tt.ptr<f32>, #blocked>
    %18 = tt.broadcast %16 : tensor<1x8xi32, #blocked> -> tensor<16x8xi32, #blocked>
    %19 = tt.addptr %17, %18 : tensor<16x8x!tt.ptr<f32>, #blocked>, tensor<16x8xi32, #blocked>
    %20 = tt.load %19 : tensor<16x8x!tt.ptr<f32>, #blocked>
    %21 = tt.trans %20 {order = array<i32: 1, 0>} : tensor<16x8xf32, #blocked> -> tensor<8x16xf32, #blocked2>
    %22 = ttg.convert_layout %21 : tensor<8x16xf32, #blocked2> -> tensor<8x16xf32, #blocked1>
    tt.store %9, %22 : tensor<8x16x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}

// Both addresses decompose the flat lane index by the LOAD tile's row length,
// 8 — the load reads `(lane / 8) * 8 + lane % 8`, the store writes the
// transposed `(lane % 8) * 16 + lane / 8`, and 16 appears only as a multiplier.
//
// Dividing the lane index by 16 is precisely the defect: that was the store
// tile's row length reaching the load's index range through the module-wide
// tile, and it made the kernel read `x[(lane / 16) * 8 + lane % 16]`.
//
// CHECK-LABEL: metal.kernel trans_two_tiles
// CHECK-NOT: arith.divsi %{{.*}}, %c16
// CHECK: metal.get_element %arg0
// CHECK-NOT: arith.divsi %{{.*}}, %c16
// CHECK: metal.store %{{.*}}, %arg1
