// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | triton-metal-translate --mlir-to-msl | FileCheck %s
//
// Two rank-1 column ranges over ONE rank-2 tile, slice-encoded over blocked
// layouts whose `order` disagrees.
//
// Triton emits this whenever a kernel's inner dim has no `tt.divisibility = 16`:
// the tile does not vectorize, so the `gamma[None, :]` cone gets a `#blocked1`
// of its own with order [0, 1] and is bridged to the [1, 0] tile by a
// `ttg.convert_layout` over a degenerate `1xN`. Both make_ranges are the COLUMN
// of the same 64x64 tile and must decompose the same way.
//
// `emitTileAxisCoord` took its divisor from the tile it found and its `order`
// from the caller's encoding. Those are the same attribute in every kernel with
// one blocked layout, and here they are not: the tile is 64 wide under [1, 0],
// so the column is `flat % 64`, while `#blocked1` under [0, 1] made it
// `flat / 64`. One tile, two column mappings — the token-embedding layernorm
// was wrong in every element but column 0 at D = 33, exact at D = 32 and D = 48.
//
// A `#blocked1` that no non-degenerate rank-2 tensor carries is scaffolding, so
// the order now comes from the real tile. Note that
// `normalizeBlockedDivergentCvt` would otherwise re-encode the whole cone and
// hide this: the column mask below is what stops it, because `%col_mask` is
// shared between the mask cone and the two loads, so neither cvt owns a
// self-contained cone. That sharing is not decoration — without it the bug does
// not reproduce.

// CHECK-LABEL: kernel void two_order_columns
// The tile's own column, from the [1, 0] range ...
// CHECK: % (int32_t)(64)
// ... and gamma's, from the [0, 1] one. Before the fix this read `/ 64`.
// CHECK: v1[{{.*}}% (int32_t)(64)
// CHECK-NOT: / (int32_t)(64)

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @two_order_columns(%x_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %g_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %o_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %D: i32) attributes {noinline = false} {
    %zero1 = arith.constant dense<0.000000e+00> : tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %c64 = arith.constant dense<64> : tensor<64x1xi32, #blocked>
    // Row and column of the 64x64 tile, both slice-encoded over #blocked.
    %rows = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %rows2d = tt.expand_dims %rows {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %rowoff = arith.muli %rows2d, %c64 : tensor<64x1xi32, #blocked>
    %cols = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %cols2d = tt.expand_dims %cols {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
    %xb = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<64x1x!tt.ptr<f32>, #blocked>
    %xr = tt.addptr %xb, %rowoff : tensor<64x1x!tt.ptr<f32>, #blocked>, tensor<64x1xi32, #blocked>
    %xrb = tt.broadcast %xr : tensor<64x1x!tt.ptr<f32>, #blocked> -> tensor<64x64x!tt.ptr<f32>, #blocked>
    %colb = tt.broadcast %cols2d : tensor<1x64xi32, #blocked> -> tensor<64x64xi32, #blocked>
    %xp = tt.addptr %xrb, %colb : tensor<64x64x!tt.ptr<f32>, #blocked>, tensor<64x64xi32, #blocked>
    %x = tt.load %xp : tensor<64x64x!tt.ptr<f32>, #blocked>
    // The SAME column, slice-encoded over #blocked1 (order [0, 1]). Its mask is
    // shared between the two loads and the 1x64 mask cone, so no single cvt has
    // a self-contained cone to re-encode.
    %gcols = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %dsplat = tt.splat %D : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %col_mask = arith.cmpi slt, %gcols, %dsplat : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %m2d = tt.expand_dims %col_mask {axis = 0 : i32} : tensor<64xi1, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi1, #blocked1>
    %mcvt = ttg.convert_layout %m2d : tensor<1x64xi1, #blocked1> -> tensor<1x64xi1, #blocked>
    %mbc = tt.broadcast %mcvt : tensor<1x64xi1, #blocked> -> tensor<64x64xi1, #blocked>
    %gb = tt.splat %g_ptr : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %gp = tt.addptr %gb, %gcols : tensor<64x!tt.ptr<f32>, #ttg.slice<{dim = 0, parent = #blocked1}>>, tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %g = tt.load %gp, %col_mask, %zero1 : tensor<64x!tt.ptr<f32>, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %g2d = tt.expand_dims %g {axis = 0 : i32} : tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xf32, #blocked1>
    %gcvt = ttg.convert_layout %g2d : tensor<1x64xf32, #blocked1> -> tensor<1x64xf32, #blocked>
    %gbc = tt.broadcast %gcvt : tensor<1x64xf32, #blocked> -> tensor<64x64xf32, #blocked>
    %b = tt.load %gp, %col_mask, %zero1 : tensor<64x!tt.ptr<f32>, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %b2d = tt.expand_dims %b {axis = 0 : i32} : tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xf32, #blocked1>
    %bcvt = ttg.convert_layout %b2d : tensor<1x64xf32, #blocked1> -> tensor<1x64xf32, #blocked>
    %bbc = tt.broadcast %bcvt : tensor<1x64xf32, #blocked> -> tensor<64x64xf32, #blocked>
    %mul = arith.mulf %x, %gbc : tensor<64x64xf32, #blocked>
    %out = arith.addf %mul, %bbc : tensor<64x64xf32, #blocked>
    %ob = tt.splat %o_ptr : !tt.ptr<f32> -> tensor<64x1x!tt.ptr<f32>, #blocked>
    %or = tt.addptr %ob, %rowoff : tensor<64x1x!tt.ptr<f32>, #blocked>, tensor<64x1xi32, #blocked>
    %orb = tt.broadcast %or : tensor<64x1x!tt.ptr<f32>, #blocked> -> tensor<64x64x!tt.ptr<f32>, #blocked>
    %op = tt.addptr %orb, %colb : tensor<64x64x!tt.ptr<f32>, #blocked>, tensor<64x64xi32, #blocked>
    tt.store %op, %out, %mbc : tensor<64x64x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
