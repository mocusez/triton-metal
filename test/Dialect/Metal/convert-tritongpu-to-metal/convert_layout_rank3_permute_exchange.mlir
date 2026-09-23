// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | FileCheck %s
//
// A rank-3 `tl.permute` written the way a rank-3 kernel actually is: with a
// `[:, None, None]` / `[None, :, None]` / `[None, None, :]` address cone, not a
// rank-1 load reshaped into a cube (that spelling is
// `convert_layout_tile_exchange.mlir`, and it folds the whole permutation into
// one `#ttg.linear` load layout instead).
//
//     a = tl.arange(0, 2)[:, None, None]
//     b = tl.arange(0, 4)[None, :, None]
//     c = tl.arange(0, 8)[None, None, :]
//     v = tl.load(X + a * 32 + b * 8 + c)
//     p = tl.permute(v, (0, 2, 1))
//     i0 = tl.arange(0, 2)[:, None, None]
//     i1 = tl.arange(0, 8)[None, :, None]
//     i2 = tl.arange(0, 4)[None, None, :]
//     tl.store(O + i0 * 32 + i1 * 4 + i2, p)
//
// Here the `tt.trans` is a relabel — `transIsFlatIdentity` holds, and four
// earlier attempts each measured a guard on it catching nothing — and the op
// that moves the data is the rank-3 blocked↔blocked `ttg.convert_layout` in
// front of the store. It needs the same full-tile phase (publish, barrier, read
// in a second loop) the reshape spelling gets, and `planTileExchange` grants
// that only to a function which already HAS a phase; a permute-and-store kernel
// has none, so every one of these was refused by name.
//
// The phase is granted here because the DESTINATION layout's flat index is the
// thread-major expression `localTid * E + iv` that the consuming store's own
// address arithmetic emits: `#blocked` puts its one register bit at row-major
// offset 1 and its lane bits at 2, 4, 8, 16, 32. Section 1 of
// `convert_layout_reject_nontrivial.mlir` is the same classification with a
// destination whose register bit sits at offset 8 instead, and it keeps its
// refusal — the read half would then ask the layout while the store asks the
// tile, which is one buffer under two index mappings.
//
// Two rank-3 tile SHAPES appear here — (2,4,8) for the load and (2,8,4) for the
// store — of the same element count, so the store's index ranges must be
// decomposed against their own cone's tile (`makeRangeOwnTileOfRank`); picking
// the largest is a coin flip when the counts tie, and it addressed the permuted
// tile with the source tile's coordinates.

#blocked = #ttg.blocked<{sizePerThread = [1, 1, 2], threadsPerWarp = [2, 8, 2], warpsPerCTA = [1, 1, 1], order = [2, 1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1, 2], threadsPerWarp = [2, 4, 4], warpsPerCTA = [1, 1, 1], order = [2, 1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 2, 1], threadsPerWarp = [2, 4, 4], warpsPerCTA = [1, 1, 1], order = [1, 2, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @permute_rank3_addr_exchange(%X: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %O: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<4> : tensor<1x8x1xi32, #blocked>
    %v = arith.constant dense<8> : tensor<1x4x1xi32, #blocked1>
    %cst_0 = arith.constant dense<32> : tensor<2x1x1xi32, #blocked1>
    %cst_1 = arith.constant dense<32> : tensor<2x1x1xi32, #blocked>
    %a = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #blocked1}>}>>
    %a_2 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #blocked}>}>>
    %a_3 = tt.expand_dims %a {axis = 1 : i32} : tensor<2xi32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #blocked1}>}>> -> tensor<2x1xi32, #ttg.slice<{dim = 2, parent = #blocked1}>>
    %a_4 = tt.expand_dims %a_2 {axis = 1 : i32} : tensor<2xi32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #blocked}>}>> -> tensor<2x1xi32, #ttg.slice<{dim = 2, parent = #blocked}>>
    %a_5 = tt.expand_dims %a_3 {axis = 2 : i32} : tensor<2x1xi32, #ttg.slice<{dim = 2, parent = #blocked1}>> -> tensor<2x1x1xi32, #blocked1>
    %a_6 = tt.expand_dims %a_4 {axis = 2 : i32} : tensor<2x1xi32, #ttg.slice<{dim = 2, parent = #blocked}>> -> tensor<2x1x1xi32, #blocked>
    %b = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 2, parent = #blocked1}>}>>
    %b_7 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 1, parent = #blocked}>}>>
    %b_8 = tt.expand_dims %b {axis = 0 : i32} : tensor<4xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 2, parent = #blocked1}>}>> -> tensor<1x4xi32, #ttg.slice<{dim = 2, parent = #blocked1}>>
    %b_9 = tt.expand_dims %b_7 {axis = 0 : i32} : tensor<4xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 1, parent = #blocked}>}>> -> tensor<1x4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %b_10 = tt.expand_dims %b_8 {axis = 2 : i32} : tensor<1x4xi32, #ttg.slice<{dim = 2, parent = #blocked1}>> -> tensor<1x4x1xi32, #blocked1>
    %c = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 1, parent = #blocked1}>}>>
    %c_11 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 2, parent = #blocked}>}>>
    %c_12 = tt.expand_dims %c {axis = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 1, parent = #blocked1}>}>> -> tensor<1x8xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %c_13 = tt.expand_dims %c_11 {axis = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 2, parent = #blocked}>}>> -> tensor<1x8xi32, #ttg.slice<{dim = 2, parent = #blocked}>>
    %c_14 = tt.expand_dims %c_12 {axis = 1 : i32} : tensor<1x8xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<1x1x8xi32, #blocked1>
    %v_15 = arith.muli %a_5, %cst_0 : tensor<2x1x1xi32, #blocked1>
    %v_16 = arith.muli %a_6, %cst_1 : tensor<2x1x1xi32, #blocked>
    %v_17 = tt.splat %X : !tt.ptr<f32> -> tensor<2x1x1x!tt.ptr<f32>, #blocked1>
    %v_18 = tt.addptr %v_17, %v_15 : tensor<2x1x1x!tt.ptr<f32>, #blocked1>, tensor<2x1x1xi32, #blocked1>
    %v_19 = arith.muli %b_10, %v : tensor<1x4x1xi32, #blocked1>
    %v_20 = tt.broadcast %v_18 : tensor<2x1x1x!tt.ptr<f32>, #blocked1> -> tensor<2x4x1x!tt.ptr<f32>, #blocked1>
    %v_21 = tt.broadcast %v_19 : tensor<1x4x1xi32, #blocked1> -> tensor<2x4x1xi32, #blocked1>
    %v_22 = tt.addptr %v_20, %v_21 : tensor<2x4x1x!tt.ptr<f32>, #blocked1>, tensor<2x4x1xi32, #blocked1>
    %v_23 = tt.broadcast %v_22 : tensor<2x4x1x!tt.ptr<f32>, #blocked1> -> tensor<2x4x8x!tt.ptr<f32>, #blocked1>
    %v_24 = tt.broadcast %c_14 : tensor<1x1x8xi32, #blocked1> -> tensor<2x4x8xi32, #blocked1>
    %v_25 = tt.addptr %v_23, %v_24 : tensor<2x4x8x!tt.ptr<f32>, #blocked1>, tensor<2x4x8xi32, #blocked1>
    %v_26 = tt.load %v_25 : tensor<2x4x8x!tt.ptr<f32>, #blocked1>
    %p = tt.trans %v_26 {order = array<i32: 0, 2, 1>} : tensor<2x4x8xf32, #blocked1> -> tensor<2x8x4xf32, #blocked2>
    %i1 = tt.expand_dims %c_13 {axis = 2 : i32} : tensor<1x8xi32, #ttg.slice<{dim = 2, parent = #blocked}>> -> tensor<1x8x1xi32, #blocked>
    %i2 = tt.expand_dims %b_9 {axis = 1 : i32} : tensor<1x4xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<1x1x4xi32, #blocked>
    %0 = tt.splat %O : !tt.ptr<f32> -> tensor<2x1x1x!tt.ptr<f32>, #blocked>
    %1 = tt.addptr %0, %v_16 : tensor<2x1x1x!tt.ptr<f32>, #blocked>, tensor<2x1x1xi32, #blocked>
    %2 = arith.muli %i1, %cst : tensor<1x8x1xi32, #blocked>
    %3 = tt.broadcast %1 : tensor<2x1x1x!tt.ptr<f32>, #blocked> -> tensor<2x8x1x!tt.ptr<f32>, #blocked>
    %4 = tt.broadcast %2 : tensor<1x8x1xi32, #blocked> -> tensor<2x8x1xi32, #blocked>
    %5 = tt.addptr %3, %4 : tensor<2x8x1x!tt.ptr<f32>, #blocked>, tensor<2x8x1xi32, #blocked>
    %6 = tt.broadcast %5 : tensor<2x8x1x!tt.ptr<f32>, #blocked> -> tensor<2x8x4x!tt.ptr<f32>, #blocked>
    %7 = tt.broadcast %i2 : tensor<1x1x4xi32, #blocked> -> tensor<2x8x4xi32, #blocked>
    %8 = tt.addptr %6, %7 : tensor<2x8x4x!tt.ptr<f32>, #blocked>, tensor<2x8x4xi32, #blocked>
    %9 = ttg.convert_layout %p : tensor<2x8x4xf32, #blocked2> -> tensor<2x8x4xf32, #blocked>
    tt.store %8, %9 : tensor<2x8x4x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// One buffer for the whole 64-element tile, allocated in the prologue so it
// dominates both loops; then publish loop, barrier, read loop, store.
//
// CHECK-LABEL: metal.kernel permute_rank3_addr_exchange
// CHECK: %[[BUF:.+]] = metal.threadgroup_alloca : !metal.memref<64 x f32>
// CHECK: scf.for
// CHECK: metal.tg_store_indexed %[[BUF]]
// CHECK: }
// CHECK: metal.barrier
// CHECK: scf.for
// CHECK: metal.tg_load_indexed %[[BUF]]
// CHECK: metal.store
