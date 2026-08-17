// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | FileCheck %s
//
// The full-tile cross-lane exchange: a relabel on a tile that holds MORE than
// one element per thread.
//
// This is the TTGIR of a rank-3 `tl.permute` at num_warps=1 — 64 elements over
// 32 threads, so two per thread:
//
//     i = tl.arange(0, 64)
//     v = tl.reshape(tl.load(x + i), (2, 4, 8))
//     tl.store(o + i, tl.reshape(tl.permute(v, (2, 0, 1)), (64,)))
//
// `ConvertLayoutLowering`'s staged transpose cannot take it. Its publish and its
// read would both sit inside the scalarized tile loop, and the slot a lane needs
// at iteration `iv` is written by another lane at some OTHER iteration — no
// barrier placed inside that loop makes that safe.
//
// So the loop is SPLIT where it is built. `planTileExchange` rewrites the cvt
// into a publish/read pair and partitions the body by dependence (everything the
// exchanged value is built from ahead of the publish, everything else after the
// read, values wanted on both sides recomputed rather than carried across), and
// `FuncOpLowering` emits two loops with a barrier between them.

#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1, 2], threadsPerWarp = [2, 4, 4], warpsPerCTA = [1, 1, 1], order = [2, 1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [2, 1, 1], threadsPerWarp = [4, 2, 4], warpsPerCTA = [1, 1, 1], order = [0, 2, 1]}>
#linear = #ttg.linear<{register = [[8]], lane = [[16], [32], [1], [2], [4]], warp = [], block = []}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @permute_exchange(%x: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %o: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %i = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked>
    %xs = tt.splat %x : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>, #blocked>
    %xa = tt.addptr %xs, %i : tensor<64x!tt.ptr<f32>, #blocked>, tensor<64xi32, #blocked>
    %v = tt.load %xa : tensor<64x!tt.ptr<f32>, #blocked>
    %v3 = tt.reshape %v : tensor<64xf32, #blocked> -> tensor<2x4x8xf32, #blocked1>
    %t = tt.trans %v3 {order = array<i32: 2, 0, 1>} : tensor<2x4x8xf32, #blocked1> -> tensor<8x2x4xf32, #blocked2>
    // Shared by the load's address and the store's: it is in the publish cone,
    // so the store's copy has to be RECOMPUTED in the second loop.
    %os = tt.splat %o : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>, #blocked>
    %oa = tt.addptr %os, %i : tensor<64x!tt.ptr<f32>, #blocked>, tensor<64xi32, #blocked>
    %r = tt.reshape %t : tensor<8x2x4xf32, #blocked2> -> tensor<64xf32, #linear>
    %c = ttg.convert_layout %r : tensor<64xf32, #linear> -> tensor<64xf32, #blocked>
    tt.store %oa, %c : tensor<64x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// One buffer for the whole tile, allocated in the PROLOGUE so it dominates both
// loops — allocating at the publish would put it inside the first one, where the
// second cannot see it. Then: loop, publish, barrier, loop, read.
//
// The barrier must sit BETWEEN the two loops. One inside the first would only
// order that iteration's writes, which is the whole reason the single-loop body
// cannot do this.
//
// CHECK-LABEL: metal.kernel permute_exchange
// CHECK: %[[BUF:.+]] = metal.threadgroup_alloca : !metal.memref<64 x f32>
// CHECK: scf.for
// CHECK: metal.tg_store_indexed %[[BUF]]
// CHECK: }
// CHECK: metal.barrier
// CHECK: scf.for
// CHECK: metal.tg_load_indexed %[[BUF]]
// CHECK: metal.store
