// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | triton-metal-translate --mlir-to-msl | FileCheck %s
//
// A rank-1 `tt.make_range` that reaches its rank-2 tile through a unit-dim
// `tt.reshape` instead of a `tt.expand_dims`.
//
// `x[:, None]` and `x[None, :]` are the same operation twice over, but only the
// `tt.expand_dims` spelling gives the range a `#ttg.slice` encoding, and that
// encoding was the ONLY thing telling `MakeRangeLowering` to decompose the flat
// index by axis. Upstream #10555 ("Replace compiler internal expand dims with
// reshape") rewrote `RewriteTensorDescriptorToPointer`'s offset expansion to
// emit one `tt.reshape`, so the range fell through to the rank-1 fallback and
// handed back a bare `localTid`. The tile then indexed its column by
// `flat % BLOCK_N` while the offset addressing it counted all the way to `tpb`:
// every lane past the first tile row failed its own bounds mask and took
// `other`, so a 2-D descriptor load returned row 0 and zeros.
//
// This is the shape the descriptor rewrite produces, cut down to the two
// reshapes and the load: an 8x16 tile over 128 threads.

// CHECK-LABEL: kernel void desc_load_2d
// The column offset wraps at BLOCK_N. Before the fix it was the bare
// `id.x - tgid.x * 128`, which runs to 127 on a 16-wide tile.
// CHECK: ((uint64_t)((int32_t)(((int32_t)((id.x - (tgid.x * 128))) % (int32_t)(16)))) * v2[0])

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#linear = #ttg.linear<{register = [], lane = [[0], [0], [0], [0], [1]], warp = [[2], [4]], block = []}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @desc_load_2d(%d: !tt.ptr<f32>, %d.stride.0: i64, %d.stride.1: i64, %o_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst_12 = arith.constant dense<16> : tensor<8x1xi32, #blocked>
    %i = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #linear>
    %j = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #blocked1>
    %0 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<8x1xi32, #blocked>
    %2 = arith.muli %1, %cst_12 : tensor<8x1xi32, #blocked>
    %3 = tt.splat %o_ptr : !tt.ptr<f32> -> tensor<8x1x!tt.ptr<f32>, #blocked>
    %4 = tt.addptr %3, %2 : tensor<8x1x!tt.ptr<f32>, #blocked>, tensor<8x1xi32, #blocked>
    %5 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %6 = tt.expand_dims %5 {axis = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x16xi32, #blocked>
    %7 = tt.broadcast %4 : tensor<8x1x!tt.ptr<f32>, #blocked> -> tensor<8x16x!tt.ptr<f32>, #blocked>
    %8 = tt.broadcast %6 : tensor<1x16xi32, #blocked> -> tensor<8x16xi32, #blocked>
    %9 = tt.addptr %7, %8 : tensor<8x16x!tt.ptr<f32>, #blocked>, tensor<8x16xi32, #blocked>
    %11 = arith.extsi %i : tensor<8xi32, #linear> to tensor<8xi64, #linear>
    %12 = tt.reshape %11 : tensor<8xi64, #linear> -> tensor<8x1xi64, #blocked>
    %13 = arith.extsi %j : tensor<16xi32, #blocked1> to tensor<16xi64, #blocked1>
    %14 = tt.reshape %13 : tensor<16xi64, #blocked1> -> tensor<1x16xi64, #blocked>
    %15 = tt.splat %d : !tt.ptr<f32> -> tensor<8x16x!tt.ptr<f32>, #blocked>
    %16 = tt.splat %d.stride.0 : i64 -> tensor<8x1xi64, #blocked>
    %17 = arith.muli %12, %16 : tensor<8x1xi64, #blocked>
    %18 = tt.broadcast %17 : tensor<8x1xi64, #blocked> -> tensor<8x16xi64, #blocked>
    %19 = tt.splat %d.stride.1 : i64 -> tensor<1x16xi64, #blocked>
    %20 = arith.muli %14, %19 : tensor<1x16xi64, #blocked>
    %21 = tt.broadcast %20 : tensor<1x16xi64, #blocked> -> tensor<8x16xi64, #blocked>
    %22 = arith.addi %18, %21 : tensor<8x16xi64, #blocked>
    %23 = tt.addptr %15, %22 : tensor<8x16x!tt.ptr<f32>, #blocked>, tensor<8x16xi64, #blocked>
    %35 = tt.load %23 : tensor<8x16x!tt.ptr<f32>, #blocked>
    tt.store %9, %35 : tensor<8x16x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
