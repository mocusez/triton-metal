// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | FileCheck %s
//
// A rank-2 axis=0 reduce whose result is pushed BACK into a 2-D tile
// (`x - mean[None, :]`), with more columns than the threadgroup has threads.
//
// The re-expansion asks each lane for the reduce's value at the column its
// RANK-2 tile position names, and no lane holds it: at one column per thread
// the republish could just publish the lane's scalar and read slot `col`, but
// above that a lane owns several columns and only ever computed one of them per
// tile-loop iteration.
//
// So the reduce folds every column into a buffer indexed BY COLUMN, and both
// readers index that one buffer: the reduce's own result at the column this
// iteration owns, the re-expansion at the column its tile position names. One
// buffer, one index space, two questions.
//
// ⚠️ Which lane fills which column is a free choice -- the fold re-derives the
// row addresses from the load's own cone -- so the fill uses `localTid + j*tpb`
// purely because it covers the columns once.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
#slice0 = #ttg.slice<{dim = 0, parent = #blocked}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @colsum_reexpanded(%in: !tt.ptr<f32>, %out: !tt.ptr<f32>) {
    %rows = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %rows2 = tt.expand_dims %rows {axis = 1 : i32} : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<8x1xi32, #blocked>
    %cN = arith.constant dense<64> : tensor<8x1xi32, #blocked>
    %rowsN = arith.muli %rows2, %cN : tensor<8x1xi32, #blocked>
    %cols = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #slice0>
    %cols2 = tt.expand_dims %cols {axis = 0 : i32} : tensor<64xi32, #slice0> -> tensor<1x64xi32, #blocked>
    %sp = tt.splat %in : !tt.ptr<f32> -> tensor<8x1x!tt.ptr<f32>, #blocked>
    %ap0 = tt.addptr %sp, %rowsN : tensor<8x1x!tt.ptr<f32>, #blocked>, tensor<8x1xi32, #blocked>
    %apb = tt.broadcast %ap0 : tensor<8x1x!tt.ptr<f32>, #blocked> -> tensor<8x64x!tt.ptr<f32>, #blocked>
    %colsb = tt.broadcast %cols2 : tensor<1x64xi32, #blocked> -> tensor<8x64xi32, #blocked>
    %ap = tt.addptr %apb, %colsb : tensor<8x64x!tt.ptr<f32>, #blocked>, tensor<8x64xi32, #blocked>
    %tile = tt.load %ap : tensor<8x64x!tt.ptr<f32>, #blocked>
    %s = "tt.reduce"(%tile) ({
    ^bb0(%a: f32, %b: f32):
      %add = arith.addf %a, %b : f32
      tt.reduce.return %add : f32
    }) {axis = 0 : i32} : (tensor<8x64xf32, #blocked>) -> tensor<64xf32, #slice0>
    %ocols = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked1>
    %osp = tt.splat %out : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>, #blocked1>
    %oap = tt.addptr %osp, %ocols : tensor<64x!tt.ptr<f32>, #blocked1>, tensor<64xi32, #blocked1>
    %sc = ttg.convert_layout %s : tensor<64xf32, #slice0> -> tensor<64xf32, #blocked1>
    tt.store %oap, %sc : tensor<64x!tt.ptr<f32>, #blocked1>
    %s2 = tt.expand_dims %s {axis = 0 : i32} : tensor<64xf32, #slice0> -> tensor<1x64xf32, #blocked>
    %sb = tt.broadcast %s2 : tensor<1x64xf32, #blocked> -> tensor<8x64xf32, #blocked>
    %dev = arith.subf %tile, %sb : tensor<8x64xf32, #blocked>
    tt.store %ap, %dev : tensor<8x64x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// CHECK-LABEL: metal.kernel colsum_reexpanded
// One slot per COLUMN, not per thread.
// CHECK: %[[BUF:.*]] = metal.threadgroup_alloca : !metal.memref<64 x f32>
// The leading barrier keeps this trip's fill off the previous trip's reads:
// under the tile loop the allocation is one static buffer reused every trip.
// CHECK: metal.barrier
// CHECK: scf.for %[[J:.*]] = %{{.*}} to %{{.*}} step %{{.*}}
// CHECK:   %[[COL:.*]] = arith.addi %{{.*}}, %{{.*}} : i32
// CHECK:   %[[SUM:.*]] = scf.for %{{.*}} iter_args(%{{.*}} = %{{.*}}) -> (f32)
// CHECK:     metal.get_element
// CHECK:     arith.addf
// CHECK:   metal.store %[[SUM]], %[[BUF]]
// CHECK: metal.barrier
// The reduce's own result: the column THIS iteration owns.
// CHECK: metal.get_element %[[BUF]][%{{.*}}] {metal.materialize}
// The re-expansion: the column this lane's rank-2 tile position names, which is
// a different index into the same buffer.
// CHECK: %[[TILECOL:.*]] = arith.remsi %{{.*}}, %{{.*}} : i32
// CHECK: %[[TILECOLU:.*]] = builtin.unrealized_conversion_cast %[[TILECOL]] : i32 to ui32
// CHECK: metal.get_element %[[BUF]][%[[TILECOLU]]]
