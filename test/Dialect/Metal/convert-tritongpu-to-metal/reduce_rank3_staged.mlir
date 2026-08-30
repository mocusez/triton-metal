// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | FileCheck %s
//
// A rank-3 `tt.reduce`. Rank > 2 used to be rejected outright ("requires
// Session L3b"), which is what made `tl.sort` and `tl.flip` unreachable: their
// hypercube is reduced axis by axis.
//
// The rank-1 butterfly and the rank-2 row scanner both encode their tile's
// shape, so at rank N there is nothing to specialise on. This takes the general
// route instead — publish the whole tile into threadgroup memory, barrier, then
// let each lane walk the reduced axis itself — which is the same shape as the
// gather's staging and costs `extent` reads instead of `log2(extent)` shuffles.
// It is registered at benefit 0, so the specialised paths still win wherever
// they apply and this only picks up what they decline.
//
// The result here feeds a rank-2 store, not a keep_dims broadcast, so each lane
// must reduce the source column that ITS OWN output element projects from —
// `base` is rebuilt from the output mapping, not from the lane's rank-3
// coordinates. Reducing at the lane's own coordinates is right only for the
// keep_dims shape, and only looks right for axis 0 in the other one, because
// dropping the slowest axis leaves the flat index alone.
//
// CHECK-LABEL: metal.kernel reduce_rank3_axis1
// The tile is published once...
// CHECK: %[[BUF:.*]] = metal.threadgroup_alloca : !metal.memref<128 x f32>
// CHECK: metal.barrier
// CHECK: metal.store %{{.*}}, %[[BUF]]
// CHECK: metal.barrier
// ...then walked: M = 4 reads combined by three adds.
// CHECK: metal.get_element %[[BUF]]
// CHECK: metal.get_element %[[BUF]]
// CHECK: metal.binary_exp %{{.*}}, %{{.*}}, addOp
// CHECK: metal.get_element %[[BUF]]
// CHECK: metal.binary_exp %{{.*}}, %{{.*}}, addOp
// CHECK: metal.get_element %[[BUF]]
// CHECK: metal.binary_exp %{{.*}}, %{{.*}}, addOp

#b3 = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [1, 4, 8], warpsPerCTA = [4, 1, 1], order = [2, 1, 0]}>
#b2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#s1 = #ttg.slice<{dim = 1, parent = #b3}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reduce_rank3_axis1(%o: !tt.ptr<f32>) {
    %x = arith.constant dense<1.000000e+00> : tensor<2x4x8xf32, #b3>
    %cst = arith.constant dense<8> : tensor<2x1xi32, #b2>
    %r = "tt.reduce"(%x) <{axis = 1 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %s = arith.addf %a, %b : f32
      tt.reduce.return %s : f32
    }) : (tensor<2x4x8xf32, #b3>) -> tensor<2x8xf32, #s1>
    %v = ttg.convert_layout %r : tensor<2x8xf32, #s1> -> tensor<2x8xf32, #b2>
    %rows = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32, #ttg.slice<{dim = 1, parent = #b2}>>
    %rows2 = tt.expand_dims %rows {axis = 1 : i32} : tensor<2xi32, #ttg.slice<{dim = 1, parent = #b2}>> -> tensor<2x1xi32, #b2>
    %rowoff = arith.muli %rows2, %cst : tensor<2x1xi32, #b2>
    %base = tt.splat %o : !tt.ptr<f32> -> tensor<2x1x!tt.ptr<f32>, #b2>
    %rowptr = tt.addptr %base, %rowoff : tensor<2x1x!tt.ptr<f32>, #b2>, tensor<2x1xi32, #b2>
    %cols = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #b2}>>
    %cols2 = tt.expand_dims %cols {axis = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #b2}>> -> tensor<1x8xi32, #b2>
    %rowb = tt.broadcast %rowptr : tensor<2x1x!tt.ptr<f32>, #b2> -> tensor<2x8x!tt.ptr<f32>, #b2>
    %colb = tt.broadcast %cols2 : tensor<1x8xi32, #b2> -> tensor<2x8xi32, #b2>
    %addr = tt.addptr %rowb, %colb : tensor<2x8x!tt.ptr<f32>, #b2>, tensor<2x8xi32, #b2>
    tt.store %addr, %v : tensor<2x8x!tt.ptr<f32>, #b2>
    tt.return
  }
}
