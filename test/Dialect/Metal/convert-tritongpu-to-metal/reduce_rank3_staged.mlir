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
// Phased rank-N reductions are committed before full conversion reaches their
// shape-changing users.
//
// The result here feeds a rank-2 store, not a keep_dims broadcast, so each lane
// must reduce the source column that ITS OWN output element projects from —
// `base` is rebuilt from the output mapping, not from the lane's rank-3
// coordinates. Reducing at the lane's own coordinates is right only for the
// keep_dims shape, and only looks right for axis 0 in the other one, because
// dropping the slowest axis leaves the flat index alone.
//
// CHECK-LABEL: metal.kernel reduce_rank3_axis1
// The result slots and source tile are both staged.
// CHECK: %[[OUT:.*]] = metal.threadgroup_alloca : !metal.memref<16 x f32>
// CHECK: %[[BUF:.*]] = metal.threadgroup_alloca : !metal.memref<64 x f32>
// CHECK: metal.barrier
// CHECK: metal.tg_store_indexed %[[BUF]]
// CHECK: metal.barrier
// ...then walked: M = 4 reads combined by three adds.
// CHECK: metal.tg_load_indexed %[[BUF]]
// CHECK: metal.tg_load_indexed %[[BUF]]
// CHECK: metal.binary_exp %{{.*}}, %{{.*}}, addOp
// CHECK: metal.tg_load_indexed %[[BUF]]
// CHECK: metal.binary_exp %{{.*}}, %{{.*}}, addOp
// CHECK: metal.tg_load_indexed %[[BUF]]
// CHECK: metal.binary_exp %{{.*}}, %{{.*}}, addOp
// CHECK: metal.tg_store_indexed %[[OUT]]

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

// Without a rank-N exact-layout phase in the function, the same slice-shaped
// arithmetic remains on the established local scalar path.  This guards the
// planner boundary: the generic phase scheduler is selected by a planned
// reduce/layout contract, not merely by recognizing this hypercube expression.
// CHECK-LABEL: metal.kernel slice_hypercube_parity
// CHECK-NOT: !metal.memref<4 x i8>
// CHECK: arith.xori
// CHECK: arith.cmpi ne
// CHECK: arith.extui
// CHECK: metal.store
#topk_parent = #ttg.blocked<{sizePerThread = [1, 1, 1, 1, 1, 1, 1, 1, 1], threadsPerWarp = [1, 1, 1, 1, 2, 2, 2, 2, 2], warpsPerCTA = [1, 2, 2, 2, 1, 1, 1, 1, 1], order = [8, 7, 6, 5, 4, 3, 2, 1, 0]}>
#topk_slice = #ttg.slice<{dim = 5, parent = #topk_parent}>
#topk_range = #ttg.linear<{register = [], lane = [[0], [0], [0], [0], [1]], warp = [[0], [0], [0]], block = []}>
#topk_range_b = #ttg.linear<{register = [], lane = [[0], [0], [1], [0], [0]], warp = [[0], [0], [0]], block = []}>
#topk_flat = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
#topk_flat8 = #ttg.blocked<{sizePerThread = [1, 1, 1, 1, 1, 1, 1, 1], threadsPerWarp = [1, 1, 1, 2, 2, 2, 2, 2], warpsPerCTA = [2, 2, 2, 1, 1, 1, 1, 1], order = [7, 6, 5, 4, 3, 2, 1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @slice_hypercube_parity(%out: !tt.ptr<i32>) {
    %a = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32, #topk_range>
    %b = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32, #topk_range_b>
    %ra = tt.reshape %a : tensor<2xi32, #topk_range> -> tensor<1x1x1x1x2x1x1x1xi32, #topk_slice>
    %rb = tt.reshape %b : tensor<2xi32, #topk_range_b> -> tensor<1x1x1x1x1x2x1x1xi32, #topk_slice>
    %ba = tt.broadcast %ra : tensor<1x1x1x1x2x1x1x1xi32, #topk_slice> -> tensor<1x1x1x1x2x2x1x1xi32, #topk_slice>
    %bb = tt.broadcast %rb : tensor<1x1x1x1x1x2x1x1xi32, #topk_slice> -> tensor<1x1x1x1x2x2x1x1xi32, #topk_slice>
    %xor = arith.xori %ba, %bb : tensor<1x1x1x1x2x2x1x1xi32, #topk_slice>
    %one = arith.constant dense<1> : tensor<1x1x1x1x2x2x1x1xi32, #topk_slice>
    %sparse = arith.cmpi ne, %xor, %one : tensor<1x1x1x1x2x2x1x1xi32, #topk_slice>
    %direction = tt.broadcast %sparse : tensor<1x1x1x1x2x2x1x1xi1, #topk_slice> -> tensor<2x2x2x2x2x2x2x2xi1, #topk_slice>
    %value = arith.extui %direction : tensor<2x2x2x2x2x2x2x2xi1, #topk_slice> to tensor<2x2x2x2x2x2x2x2xi32, #topk_slice>
    %offset = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #topk_flat>
    %offsets0 = tt.reshape %offset : tensor<256xi32, #topk_flat> -> tensor<2x2x2x2x2x2x2x2xi32, #topk_flat8>
    %offsets = ttg.convert_layout %offsets0 : tensor<2x2x2x2x2x2x2x2xi32, #topk_flat8> -> tensor<2x2x2x2x2x2x2x2xi32, #topk_slice>
    %ptrs = tt.splat %out : !tt.ptr<i32> -> tensor<2x2x2x2x2x2x2x2x!tt.ptr<i32>, #topk_slice>
    %addrs = tt.addptr %ptrs, %offsets : tensor<2x2x2x2x2x2x2x2x!tt.ptr<i32>, #topk_slice>, tensor<2x2x2x2x2x2x2x2xi32, #topk_slice>
    tt.store %addrs, %value : tensor<2x2x2x2x2x2x2x2x!tt.ptr<i32>, #topk_slice>
    tt.return
  }
}

// A four-band rank-4 tile followed by a slice-encoded rank-3 tile requires two
// complete materializations.  Only two f32 buffers are live: the scheduler
// alternates them instead of allocating one full tile per reduction stage.
// CHECK-LABEL: metal.kernel reduce_two_multiband_stages
// CHECK: %[[PHASE0:.*]] = metal.threadgroup_alloca : !metal.memref<512 x f32>
// CHECK: %[[PHASE1:.*]] = metal.threadgroup_alloca : !metal.memref<128 x f32>
// CHECK: scf.for
// CHECK: metal.tg_store_indexed %[[PHASE0]]
// CHECK: }
// CHECK: metal.barrier
// CHECK: scf.for
// CHECK: metal.tg_load_indexed %[[PHASE0]]
// CHECK: metal.binary_exp
// CHECK: metal.tg_store_indexed %[[PHASE1]]
// CHECK: }
// CHECK: metal.barrier
// CHECK: scf.for
// CHECK: metal.tg_load_indexed %[[PHASE1]]
// CHECK: metal.binary_exp

#b4 = #ttg.blocked<{sizePerThread = [1, 1, 1, 1], threadsPerWarp = [1, 1, 4, 8], warpsPerCTA = [1, 4, 1, 1], order = [3, 2, 1, 0]}>
#s3 = #ttg.slice<{dim = 3, parent = #b4}>
#s2 = #ttg.slice<{dim = 2, parent = #s3}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reduce_two_multiband_stages() {
    %x = arith.constant dense<1.000000e+00> : tensor<4x4x8x4xf32, #b4>
    %r0 = "tt.reduce"(%x) <{axis = 3 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %sum = arith.addf %a, %b : f32
      tt.reduce.return %sum : f32
    }) : (tensor<4x4x8x4xf32, #b4>) -> tensor<4x4x8xf32, #s3>
    %r1 = "tt.reduce"(%r0) <{axis = 2 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %sum = arith.addf %a, %b : f32
      tt.reduce.return %sum : f32
    }) : (tensor<4x4x8xf32, #s3>) -> tensor<4x4xf32, #s2>
    tt.return
  }
}
