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

// -----
// A rank-3 tile whose blocked layout WRAPS into a second register band: 512
// elements over 32 threads at sizePerThread 4, so the layout's own flat index
// and the `localTid * elemPerThread + iv` order the tile's ADDRESSES use are
// different mappings. The staged scratch is relabelled into the address order,
// so one label serves the load, the publish and the fold. Indexing the scratch
// by the layout instead made the fold reduce a group the kernel never
// addressed, silently, on every axis.
#blocked = #ttg.blocked<{sizePerThread = [1, 1, 4], threadsPerWarp = [1, 8, 4], warpsPerCTA = [1, 1, 1], order = [2, 1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reduce_rank3_multi_register_band(%X: !tt.ptr<f32>, %O: !tt.ptr<f32>) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<4x8x16xf32, #blocked>
    %off = arith.constant dense<8> : tensor<4x1x1xi32, #blocked>
    %off_0 = arith.constant dense<16> : tensor<4x1x1xi32, #blocked>
    %off_1 = arith.constant dense<16> : tensor<1x8x1xi32, #blocked>
    %i = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #blocked}>}>>
    %i_2 = tt.expand_dims %i {axis = 1 : i32} : tensor<4xi32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #blocked}>}>> -> tensor<4x1xi32, #ttg.slice<{dim = 2, parent = #blocked}>>
    %i_3 = tt.expand_dims %i_2 {axis = 2 : i32} : tensor<4x1xi32, #ttg.slice<{dim = 2, parent = #blocked}>> -> tensor<4x1x1xi32, #blocked>
    %j = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 2, parent = #blocked}>}>>
    %j_4 = tt.expand_dims %j {axis = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 2, parent = #blocked}>}>> -> tensor<1x8xi32, #ttg.slice<{dim = 2, parent = #blocked}>>
    %j_5 = tt.expand_dims %j_4 {axis = 2 : i32} : tensor<1x8xi32, #ttg.slice<{dim = 2, parent = #blocked}>> -> tensor<1x8x1xi32, #blocked>
    %k = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 1, parent = #blocked}>}>>
    %k_6 = tt.expand_dims %k {axis = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 1, parent = #blocked}>}>> -> tensor<1x16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %k_7 = tt.expand_dims %k_6 {axis = 1 : i32} : tensor<1x16xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<1x1x16xi32, #blocked>
    %off_8 = arith.muli %i_3, %off : tensor<4x1x1xi32, #blocked>
    %off_9 = arith.muli %off_8, %off_0 : tensor<4x1x1xi32, #blocked>
    %off_10 = arith.muli %j_5, %off_1 : tensor<1x8x1xi32, #blocked>
    %off_11 = tt.broadcast %off_9 : tensor<4x1x1xi32, #blocked> -> tensor<4x8x1xi32, #blocked>
    %off_12 = tt.broadcast %off_10 : tensor<1x8x1xi32, #blocked> -> tensor<4x8x1xi32, #blocked>
    %off_13 = arith.addi %off_11, %off_12 : tensor<4x8x1xi32, #blocked>
    %off_14 = tt.broadcast %off_13 : tensor<4x8x1xi32, #blocked> -> tensor<4x8x16xi32, #blocked>
    %off_15 = tt.broadcast %k_7 : tensor<1x1x16xi32, #blocked> -> tensor<4x8x16xi32, #blocked>
    %off_16 = arith.addi %off_14, %off_15 : tensor<4x8x16xi32, #blocked>
    %v = tt.splat %X : !tt.ptr<f32> -> tensor<4x8x16x!tt.ptr<f32>, #blocked>
    %v_17 = tt.addptr %v, %off_16 : tensor<4x8x16x!tt.ptr<f32>, #blocked>, tensor<4x8x16xi32, #blocked>
    %v_18 = tt.load %v_17 : tensor<4x8x16x!tt.ptr<f32>, #blocked>
    %0 = tt.splat %O : !tt.ptr<f32> -> tensor<4x8x16x!tt.ptr<f32>, #blocked>
    %1 = tt.addptr %0, %off_16 : tensor<4x8x16x!tt.ptr<f32>, #blocked>, tensor<4x8x16xi32, #blocked>
    %2 = "tt.reduce"(%v_18) <{axis = 2 : i32}> ({
    ^bb0(%arg2: f32, %arg3: f32):
      %7 = arith.addf %arg2, %arg3 : f32
      tt.reduce.return %7 : f32
    }) : (tensor<4x8x16xf32, #blocked>) -> tensor<4x8xf32, #ttg.slice<{dim = 2, parent = #blocked}>>
    %3 = tt.expand_dims %2 {axis = 2 : i32} : tensor<4x8xf32, #ttg.slice<{dim = 2, parent = #blocked}>> -> tensor<4x8x1xf32, #blocked>
    %4 = arith.mulf %v_18, %cst : tensor<4x8x16xf32, #blocked>
    %5 = tt.broadcast %3 : tensor<4x8x1xf32, #blocked> -> tensor<4x8x16xf32, #blocked>
    %6 = arith.addf %5, %4 : tensor<4x8x16xf32, #blocked>
    tt.store %1, %6 : tensor<4x8x16x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
// CHECK-LABEL: metal.kernel reduce_rank3_multi_register_band
// CHECK: metal.threadgroup_alloca : !metal.memref<512 x f32>
// The publish loop runs one trip per element per thread.
// CHECK: scf.for [[IV:%.*]] = %{{.*}} to %{{.*}} step
// The reduced axis (extent 16, row-major stride 1) occupies the LOW bits of the
// label: bit b of the iteration contributes 1 << b, i.e. the label is
// localTid*16 + iv. A layout-indexed scratch put iteration bits 2 and 3 on
// dimension 0 instead, at stride 128.
// CHECK: arith.shrsi [[IV]], %c3
// CHECK: arith.muli %{{.*}}, %c8_i32
// CHECK: metal.tg_store_indexed
// CHECK: metal.barrier
// CHECK: scf.for
// CHECK: metal.binary_exp {{.*}}, {{.*}}, addOp
