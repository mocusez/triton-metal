// RUN: triton-metal-opt --convert-tritongpu-to-metal --split-input-file %s | FileCheck %s
//
// Rank-2 axis=1 `tt.reduce` with a MAX combine (arith.maxnumf — Triton's
// tl.max — or arith.maximumf, f32) lowers through the same self-contained
// per-row reduction body as the sum combine (L3a-tileloop-2), with the
// scf.for iter_arg seeded from the first actual element.
//
// The two float extrema differ ONLY in what a NaN element does, and the fold
// operator is where that split lands: maxnumf (NaN ignored) keeps
// `metal.binary_exp ... maxOp` (MSL `max(a, b)`, itself number-preferring),
// while maximumf (IEEE-754 maximum, `propagate_nan=ALL`) must emit maxNanOp so
// a NaN row element survives the fold. A synthetic -infinity would discard an
// all-NaN maxNum row, so both operators start from column zero instead.

// -----
// f32 / arith.maxnumf, shape <8x16xf32>, axis=1. M=8 < tpb=128, so there is no
// outer tile loop and the M<tpb output store is wrapped in a bounds guard.
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#slice1 = #ttg.slice<{dim = 1, parent = #blocked}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reduce_maxf_axis1_f32(%x_ptr: !tt.ptr<f32>, %out_ptr: !tt.ptr<f32>) {
    %cst = arith.constant dense<16> : tensor<8x1xi32, #blocked>
    %offs_m = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #blocked1>
    %r0 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #slice1>
    %r1 = tt.expand_dims %r0 {axis = 1 : i32} : tensor<8xi32, #slice1> -> tensor<8x1xi32, #blocked>
    %r2 = arith.muli %r1, %cst : tensor<8x1xi32, #blocked>
    %r3 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %r4 = tt.expand_dims %r3 {axis = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x16xi32, #blocked>
    %r5 = tt.broadcast %r2 : tensor<8x1xi32, #blocked> -> tensor<8x16xi32, #blocked>
    %r6 = tt.broadcast %r4 : tensor<1x16xi32, #blocked> -> tensor<8x16xi32, #blocked>
    %r7 = arith.addi %r5, %r6 : tensor<8x16xi32, #blocked>
    %sp = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<8x16x!tt.ptr<f32>, #blocked>
    %ap = tt.addptr %sp, %r7 : tensor<8x16x!tt.ptr<f32>, #blocked>, tensor<8x16xi32, #blocked>
    %x = tt.load %ap : tensor<8x16x!tt.ptr<f32>, #blocked>
    %s = "tt.reduce"(%x) ({
    ^bb0(%a: f32, %b: f32):
      %mx = arith.maxnumf %a, %b : f32
      tt.reduce.return %mx : f32
    }) {axis = 1 : i32} : (tensor<8x16xf32, #blocked>) -> tensor<8xf32, #slice1>
    %osp = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>, #blocked1>
    %oap = tt.addptr %osp, %offs_m : tensor<8x!tt.ptr<f32>, #blocked1>, tensor<8xi32, #blocked1>
    %cv = ttg.convert_layout %s : tensor<8xf32, #slice1> -> tensor<8xf32, #blocked1>
    tt.store %oap, %cv : tensor<8x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}
// CHECK-LABEL: metal.kernel reduce_maxf_axis1_f32
// rowBuf is M (= 8) elements, not M*N.
// CHECK: metal.threadgroup_alloca : !metal.memref<8 x f32>
// Grid-stride row loop + per-row `r < M` guard + per-row column scan.
// CHECK: scf.for
// CHECK: arith.cmpi slt
// CHECK: scf.if
// The first real element seeds the fold, retaining all-NaN and all--inf rows.
// CHECK: %[[MAXNUM_SEED:.*]] = metal.get_element {{.*}} -> f32
// CHECK: scf.for {{.*}} iter_args({{.*}} = %[[MAXNUM_SEED]]) -> (f32)
// CHECK: metal.get_element
// The combine is maxOp, not addOp.
// CHECK: metal.binary_exp {{.*}}, {{.*}}, maxOp
// CHECK: scf.yield
// CHECK: metal.store {{.*}} !metal.memref<8 x f32>
// CHECK: metal.barrier
// Per-row result read + sub-tpb (M < tpb) output store bounds guard.
// CHECK: metal.get_element {{.*}}memref<8 x f32>
// CHECK: metal.return

// -----
// f32 / arith.maximumf (IEEE maximum) takes the same path as maxnumf but folds
// with the NaN-propagating operator.
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#slice1 = #ttg.slice<{dim = 1, parent = #blocked}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reduce_maximumf_axis1_f32(%x_ptr: !tt.ptr<f32>, %out_ptr: !tt.ptr<f32>) {
    %cst = arith.constant dense<16> : tensor<8x1xi32, #blocked>
    %offs_m = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #blocked1>
    %r0 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #slice1>
    %r1 = tt.expand_dims %r0 {axis = 1 : i32} : tensor<8xi32, #slice1> -> tensor<8x1xi32, #blocked>
    %r2 = arith.muli %r1, %cst : tensor<8x1xi32, #blocked>
    %r3 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %r4 = tt.expand_dims %r3 {axis = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x16xi32, #blocked>
    %r5 = tt.broadcast %r2 : tensor<8x1xi32, #blocked> -> tensor<8x16xi32, #blocked>
    %r6 = tt.broadcast %r4 : tensor<1x16xi32, #blocked> -> tensor<8x16xi32, #blocked>
    %r7 = arith.addi %r5, %r6 : tensor<8x16xi32, #blocked>
    %sp = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<8x16x!tt.ptr<f32>, #blocked>
    %ap = tt.addptr %sp, %r7 : tensor<8x16x!tt.ptr<f32>, #blocked>, tensor<8x16xi32, #blocked>
    %x = tt.load %ap : tensor<8x16x!tt.ptr<f32>, #blocked>
    %s = "tt.reduce"(%x) ({
    ^bb0(%a: f32, %b: f32):
      %mx = arith.maximumf %a, %b : f32
      tt.reduce.return %mx : f32
    }) {axis = 1 : i32} : (tensor<8x16xf32, #blocked>) -> tensor<8xf32, #slice1>
    %osp = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>, #blocked1>
    %oap = tt.addptr %osp, %offs_m : tensor<8x!tt.ptr<f32>, #blocked1>, tensor<8xi32, #blocked1>
    %cv = ttg.convert_layout %s : tensor<8xf32, #slice1> -> tensor<8xf32, #blocked1>
    tt.store %oap, %cv : tensor<8x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}
// CHECK-LABEL: metal.kernel reduce_maximumf_axis1_f32
// CHECK: metal.threadgroup_alloca : !metal.memref<8 x f32>
// First-element seeding also preserves the NaN-propagating operator.
// CHECK: scf.for
// CHECK: arith.cmpi slt
// CHECK: scf.if
// CHECK: %[[MAXIMUM_SEED:.*]] = metal.get_element {{.*}} -> f32
// CHECK: scf.for {{.*}} iter_args({{.*}} = %[[MAXIMUM_SEED]]) -> (f32)
// CHECK: metal.binary_exp {{.*}}, {{.*}}, maxNanOp
// CHECK-NOT: metal.binary_exp {{.*}}, {{.*}}, maxOp
// CHECK: metal.return

// -----
// f32 / arith.minnumf, shape <8x16xf32>, axis=1. This is the `tl.min` shape
// used by nearest_neighbor before the paired argmin.
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#slice1 = #ttg.slice<{dim = 1, parent = #blocked}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reduce_minnumf_axis1_f32(%x_ptr: !tt.ptr<f32>, %out_ptr: !tt.ptr<f32>) {
    %cst = arith.constant dense<16> : tensor<8x1xi32, #blocked>
    %offs_m = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #blocked1>
    %r0 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #slice1>
    %r1 = tt.expand_dims %r0 {axis = 1 : i32} : tensor<8xi32, #slice1> -> tensor<8x1xi32, #blocked>
    %r2 = arith.muli %r1, %cst : tensor<8x1xi32, #blocked>
    %r3 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %r4 = tt.expand_dims %r3 {axis = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x16xi32, #blocked>
    %r5 = tt.broadcast %r2 : tensor<8x1xi32, #blocked> -> tensor<8x16xi32, #blocked>
    %r6 = tt.broadcast %r4 : tensor<1x16xi32, #blocked> -> tensor<8x16xi32, #blocked>
    %r7 = arith.addi %r5, %r6 : tensor<8x16xi32, #blocked>
    %sp = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<8x16x!tt.ptr<f32>, #blocked>
    %ap = tt.addptr %sp, %r7 : tensor<8x16x!tt.ptr<f32>, #blocked>, tensor<8x16xi32, #blocked>
    %x = tt.load %ap : tensor<8x16x!tt.ptr<f32>, #blocked>
    %s = "tt.reduce"(%x) ({
    ^bb0(%a: f32, %b: f32):
      %mn = arith.minnumf %a, %b : f32
      tt.reduce.return %mn : f32
    }) {axis = 1 : i32} : (tensor<8x16xf32, #blocked>) -> tensor<8xf32, #slice1>
    %osp = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>, #blocked1>
    %oap = tt.addptr %osp, %offs_m : tensor<8x!tt.ptr<f32>, #blocked1>, tensor<8xi32, #blocked1>
    %cv = ttg.convert_layout %s : tensor<8xf32, #slice1> -> tensor<8xf32, #blocked1>
    tt.store %oap, %cv : tensor<8x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}
// CHECK-LABEL: metal.kernel reduce_minnumf_axis1_f32
// CHECK: metal.threadgroup_alloca : !metal.memref<8 x f32>
// CHECK-NOT: arith.constant 3.40282347E+38 : f32
// CHECK: metal.binary_exp {{.*}}, {{.*}}, minOp
// CHECK: metal.return

// -----
// Raw bf16 keeps its own accumulator type; the public Python max API promotes
// bf16 inputs to f32. Both still require the first actual element as seed.
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#slice1 = #ttg.slice<{dim = 1, parent = #blocked}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reduce_maxnumf_axis1_bf16(%x_ptr: !tt.ptr<bf16>, %out_ptr: !tt.ptr<bf16>) {
    %cst = arith.constant dense<16> : tensor<8x1xi32, #blocked>
    %offs_m = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #blocked1>
    %r0 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #slice1>
    %r1 = tt.expand_dims %r0 {axis = 1 : i32} : tensor<8xi32, #slice1> -> tensor<8x1xi32, #blocked>
    %r2 = arith.muli %r1, %cst : tensor<8x1xi32, #blocked>
    %r3 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %r4 = tt.expand_dims %r3 {axis = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x16xi32, #blocked>
    %r5 = tt.broadcast %r2 : tensor<8x1xi32, #blocked> -> tensor<8x16xi32, #blocked>
    %r6 = tt.broadcast %r4 : tensor<1x16xi32, #blocked> -> tensor<8x16xi32, #blocked>
    %r7 = arith.addi %r5, %r6 : tensor<8x16xi32, #blocked>
    %sp = tt.splat %x_ptr : !tt.ptr<bf16> -> tensor<8x16x!tt.ptr<bf16>, #blocked>
    %ap = tt.addptr %sp, %r7 : tensor<8x16x!tt.ptr<bf16>, #blocked>, tensor<8x16xi32, #blocked>
    %x = tt.load %ap : tensor<8x16x!tt.ptr<bf16>, #blocked>
    %s = "tt.reduce"(%x) ({
    ^bb0(%a: bf16, %b: bf16):
      %mx = arith.maxnumf %a, %b : bf16
      tt.reduce.return %mx : bf16
    }) {axis = 1 : i32} : (tensor<8x16xbf16, #blocked>) -> tensor<8xbf16, #slice1>
    %osp = tt.splat %out_ptr : !tt.ptr<bf16> -> tensor<8x!tt.ptr<bf16>, #blocked1>
    %oap = tt.addptr %osp, %offs_m : tensor<8x!tt.ptr<bf16>, #blocked1>, tensor<8xi32, #blocked1>
    %cv = ttg.convert_layout %s : tensor<8xbf16, #slice1> -> tensor<8xbf16, #blocked1>
    tt.store %oap, %cv : tensor<8x!tt.ptr<bf16>, #blocked1>
    tt.return
  }
}
// CHECK-LABEL: metal.kernel reduce_maxnumf_axis1_bf16
// CHECK: metal.threadgroup_alloca : !metal.memref<8 x bf16>
// CHECK: scf.for
// CHECK: arith.cmpi slt
// CHECK: scf.if
// CHECK: %[[BF16_SEED:.*]] = metal.get_element {{.*}} -> bf16
// CHECK: scf.for {{.*}} iter_args({{.*}} = %[[BF16_SEED]]) -> (bf16)
// CHECK: metal.binary_exp {{.*}}, {{.*}}, maxOp : (bf16, bf16) -> bf16
// CHECK: metal.return
