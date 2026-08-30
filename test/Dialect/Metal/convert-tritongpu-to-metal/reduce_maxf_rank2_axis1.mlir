// RUN: triton-metal-opt --convert-tritongpu-to-metal --split-input-file %s | FileCheck %s
//
// Rank-2 axis=1 `tt.reduce` with a MAX combine (arith.maxnumf — Triton's
// tl.max — or arith.maximumf, f32) lowers through the same self-contained
// per-row reduction body as the sum combine (L3a-tileloop-2), but with the
// combine emitted as `metal.binary_exp ... maxOp` (MSL max(a, b)) and the
// scf.for iter_arg identity-initialised to exact -infinity.

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
// Identity init is exact -infinity, so an all--inf row remains -inf.
// CHECK: arith.constant 0xFF800000 : f32
// CHECK: scf.for {{.*}} iter_args({{.*}} = {{.*}}) -> (f32)
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
// f32 / arith.maximumf (IEEE maximum) accepted on the same path as maxnumf.
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
// CHECK: metal.binary_exp {{.*}}, {{.*}}, maxOp
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
