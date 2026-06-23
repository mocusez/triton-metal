// RUN: triton-metal-opt --convert-tritongpu-to-metal --split-input-file %s | FileCheck %s
//
// Rank-2 axis=1 `tt.reduce` (combine ∈ {arith.addf, arith.addi}, T ∈
// {f32, i32}) lowers to the self-contained per-row reduction body
// (L3a-tileloop-2): walk back to the producing `tt.load`, allocate a
// per-row threadgroup buffer `rowBuf[M]`, and have each thread reduce a
// grid-strided set of whole rows directly from device memory into rowBuf;
// after a barrier each thread reads the row its downstream store targets.
// The staging is hoisted above any FuncOpLowering tile loop so it runs once.

// -----
// f32 / arith.addf, shape <8x16xf32>, axis=1. M=8 < tpb=128, so there is no
// outer tile loop and the M<tpb output store is wrapped in a bounds guard.
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#slice1 = #ttg.slice<{dim = 1, parent = #blocked}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reduce_sum_axis1_f32(%x_ptr: !tt.ptr<f32>, %out_ptr: !tt.ptr<f32>) {
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
      %add = arith.addf %a, %b : f32
      tt.reduce.return %add : f32
    }) {axis = 1 : i32} : (tensor<8x16xf32, #blocked>) -> tensor<8xf32, #slice1>
    %osp = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>, #blocked1>
    %oap = tt.addptr %osp, %offs_m : tensor<8x!tt.ptr<f32>, #blocked1>, tensor<8xi32, #blocked1>
    %cv = ttg.convert_layout %s : tensor<8xf32, #slice1> -> tensor<8xf32, #blocked1>
    tt.store %oap, %cv : tensor<8x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}
// CHECK-LABEL: metal.kernel reduce_sum_axis1_f32
// rowBuf is M (= 8) elements, not M*N.
// CHECK: metal.threadgroup_alloca : !metal.memref<8 x f32>
// Grid-stride row loop + per-row `r < M` guard + per-row column scan.
// CHECK: scf.for
// CHECK: arith.cmpi slt
// CHECK: scf.if
// CHECK: scf.for {{.*}} iter_args({{.*}} = {{.*}}) -> (f32)
// CHECK: metal.get_element
// CHECK: metal.binary_exp {{.*}}, {{.*}}, addOp
// CHECK: scf.yield
// CHECK: metal.store {{.*}} !metal.memref<8 x f32>
// CHECK: metal.barrier
// Per-row result read + sub-tpb (M < tpb) output store bounds guard.
// CHECK: metal.get_element {{.*}}memref<8 x f32>
// CHECK: arith.cmpi slt
// CHECK: scf.if
// CHECK: metal.store
// CHECK: metal.return

// -----
// i32 / arith.addi, shape <8x16xi32>, axis=1. The i32 path routes the row
// buffer + column scan through ui32 storage (Metal_Type rejects signless
// i32) and emits the column scan UNROLLED (the translator only threads f32
// scf.for iter_args).
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#slice1 = #ttg.slice<{dim = 1, parent = #blocked}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reduce_sum_axis1_i32(%x_ptr: !tt.ptr<i32>, %out_ptr: !tt.ptr<i32>) {
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
    %sp = tt.splat %x_ptr : !tt.ptr<i32> -> tensor<8x16x!tt.ptr<i32>, #blocked>
    %ap = tt.addptr %sp, %r7 : tensor<8x16x!tt.ptr<i32>, #blocked>, tensor<8x16xi32, #blocked>
    %x = tt.load %ap : tensor<8x16x!tt.ptr<i32>, #blocked>
    %s = "tt.reduce"(%x) ({
    ^bb0(%a: i32, %b: i32):
      %add = arith.addi %a, %b : i32
      tt.reduce.return %add : i32
    }) {axis = 1 : i32} : (tensor<8x16xi32, #blocked>) -> tensor<8xi32, #slice1>
    %osp = tt.splat %out_ptr : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>, #blocked1>
    %oap = tt.addptr %osp, %offs_m : tensor<8x!tt.ptr<i32>, #blocked1>, tensor<8xi32, #blocked1>
    %cv = ttg.convert_layout %s : tensor<8xi32, #slice1> -> tensor<8xi32, #blocked1>
    tt.store %oap, %cv : tensor<8x!tt.ptr<i32>, #blocked1>
    tt.return
  }
}
// CHECK-LABEL: metal.kernel reduce_sum_axis1_i32
// i32 row buffer routed through ui32 storage.
// CHECK: metal.threadgroup_alloca : !metal.memref<8 x ui32>
// Unrolled ui32 column scan.
// CHECK: metal.binary_exp {{.*}}, {{.*}}, addOp : (ui32, ui32) -> ui32
// CHECK: metal.barrier
// CHECK: metal.return
