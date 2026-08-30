// RUN: triton-metal-opt --convert-tritongpu-to-metal --split-input-file %s | FileCheck %s

// Rank-2 axis=1 signed-i32 extrema must compare in signless i32 space and
// bridge back to the backend's ui32 storage type.  Using metal.binary_exp on
// ui32 would implement unsigned extrema and miscompile negative inputs.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#slice1 = #ttg.slice<{dim = 1, parent = #blocked}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reduce_maxsi_axis1_i32(%x_ptr: !tt.ptr<i32>, %out_ptr: !tt.ptr<i32>) {
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
      %mx = arith.maxsi %a, %b : i32
      tt.reduce.return %mx : i32
    }) {axis = 1 : i32} : (tensor<8x16xi32, #blocked>) -> tensor<8xi32, #slice1>
    %osp = tt.splat %out_ptr : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>, #blocked1>
    %oap = tt.addptr %osp, %offs_m : tensor<8x!tt.ptr<i32>, #blocked1>, tensor<8xi32, #blocked1>
    %cv = ttg.convert_layout %s : tensor<8xi32, #slice1> -> tensor<8xi32, #blocked1>
    tt.store %oap, %cv : tensor<8x!tt.ptr<i32>, #blocked1>
    tt.return
  }
}
// CHECK-LABEL: metal.kernel reduce_maxsi_axis1_i32
// CHECK: metal.threadgroup_alloca : !metal.memref<8 x ui32>
// CHECK: arith.cmpi sgt
// CHECK: arith.select
// CHECK-NOT: metal.binary_exp {{.*}} maxOp
// CHECK: metal.return

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#slice1 = #ttg.slice<{dim = 1, parent = #blocked}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reduce_muli_axis1_i32(%x_ptr: !tt.ptr<i32>, %out_ptr: !tt.ptr<i32>) {
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
      %product = arith.muli %a, %b : i32
      tt.reduce.return %product : i32
    }) {axis = 1 : i32} : (tensor<8x16xi32, #blocked>) -> tensor<8xi32, #slice1>
    %osp = tt.splat %out_ptr : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>, #blocked1>
    %oap = tt.addptr %osp, %offs_m : tensor<8x!tt.ptr<i32>, #blocked1>, tensor<8xi32, #blocked1>
    %cv = ttg.convert_layout %s : tensor<8xi32, #slice1> -> tensor<8xi32, #blocked1>
    tt.store %oap, %cv : tensor<8x!tt.ptr<i32>, #blocked1>
    tt.return
  }
}
// CHECK-LABEL: metal.kernel reduce_muli_axis1_i32
// CHECK: metal.threadgroup_alloca : !metal.memref<8 x ui32>
// CHECK: metal.binary_exp {{.*}}, {{.*}}, mulOp
// CHECK: metal.return

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#slice1 = #ttg.slice<{dim = 1, parent = #blocked}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reduce_minsi_axis1_i32(%x_ptr: !tt.ptr<i32>, %out_ptr: !tt.ptr<i32>) {
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
      %mn = arith.minsi %a, %b : i32
      tt.reduce.return %mn : i32
    }) {axis = 1 : i32} : (tensor<8x16xi32, #blocked>) -> tensor<8xi32, #slice1>
    %osp = tt.splat %out_ptr : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>, #blocked1>
    %oap = tt.addptr %osp, %offs_m : tensor<8x!tt.ptr<i32>, #blocked1>, tensor<8xi32, #blocked1>
    %cv = ttg.convert_layout %s : tensor<8xi32, #slice1> -> tensor<8xi32, #blocked1>
    tt.store %oap, %cv : tensor<8x!tt.ptr<i32>, #blocked1>
    tt.return
  }
}
// CHECK-LABEL: metal.kernel reduce_minsi_axis1_i32
// CHECK: metal.threadgroup_alloca : !metal.memref<8 x ui32>
// CHECK: arith.cmpi slt
// CHECK: arith.select
// CHECK-NOT: metal.binary_exp {{.*}} minOp
// CHECK: metal.return
