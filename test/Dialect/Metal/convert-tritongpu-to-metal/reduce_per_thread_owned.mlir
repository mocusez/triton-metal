// RUN: triton-metal-opt --convert-tritongpu-to-metal --split-input-file %s | FileCheck %s
//
// Rank-2 axis=1 reduce on the conv1d-style layout where the reduce axis is
// fully owned per-thread (`threadsPerCTA[axis_dim] == 1`). The L3a-tileloop-2
// body handles this with the SAME self-contained per-row reduction used for
// every other rank-2 axis=1 reduce (walk back to tt.load, reduce whole rows
// from device memory into rowBuf[M], hoisted above the output tile loop):
// there is no longer a separate "per-thread register chain" branch.

// -----
// f32 / arith.addf, conv1d-shape <1024x64xf32> with sPT=[1,1], tPW=[32,1],
// wPC=[4,1] (threadsPerCTA[1] == 1). tpb=128, E_out = 1024/128 = 8.
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#slice1 = #ttg.slice<{dim = 1, parent = #blocked}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reduce_per_thread_owned_f32(%x_ptr: !tt.ptr<f32>, %out_ptr: !tt.ptr<f32>) {
    %cst = arith.constant dense<64> : tensor<1024x1xi32, #blocked>
    %offs_m = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked1>
    %r0 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #slice1>
    %r1 = tt.expand_dims %r0 {axis = 1 : i32} : tensor<1024xi32, #slice1> -> tensor<1024x1xi32, #blocked>
    %r2 = arith.muli %r1, %cst : tensor<1024x1xi32, #blocked>
    %r3 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %r4 = tt.expand_dims %r3 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
    %r5 = tt.broadcast %r2 : tensor<1024x1xi32, #blocked> -> tensor<1024x64xi32, #blocked>
    %r6 = tt.broadcast %r4 : tensor<1x64xi32, #blocked> -> tensor<1024x64xi32, #blocked>
    %r7 = arith.addi %r5, %r6 : tensor<1024x64xi32, #blocked>
    %sp = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<1024x64x!tt.ptr<f32>, #blocked>
    %ap = tt.addptr %sp, %r7 : tensor<1024x64x!tt.ptr<f32>, #blocked>, tensor<1024x64xi32, #blocked>
    %x = tt.load %ap : tensor<1024x64x!tt.ptr<f32>, #blocked>
    %s = "tt.reduce"(%x) ({
    ^bb0(%a: f32, %b: f32):
      %add = arith.addf %a, %b : f32
      tt.reduce.return %add : f32
    }) {axis = 1 : i32} : (tensor<1024x64xf32, #blocked>) -> tensor<1024xf32, #slice1>
    %osp = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked1>
    %oap = tt.addptr %osp, %offs_m : tensor<1024x!tt.ptr<f32>, #blocked1>, tensor<1024xi32, #blocked1>
    %cv = ttg.convert_layout %s : tensor<1024xf32, #slice1> -> tensor<1024xf32, #blocked1>
    tt.store %oap, %cv : tensor<1024x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}
// CHECK-LABEL: metal.kernel reduce_per_thread_owned_f32
// Unified self-contained body: a per-row threadgroup buffer + barrier (NOT a
// barrier-free per-thread register chain).
// CHECK: metal.threadgroup_alloca : !metal.memref<1024 x f32>
// CHECK: scf.for
// CHECK: scf.if
// CHECK: scf.for {{.*}} iter_args({{.*}} = {{.*}}) -> (f32)
// CHECK: metal.binary_exp {{.*}}, {{.*}}, addOp
// CHECK: scf.yield
// CHECK: metal.barrier
// CHECK: metal.get_element {{.*}}memref<1024 x f32>
// CHECK: metal.return
