// RUN: triton-metal-opt --convert-tritongpu-to-metal --split-input-file %s | FileCheck %s
//
// Rank-2 axis=1 reduce over a LARGE tile (M*N >> tpb). The self-contained
// per-row body (L3a-tileloop-2) handles this with no special "chunking":
// rowBuf is sized M (the row count), and each thread reduces a grid-strided
// set of whole rows from device memory. The staging is hoisted above the
// FuncOpLowering output tile loop (now sized from the reduce OUTPUT, E_out),
// and the per-row result is read inside that loop. Threadgroup memory is
// M * sizeof(T) (4 KiB for M=1024) — independent of N — so the old chunked
// body's 32 KiB over-allocation is gone.

// -----
// f32 / arith.addf, shape <1024x64xf32>, axis=1. tpb=128, E_out = 1024/128 = 8.
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#slice1 = #ttg.slice<{dim = 1, parent = #blocked}>
#blocked1 = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reduce_chunked_axis1_f32(%x_ptr: !tt.ptr<f32>, %out_ptr: !tt.ptr<f32>) {
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
// CHECK-LABEL: metal.kernel reduce_chunked_axis1_f32
// rowBuf is M = 1024 elements (4 KiB), NOT M*N.
// CHECK: metal.threadgroup_alloca : !metal.memref<1024 x f32>
// Hoisted staging: grid-stride row loop, per-row guard, rerolled column scan.
// CHECK: scf.for
// CHECK: arith.cmpi slt
// CHECK: scf.if
// CHECK: scf.for {{.*}} iter_args({{.*}} = {{.*}}) -> (f32)
// CHECK: metal.get_element
// CHECK: metal.binary_exp {{.*}}, {{.*}}, addOp
// CHECK: scf.yield
// CHECK: metal.store {{.*}}memref<1024 x f32>
// CHECK: metal.barrier
// Output tile loop (E_out = 8) reads rowBuf and stores; M == tpb*E_out so the
// store is in-bounds (no sub-tpb guard).
// CHECK: scf.for
// CHECK: metal.get_element {{.*}}memref<1024 x f32>
// CHECK: metal.store
// CHECK: metal.return
