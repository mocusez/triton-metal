// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | FileCheck %s
//
// Wall 8 lit fixture: masked rank-1 reduce, spt=[1], BLOCK=1024, num_warps=8.
// Direct B2.3 spt=[1] path: E = BLOCK/tpb = 1024/256 = 4 elements per thread
// at stride tpb (256). idx_k = tid + k*256 for k in 0..3.
// Per-k scf.if guards the load with the canonical mask cond_k = idx_k < N.
// Independent of Wall 6's cvt insertion — exercised entirely via
// convert-tritongpu-to-metal on spt=[1] source IR.
// See .omc/plans/tutorial02-wall8-spt1-direct.md AC5.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @rank1_reduce_addf_block1024_spt1_masked(%x_ptr: !tt.ptr<f32>, %n_cols: i32) {
    %arange = tt.make_range {start = 0 : i32, end = 1024 : i32} : tensor<1024xi32, #blocked>
    %x_ptrs = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %x_ptrs_off = tt.addptr %x_ptrs, %arange : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    %n_splat = tt.splat %n_cols : i32 -> tensor<1024xi32, #blocked>
    %mask = arith.cmpi slt, %arange, %n_splat : tensor<1024xi32, #blocked>
    %other = arith.constant dense<0.0> : tensor<1024xf32, #blocked>
    %x = tt.load %x_ptrs_off, %mask, %other : tensor<1024x!tt.ptr<f32>, #blocked>
    %r = "tt.reduce"(%x) ({
    ^bb0(%a: f32, %b: f32):
      %s = arith.addf %a, %b : f32
      tt.reduce.return %s : f32
    }) {axis = 0 : i32} : (tensor<1024xf32, #blocked>) -> f32
    tt.return
  }
}
// CHECK-LABEL: metal.kernel rank1_reduce_addf_block1024_spt1_masked
// Wall 15: scf.for + iter_args replaces the 4-unroll. Masked scf.if
// lives INSIDE the loop body.
// CHECK: scf.for {{.*}} iter_args({{.*}} = {{.*}}) -> (f32)
// CHECK: scf.if
// CHECK: metal.get_element
// CHECK: arith.constant 0.000000e+00 : f32
// CHECK: metal.binary_exp {{.*}}, {{.*}}, addOp
// CHECK: scf.yield
// Threadgroup buffer size == tpb (256).
// CHECK: metal.threadgroup_alloca : !metal.memref<256 x f32>
// CHECK: metal.return
