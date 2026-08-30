// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | FileCheck %s
//
// Phase D lit fixture: rank-1 reduce, arith.addf combine, BLOCK=1024, num_warps=8.
// spt = sizePerThread[0] = 4; tpb = 32*8 = 256; BLOCK == tpb*spt (1024 == 256*4).
// Verifies spt-fold path: 4 inline metal.get_element + 3 inline
// metal.binary_exp folds before the standard B2.4 butterfly.
// See the implementation notes

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @rank1_reduce_addf_block1024(%x_ptr: !tt.ptr<f32>) {
    %arange = tt.make_range {start = 0 : i32, end = 1024 : i32} : tensor<1024xi32, #blocked>
    %x_ptrs = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %x_ptrs_off = tt.addptr %x_ptrs, %arange : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    %x = tt.load %x_ptrs_off : tensor<1024x!tt.ptr<f32>, #blocked>
    %r = "tt.reduce"(%x) ({
    ^bb0(%a: f32, %b: f32):
      %s = arith.addf %a, %b : f32
      tt.reduce.return %s : f32
    }) {axis = 0 : i32} : (tensor<1024xf32, #blocked>) -> f32
    tt.return
  }
}
// CHECK-LABEL: metal.kernel rank1_reduce_addf_block1024
// Wall 15: scf.for + iter_args replaces the per-k unroll. spt=4, E=4 ⇒
// single loop trip count 4 emission instead of 4 unrolled get_element +
// 3 linear-fold combines.
// CHECK: scf.for {{.*}} iter_args({{.*}} = {{.*}}) -> (f32)
// CHECK: metal.get_element
// CHECK: metal.binary_exp {{.*}}, {{.*}}, addOp
// CHECK: scf.yield
// B2.4: threadgroup buffer size == tpb (256).
// CHECK: metal.threadgroup_alloca : !metal.memref<256 x f32>
// CHECK: metal.store
// CHECK: metal.barrier
// B2.4: butterfly with metal.binary_exp addOp — at least one stage.
// CHECK: metal.binary_exp {{.*}}, {{.*}}, addOp
// CHECK: metal.return
