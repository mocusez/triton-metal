// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | FileCheck %s
//
// Wall 14 fixture (updated for Wall 15): BLOCK > tpb masked load,
// E=BLOCK/tpb=8. Wall 7 canonical mask `cmpi slt make_range splat(n_cols)`
// + splat-other=0.0. Wall 15 nests the per-k scf.if INSIDE the new
// scf.for body — single occurrence per iteration instead of 8 unrolled.
//
// See the implementation notes AC8.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @rank1_reduce_addf_block2048_masked(%x_ptr: !tt.ptr<f32>, %n_cols: i32) {
    %offsets = tt.make_range {end = 2048 : i32, start = 0 : i32} : tensor<2048xi32, #blocked>
    %n_splat = tt.splat %n_cols : i32 -> tensor<2048xi32, #blocked>
    %mask = arith.cmpi slt, %offsets, %n_splat : tensor<2048xi32, #blocked>
    %zero = arith.constant dense<0.000000e+00> : tensor<2048xf32, #blocked>
    %x_splat = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<2048x!tt.ptr<f32>, #blocked>
    %x_addr = tt.addptr %x_splat, %offsets : tensor<2048x!tt.ptr<f32>, #blocked>, tensor<2048xi32, #blocked>
    %row = tt.load %x_addr, %mask, %zero : tensor<2048x!tt.ptr<f32>, #blocked>
    %r = "tt.reduce"(%row) ({
    ^bb0(%a: f32, %b: f32):
      %s = arith.addf %a, %b : f32
      tt.reduce.return %s : f32
    }) {axis = 0 : i32} : (tensor<2048xf32, #blocked>) -> f32
    tt.return
  }
}
// CHECK-LABEL: metal.kernel rank1_reduce_addf_block2048_masked
// Wall 15: scf.for + iter_args, masked scf.if inside the body.
// CHECK: scf.for {{.*}} iter_args({{.*}} = {{.*}}) -> (f32)
// CHECK: scf.if
// CHECK: metal.get_element
// CHECK: metal.binary_exp {{.*}}, {{.*}}, addOp
// CHECK: scf.yield
// Threadgroup butterfly buffer at tpb=256.
// CHECK: metal.threadgroup_alloca : !metal.memref<256 x f32>
// CHECK: metal.return
