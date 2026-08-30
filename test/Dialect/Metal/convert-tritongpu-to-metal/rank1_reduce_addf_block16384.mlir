// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | FileCheck %s
//
// Wall 14 fixture (updated for Wall 15): BLOCK > tpb at tutorial02
// autotune endpoint. num_warps=8, threads_per_warp=32 ⇒ tpb=256;
// BLOCK=16384 ⇒ E=64. Wall 15 captures E=64 as the scf.for trip count
// instead of unrolling — MSL output stays O(1) in E.
//
// See the implementation notes AC8.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @rank1_reduce_addf_block16384(%x_ptr: !tt.ptr<f32>) {
    %arange = tt.make_range {start = 0 : i32, end = 16384 : i32} : tensor<16384xi32, #blocked>
    %x_ptrs = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<16384x!tt.ptr<f32>, #blocked>
    %x_ptrs_off = tt.addptr %x_ptrs, %arange : tensor<16384x!tt.ptr<f32>, #blocked>, tensor<16384xi32, #blocked>
    %x = tt.load %x_ptrs_off : tensor<16384x!tt.ptr<f32>, #blocked>
    %r = "tt.reduce"(%x) ({
    ^bb0(%a: f32, %b: f32):
      %s = arith.addf %a, %b : f32
      tt.reduce.return %s : f32
    }) {axis = 0 : i32} : (tensor<16384xf32, #blocked>) -> f32
    tt.return
  }
}
// CHECK-LABEL: metal.kernel rank1_reduce_addf_block16384
// Wall 15: single scf.for; the upper bound encodes E=64.
// Multi-accumulator reduce (K=8, metal-multiacc-reduce-plan.md): the scf.for
// upper bound encodes E=64 and it steps by 8 with 8 f32 iter_args.
// CHECK: arith.constant 64 : i32
// CHECK: scf.for {{.*}} step {{.*}} iter_args({{.*}}) -> (f32, f32, f32, f32, f32, f32, f32, f32)
// CHECK: metal.get_element
// CHECK-COUNT-8: metal.binary_exp {{.*}}, {{.*}}, addOp
// CHECK: scf.yield
// Threadgroup butterfly buffer at tpb=256 (unchanged regardless of E).
// CHECK: metal.threadgroup_alloca : !metal.memref<256 x f32>
// CHECK: metal.return
