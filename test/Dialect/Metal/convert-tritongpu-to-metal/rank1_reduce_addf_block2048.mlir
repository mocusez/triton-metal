// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | FileCheck %s
//
// Wall 14 fixture (updated for Wall 15 scf.for re-roll): BLOCK > tpb,
// spt=[1], unmasked direct load. num_warps=8, threads_per_warp=32 ⇒
// tpb=256; BLOCK=2048 ⇒ E=BLOCK/tpb=8.
//
// Wall 15 re-roll: the per-k unroll is now a single scf.for + f32
// iter_arg accumulator. The translator emits `float vN = init;` BEFORE
// the for line and `vN = ...;` at scf.yield, making MSL emission O(1)
// in E.
//
// See .omc/plans/tutorial02-wall15-iter-args-translator-consensus.md AC8.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @rank1_reduce_addf_block2048(%x_ptr: !tt.ptr<f32>) {
    %arange = tt.make_range {start = 0 : i32, end = 2048 : i32} : tensor<2048xi32, #blocked>
    %x_ptrs = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<2048x!tt.ptr<f32>, #blocked>
    %x_ptrs_off = tt.addptr %x_ptrs, %arange : tensor<2048x!tt.ptr<f32>, #blocked>, tensor<2048xi32, #blocked>
    %x = tt.load %x_ptrs_off : tensor<2048x!tt.ptr<f32>, #blocked>
    %r = "tt.reduce"(%x) ({
    ^bb0(%a: f32, %b: f32):
      %s = arith.addf %a, %b : f32
      tt.reduce.return %s : f32
    }) {axis = 0 : i32} : (tensor<2048xf32, #blocked>) -> f32
    tt.return
  }
}
// CHECK-LABEL: metal.kernel rank1_reduce_addf_block2048
// Multi-accumulator reduce (K=8, metal-multiacc-reduce-plan.md): the scf.for
// steps by 8 with 8 f32 iter_args (breaks the serial FADD chain); a balanced
// tree-combine of the 8 results then feeds the threadgroup butterfly.
// CHECK: scf.for {{.*}} step {{.*}} iter_args({{.*}}) -> (f32, f32, f32, f32, f32, f32, f32, f32)
// CHECK: metal.get_element
// CHECK-COUNT-8: metal.binary_exp {{.*}}, {{.*}}, addOp
// CHECK: scf.yield
// CHECK: metal.binary_exp {{.*}}, {{.*}}, addOp
// Threadgroup butterfly buffer at tpb=256.
// CHECK: metal.threadgroup_alloca : !metal.memref<256 x f32>
// CHECK: metal.return
