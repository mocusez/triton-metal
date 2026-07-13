// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | FileCheck %s
//
// Wall 14 fixture (updated for Wall 15): BLOCK > tpb + Wall-11 walker
// chain (softmax 2nd-reduce shape). num_warps=8, threads_per_warp=32 ⇒
// tpb=256; BLOCK=4096 ⇒ E=16. Chain: masked tt.load → arith.subf
// splat(rmax) → math.exp → tt.reduce addf.
//
// Wall 15: the Wall-11 chain re-emission now runs ONCE PER LOOP ITER
// inside the scf.for body. Combiner-identity arith.select fires inside
// the body after the chain, before the combine step.
//
// See .omc/plans/tutorial02-wall15-iter-args-translator-consensus.md AC8.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @rank1_reduce_addf_block4096_chain(%x_ptr: !tt.ptr<f32>, %n_cols: i32, %rmax: f32) {
    %offsets = tt.make_range {end = 4096 : i32, start = 0 : i32} : tensor<4096xi32, #blocked>
    %n_splat = tt.splat %n_cols : i32 -> tensor<4096xi32, #blocked>
    %mask = arith.cmpi slt, %offsets, %n_splat : tensor<4096xi32, #blocked>
    %neg_inf = arith.constant dense<0xFF800000> : tensor<4096xf32, #blocked>
    %x_splat = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<4096x!tt.ptr<f32>, #blocked>
    %x_addr = tt.addptr %x_splat, %offsets : tensor<4096x!tt.ptr<f32>, #blocked>, tensor<4096xi32, #blocked>
    %row = tt.load %x_addr, %mask, %neg_inf : tensor<4096x!tt.ptr<f32>, #blocked>
    %rmax_t = tt.splat %rmax : f32 -> tensor<4096xf32, #blocked>
    %diff = arith.subf %row, %rmax_t : tensor<4096xf32, #blocked>
    %ex = math.exp %diff : tensor<4096xf32, #blocked>
    %sum1 = "tt.reduce"(%ex) ({
    ^bb0(%a: f32, %b: f32):
      %s = arith.addf %a, %b : f32
      tt.reduce.return %s : f32
    }) {axis = 0 : i32} : (tensor<4096xf32, #blocked>) -> f32
    tt.return
  }
}
// CHECK-LABEL: metal.kernel rank1_reduce_addf_block4096_chain
// Multi-accumulator reduce (K=8, metal-multiacc-reduce-plan.md): scf.for + 8
// f32 iter_args, with the Wall-11 chain ops nested inside the body (per
// accumulator). The CHECKs below match the first accumulator's chain.
// CHECK: scf.for {{.*}} step {{.*}} iter_args({{.*}}) -> (f32, f32, f32, f32, f32, f32, f32, f32)
// CHECK: metal.binary_exp {{.*}}, {{.*}}, subOp
// CHECK: metal.unary_exp {{.*}}, expOp
// CHECK: arith.select
// CHECK: metal.binary_exp {{.*}}, {{.*}}, addOp
// CHECK: scf.yield
// CHECK: metal.threadgroup_alloca : !metal.memref<256 x f32>
// CHECK: metal.return
