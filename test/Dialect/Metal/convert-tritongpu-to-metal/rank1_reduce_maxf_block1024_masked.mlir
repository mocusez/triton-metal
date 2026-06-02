// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | FileCheck %s
//
// Wall 7 lit fixture: masked rank-1 reduce with maxnumf combine.
// BLOCK=1024, num_warps=8, spt=4. Masked tt.load with `other=-INFINITY`.
// Exercises the float-literal emitter regression path (wall-5
// `emitFloatLiteral` at ModuleTranslation.cpp:36) — the else-branch
// arith.constant must carry the -inf bit pattern through.
// Mirror: rank1_reduce_maxf_block32.mlir (combine block) +
// rank1_reduce_addf_block1024.mlir (layout). See plan AC6.

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @rank1_reduce_maxf_block1024_masked(%x_ptr: !tt.ptr<f32>, %n_cols: i32) {
    %arange = tt.make_range {start = 0 : i32, end = 1024 : i32} : tensor<1024xi32, #blocked>
    %x_ptrs = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %x_ptrs_off = tt.addptr %x_ptrs, %arange : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    %n_splat = tt.splat %n_cols : i32 -> tensor<1024xi32, #blocked>
    %mask = arith.cmpi slt, %arange, %n_splat : tensor<1024xi32, #blocked>
    // Canonical -INFINITY bit pattern per arith_constant_neg_inf_f32.mlir.
    %other = arith.constant dense<0xFF800000> : tensor<1024xf32, #blocked>
    %x = tt.load %x_ptrs_off, %mask, %other : tensor<1024x!tt.ptr<f32>, #blocked>
    %r = "tt.reduce"(%x) ({
    ^bb0(%a: f32, %b: f32):
      %s = arith.maxnumf %a, %b : f32
      tt.reduce.return %s : f32
    }) {axis = 0 : i32} : (tensor<1024xf32, #blocked>) -> f32
    tt.return
  }
}
// CHECK-LABEL: metal.kernel rank1_reduce_maxf_block1024_masked
// Wall 15: scf.for + iter_args replaces the 4-unroll. Masked scf.if
// lives INSIDE the loop body; init uses -FLT_MAX (MSL doesn't accept ±inf).
// CHECK: scf.for {{.*}} iter_args({{.*}} = {{.*}}) -> (f32)
// CHECK: scf.if
// CHECK: metal.get_element
// CHECK: arith.constant {{(0xFF800000|-INFINITY|.*INF.*)}} : f32
// CHECK: metal.binary_exp {{.*}}, {{.*}}, maxOp
// CHECK: scf.yield
// Threadgroup buffer size == tpb (256).
// CHECK: metal.threadgroup_alloca : !metal.memref<256 x f32>
// CHECK: metal.return
