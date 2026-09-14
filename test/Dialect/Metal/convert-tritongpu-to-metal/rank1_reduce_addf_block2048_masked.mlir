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

// -----
// Contiguous masked f32 sum exemption: E=BLOCK/tpb=1024 is allowed only for
// zero-other sum with the contiguous masked inspector. The compound row/column
// mask keeps the optional row predicate in front of the runtime column bound.
#blocked_e1024 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @rank1_reduce_addf_block32768_masked_compound(%x_ptr: !tt.ptr<f32>, %out_ptr: !tt.ptr<f32>, %row: i32, %n_cols: i32) {
    %offsets = tt.make_range {end = 32768 : i32, start = 0 : i32} : tensor<32768xi32, #blocked_e1024>
    %c0 = arith.constant 0 : i32
    %row_ok_scalar = arith.cmpi sge, %row, %c0 : i32
    %row_ok = tt.splat %row_ok_scalar : i1 -> tensor<32768xi1, #blocked_e1024>
    %n_splat = tt.splat %n_cols : i32 -> tensor<32768xi32, #blocked_e1024>
    %col_ok = arith.cmpi sle, %offsets, %n_splat : tensor<32768xi32, #blocked_e1024>
    %mask = arith.andi %row_ok, %col_ok : tensor<32768xi1, #blocked_e1024>
    %zero = arith.constant dense<0.000000e+00> : tensor<32768xf32, #blocked_e1024>
    %x_splat = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<32768x!tt.ptr<f32>, #blocked_e1024>
    %x_addr = tt.addptr %x_splat, %offsets : tensor<32768x!tt.ptr<f32>, #blocked_e1024>, tensor<32768xi32, #blocked_e1024>
    %row_values = tt.load %x_addr, %mask, %zero : tensor<32768x!tt.ptr<f32>, #blocked_e1024>
    %r = "tt.reduce"(%row_values) ({
    ^bb0(%a: f32, %b: f32):
      %s = arith.addf %a, %b : f32
      tt.reduce.return %s : f32
    }) {axis = 0 : i32} : (tensor<32768xf32, #blocked_e1024>) -> f32
    tt.store %out_ptr, %r : !tt.ptr<f32>
    tt.return
  }
}

// CHECK-LABEL: metal.kernel rank1_reduce_addf_block32768_masked_compound
// CHECK: [[ROW_OK:%.*]] = arith.cmpi sge
// CHECK: [[E:%.*]] = arith.constant 1024 : i32
// CHECK: scf.for {{.*}} to [[E]] {{.*}} iter_args({{.*}} = {{.*}}) -> (f32)
// CHECK: [[COL_OK:%.*]] = arith.cmpi sle
// CHECK: [[MASK:%.*]] = arith.andi [[COL_OK]], [[ROW_OK]] : i1
// CHECK: scf.if [[MASK]]
// CHECK: metal.get_element
// CHECK: metal.binary_exp {{.*}}, {{.*}}, addOp
// CHECK: scf.yield
// CHECK: metal.threadgroup_alloca : !metal.memref<32 x f32>
// CHECK: metal.barrier
// CHECK: metal.store
// CHECK: metal.barrier
// CHECK: metal.get_element
// CHECK: metal.store {{.*}} : f32, !metal.memref<? x f32>, ui32
// CHECK: metal.return
