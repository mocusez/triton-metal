// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | FileCheck %s --implicit-check-not=tt.reduce
//
// The address twin of this file is rank1_reduce_replayed_address.mlir. Above one
// element per thread the legacy single-load fold rebuilds only
// `cmpi slt(make_range(0, BLOCK) [+ splat(base)], splat(N))` with a splat
// CONSTANT `other`. Every other mask predicate, compound mask, mask index and
// runtime `other` is handed to the general per-element cone evaluator, which
// replays mask and `other` like any other value in the cone.
// The neighbour that stays refused is
// reduce_unsupported.mlir::@reduce_rank1_masked_loop_carried_deferred, a mask
// the evaluator cannot rebuild either.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  // A compound mask: two predicates combined with `andi`.
  tt.func public @reduce_rank1_masked_compound_max(%x_ptr: !tt.ptr<f32>, %n: i32, %row: i32) {
    %offsets = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %n_splat = tt.splat %n : i32 -> tensor<128xi32, #blocked>
    %col_ok = arith.cmpi slt, %offsets, %n_splat : tensor<128xi32, #blocked>
    %zero_i32 = arith.constant 0 : i32
    %row_ok_scalar = arith.cmpi sge, %row, %zero_i32 : i32
    %row_ok = tt.splat %row_ok_scalar : i1 -> tensor<128xi1, #blocked>
    %mask = arith.andi %col_ok, %row_ok : tensor<128xi1, #blocked>
    %other = arith.constant dense<0.0> : tensor<128xf32, #blocked>
    %x_splat = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %x_addr = tt.addptr %x_splat, %offsets : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    %x = tt.load %x_addr, %mask, %other : tensor<128x!tt.ptr<f32>, #blocked>
    %r = "tt.reduce"(%x) ({
    ^bb0(%a: f32, %b: f32):
      %m = arith.maxnumf %a, %b : f32
      tt.reduce.return %m : f32
    }) {axis = 0 : i32} : (tensor<128xf32, #blocked>) -> f32
    tt.return
  }

  // A predicate the legacy fold does not admit, over a mask index that is not
  // the bare range.
  tt.func public @reduce_rank1_masked_sge_scaled(%x_ptr: !tt.ptr<f32>, %n: i32) {
    %offsets = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %two = arith.constant dense<2> : tensor<128xi32, #blocked>
    %scaled = arith.muli %offsets, %two : tensor<128xi32, #blocked>
    %n_splat = tt.splat %n : i32 -> tensor<128xi32, #blocked>
    %mask = arith.cmpi sge, %scaled, %n_splat : tensor<128xi32, #blocked>
    %other = arith.constant dense<7.500000e+00> : tensor<128xf32, #blocked>
    %x_splat = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %x_addr = tt.addptr %x_splat, %offsets : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    %x = tt.load %x_addr, %mask, %other : tensor<128x!tt.ptr<f32>, #blocked>
    %r = "tt.reduce"(%x) ({
    ^bb0(%a: f32, %b: f32):
      %s = arith.addf %a, %b : f32
      tt.reduce.return %s : f32
    }) {axis = 0 : i32} : (tensor<128xf32, #blocked>) -> f32
    tt.return
  }

  // A runtime `other`: uniform across lanes, but not a constant, so the legacy
  // fold could not materialize it.
  tt.func public @reduce_rank1_masked_runtime_other(%x_ptr: !tt.ptr<f32>, %n: i32, %fill: f32) {
    %offsets = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %n_splat = tt.splat %n : i32 -> tensor<128xi32, #blocked>
    %mask = arith.cmpi slt, %offsets, %n_splat : tensor<128xi32, #blocked>
    %other = tt.splat %fill : f32 -> tensor<128xf32, #blocked>
    %x_splat = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %x_addr = tt.addptr %x_splat, %offsets : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    %x = tt.load %x_addr, %mask, %other : tensor<128x!tt.ptr<f32>, #blocked>
    %r = "tt.reduce"(%x) ({
    ^bb0(%a: f32, %b: f32):
      %s = arith.addf %a, %b : f32
      tt.reduce.return %s : f32
    }) {axis = 0 : i32} : (tensor<128xf32, #blocked>) -> f32
    tt.return
  }
}

// CHECK-LABEL: metal.kernel reduce_rank1_masked_compound_max
// Both predicates survive into the per-element guard, and the fold is a max.
// CHECK: arith.cmpi
// CHECK: arith.andi
// CHECK: scf.if
// CHECK: metal.binary_exp {{.*}}, {{.*}}, max

// CHECK-LABEL: metal.kernel reduce_rank1_masked_sge_scaled
// The mask index keeps its multiply, and the predicate stays `sge`.
// CHECK: arith.muli
// CHECK: arith.cmpi sge
// CHECK: scf.if

// CHECK-LABEL: metal.kernel reduce_rank1_masked_runtime_other
// The masked-off lane takes the kernel argument, not a materialized constant.
// CHECK: scf.if
// CHECK: metal.binary_exp {{.*}}, {{.*}}, addOp
