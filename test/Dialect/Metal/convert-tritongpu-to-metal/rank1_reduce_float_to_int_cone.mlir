// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | FileCheck %s --implicit-check-not=tt.reduce
//
// The cone evaluator has always rebuilt int -> float and every width cast; the
// float -> int direction was missing from it, from its dry-run predicate, and
// from the index pair that scalarizes a load address. Above one element per
// thread that refused `tl.load(Y + o).to(tl.int32)` wherever it appeared.
// ⚠️ Widening the predicates without the two emitters is what turns this into a
// silent wrong answer: the evaluator returns null, the fold keeps the
// accumulator, and the reduce comes back as the identity. Both halves land
// together, and the fold now aborts by name if they ever drift again.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  // The gather index lives in a float buffer.
  tt.func public @reduce_rank1_float_to_int_address(%x_ptr: !tt.ptr<f32>, %y_ptr: !tt.ptr<f32>) {
    %offsets = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %y_splat = tt.splat %y_ptr : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %y_addr = tt.addptr %y_splat, %offsets : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    %y = tt.load %y_addr : tensor<128x!tt.ptr<f32>, #blocked>
    %yi = arith.fptosi %y : tensor<128xf32, #blocked> to tensor<128xi32, #blocked>
    %x_splat = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %x_addr = tt.addptr %x_splat, %yi : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    %x = tt.load %x_addr : tensor<128x!tt.ptr<f32>, #blocked>
    %r = "tt.reduce"(%x) ({
    ^bb0(%a: f32, %b: f32):
      %s = arith.addf %a, %b : f32
      tt.reduce.return %s : f32
    }) {axis = 0 : i32} : (tensor<128xf32, #blocked>) -> f32
    tt.return
  }

  // The mask bound is a float tensor cast to int.
  tt.func public @reduce_rank1_float_to_int_mask_bound(%x_ptr: !tt.ptr<f32>, %y_ptr: !tt.ptr<f32>) {
    %offsets = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %y_splat = tt.splat %y_ptr : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %y_addr = tt.addptr %y_splat, %offsets : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    %y = tt.load %y_addr : tensor<128x!tt.ptr<f32>, #blocked>
    %bound = arith.fptosi %y : tensor<128xf32, #blocked> to tensor<128xi32, #blocked>
    %mask = arith.cmpi slt, %offsets, %bound : tensor<128xi32, #blocked>
    %other = arith.constant dense<7.500000e+00> : tensor<128xf32, #blocked>
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
}

// CHECK-LABEL: metal.kernel reduce_rank1_float_to_int_address
// The cast reaches the address: one element is read, converted, and used as the
// index of the second read.
// CHECK: metal.get_element
// CHECK: arith.fptosi
// CHECK: metal.get_element
// CHECK: metal.binary_exp {{.*}}, {{.*}}, addOp

// CHECK-LABEL: metal.kernel reduce_rank1_float_to_int_mask_bound
// The cast reaches the per-element guard.
// CHECK: arith.fptosi
// CHECK: arith.cmpi slt
// CHECK: scf.if
