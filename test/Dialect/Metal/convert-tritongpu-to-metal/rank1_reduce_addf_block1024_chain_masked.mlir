// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | FileCheck %s
//
// Wall 11 masked-chain fixture: mirrors the softmax tutorial's actual shape.
// Masked tt.load with `other=-inf`, followed by chain: subf(splat) → exp →
// divf(splat) → reduce.
//
// A masked-off lane carries the load's `other` THROUGH the chain, which is
// Triton's contract. This fixture used to require the opposite -- the whole
// chained value replaced by the combine identity -- which happens to agree
// here, because exp(-inf - max) is exactly +0.0, the additive identity. It
// disagreed for every other `other`, which is how an explicit one came to be
// dropped silently. The identity substitution now applies only to a mask with
// NO `other`, where the masked-off value is unspecified: the second kernel.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @softmax_chain_masked(%x_ptr: !tt.ptr<f32>, %n_cols: i32, %rmax: f32) {
    %offsets = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %n_splat = tt.splat %n_cols : i32 -> tensor<1024xi32, #blocked>
    %mask = arith.cmpi slt, %offsets, %n_splat : tensor<1024xi32, #blocked>
    %neg_inf = arith.constant dense<0xFF800000> : tensor<1024xf32, #blocked>
    %x_splat = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %x_addr = tt.addptr %x_splat, %offsets : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    %row = tt.load %x_addr, %mask, %neg_inf : tensor<1024x!tt.ptr<f32>, #blocked>
    %rmax_t = tt.splat %rmax : f32 -> tensor<1024xf32, #blocked>
    %diff = arith.subf %row, %rmax_t : tensor<1024xf32, #blocked>
    %ex = math.exp %diff : tensor<1024xf32, #blocked>
    %sum1 = "tt.reduce"(%ex) ({
    ^bb0(%a: f32, %b: f32):
      %s = arith.addf %a, %b : f32
      tt.reduce.return %s : f32
    }) {axis = 0 : i32} : (tensor<1024xf32, #blocked>) -> f32
    tt.return
  }

  tt.func public @softmax_chain_masked_no_other(%x_ptr: !tt.ptr<f32>, %n_cols: i32, %rmax: f32) {
    %offsets = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %n_splat = tt.splat %n_cols : i32 -> tensor<1024xi32, #blocked>
    %mask = arith.cmpi slt, %offsets, %n_splat : tensor<1024xi32, #blocked>
    %x_splat = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %x_addr = tt.addptr %x_splat, %offsets : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    %row = tt.load %x_addr, %mask : tensor<1024x!tt.ptr<f32>, #blocked>
    %rmax_t = tt.splat %rmax : f32 -> tensor<1024xf32, #blocked>
    %diff = arith.subf %row, %rmax_t : tensor<1024xf32, #blocked>
    %ex = math.exp %diff : tensor<1024xf32, #blocked>
    %sum1 = "tt.reduce"(%ex) ({
    ^bb0(%a: f32, %b: f32):
      %s = arith.addf %a, %b : f32
      tt.reduce.return %s : f32
    }) {axis = 0 : i32} : (tensor<1024xf32, #blocked>) -> f32
    tt.return
  }
}

// CHECK-LABEL: metal.kernel softmax_chain_masked
// Per-k iter: scf.if then=get_element, else=-INFINITY; the chain is re-emitted
// AFTER the scf.if and folded straight into the accumulator -- no select,
// because -inf is what the masked lane is defined to hold and the chain maps
// it onto the identity by itself.
// CHECK: scf.if
// CHECK: metal.binary_exp {{.*}}, {{.*}}, subOp
// CHECK: metal.unary_exp {{.*}}, expOp
// CHECK-NOT: arith.select
// CHECK: metal.binary_exp {{.*}}, {{.*}}, addOp
// CHECK: metal.return

// CHECK-LABEL: metal.kernel softmax_chain_masked_no_other
// With no `other` the masked-off value is unspecified, so the identity
// substitution stays and is what keeps this reduce deterministic.
// CHECK: scf.if
// CHECK: metal.unary_exp {{.*}}, expOp
// CHECK: arith.select
// CHECK: metal.return
