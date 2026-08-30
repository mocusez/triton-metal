// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | FileCheck %s
//
// Wall 11 masked-chain fixture: mirrors the softmax tutorial's actual shape.
// Masked tt.load with `other=-inf`, followed by chain: subf(splat) → exp →
// divf(splat) → reduce. Architect F1 / Critic D1: masked-off lanes must
// contribute combine identity, not chain-applied else-attr.

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
}

// CHECK-LABEL: metal.kernel softmax_chain_masked
// Per-k iter: scf.if then=get_element, else=-INFINITY; chain re-emitted AFTER
// scf.if; arith.select rewrites masked-off lanes to 0.0 (addf identity).
// CHECK: scf.if
// CHECK: metal.binary_exp {{.*}}, {{.*}}, subOp
// CHECK: metal.unary_exp {{.*}}, expOp
// CHECK: arith.select
// CHECK: metal.return
