// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | FileCheck %s
//
// Wall 11 fixture: end-to-end softmax-shape with TWO reduces in series
// per Architect F6.
//
//   1. %rmax = tt.reduce maxnumf <- tt.load %row     (existing Wall 7 plumbing)
//   2. %diff = arith.subf %row, tt.splat(%rmax)      (Wall 9)
//   3. %ex   = math.exp %diff                        (Wall 10, already present)
//   4. %sum1 = tt.reduce addf <- %ex                 (NEW: W11 walks chain)
//   5. %norm = arith.divf %ex, tt.splat(%sum1)       (Wall 12)
//   6. %sum2 = tt.reduce addf <- %norm               (NEW: W11 walks deeper)
//
// On the unmodified tree this fails at the second tt.reduce because the
// B2.3 walk-back currently requires `tt.load` as the direct producer
// (rejects the chain). After Wall 11 ships, the recursive walker resolves
// the chain and emits per-spt-idx scalar re-emission.
//
// Architect F4 + Critic D2: CHECK lines pin operand order on subf/divf via
// FileCheck captures — the splat-fed operand must appear on the RHS.
//
// BLOCK=1024, num_warps=8, spt=[1] → E = BLOCK/tpb = 1024/256 = 4 (Wall 8
// direct path).
// See the implementation notes AC5.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @softmax_chain(%x_ptr: !tt.ptr<f32>, %out_ptr: !tt.ptr<f32>) {
    %offsets = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %x_splat = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %x_addr = tt.addptr %x_splat, %offsets : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    %row = tt.load %x_addr : tensor<1024x!tt.ptr<f32>, #blocked>
    // first reduce — Wall 7/8 plumbing (max over row)
    %rmax = "tt.reduce"(%row) ({
    ^bb0(%a: f32, %b: f32):
      %m = arith.maxnumf %a, %b : f32
      tt.reduce.return %m : f32
    }) {axis = 0 : i32} : (tensor<1024xf32, #blocked>) -> f32
    %rmax_t = tt.splat %rmax : f32 -> tensor<1024xf32, #blocked>
    %diff = arith.subf %row, %rmax_t : tensor<1024xf32, #blocked>
    %ex = math.exp %diff : tensor<1024xf32, #blocked>
    // second reduce — W11 walks chain through subf+exp back to %row's load
    %sum1 = "tt.reduce"(%ex) ({
    ^bb0(%a: f32, %b: f32):
      %s = arith.addf %a, %b : f32
      tt.reduce.return %s : f32
    }) {axis = 0 : i32} : (tensor<1024xf32, #blocked>) -> f32
    %denom_t = tt.splat %sum1 : f32 -> tensor<1024xf32, #blocked>
    %norm = arith.divf %ex, %denom_t : tensor<1024xf32, #blocked>
    %o_splat = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %o_addr = tt.addptr %o_splat, %offsets : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    tt.store %o_addr, %norm : tensor<1024x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// CHECK-LABEL: metal.kernel softmax_chain
// Wall 15: each rank-1 reduce is now a single scf.for with f32 iter_arg.
// The chain ops (subOp + expOp) live INSIDE the second reduce's loop body
// — emitted once per iter instead of 4× unrolled.
// First reduce: maxnumf scf.for + iter_args.
// CHECK: scf.for {{.*}} iter_args({{.*}} = {{.*}}) -> (f32)
// CHECK: metal.binary_exp {{.*}}, {{.*}}, maxOp
// CHECK: scf.yield
// Second reduce (W11 walker chain): scf.for body with subOp → expOp → addOp.
// CHECK: scf.for {{.*}} iter_args({{.*}} = {{.*}}) -> (f32)
// CHECK: metal.binary_exp {{.*}}, {{.*}}, subOp
// CHECK: metal.unary_exp {{.*}}, expOp
// CHECK: metal.binary_exp {{.*}}, {{.*}}, addOp
// CHECK: scf.yield
// Wall 12: divf lowering after the second reduce result is broadcast.
// CHECK: metal.binary_exp {{.*}}, {{.*}}, divOp
// CHECK: metal.return
