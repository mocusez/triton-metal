// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | FileCheck %s
// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | triton-metal-translate --mlir-to-msl | FileCheck %s --check-prefix=MSL
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

  // A prior observable write is a hard placement boundary.  The reduction
  // remains inside the synthetic output loop rather than moving ahead of the
  // side-effecting store.
  tt.func public @rank1_reduce_after_store(%x_ptr: !tt.ptr<f32>, %side_ptr: !tt.ptr<f32>, %out_ptr: !tt.ptr<f32>) {
    %offsets = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %x_splat = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %x_addr = tt.addptr %x_splat, %offsets : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    %row = tt.load %x_addr : tensor<1024x!tt.ptr<f32>, #blocked>
    %side_splat = tt.splat %side_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %side_addr = tt.addptr %side_splat, %offsets : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    tt.store %side_addr, %row : tensor<1024x!tt.ptr<f32>, #blocked>
    %rmax = "tt.reduce"(%row) ({
    ^bb0(%a: f32, %b: f32):
      %m = arith.maxnumf %a, %b : f32
      tt.reduce.return %m : f32
    }) {axis = 0 : i32} : (tensor<1024xf32, #blocked>) -> f32
    %rmax_t = tt.splat %rmax : f32 -> tensor<1024xf32, #blocked>
    %o_splat = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %o_addr = tt.addptr %o_splat, %offsets : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    tt.store %o_addr, %rmax_t : tensor<1024x!tt.ptr<f32>, #blocked>
    tt.return
  }

  // An unrelated earlier device read is also an ordering boundary.  It is
  // consumed only by the output suffix, not by the aggregate replay cone.
  tt.func public @rank1_reduce_after_unrelated_load(%x_ptr: !tt.ptr<f32>, %bias_ptr: !tt.ptr<f32>, %out_ptr: !tt.ptr<f32>) {
    %bias = tt.load %bias_ptr : !tt.ptr<f32>
    %offsets = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %x_splat = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %x_addr = tt.addptr %x_splat, %offsets : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    %row = tt.load %x_addr : tensor<1024x!tt.ptr<f32>, #blocked>
    %rmax = "tt.reduce"(%row) ({
    ^bb0(%a: f32, %b: f32):
      %m = arith.maxnumf %a, %b : f32
      tt.reduce.return %m : f32
    }) {axis = 0 : i32} : (tensor<1024xf32, #blocked>) -> f32
    %biased = arith.addf %rmax, %bias : f32
    %biased_t = tt.splat %biased : f32 -> tensor<1024xf32, #blocked>
    %o_splat = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %o_addr = tt.addptr %o_splat, %offsets : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    tt.store %o_addr, %biased_t : tensor<1024x!tt.ptr<f32>, #blocked>
    tt.return
  }

  // A CTA barrier is an iteration/visibility boundary even when the reduce's
  // device-load cone itself is otherwise replayable.
  tt.func public @rank1_reduce_after_barrier(%x_ptr: !tt.ptr<f32>, %out_ptr: !tt.ptr<f32>) {
    %offsets = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %x_splat = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %x_addr = tt.addptr %x_splat, %offsets : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    %row = tt.load %x_addr : tensor<1024x!tt.ptr<f32>, #blocked>
    ttg.barrier local
    %rmax = "tt.reduce"(%row) ({
    ^bb0(%a: f32, %b: f32):
      %m = arith.maxnumf %a, %b : f32
      tt.reduce.return %m : f32
    }) {axis = 0 : i32} : (tensor<1024xf32, #blocked>) -> f32
    %rmax_t = tt.splat %rmax : f32 -> tensor<1024xf32, #blocked>
    %o_splat = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %o_addr = tt.addptr %o_splat, %offsets : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    tt.store %o_addr, %rmax_t : tensor<1024x!tt.ptr<f32>, #blocked>
    tt.return
  }

  // A user-authored loop before the aggregate is a region/iteration boundary.
  // Its scalar result remains live in the output suffix so the loop cannot be
  // canonicalized away and accidentally weaken this regression.
  tt.func public @rank1_reduce_after_user_loop(%x_ptr: !tt.ptr<f32>, %out_ptr: !tt.ptr<f32>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %zero = arith.constant 0.0 : f32
    %one = arith.constant 1.0 : f32
    %loop_value = scf.for %i = %c0 to %c2 step %c1 iter_args(%acc = %zero) -> (f32) : i32 {
      %next = arith.addf %acc, %one : f32
      scf.yield %next : f32
    }
    %offsets = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %x_splat = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %x_addr = tt.addptr %x_splat, %offsets : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    %row = tt.load %x_addr : tensor<1024x!tt.ptr<f32>, #blocked>
    %rmax = "tt.reduce"(%row) ({
    ^bb0(%a: f32, %b: f32):
      %m = arith.maxnumf %a, %b : f32
      tt.reduce.return %m : f32
    }) {axis = 0 : i32} : (tensor<1024xf32, #blocked>) -> f32
    %biased = arith.addf %rmax, %loop_value : f32
    %biased_t = tt.splat %biased : f32 -> tensor<1024xf32, #blocked>
    %o_splat = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %o_addr = tt.addptr %o_splat, %offsets : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    tt.store %o_addr, %biased_t : tensor<1024x!tt.ptr<f32>, #blocked>
    tt.return
  }

  // Any preceding region-bearing control flow closes the hoistable prefix.
  // The block-uniform predicate keeps this a valid Triton program while still
  // exercising the selector's control-flow boundary.
  tt.func public @rank1_reduce_after_top_level_if(%x_ptr: !tt.ptr<f32>, %out_ptr: !tt.ptr<f32>) {
    %c0 = arith.constant 0 : i32
    %one = arith.constant 1.0 : f32
    %two = arith.constant 2.0 : f32
    %pid = tt.get_program_id x : i32
    %is_first = arith.cmpi eq, %pid, %c0 : i32
    %choice = scf.if %is_first -> (f32) {
      scf.yield %one : f32
    } else {
      scf.yield %two : f32
    }
    %offsets = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %x_splat = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %x_addr = tt.addptr %x_splat, %offsets : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    %row = tt.load %x_addr : tensor<1024x!tt.ptr<f32>, #blocked>
    %rmax = "tt.reduce"(%row) ({
    ^bb0(%a: f32, %b: f32):
      %m = arith.maxnumf %a, %b : f32
      tt.reduce.return %m : f32
    }) {axis = 0 : i32} : (tensor<1024xf32, #blocked>) -> f32
    %biased = arith.addf %rmax, %choice : f32
    %biased_t = tt.splat %biased : f32 -> tensor<1024xf32, #blocked>
    %o_splat = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %o_addr = tt.addptr %o_splat, %offsets : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    tt.store %o_addr, %biased_t : tensor<1024x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// CHECK-LABEL: metal.kernel softmax_chain
// P4.2: both whole-block aggregates execute before the synthetic output-band
// loop.  The first two loops carry f32 accumulators; the later loop has no
// iter_args and only materializes/stores one output band per trip.
// First reduce: maxnumf scf.for + iter_args.
// CHECK: scf.for {{.*}} iter_args({{.*}} = {{.*}}) -> (f32)
// CHECK: metal.binary_exp {{.*}}, {{.*}}, maxOp
// CHECK: scf.yield
// Second reduce (W11 walker chain): the shared row/max cone is replayed once
// per logical position, before the output-band loop.
// CHECK: scf.for {{.*}} iter_args({{.*}} = {{.*}}) -> (f32)
// CHECK: metal.binary_exp {{.*}}, {{.*}}, subOp
// CHECK: metal.unary_exp {{.*}}, expOp
// CHECK: metal.binary_exp {{.*}}, {{.*}}, addOp
// CHECK: scf.yield
// Synthetic output loop: reductions and their threadgroup synchronization
// must not be nested here.
// CHECK: scf.for {{.*}} : i32 {
// CHECK-NOT: metal.threadgroup_alloca
// CHECK-NOT: metal.barrier
// CHECK: metal.binary_exp {{.*}}, {{.*}}, divOp
// CHECK: metal.store
// CHECK: metal.return

// MSL-LABEL: kernel void softmax_chain(
// The max fold and its threadgroup reduction complete before the sum fold.
// MSL: for (int
// MSL: threadgroup float
// MSL: threadgroup_barrier
// MSL: float [[MAX_RESULT:v[0-9]+]] = {{v[0-9]+}}[0];
// MSL: for (int
// MSL: metal::precise::exp
// MSL: threadgroup float
// MSL: threadgroup_barrier
// MSL: float [[SUM_RESULT:v[0-9]+]] = {{v[0-9]+}}[0];
// The final output loop contains only elementwise replay/store work.  Neither
// aggregate scratch nor a barrier may be emitted inside it.
// MSL: for (int
// MSL-NOT: threadgroup float
// MSL-NOT: threadgroup_barrier
// MSL: metal::precise::exp
// MSL-SAME: ([[MAX_RESULT]])
// MSL-SAME: / ([[SUM_RESULT]])
// MSL: return;

// CHECK-LABEL: metal.kernel rank1_reduce_after_store
// The output-band loop precedes and contains the rank-1 fold, proving the
// aggregate was not moved across the earlier side-effecting store.
// CHECK: scf.for {{.*}} : i32 {
// CHECK: metal.store
// CHECK: scf.for {{.*}} iter_args({{.*}} = {{.*}}) -> (f32)
// CHECK: metal.binary_exp {{.*}}, {{.*}}, maxOp
// CHECK: metal.threadgroup_alloca
// CHECK: metal.store
// CHECK: metal.return

// CHECK-LABEL: metal.kernel rank1_reduce_after_unrelated_load
// The unselected scalar load stays inside the output-band loop and precedes
// the nested aggregate fold; the aggregate was not moved across it.
// CHECK: scf.for {{.*}} : i32 {
// CHECK: metal.get_element
// CHECK: scf.for {{.*}} iter_args({{.*}} = {{.*}}) -> (f32)
// CHECK: metal.binary_exp {{.*}}, {{.*}}, maxOp
// CHECK: metal.threadgroup_alloca
// CHECK: metal.return

// CHECK-LABEL: metal.kernel rank1_reduce_after_barrier
// The synthetic output loop contains both the source barrier and nested fold;
// the aggregate was not moved to a prologue ahead of the barrier.
// CHECK: scf.for {{.*}} : i32 {
// CHECK: metal.barrier
// CHECK: scf.for {{.*}} iter_args({{.*}} = {{.*}}) -> (f32)
// CHECK: metal.binary_exp {{.*}}, {{.*}}, maxOp
// CHECK: metal.threadgroup_alloca
// CHECK: metal.return

// CHECK-LABEL: metal.kernel rank1_reduce_after_user_loop
// Both source regions stay ordered inside the synthetic output loop: first the
// user loop, then the nested rank-1 fold.
// CHECK: scf.for {{.*}} : i32 {
// CHECK: scf.for {{.*}} iter_args({{.*}} = {{.*}}) -> (f32)
// CHECK: arith.addf
// CHECK: scf.for {{.*}} iter_args({{.*}} = {{.*}}) -> (f32)
// CHECK: metal.binary_exp {{.*}}, {{.*}}, maxOp
// CHECK: metal.threadgroup_alloca
// CHECK: metal.return

// CHECK-LABEL: metal.kernel rank1_reduce_after_top_level_if
// The block-uniform source if remains before the aggregate inside the output
// loop, proving that region-bearing control flow closes the hoistable prefix.
// CHECK: scf.for {{.*}} : i32 {
// CHECK: scf.if
// CHECK: scf.yield
// CHECK: scf.for {{.*}} iter_args({{.*}} = {{.*}}) -> (f32)
// CHECK: metal.binary_exp {{.*}}, {{.*}}, maxOp
// CHECK: metal.threadgroup_alloca
// CHECK: metal.return
