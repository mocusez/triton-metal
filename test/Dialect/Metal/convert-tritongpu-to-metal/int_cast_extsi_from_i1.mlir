// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | FileCheck %s
//
// `arith.extsi` widening a PREDICATE (i1 -> i32).
//
// The sibling `arith.extui` lowers to the plain constructor cast the MSL emitter
// spells as `(int)(x)`, which on an MSL `bool` yields 0/1 — correct
// zero-extension. That same cast is WRONG for extsi: the op is defined to
// produce all-ones (-1) for a true predicate, not +1. `ArithIntCastLowering`
// therefore special-cases this one shape and emits `select(pred, -1, 0)`
// instead of a cast, which is exactly extsi's semantics and uses only ops the
// emitter already handles.
//
// Covered here rather than in pytest because the Triton frontend casts i1 to an
// integer via extUI — there is no Python-level construct that reaches extsi from
// a predicate — so a lit fixture is the only way to pin the behaviour.

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @extsi_from_pred(%src: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %dst: !tt.ptr<i32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0> : tensor<1024xi32, #blocked>
    %o = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %sp = tt.splat %src : !tt.ptr<i32> -> tensor<1024x!tt.ptr<i32>, #blocked>
    %sa = tt.addptr %sp, %o : tensor<1024x!tt.ptr<i32>, #blocked>, tensor<1024xi32, #blocked>
    %v = tt.load %sa : tensor<1024x!tt.ptr<i32>, #blocked>
    %dp = tt.splat %dst : !tt.ptr<i32> -> tensor<1024x!tt.ptr<i32>, #blocked>
    %da = tt.addptr %dp, %o : tensor<1024x!tt.ptr<i32>, #blocked>, tensor<1024xi32, #blocked>
    %pred = arith.cmpi sgt, %v, %cst : tensor<1024xi32, #blocked>
    %ext = arith.extsi %pred : tensor<1024xi1, #blocked> to tensor<1024xi32, #blocked>
    tt.store %da, %ext : tensor<1024x!tt.ptr<i32>, #blocked>
    tt.return
  }
}

// No cast survives: the widening becomes a select of all-ones vs zero.
// CHECK-LABEL: metal.kernel
// CHECK-NOT: arith.extsi
// CHECK: %[[ONES:.*]] = arith.constant -1 : i32
// CHECK: %[[ZERO:.*]] = arith.constant 0 : i32
// CHECK: arith.select %{{.*}}, %[[ONES]], %[[ZERO]] : i32
// CHECK: metal.store
