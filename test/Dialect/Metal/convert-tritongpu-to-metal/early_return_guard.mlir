// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | FileCheck %s
//
// A Python early-return guard (`if pid >= B: return`) lowers to an unstructured
// `cf.cond_br %c, ^ret(tt.return), ^cont` that the structured-only MSL emitter
// cannot handle (and `--lift-cf-to-scf` cannot recover, a void early-return not
// reconverging). `structureEarlyReturns` rewrites it, KEEPING the original
// condition polarity, to `scf.if %c { } else { <cont> }` (continuation in the
// else arm since cond_br branches to ^cont on false). No `arith.xori` negation
// (that mis-lowers to MSL bitwise `%c ^ -1`, always truthy).

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @early_return_guard(%out: !tt.ptr<i32>, %B: i32) {
    %pid = tt.get_program_id x : i32
    %c = arith.cmpi sge, %pid, %B : i32
    cf.cond_br %c, ^bb1, ^bb2
  ^bb1:  // early return
    tt.return
  ^bb2:  // continuation
    %z = arith.constant 0 : i32
    %p = tt.addptr %out, %pid : !tt.ptr<i32>, i32
    tt.store %p, %z : !tt.ptr<i32>
    tt.return
  }
}
// CHECK-LABEL: metal.kernel early_return_guard
// The unstructured branch is gone; the guard is a structured scf.if.
// CHECK-NOT: cf.cond_br
// CHECK: scf.if
// The store lands in the else arm (entered when pid < B) as a metal.store.
// CHECK: else
// CHECK: metal.store
// CHECK: metal.return
