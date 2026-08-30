// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | FileCheck %s
//
// `tl.assume(pid >= 0)` lowers to `llvm.intr.assume` (python/src/ir.cc
// `create_assume`). It is a result-less optimizer hint with no MSL equivalent,
// and leaving it in place fails the conversion outright ("failed to legalize
// operation 'llvm.intr.assume'") — the first of the two walls that blocked
// leet-triton/medium-count_array_element.py.
//
// `eraseAssumeHints` drops the hint AND whatever predicate cone it kept alive.
// The second half matters: a dead-but-legal `arith.cmpi` survives conversion
// and reaches the MSL emitter, whose `isStatementPrintable` default branch
// prints any use-less op as a bare statement.

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @assume_hint_erased(%out_ptr: !tt.ptr<f32>) {
    %pid = tt.get_program_id x : i32
    %c0 = arith.constant 0 : i32
    %ge = arith.cmpi sge, %pid, %c0 : i32
    llvm.intr.assume %ge : i1
    %f = arith.sitofp %pid : i32 to f32
    tt.store %out_ptr, %f : !tt.ptr<f32>
    tt.return
  }
}

// CHECK-LABEL: metal.kernel assume_hint_erased
// The hint and its now-dead predicate are both gone; the real body survives.
// CHECK-NOT: llvm.intr.assume
// CHECK-NOT: arith.cmpi
// CHECK: metal.store
