// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | FileCheck %s
//
// Scalar INTEGER `tl.atomic_add(ptr, i32)` — `tt.atomic_rmw add` (RMWOp::ADD),
// not `fadd`. `AtomicRmwLowering` used to accept FADD/f32 only, so
// `tl.atomic_add(out, tl.sum(x == K))` (whose reduce result is i32) failed to
// legalize: the second wall in leet-triton/medium-count_array_element.py.
//
// The payload must enter `metal.atomic_rmw` as the memref's ui32 STORAGE type,
// not as signless i32 — `Metal_Type` (MetalOps.td:17) admits ui32 but not
// signless i32, so an i32 operand fails the op's own verifier. The emitter then
// picks `atomic_uint` over `atomic_float` from that element type.
//
// The `localTid == 0` guard is the same one the f32 scalar form uses: a scalar
// atomic is one add per PROGRAM, but every Metal thread runs the kernel body.

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @atomic_add_scalar_i32(%out_ptr: !tt.ptr<i32>, %v: i32) {
    %r = tt.atomic_rmw add, acq_rel, gpu, %out_ptr, %v : (!tt.ptr<i32>, i32) -> i32
    tt.return
  }
}

// CHECK-LABEL: metal.kernel atomic_add_scalar_i32
// Single-lane guard, then the atomic on the ui32 buffer.
// CHECK: arith.cmpi eq
// CHECK: scf.if
// CHECK: metal.atomic_rmw {{.*}} : (ui32, !metal.memref<? x ui32>, ui32) -> ui32
