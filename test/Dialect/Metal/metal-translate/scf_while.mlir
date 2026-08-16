// RUN: triton-metal-translate --mlir-to-msl %s 2>&1 | FileCheck %s
//
// Translator-side fixture for `scf.while`, the shape a Python `while` in a
// @triton.jit kernel lowers to. Before this the emitter had no case at all and
// aborted the process on `llvm_unreachable("Unexpected operation")` rather than
// reporting anything.
//
// MSL has no multi-value loop carry, so each carried value is a temp declared
// ahead of the loop and the loop is an unconditional `while (true)` with the
// predicate re-evaluated at the top of each trip and an explicit `break`. A
// C-style `while (<cond>)` could not express this: the predicate reads the very
// temps the body updates, and the before-region ops that compute it must run
// again every trip.

module {
  metal.module {
    // Single f32 accumulator plus an i32 trip counter — the canonical
    // `while n < N: acc += ...; n += 1` shape.
    metal.kernel while_accumulate address_space_device [false] {
    ^bb0(%out: !metal.memref<? x f32>):
      %c0_ui = metal.constant 0 : ui32
      %c0_i32 = arith.constant 0 : i32
      %c1_i32 = arith.constant 1 : i32
      %c8_i32 = arith.constant 8 : i32
      %finit = metal.constant 0x00000000 : f32
      %r:2 = scf.while (%a = %finit, %n = %c0_i32) : (f32, i32) -> (f32, i32) {
        %cond = arith.cmpi slt, %n, %c8_i32 : i32
        scf.condition(%cond) %a, %n : f32, i32
      } do {
      ^bb0(%a2: f32, %n2: i32):
        %one_f = metal.constant 0x3F800000 : f32
        %next_a = metal.binary_exp %a2, %one_f, addOp : (f32, f32) -> f32
        %next_n = arith.addi %n2, %c1_i32 : i32
        scf.yield %next_a, %next_n : f32, i32
      }
      metal.store %r#0, %out[%c0_ui] : f32, !metal.memref<? x f32>, ui32
      metal.return
    }
    metal.module_end
  }
}

// CHECK: kernel void while_accumulate(
// Carried values are declared BEFORE the loop, so a zero-trip loop yields the
// inits unchanged.
// CHECK: float v[[ACC:[0-9]+]] = 0.0
// CHECK: int v[[N:[0-9]+]] = 0
// The predicate temp is declared outside too: it is written by the before
// region (its own C scope) and read after that scope closes.
// CHECK: bool v[[COND:[0-9]+]] = false
// CHECK: while (true)
// CHECK: v[[COND]] = (v[[N]] < 8)
// CHECK: if (!v[[COND]]) break;
// CHECK: v[[ACC]] = (v[[ACC]]) + (1.0
// CHECK: v[[N]] = (v[[N]] + 1)
// The loop result reads the same temp the carry writes.
// CHECK: v{{[0-9]+}}[0] = v[[ACC]]
// CHECK: return;
