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
    // `a, b = b, a + b`: the yield is a simultaneous tuple assignment. With one
    // temp per position it wrote `a` and then read the NEW `a` for `a + b`.
    metal.kernel while_swap address_space_device [false] {
    ^bb0(%out: !metal.memref<? x f32>):
      %c0_ui = metal.constant 0 : ui32
      %c1_ui = metal.constant 1 : ui32
      %c0_i32 = arith.constant 0 : i32
      %c1_i32 = arith.constant 1 : i32
      %c10_i32 = arith.constant 10 : i32
      %f0 = metal.constant 0x00000000 : f32
      %f1 = metal.constant 0x3F800000 : f32
      %r:3 = scf.while (%a = %f0, %b = %f1, %n = %c0_i32) : (f32, f32, i32) -> (f32, f32, i32) {
        %cond = arith.cmpi slt, %n, %c10_i32 : i32
        scf.condition(%cond) %a, %b, %n : f32, f32, i32
      } do {
      ^bb0(%a2: f32, %b2: f32, %n2: i32):
        %sum = metal.binary_exp %a2, %b2, addOp : (f32, f32) -> f32
        %next_n = arith.addi %n2, %c1_i32 : i32
        scf.yield %b2, %sum, %next_n : f32, f32, i32
      }
      metal.store %r#0, %out[%c0_ui] : f32, !metal.memref<? x f32>, ui32
      metal.store %r#1, %out[%c1_ui] : f32, !metal.memref<? x f32>, ui32
      metal.return
    }
    // A condition that forwards a value computed in the before region, not its
    // block argument. That aborted the process; the value is copied out inside
    // the before region's own C scope.
    metal.kernel while_forward_computed address_space_device [false] {
    ^bb0(%out: !metal.memref<? x f32>):
      %c0_ui = metal.constant 0 : ui32
      %f0 = metal.constant 0x00000000 : f32
      %f1 = metal.constant 0x3F800000 : f32
      %f5 = metal.constant 0x40A00000 : f32
      %r = scf.while (%x = %f0) : (f32) -> (f32) {
        %next = metal.binary_exp %x, %f1, addOp : (f32, f32) -> f32
        %cond = arith.cmpf olt, %next, %f5 : f32
        scf.condition(%cond) %next : f32
      } do {
      ^bb0(%y: f32):
        scf.yield %y : f32
      }
      metal.store %r, %out[%c0_ui] : f32, !metal.memref<? x f32>, ui32
      metal.return
    }
    metal.module_end
  }
}

// CHECK: kernel void while_accumulate(
// The before-region args are declared BEFORE the loop with their inits.
// CHECK: float v[[ACC:[0-9]+]] = 0.0
// CHECK: int v[[N:[0-9]+]] = 0
// What the condition forwards -- the after-region args and the results -- has
// temps of its own.
// CHECK: float v[[FACC:[0-9]+]];
// CHECK: int v[[FN:[0-9]+]];
// The predicate temp is declared outside too: it is written by the before
// region (its own C scope) and read after that scope closes.
// CHECK: bool v[[COND:[0-9]+]] = false
// CHECK: while (true)
// CHECK: v[[COND]] = ((int32_t)(v[[N]]) < (int32_t)(8))
// CHECK: v[[FACC]] = v[[ACC]];
// CHECK: v[[FN]] = v[[N]];
// CHECK: if (!v[[COND]]) break;
// The body reads the forwarded temps and writes the before-region ones.
// CHECK: v[[ACC]] = ((v[[FACC]]) + (1.0
// CHECK: v[[N]] = (v[[FN]] + 1)
// The loop result is the last forwarded value.
// CHECK: v{{[0-9]+}}[0] = v[[FACC]]
// CHECK: return;

// CHECK: kernel void while_swap(
// CHECK: float v[[A:[0-9]+]] = 0.0
// CHECK: float v[[B:[0-9]+]] = 1.0
// CHECK: float v[[FA:[0-9]+]];
// CHECK: float v[[FB:[0-9]+]];
// CHECK: v[[FA]] = v[[A]];
// CHECK: v[[FB]] = v[[B]];
// CHECK: if (!v{{[0-9]+}}) break;
// Each yielded value reads only the forwarded temps, so the order of the
// writes cannot matter.
// CHECK: v[[A]] = v[[FB]];
// CHECK: v[[B]] = ((v[[FA]]) + (v[[FB]]));
// CHECK: return;

// CHECK: kernel void while_forward_computed(
// CHECK: float v[[I:[0-9]+]] = 0.0
// CHECK: float v[[F:[0-9]+]];
// CHECK: while (true)
// CHECK: float v[[T:[0-9]+]] = ((v[[I]]) + (1.0
// CHECK: v[[F]] = v[[T]];
// CHECK: if (!v{{[0-9]+}}) break;
// CHECK: v[[I]] = v[[F]];
// CHECK: v{{[0-9]+}}[0] = v[[F]];
// CHECK: return;
