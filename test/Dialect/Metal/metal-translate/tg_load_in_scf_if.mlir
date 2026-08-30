// RUN: triton-metal-opt %s | triton-metal-translate --mlir-to-msl | FileCheck %s
//
// L1d2b canary fixture (the implementation notes):
// Asserts the MSL emitter's inline-barrier contract for the boundary op
// `metal.tg_load_indexed`.
//
// Bug class: when a `metal.tg_load_indexed` result is consumed inside an
// `scf.if` body (e.g. emitted by `MaskedStoreLowering`), naive emitter
// behaviour inlines the load expression `<buf>[<idx>]` into the if-body,
// placing it AFTER the trailing `threadgroup_barrier` in execution order
// on every masked-true lane. Apple's Metal shading compiler miscompiles
// that pattern (drops higher-warp stores + warp-0 race). The fix
// force-materialises the load as a named MSL let-binding at its IR
// position; subsequent uses inside the `scf.if` render as the
// let-binding name.
//
// This fixture is the regression test for the contract: the let-binding
// must appear BEFORE the `if (...)` block, and the body must reference
// the let-binding name rather than re-evaluate the load expression.

module {
  metal.module {
    metal.kernel tg_load_in_scf_if address_space_device [true] {
    ^bb0(%arg0: !metal.memref<? x f32>):
      %i = metal.constant 0 : ui32
      %t = metal.constant true
      %buf = metal.threadgroup_alloca : !metal.memref<16 x f32>
      %v = metal.get_element %arg0[%i] : (!metal.memref<? x f32>, ui32) -> f32
      metal.tg_store_indexed %buf[%i], %v : !metal.memref<16 x f32>, ui32, f32
      metal.barrier
      %w = metal.tg_load_indexed %buf[%i] : (!metal.memref<16 x f32>, ui32) -> f32
      scf.if %t {
        metal.store %w, %arg0[%i] : f32, !metal.memref<? x f32>, ui32
      }
      metal.return
    }
    metal.module_end
  }
}

// The contract: the let-binding for `tg_load_indexed` must be emitted at
// the load's IR position (i.e. BEFORE the `if (...) {` block opens), and
// the consumer inside the body must reference the let-binding name —
// never a re-inlined `<buf>[<idx>]` expression.

// CHECK: kernel void tg_load_in_scf_if
// CHECK: threadgroup float v[[#TG:]][16];
// CHECK: float v[[#LET:]] = v[[#TG]][{{[^]]+}}];
// CHECK-NEXT: if (
// CHECK: v{{[0-9]+}}[{{[^]]+}}] = v[[#LET]];
// CHECK: return;
