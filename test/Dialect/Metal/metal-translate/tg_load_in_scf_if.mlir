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

    // A result-bearing scf.if needs one persistent MSL temp per result. Keep
    // unlike result types here so accidentally mapping both yields to the last
    // declared temp is rejected by both FileCheck and the Metal compiler.
    metal.kernel multi_result_scf_if address_space_device [true, true] {
    ^bb0(%out_f: !metal.memref<? x f32>, %out_i: !metal.memref<? x ui32>):
      %idx = metal.constant 0 : ui32
      %cond = metal.constant true
      %then_f = metal.constant 0x3F800000 : f32
      %else_f = metal.constant 0x40000000 : f32
      %then_i = metal.constant 7 : ui32
      %else_i = metal.constant 9 : ui32
      %pair:2 = scf.if %cond -> (f32, ui32) {
        scf.yield %then_f, %then_i : f32, ui32
      } else {
        scf.yield %else_f, %else_i : f32, ui32
      }
      metal.store %pair#0, %out_f[%idx] : f32, !metal.memref<? x f32>, ui32
      metal.store %pair#1, %out_i[%idx] : ui32, !metal.memref<? x ui32>, ui32
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

// CHECK-LABEL: kernel void multi_result_scf_if
// CHECK: float v[[#F:]];
// CHECK-NEXT: uint32_t v[[#I:]];
// CHECK: if (true) {
// CHECK: v[[#F]] = 1.000000e+00;
// CHECK: v[[#I]] = 7;
// CHECK: } else {
// CHECK: v[[#F]] = 2.000000e+00;
// CHECK: v[[#I]] = 9;
// CHECK: v{{[0-9]+}}[0] = v[[#F]];
// CHECK: v{{[0-9]+}}[0] = v[[#I]];
