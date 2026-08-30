// RUN: triton-metal-opt %s | FileCheck %s
// RUN: triton-metal-opt %s | triton-metal-translate --mlir-to-msl | FileCheck %s --check-prefix=MSL
//
// Session L1d Phase A (the implementation notes):
// Smoke test for the two new staging ops `metal.tg_store_indexed` and
// `metal.tg_load_indexed`, composed with the L3-shipped `metal.threadgroup_alloca`
// and `metal.barrier`. These will be consumed by §3.5 staged-transpose Phase C
// (L1d2). This fixture only validates that the ops parse, verify, and round-trip
// through the MSL emitter.

module {
  metal.module {
    metal.kernel tg_indexed_smoke address_space_device [true] {
    ^bb0(%arg0: !metal.memref<? x f32>):
      %i = metal.constant 0 : ui32
      %buf = metal.threadgroup_alloca : !metal.memref<16 x f32>
      %v = metal.get_element %arg0[%i] : (!metal.memref<? x f32>, ui32) -> f32
      metal.tg_store_indexed %buf[%i], %v : !metal.memref<16 x f32>, ui32, f32
      metal.barrier
      %w = metal.tg_load_indexed %buf[%i] : (!metal.memref<16 x f32>, ui32) -> f32
      metal.barrier
      metal.store %w, %arg0[%i] : f32, !metal.memref<? x f32>, ui32
      metal.return
    }
    metal.module_end
  }
}

// CHECK: metal.kernel tg_indexed_smoke
// CHECK: metal.threadgroup_alloca
// CHECK: metal.tg_store_indexed
// CHECK: metal.barrier
// CHECK: metal.tg_load_indexed
// CHECK: metal.barrier
// CHECK: metal.return

// MSL: kernel void tg_indexed_smoke
// MSL: threadgroup float v{{[0-9]+}}[16];
// The device read is let-bound at its IR position rather than inlined into the
// `tg_store_indexed` below it: `%arg0` is WRITTEN by the `metal.store` at the
// end of this kernel, and an inlined read would be emitted at its use, which
// for a use placed after such a store observes the new contents. See the
// read-after-overwrite guard in `ModuleTranslation::translate(Region &)`.
// MSL: float v{{[0-9]+}} = v{{[0-9]+}}[{{[0-9]+}}];
// MSL: v{{[0-9]+}}[{{[0-9]+}}] = v{{[0-9]+}};
// MSL: threadgroup_barrier(mem_flags::mem_threadgroup);
// L1d2b inline-barrier contract: `metal.tg_load_indexed` is now
// force-materialised as a named let-binding at its IR position rather
// than inlined into the consumer's emission point. The downstream
// `metal.store` therefore reads the let-binding name. See
// the implementation notes.
// MSL: float v{{[0-9]+}} = v{{[0-9]+}}[{{[0-9]+}}];
// MSL: threadgroup_barrier(mem_flags::mem_threadgroup);
// MSL: v{{[0-9]+}}[{{[0-9]+}}] = v{{[0-9]+}};
// MSL: return;
