// RUN: triton-metal-opt %s | FileCheck %s
// RUN: triton-metal-opt %s | triton-metal-translate --mlir-to-msl | FileCheck %s --check-prefix=MSL
//
// Session L3 (`.omc/specs/deep-interview-leet-triton-l3-reduce-axis-2d.md`):
// Smoke test for the two new shared metal-dialect ops, `metal.threadgroup_alloca`
// and `metal.barrier`. These will be consumed by §3.3 reduce lowering (Phase C,
// L3a) and §3.5 staged-transpose (future). This fixture only validates that the
// ops parse, verify, and round-trip through the MSL emitter.

module {
  metal.module {
    metal.kernel tg_smoke address_space_device [true] {
    ^bb0(%arg0: !metal.memref<? x f32>):
      %0 = metal.constant 0 : ui32
      %tg = metal.threadgroup_alloca : !metal.memref<16 x f32>
      %v = metal.get_element %arg0[%0] : (!metal.memref<? x f32>, ui32) -> f32
      metal.store %v, %tg[%0] : f32, !metal.memref<16 x f32>, ui32
      metal.barrier
      %r = metal.get_element %tg[%0] : (!metal.memref<16 x f32>, ui32) -> f32
      metal.store %r, %arg0[%0] : f32, !metal.memref<? x f32>, ui32
      metal.return
    }
    metal.module_end
  }
}

// CHECK: metal.kernel tg_smoke
// CHECK: metal.threadgroup_alloca
// CHECK: metal.barrier
// CHECK: metal.return

// MSL: kernel void tg_smoke
// MSL: threadgroup float v{{[0-9]+}}[16];
// MSL: threadgroup_barrier(mem_flags::mem_threadgroup);
// MSL: return;
