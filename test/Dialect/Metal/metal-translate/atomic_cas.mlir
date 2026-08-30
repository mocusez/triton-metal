// RUN: triton-metal-translate --mlir-to-msl %s | FileCheck %s
//
// P0d translation coverage: Metal CAS emits MSL compare-exchange and returns
// Triton's old value through an explicit expected temporary.

module {
  metal.module {
    metal.kernel atomic_cas_i32 address_space_device [false, true] {
    ^bb0(%in: !metal.memref<? x si32>, %out: !metal.memref<? x si32>):
      %id = metal.thread_id "x" : ui32
      %cmp = metal.get_element %in[%id] : (!metal.memref<? x si32>, ui32) -> si32
      %one = metal.constant 1 : si32
      %old = metal.atomic_cas %cmp, %one, %out[%id] : (si32, si32, !metal.memref<? x si32>, ui32) -> si32
      metal.store %old, %out[%id] : si32, !metal.memref<? x si32>, ui32
      metal.return
    }
    metal.module_end
  }
}

// CHECK: kernel void atomic_cas_i32
// CHECK: int32_t v[[EXPECTED:[0-9]+]] = v0[id.x];
// MSL documents weak compare-exchange only. Retry while the failure was
// spurious (`expected` still equals cmp), which recovers strong CAS semantics.
// CHECK: while (!atomic_compare_exchange_weak_explicit((device atomic_int*)&v1[id.x], &v[[EXPECTED]], 1, memory_order_relaxed, memory_order_relaxed) && v[[EXPECTED]] == v0[id.x]) {};
// CHECK: v1[id.x] = v[[EXPECTED]];
