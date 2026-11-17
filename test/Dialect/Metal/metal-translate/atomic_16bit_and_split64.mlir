// RUN: triton-metal-translate --mlir-to-msl %s | FileCheck %s
//
// MSL has no 16-bit atomics and atomic_ulong has only void min/max. bf16 add
// and every 16-bit CAS run as a compare-exchange loop over the 32-bit word
// that contains the element, found from the element's ABSOLUTE address (a view
// may start two bytes into a word). A 64-bit add/and/or/xor with an unused old
// value is two 32-bit atomics on the low and high words.

module {
  metal.module {
    metal.kernel atomic_bf16_add address_space_device [false, true] {
    ^bb0(%in: !metal.memref<? x bf16>, %out: !metal.memref<? x bf16>):
      %id = metal.thread_id "x" : ui32
      %v = metal.get_element %in[%id] : (!metal.memref<? x bf16>, ui32) -> bf16
      %old = metal.atomic_rmw Add %v, %out[%id] : (bf16, !metal.memref<? x bf16>, ui32) -> bf16
      metal.store %old, %out[%id] : bf16, !metal.memref<? x bf16>, ui32
      metal.return
    }
    metal.kernel atomic_cas_16 address_space_device [false, true] {
    ^bb0(%in: !metal.memref<? x f16>, %out: !metal.memref<? x f16>):
      %id = metal.thread_id "x" : ui32
      %c = metal.get_element %in[%id] : (!metal.memref<? x f16>, ui32) -> f16
      %cmp = metal.bitcast %c : (f16) -> ui16
      %one = metal.constant 15360 : ui16
      %old = metal.atomic_cas %cmp, %one, %out[%id] : (ui16, ui16, !metal.memref<? x f16>, ui32) -> ui16
      %oldh = metal.bitcast %old : (ui16) -> f16
      metal.store %oldh, %out[%id] : f16, !metal.memref<? x f16>, ui32
      metal.return
    }
    metal.kernel atomic_split64 address_space_device [false, true] {
    ^bb0(%in: !metal.memref<? x si64>, %out: !metal.memref<? x si64>):
      %id = metal.thread_id "x" : ui32
      %v = metal.get_element %in[%id] : (!metal.memref<? x si64>, ui32) -> si64
      %a = metal.atomic_rmw Add %v, %out[%id] : (si64, !metal.memref<? x si64>, ui32) -> si64
      %x = metal.atomic_rmw Xor %v, %out[%id] : (si64, !metal.memref<? x si64>, ui32) -> si64
      metal.return
    }
    metal.module_end
  }
}

// The bf16 helper adds on the f32 bit pattern; `bfloat2` arithmetic in a loop
// crashes Apple's shader compiler service.
// CHECK-LABEL: static inline bfloat __triton_atomic_add_bfloat(device bfloat* base, uint idx, bfloat val) {
// CHECK: ulong addr = reinterpret_cast<ulong>(base + idx);
// CHECK: uint shift = uint(addr & 2ul) * 8u;
// CHECK-NOT: bfloat2
// CHECK: return as_type<bfloat>(ushort(prev));
// CHECK-LABEL: static inline ushort __triton_atomic_cas_16(ulong addr, ushort cmp, ushort val) {
// CHECK: reinterpret_cast<device atomic_uint*>(addr & ~3ul);
// CHECK: if (cur != cmp)
// CHECK: if (atomic_compare_exchange_weak_explicit(word, &old, desired, memory_order_relaxed, memory_order_relaxed))
// CHECK-LABEL: kernel void atomic_bf16_add
// CHECK: = __triton_atomic_add_bfloat(v1, id.x, v0[id.x]);
// CHECK-LABEL: kernel void atomic_cas_16
// CHECK: = __triton_atomic_cas_16(reinterpret_cast<ulong>(&v1[id.x]),
// CHECK-LABEL: kernel void atomic_split64
// CHECK: device atomic_uint* __w64_[[A:[0-9]+]] = reinterpret_cast<device atomic_uint*>(&v1[id.x]);
// CHECK: uint __l64_[[A]] = atomic_fetch_add_explicit(__w64_[[A]], uint(__b64_[[A]]), memory_order_relaxed);
// CHECK: atomic_fetch_add_explicit(__w64_[[A]] + 1, uint(__b64_[[A]] >> 32) + (__l64_[[A]] + uint(__b64_[[A]]) < __l64_[[A]] ? 1u : 0u), memory_order_relaxed);
// CHECK: atomic_fetch_xor_explicit(__w64_[[X:[0-9]+]], uint(__b64_[[X]]), memory_order_relaxed);
// CHECK: atomic_fetch_xor_explicit(__w64_[[X]] + 1, uint(__b64_[[X]] >> 32), memory_order_relaxed);
