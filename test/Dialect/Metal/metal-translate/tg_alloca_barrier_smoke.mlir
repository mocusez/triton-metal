// RUN: triton-metal-opt %s | FileCheck %s
// RUN: triton-metal-opt %s | triton-metal-translate --mlir-to-msl | FileCheck %s --check-prefix=MSL
//
// Session L3 (`.omc/specs/deep-interview-leet-triton-l3-reduce-axis-2d.md`):
// Smoke test for threadgroup allocation/barriers and the reverse prefix-sum
// emitter. Reverse scan walks chunks from the end, scans reversed lane order,
// then scatters each result back to its original logical index.

module {
  metal.module {
    metal.kernel tg_smoke address_space_device [true] {
    ^bb0(%arg0: !metal.memref<? x f32>):
      %0 = metal.constant 0 : ui32
      %tg = metal.threadgroup_alloca : !metal.memref<16 x f32>
      %scan_in = metal.threadgroup_alloca : !metal.memref<1024 x f32>
      %scan_out = metal.threadgroup_alloca : !metal.memref<1024 x f32>
      %v = metal.get_element %arg0[%0] : (!metal.memref<? x f32>, ui32) -> f32
      metal.store %v, %tg[%0] : f32, !metal.memref<16 x f32>, ui32
      metal.barrier
      metal.threadgroup_prefix_sum %scan_in, %scan_out {block = 1024 : i64, reverse, tpb = 128 : i64} : !metal.memref<1024 x f32>, !metal.memref<1024 x f32>
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
// CHECK: metal.threadgroup_prefix_sum
// CHECK-SAME: reverse
// CHECK: metal.return

// MSL: kernel void tg_smoke
// MSL: threadgroup float v{{[0-9]+}}[16];
// MSL: threadgroup_barrier(mem_flags::mem_threadgroup);
// MSL: float _ps_carry = 0.0f;
// MSL: for (uint _ps_k = 0u; _ps_k < 8u; ++_ps_k) {
// MSL: uint _ps_base = 1024u - (_ps_k + 1u) * 128u;
// MSL: uint _ps_orig = _ps_base + 127u - _ps_tid;
// MSL: float _ps_total =
// MSL: _ps_carry += _ps_total;
// MSL: float _ps_result =
// MSL: [_ps_orig] = _ps_result;
// MSL: return;
