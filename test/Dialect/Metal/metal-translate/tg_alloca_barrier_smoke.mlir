// RUN: triton-metal-opt %s | FileCheck %s
// RUN: triton-metal-opt %s | triton-metal-translate --mlir-to-msl | FileCheck %s --check-prefix=MSL
//
// Session L3 (the implementation notes):
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
    // The other monoid: `combine mulOp` is a cumprod, so the carry, the
    // Hillis-Steele neighbour and the fold all switch to the multiplicative
    // identity and operator. Padding a cumprod with the add identity 0 instead
    // would zero every prefix.
    metal.kernel tg_cumprod_smoke address_space_device [true] {
    ^bb0(%arg0: !metal.memref<? x f32>):
      %in = metal.threadgroup_alloca : !metal.memref<256 x f32>
      %out = metal.threadgroup_alloca : !metal.memref<256 x f32>
      metal.threadgroup_prefix_sum %in, %out combine mulOp {block = 256 : i64, tpb = 128 : i64} : !metal.memref<256 x f32>, !metal.memref<256 x f32>
      metal.threadgroup_prefix_sum %in, %out combine mulOp {block = 256 : i64, reverse, tpb = 128 : i64} : !metal.memref<256 x f32>, !metal.memref<256 x f32>
      metal.return
    }
    // Reverse affine scan: the combine is NOT commutative, so each chunk is
    // mirrored in place (barriered on both sides), scanned with the verbatim
    // forward template, then mirrored back onto its original positions.
    metal.kernel tg_affine_reverse_smoke address_space_device [true] {
    ^bb0(%arg0: !metal.memref<? x f32>):
      %a = metal.threadgroup_alloca : !metal.memref<256 x f32>
      %x = metal.threadgroup_alloca : !metal.memref<256 x f32>
      metal.threadgroup_affine_prefix_scan %a, %x {block = 256 : i64, reverse, tpb = 128 : i64} : !metal.memref<256 x f32>, !metal.memref<256 x f32>
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
// The mirrored read is CROSS-THREAD — thread `t` reads the slot thread
// `tpb-1-t` filled — so it must be barriered from the fill that precedes it.
// The forward path reads only the slot the same thread wrote and needs none.
// MSL: threadgroup_barrier(mem_flags::mem_threadgroup);
// MSL: v{{[0-9]+}}[_ps_base + _ps_tid] = v{{[0-9]+}}[_ps_orig];
// MSL: float _ps_total =
// MSL: _ps_carry += _ps_total;
// MSL: float _ps_result =
// MSL: [_ps_orig] = _ps_result;

// MSL: kernel void tg_cumprod_smoke
// MSL: float _ps_carry = 1.0f;
// MSL: uint _ps_base = _ps_k * 128u;
// MSL: float _ps_add = (_ps_tid >= _ps_off) ? v{{[0-9]+}}[_ps_base + _ps_tid - _ps_off] : 1.0f;
// MSL: v{{[0-9]+}}[_ps_base + _ps_tid] *= _ps_add;
// MSL: v{{[0-9]+}}[_ps_base + _ps_tid] *= _ps_carry;
// MSL: _ps_carry *= _ps_total;
// Reverse cumprod: mirrored indices AND the multiplicative identity/operator.
// MSL: float _ps_carry = 1.0f;
// MSL: uint _ps_base = 256u - (_ps_k + 1u) * 128u;
// MSL: uint _ps_orig = _ps_base + 127u - _ps_tid;
// MSL: _ps_carry *= _ps_total;
// MSL: v{{[0-9]+}}[_ps_orig] = _ps_result;

// MSL: kernel void tg_affine_reverse_smoke
// MSL: uint _aps_base = 256u - (_aps_k + 1u) * 128u;
// MSL: uint _aps_orig = _aps_base + 127u - _aps_tid;
// Mirror in: read both buffers at the mirrored slot, barrier, write back to the
// forward slot. Read-then-barrier-then-write is what makes an in-place
// cross-thread permutation safe.
// MSL: float _aps_mir_a = v{{[0-9]+}}[_aps_orig];
// MSL: float _aps_mir_x = v{{[0-9]+}}[_aps_orig];
// MSL: threadgroup_barrier(mem_flags::mem_threadgroup);
// MSL: v{{[0-9]+}}[_aps_base + _aps_tid] = _aps_mir_a;
// MSL: v{{[0-9]+}}[_aps_base + _aps_tid] = _aps_mir_x;
// The forward chunk template is reused verbatim between the two mirrors.
// MSL: _aps_carry_a *= _aps_total_a;
// Mirror out, so results land back on their original logical positions.
// MSL: float _aps_out_a = v{{[0-9]+}}[_aps_base + _aps_tid];
// MSL: float _aps_out_x = v{{[0-9]+}}[_aps_base + _aps_tid];
// MSL: threadgroup_barrier(mem_flags::mem_threadgroup);
// MSL: v{{[0-9]+}}[_aps_orig] = _aps_out_a;
// MSL: v{{[0-9]+}}[_aps_orig] = _aps_out_x;
// MSL: return;
