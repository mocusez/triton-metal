// RUN: triton-metal-translate --mlir-to-msl %s | FileCheck %s
//
// P0c translation coverage for direct MSL spellings. The FMA check requires a
// fused call and must not pass if the emitter splits it into multiply plus add.

module {
  metal.module {
    metal.kernel direct_math_ops address_space_device [false, false, false, true, true, true] {
    ^bb0(%a_buf: !metal.memref<? x f32>, %b_buf: !metal.memref<? x f32>, %c_buf: !metal.memref<? x f32>, %out: !metal.memref<? x f32>, %clamp_none_out: !metal.memref<? x f32>, %clamp_all_out: !metal.memref<? x f32>):
      %id = metal.thread_id "x" : ui32
      %a = metal.get_element %a_buf[%id] : (!metal.memref<? x f32>, ui32) -> f32
      %b = metal.get_element %b_buf[%id] : (!metal.memref<? x f32>, ui32) -> f32
      %c = metal.get_element %c_buf[%id] : (!metal.memref<? x f32>, ui32) -> f32
      %l2 = metal.unary_exp %a, log2Op : (f32) -> f32
      %abs = metal.unary_exp %b, absOp : (f32) -> f32
      %f = metal.fma %l2, %abs, %c : (f32, f32, f32) -> f32
      %clamp_none = metal.clampf %a, %b, %c, propagate_nan = false : (f32, f32, f32) -> f32
      %clamp_all = metal.clampf %a, %b, %c, propagate_nan = true : (f32, f32, f32) -> f32
      metal.store %f, %out[%id] : f32, !metal.memref<? x f32>, ui32
      metal.store %clamp_none, %clamp_none_out[%id] : f32, !metal.memref<? x f32>, ui32
      metal.store %clamp_all, %clamp_all_out[%id] : f32, !metal.memref<? x f32>, ui32
      metal.return
    }
    // IEEE-754 maximum/minimum (`tl.maximum`/`tl.minimum` with
    // `propagate_nan=tl.PropagateNan.ALL`) against the number-preferring
    // maxOp/minOp that the default policy uses. MSL's own `max`/`min` are
    // number-preferring, so only the first pair may reach the helper.
    metal.kernel nan_propagating_extrema address_space_device [false, false, true, true, true, true] {
    ^bb0(%x_buf: !metal.memref<? x f32>, %y_buf: !metal.memref<? x f32>, %max_nan_out: !metal.memref<? x f32>, %min_nan_out: !metal.memref<? x f32>, %max_num_out: !metal.memref<? x f32>, %min_num_out: !metal.memref<? x f32>):
      %tid = metal.thread_id "x" : ui32
      %x = metal.get_element %x_buf[%tid] : (!metal.memref<? x f32>, ui32) -> f32
      %y = metal.get_element %y_buf[%tid] : (!metal.memref<? x f32>, ui32) -> f32
      %max_nan = metal.binary_exp %x, %y, maxNanOp : (f32, f32) -> f32
      %min_nan = metal.binary_exp %x, %y, minNanOp : (f32, f32) -> f32
      %max_num = metal.binary_exp %x, %y, maxOp : (f32, f32) -> f32
      %min_num = metal.binary_exp %x, %y, minOp : (f32, f32) -> f32
      metal.store %max_nan, %max_nan_out[%tid] : f32, !metal.memref<? x f32>, ui32
      metal.store %min_nan, %min_nan_out[%tid] : f32, !metal.memref<? x f32>, ui32
      metal.store %max_num, %max_num_out[%tid] : f32, !metal.memref<? x f32>, ui32
      metal.store %min_num, %min_num_out[%tid] : f32, !metal.memref<? x f32>, ui32
      metal.return
    }
    // Narrow-width clamp. MSL's fmin/fmax carry `half` natively but have no
    // `bfloat` overload: both operands promote to `float` and the result will
    // not implicitly convert back, so the bf16 form must be wrapped.
    metal.kernel clampf_narrow_widths address_space_device [false, true, false, true] {
    ^bb0(%h_in: !metal.memref<? x f16>, %h_out: !metal.memref<? x f16>, %b_in: !metal.memref<? x bf16>, %b_out: !metal.memref<? x bf16>):
      %tid = metal.thread_id "x" : ui32
      %hv = metal.get_element %h_in[%tid] : (!metal.memref<? x f16>, ui32) -> f16
      %hlo = metal.constant -1.000000e+00 : f16
      %hhi = metal.constant 1.000000e+00 : f16
      %hc = metal.clampf %hv, %hlo, %hhi, propagate_nan = true : (f16, f16, f16) -> f16
      metal.store %hc, %h_out[%tid] : f16, !metal.memref<? x f16>, ui32
      %bv = metal.get_element %b_in[%tid] : (!metal.memref<? x bf16>, ui32) -> bf16
      %blo = metal.constant -1.000000e+00 : bf16
      %bhi = metal.constant 1.000000e+00 : bf16
      %bc = metal.clampf %bv, %blo, %bhi, propagate_nan = false : (bf16, bf16, bf16) -> bf16
      metal.store %bc, %b_out[%tid] : bf16, !metal.memref<? x bf16>, ui32
      metal.return
    }

    metal.kernel powi_helper address_space_device [false, false, true] {
    ^bb0(%x_buf: !metal.memref<? x f32>, %n_buf: !metal.memref<? x si32>, %out: !metal.memref<? x f32>):
      %tid = metal.thread_id "x" : ui32
      %x = metal.get_element %x_buf[%tid] : (!metal.memref<? x f32>, ui32) -> f32
      %n = metal.get_element %n_buf[%tid] : (!metal.memref<? x si32>, ui32) -> si32
      %p = metal.math_intrinsic "__triton_powi"(%x, %n) : (f32, si32) -> f32
      metal.store %p, %out[%tid] : f32, !metal.memref<? x f32>, ui32
      metal.return
    }
    // A prefix operator is the one context that does not bracket the expression
    // it embeds, so it is the one that can be broken by `metal.binary_exp`
    // bracketing only its operands. Both spellings of the prefix minus are
    // pinned, and each sum is used once so the translator inlines it rather
    // than naming a temp (a named temp would hide the defect).
    metal.kernel prefix_minus_over_binary address_space_device [false, false, true, true] {
    ^bb0(%a_buf: !metal.memref<? x f32>, %b_buf: !metal.memref<? x f32>, %neg_out: !metal.memref<? x f32>, %minus_out: !metal.memref<? x f32>):
      %tid = metal.thread_id "x" : ui32
      %a = metal.get_element %a_buf[%tid] : (!metal.memref<? x f32>, ui32) -> f32
      %b = metal.get_element %b_buf[%tid] : (!metal.memref<? x f32>, ui32) -> f32
      %diff = metal.binary_exp %a, %b, subOp : (f32, f32) -> f32
      %negated = arith.negf %diff : f32
      metal.store %negated, %neg_out[%tid] : f32, !metal.memref<? x f32>, ui32
      %sum = metal.binary_exp %a, %b, addOp : (f32, f32) -> f32
      %minus = metal.unary_exp %sum, minusOp : (f32) -> f32
      metal.store %minus, %minus_out[%tid] : f32, !metal.memref<? x f32>, ui32
      metal.return
    }
    metal.module_end
  }
}


// CHECK: static inline float __triton_powi(float x, int n)
// CHECK: if (n == 2) return x * x
// CHECK: uint bits = as_type<uint>(x)
// CHECK: uint e = n < 0 ? 0u - uint(n) : uint(n)
// CHECK: while (e != 0u)
// CHECK: if (n < 0) acc = 1.0f / acc
// CHECK: uint sign = (uint(n) & 1u) ? bits & 0x80000000u : 0u
// CHECK: return as_type<float>(as_type<uint>(acc) | sign)

// The two helpers are defined once, in the preamble, ahead of every kernel.
// Both bodies are pinned, not just their signatures: the NaN guard, and the
// signed-zero branch, which is the half a reader cannot check by inspection and
// the only place the two helpers differ. IEEE-754 maximum returns the POSITIVE
// zero and minimum the NEGATIVE one, so the ternaries must not be exchanged.
// CHECK: static inline T __triton_maximumf(T a, T b)
// CHECK: metal::isnan(a)
// CHECK: metal::isnan(b)
// CHECK: a == b) return metal::signbit(a) ? b : a
// CHECK: return a < b ? b : a
// CHECK: static inline T __triton_minimumf(T a, T b)
// CHECK: metal::isnan(a)
// CHECK: metal::isnan(b)
// CHECK: a == b) return metal::signbit(a) ? a : b
// CHECK: return b < a ? b : a
// CHECK: kernel void direct_math_ops
// CHECK: metal::precise::fma(metal::precise::log2(
// CHECK-SAME: metal::fabs(
// CHECK-NOT: *{{.*}}+
// CHECK: metal::fmin(metal::fmax(
// CHECK: metal::isnan(
// CHECK-SAME: ?
// CHECK-SAME: metal::fmin(metal::fmax(

// CHECK: kernel void nan_propagating_extrema
// The ALL policy calls the helper; the default policy keeps the MSL builtin.
// CHECK-DAG: __triton_maximumf(float(
// CHECK-DAG: __triton_minimumf(float(
// CHECK-DAG: = max(float(
// CHECK-DAG: = min(float(

// CHECK: kernel void clampf_narrow_widths
// half rides the native overload, unwrapped, exactly like f32.
// CHECK: metal::isnan(
// CHECK-SAME: metal::fmin(metal::fmax(
// CHECK-NOT: bfloat(metal::fmin
// bfloat must convert the promoted float result back.
// CHECK: bfloat(metal::fmin(metal::fmax(

// CHECK: kernel void powi_helper
// CHECK: constant int32_t *
// CHECK: __triton_powi({{.*}}, {{.*}})

// CHECK: kernel void prefix_minus_over_binary
// Both prefix spellings must bracket the WHOLE operand. Binding the minus to
// the left operand alone turns `-(a - b)` into `-a - b`, which is a wrong
// answer for every sum and difference and right by accident for products.
// CHECK: = (-((v{{[0-9]+}}) - (v{{[0-9]+}})));
// CHECK: = (-((v{{[0-9]+}}) + (v{{[0-9]+}})));
