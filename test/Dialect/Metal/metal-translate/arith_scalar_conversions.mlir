// RUN: triton-metal-translate --mlir-to-msl %s 2>&1 | FileCheck %s
//
// Exercises scalar numeric conversions in `translateValue`
// (ModuleTranslation.cpp). Before this case, the conversion-op family hit the
// TypeSwitch `.Default` -> `llvm_unreachable("Unexpected operation")`, so any
// scalar `.to(tl.float32)` / `.to(tl.int32)` surviving to translation crashed
// the compiler (tensor conversions are folded into metal ops by the conversion
// pass; only scalar ones reach here, e.g. `program_id(0).to(tl.float32)`).
//
// Each conversion must emit an MSL constructor cast `T(x)` (mirroring
// metal.cast), with float->int using C truncation toward zero:
//   arith.fptosi f32->i32  -> (int32_t)(...)
//   arith.sitofp i32->f32  -> float(...)
//   arith.truncf f32->f16  -> half(...)
//   arith.extf   f16->f32  -> float(...)
// Float-to-integer result signedness comes from the operation, including
// when the MLIR result type is signless. Other numeric casts retain their
// existing input-signedness and result-width handling.

module {
  metal.module {
    metal.kernel conv_casts address_space_device [true, true] {
    ^bb0(%in: !metal.memref<? x f32>, %out: !metal.memref<? x f32>):
      %i0 = metal.constant 0 : ui32
      %v = metal.get_element %in[%i0] : (!metal.memref<? x f32>, ui32) -> f32
      %iv = arith.fptosi %v : f32 to i32
      %fv = arith.sitofp %iv : i32 to f32
      %hv = arith.truncf %fv : f32 to f16
      %wv = arith.extf %hv : f16 to f32
      metal.store %wv, %out[%i0] : f32, !metal.memref<? x f32>, ui32
      metal.return
    }
    metal.kernel float_to_integer_signedness address_space_device [false, true, true, true, true, true, true, true, true] {
    ^bb0(%in: !metal.memref<? x f32>, %out_s8: !metal.memref<? x ui8>, %out_s16: !metal.memref<? x ui16>, %out_s32: !metal.memref<? x ui32>, %out_s64: !metal.memref<? x ui64>, %out_u8: !metal.memref<? x ui8>, %out_u16: !metal.memref<? x ui16>, %out_u32: !metal.memref<? x ui32>, %out_u64: !metal.memref<? x ui64>):
      %i0 = metal.constant 0 : ui32
      %v = metal.get_element %in[%i0] : (!metal.memref<? x f32>, ui32) -> f32
      %s8 = arith.fptosi %v : f32 to i8
      %bits_s8 = builtin.unrealized_conversion_cast %s8 : i8 to ui8
      metal.store %bits_s8, %out_s8[%i0] : ui8, !metal.memref<? x ui8>, ui32
      %s16 = arith.fptosi %v : f32 to i16
      %bits_s16 = builtin.unrealized_conversion_cast %s16 : i16 to ui16
      metal.store %bits_s16, %out_s16[%i0] : ui16, !metal.memref<? x ui16>, ui32
      %s32 = arith.fptosi %v : f32 to i32
      %bits_s32 = builtin.unrealized_conversion_cast %s32 : i32 to ui32
      metal.store %bits_s32, %out_s32[%i0] : ui32, !metal.memref<? x ui32>, ui32
      %s64 = arith.fptosi %v : f32 to i64
      %bits_s64 = builtin.unrealized_conversion_cast %s64 : i64 to ui64
      metal.store %bits_s64, %out_s64[%i0] : ui64, !metal.memref<? x ui64>, ui32
      %u8 = arith.fptoui %v : f32 to i8
      %bits_u8 = builtin.unrealized_conversion_cast %u8 : i8 to ui8
      metal.store %bits_u8, %out_u8[%i0] : ui8, !metal.memref<? x ui8>, ui32
      %u16 = arith.fptoui %v : f32 to i16
      %bits_u16 = builtin.unrealized_conversion_cast %u16 : i16 to ui16
      metal.store %bits_u16, %out_u16[%i0] : ui16, !metal.memref<? x ui16>, ui32
      %u32 = arith.fptoui %v : f32 to i32
      %bits_u32 = builtin.unrealized_conversion_cast %u32 : i32 to ui32
      metal.store %bits_u32, %out_u32[%i0] : ui32, !metal.memref<? x ui32>, ui32
      %u64 = arith.fptoui %v : f32 to i64
      %bits_u64 = builtin.unrealized_conversion_cast %u64 : i64 to ui64
      metal.store %bits_u64, %out_u64[%i0] : ui64, !metal.memref<? x ui64>, ui32
      metal.return
    }
    metal.kernel mulhi_ui32 address_space_device [false, false, true] {
    ^bb0(%x_in: !metal.memref<? x ui32>, %y_in: !metal.memref<? x ui32>, %out: !metal.memref<? x ui32>):
      %i0 = metal.constant 0 : ui32
      %x = metal.get_element %x_in[%i0] : (!metal.memref<? x ui32>, ui32) -> ui32
      %y = metal.get_element %y_in[%i0] : (!metal.memref<? x ui32>, ui32) -> ui32
      %hi = metal.mulhi_ui %x, %y : (ui32, ui32) -> ui32
      metal.store %hi, %out[%i0] : ui32, !metal.memref<? x ui32>, ui32
      metal.return
    }
    metal.kernel mulhi_ui64 address_space_device [false, false, true] {
    ^bb0(%x_in: !metal.memref<? x ui64>, %y_in: !metal.memref<? x ui64>, %out: !metal.memref<? x ui64>):
      %i0 = metal.constant 0 : ui32
      %x = metal.get_element %x_in[%i0] : (!metal.memref<? x ui64>, ui32) -> ui64
      %y = metal.get_element %y_in[%i0] : (!metal.memref<? x ui64>, ui32) -> ui64
      %hi = metal.mulhi_ui %x, %y : (ui64, ui64) -> ui64
      metal.store %hi, %out[%i0] : ui64, !metal.memref<? x ui64>, ui32
      metal.return
    }
    metal.kernel cmpf_nan_predicates address_space_device [false, false, true, true, true, true, true, true, true, true, true, true] {
    ^bb0(%a_in: !metal.memref<? x f32>, %b_in: !metal.memref<? x f32>,
         %one_out: !metal.memref<? x i1>, %ueq_out: !metal.memref<? x i1>,
         %ugt_out: !metal.memref<? x i1>, %uge_out: !metal.memref<? x i1>,
         %ult_out: !metal.memref<? x i1>, %ule_out: !metal.memref<? x i1>,
         %ord_out: !metal.memref<? x i1>, %uno_out: !metal.memref<? x i1>,
         %true_out: !metal.memref<? x i1>, %false_out: !metal.memref<? x i1>):
      %i0 = metal.constant 0 : ui32
      %a = metal.get_element %a_in[%i0] : (!metal.memref<? x f32>, ui32) -> f32
      %b = metal.get_element %b_in[%i0] : (!metal.memref<? x f32>, ui32) -> f32
      %one = arith.cmpf one, %a, %b : f32
      %ueq = arith.cmpf ueq, %a, %b : f32
      %ugt = arith.cmpf ugt, %a, %b : f32
      %uge = arith.cmpf uge, %a, %b : f32
      %ult = arith.cmpf ult, %a, %b : f32
      %ule = arith.cmpf ule, %a, %b : f32
      %ord = arith.cmpf ord, %a, %b : f32
      %uno = arith.cmpf uno, %a, %b : f32
      %true = arith.cmpf true, %a, %b : f32
      %false = arith.cmpf false, %a, %b : f32
      metal.store %one, %one_out[%i0] : i1, !metal.memref<? x i1>, ui32
      metal.store %ueq, %ueq_out[%i0] : i1, !metal.memref<? x i1>, ui32
      metal.store %ugt, %ugt_out[%i0] : i1, !metal.memref<? x i1>, ui32
      metal.store %uge, %uge_out[%i0] : i1, !metal.memref<? x i1>, ui32
      metal.store %ult, %ult_out[%i0] : i1, !metal.memref<? x i1>, ui32
      metal.store %ule, %ule_out[%i0] : i1, !metal.memref<? x i1>, ui32
      metal.store %ord, %ord_out[%i0] : i1, !metal.memref<? x i1>, ui32
      metal.store %uno, %uno_out[%i0] : i1, !metal.memref<? x i1>, ui32
      metal.store %true, %true_out[%i0] : i1, !metal.memref<? x i1>, ui32
      metal.store %false, %false_out[%i0] : i1, !metal.memref<? x i1>, ui32
      metal.return
    }
    metal.module_end
  }
}

// CHECK: kernel void conv_casts
// Numeric conversions emit a C-style cast `(T)(x)` (not functional `T(x)`, which
// vexing-parses as a variable decl when x is a bare identifier — see the ExtFOp
// et al. case in translateValue).
// CHECK-DAG: (int32_t)(
// CHECK-DAG: (half)(
// CHECK-DAG: (float)(
// CHECK-NOT: Unexpected operation

// CHECK-LABEL: kernel void float_to_integer_signedness
// The same signless result width must use signed or unsigned MSL casts
// according to fptosi/fptoui, independently of unsigned buffer storage.
// CHECK: (int8_t)(
// CHECK: (int16_t)(
// CHECK: (int32_t)(
// CHECK: (int64_t)(
// CHECK: (uint8_t)(
// CHECK: (uint16_t)(
// CHECK: (uint32_t)(
// CHECK: (uint64_t)(

// CHECK-LABEL: kernel void mulhi_ui32
// CHECK: (uint32_t)((((uint64_t)(
// CHECK-SAME: *
// CHECK-SAME: >> 32u)

// CHECK-LABEL: kernel void mulhi_ui64
// Exact high-half multiplication uses four 32-bit limbs and uint64_t
// intermediates; no non-MSL 128-bit type is required.
// CHECK: v{{[0-9]+}}[0] =
// CHECK-SAME: >> 32u
// CHECK-SAME: *
// CHECK-SAME: >> 32u
// CHECK-SAME: +
// CHECK-SAME: & 0xfffffffful
// CHECK-SAME: >> 32u
// CHECK-SAME: +
// CHECK-SAME: & 0xfffffffful
// CHECK-SAME: >> 32u

// CHECK-LABEL: kernel void cmpf_nan_predicates
// Ordered-not-equal must reject NaNs; each unordered relation must accept a
// NaN in either operand. ORD/UNO and the constant predicates are explicit.
// CHECK: !metal::isnan(
// CHECK-SAME: && !metal::isnan(
// CHECK-SAME: &&
// CHECK-SAME: !=
// CHECK: metal::isnan(
// CHECK-SAME: || metal::isnan(
// CHECK-SAME: ||
// CHECK-SAME: ==
// CHECK: metal::isnan(
// CHECK-SAME: || metal::isnan(
// CHECK-SAME: ||
// CHECK-SAME: >
// CHECK: metal::isnan(
// CHECK-SAME: || metal::isnan(
// CHECK-SAME: ||
// CHECK-SAME: >=
// CHECK: metal::isnan(
// CHECK-SAME: || metal::isnan(
// CHECK-SAME: ||
// CHECK-SAME: <
// CHECK: metal::isnan(
// CHECK-SAME: || metal::isnan(
// CHECK-SAME: ||
// CHECK-SAME: <=
// CHECK: !metal::isnan(
// CHECK-SAME: && !metal::isnan(
// CHECK: metal::isnan(
// CHECK-SAME: || metal::isnan(
// CHECK: = true;
// CHECK: = false;
