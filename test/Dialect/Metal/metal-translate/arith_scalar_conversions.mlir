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
//   arith.fptosi f32->i32  -> int(...)
//   arith.sitofp i32->f32  -> float(...)
//   arith.truncf f32->f16  -> half(...)
//   arith.extf   f16->f32  -> float(...)
// The remaining ops in the same TypeSwitch Case (uitofp/fptoui/extsi/extui/
// trunci) share this exact emission path (typeToString(result)(operand)).

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
    metal.module_end
  }
}

// CHECK: kernel void conv_casts
// Numeric conversions emit a C-style cast `(T)(x)` (not functional `T(x)`, which
// vexing-parses as a variable decl when x is a bare identifier — see the ExtFOp
// et al. case in translateValue).
// CHECK-DAG: (int)(
// CHECK-DAG: (half)(
// CHECK-DAG: (float)(
// CHECK-NOT: Unexpected operation
