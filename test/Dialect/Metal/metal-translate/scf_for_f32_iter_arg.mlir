// RUN: triton-metal-translate --mlir-to-msl %s 2>&1 | FileCheck %s
//
// Wall 15 AC3: synthetic translator-side fixture verifying that
// `translate(scf::ForOp)` + `translate(scf::YieldOp)` emit a single-f32
// iter_arg accumulator as `float vN = init;` BEFORE the C-style for line
// and `vN = ...;` inside the body. Mirrors the rank-1 reduce shape that
// `lowerRank1Reduce` produces after the Wall 15 re-roll.
//
// See .omc/plans/tutorial02-wall15-iter-args-translator-consensus.md AC3.

module {
  metal.module {
    metal.kernel sum_for_iter_arg address_space_device [false] {
    ^bb0(%out: !metal.memref<? x f32>):
      %c0_ui = metal.constant 0 : ui32
      %c0_i32 = arith.constant 0 : i32
      %c1_i32 = arith.constant 1 : i32
      %c8_i32 = arith.constant 8 : i32
      %finit = metal.constant 0x00000000 : f32
      %acc = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32
                     iter_args(%a = %finit) -> (f32) : i32 {
        %one_f = metal.constant 0x3F800000 : f32
        %next = metal.binary_exp %a, %one_f, addOp : (f32, f32) -> f32
        scf.yield %next : f32
      }
      metal.store %acc, %out[%c0_ui] : f32, !metal.memref<? x f32>, ui32
      metal.return
    }

    // A consumer of one specific result of a multi-result loop must resolve
    // through that result's accumulator mapping. The ui32 storage bridge is
    // the exact shape used when MoE stores its loop-carried i32 top-k index.
    metal.kernel multi_result_for_cast address_space_device [false] {
    ^bb0(%out: !metal.memref<? x ui32>):
      %c0_ui = metal.constant 0 : ui32
      %c0_i32 = arith.constant 0 : i32
      %c1_i32 = arith.constant 1 : i32
      %c4_i32 = arith.constant 4 : i32
      %finit = metal.constant 0x00000000 : f32
      %result:3 = scf.for %iv = %c0_i32 to %c4_i32 step %c1_i32
          iter_args(%a = %finit, %b = %finit, %i = %c0_i32) -> (f32, f32, i32) : i32 {
        %next_i = arith.addi %i, %c1_i32 : i32
        scf.yield %a, %b, %next_i : f32, f32, i32
      }
      %stored = builtin.unrealized_conversion_cast %result#2 : i32 to ui32
      metal.store %stored, %out[%c0_ui] : ui32, !metal.memref<? x ui32>, ui32
      metal.return
    }
    metal.module_end
  }
}

// CHECK: kernel void sum_for_iter_arg(
// CHECK: float v{{[0-9]+}} = 0.0
// CHECK: for (int v{{[0-9]+}} = 0; v{{[0-9]+}} < 8; v{{[0-9]+}} += 1)
// CHECK: v{{[0-9]+}} =
// CHECK: return;

// CHECK-LABEL: kernel void multi_result_for_cast(
// CHECK: float v{{[0-9]+}} = 0.0
// CHECK: float v{{[0-9]+}} = 0.0
// CHECK: int v{{[0-9]+}} = 0
// CHECK: for (int v{{[0-9]+}} = 0; v{{[0-9]+}} < 4; v{{[0-9]+}} += 1)
// CHECK: v{{[0-9]+}} = v{{[0-9]+}}
// CHECK: return;
