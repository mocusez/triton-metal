// RUN: triton-metal-translate --mlir-to-msl %s 2>&1 | FileCheck %s
//
// Wall 16-A AC7: synthetic translator-side fixture verifying that the
// rank-2 row-scan inner reroll emits a single C `for` per chunk with an
// f32 iter_arg accumulator: `float vN = 0.0;` BEFORE the for line,
// `vN = ...;` inside the body, and NO nested `for`. Mirrors the per-chunk
// `scf.for` that `lowerRank2Reduce` produces after the Wall 16-A reroll
// (chunk_size = 16).
//
// Sibling of Wall 15's `scf_for_f32_iter_arg.mlir`; the only emission
// delta vs. that fixture is the loop-bound literal (16 here vs. 8 there)
// and the body reads a threadgroup buffer slot via metal.get_element.
//
// See .omc/plans/tutorial02-wall16a-rank2-rowscan-reroll-consensus.md AC7.

module {
  metal.module {
    metal.kernel rank2_rowscan_iter_arg address_space_device [true] {
    ^bb0(%out: !metal.memref<? x f32>):
      %c0_ui = metal.constant 0 : ui32
      %c0_i32 = arith.constant 0 : i32
      %c1_i32 = arith.constant 1 : i32
      %c16_i32 = arith.constant 16 : i32
      %finit = metal.constant 0x00000000 : f32
      %buf = metal.threadgroup_alloca : !metal.memref<16 x f32>
      %acc = scf.for %iv = %c0_i32 to %c16_i32 step %c1_i32
                     iter_args(%a = %finit) -> (f32) : i32 {
        %idx = builtin.unrealized_conversion_cast %iv : i32 to ui32
        %elt = metal.get_element %buf[%idx] : (!metal.memref<16 x f32>, ui32) -> f32
        %next = metal.binary_exp %a, %elt, addOp : (f32, f32) -> f32
        scf.yield %next : f32
      }
      metal.store %acc, %out[%c0_ui] : f32, !metal.memref<? x f32>, ui32
      metal.return
    }
    metal.module_end
  }
}

// CHECK: kernel void rank2_rowscan_iter_arg(
// CHECK: float v{{[0-9]+}} = 0.0
// CHECK: for (int v{{[0-9]+}} = 0; v{{[0-9]+}} < 16; v{{[0-9]+}} += 1)
// CHECK-NOT: for (int
// CHECK: v{{[0-9]+}} =
// CHECK: return;
