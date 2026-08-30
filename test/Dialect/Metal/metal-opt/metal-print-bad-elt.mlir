// RUN: triton-metal-opt --verify-diagnostics --split-input-file %s

// AC3 R-P1: element type must be one of f32/f16/bf16. i32 is rejected.

func.func @bad_elt(%mem: memref<?xi32>, %n: i64) {
  // expected-error @+1 {{metal.print requires f32/f16/bf16 element type}}
  metal.print %mem, %n : memref<?xi32>, i64
  return
}
