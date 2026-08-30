// RUN: triton-metal-opt --verify-diagnostics --split-input-file %s

// AC3 R-P2: memref operand must be rank-1. memref<2x3xf32> is rejected.

func.func @bad_rank(%mem: memref<2x3xf32>, %n: i64) {
  // expected-error @+1 {{metal.print operand must be rank-1 memref}}
  metal.print %mem, %n : memref<2x3xf32>, i64
  return
}
