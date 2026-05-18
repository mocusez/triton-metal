// RUN: not triton-metal-translate --mlir-to-msl %s 2>&1 | FileCheck %s

module {
  metal.module {
    metal.kernel invalid address_space_device [true] {
    ^bb0(%arg0: !metal.memref<? x i1>):
      %c0 = metal.constant 0 : ui32
      %x = metal.get_element %arg0[%c0] : (!metal.memref<? x i1>, ui32) -> i1
      %bad = metal.unary_exp %x, sqrtOp : (i1) -> i1
      metal.return
    }
    metal.module_end
  }
}

// CHECK: error: 'metal.unary_exp' op argument type must be f32
