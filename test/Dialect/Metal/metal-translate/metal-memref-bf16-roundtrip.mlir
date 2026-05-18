// RUN: triton-metal-translate --mlir-to-msl %s | FileCheck %s

module {
  metal.module {
    metal.kernel memref_bf16_kernel address_space_device [false, true] {
    ^bb0(%arg0: !metal.memref<? x bf16>, %arg1: !metal.memref<? x bf16>):
      %id = metal.thread_id "x" : ui32
      %v = metal.get_element %arg0[%id] : (!metal.memref<? x bf16>, ui32) -> bf16
      metal.store %v, %arg1[%id] : bf16, !metal.memref<? x bf16>, ui32
      metal.return
    }
    metal.module_end
  }
}

// CHECK: kernel void memref_bf16_kernel(
// CHECK: constant bfloat *v0 {{\[\[buffer\(0\)\]\]}},
// CHECK: device bfloat *v1 {{\[\[buffer\(1\)\]\]}},
// CHECK: uint3 id {{\[\[thread_position_in_grid\]\]}})
// CHECK: v1[id.x] = v0[id.x];
