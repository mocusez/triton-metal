// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | FileCheck %s
//
// AC3 MVP-1 smoke per `.omc/specs/deep-interview-ac3-half-slice.md`:
// proves the generic-ConversionPattern + TypeConverter shell works
// end-to-end on the FuncOp + ReturnOp patterns. The remaining 7 patterns
// (get_program_id, make_range, splat, addptr+load, addf, store) plus the
// captured-fixture run land in the next session.

module {
  tt.func public @add_kernel(%x_ptr: !tt.ptr<f32>, %y_ptr: !tt.ptr<f32>, %output_ptr: !tt.ptr<f32>) {
    tt.return
  }
}

// CHECK: metal.module
// CHECK: metal.kernel add_kernel address_space_device [true, true, true]
// CHECK: !metal.memref<? x f32>
// CHECK: metal.return
// CHECK: metal.module_end
