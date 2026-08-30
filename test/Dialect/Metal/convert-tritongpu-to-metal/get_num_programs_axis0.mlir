// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | FileCheck %s
// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | triton-metal-translate --mlir-to-msl | FileCheck %s --check-prefix=MSL
//
// tt.get_num_programs axes 0/1/2 must lower to the matching x/y/z component
// of [[threadgroups_per_grid]], with a ui32 -> i32 bridge for arith users.

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @get_num_programs_axis0(%out_ptr: !tt.ptr<f32>) {
    %n = tt.get_num_programs x : i32
    %f = arith.uitofp %n : i32 to f32
    tt.store %out_ptr, %f : !tt.ptr<f32>
    tt.return
  }

  tt.func public @get_num_programs_axis1(%out_ptr: !tt.ptr<f32>) {
    %n = tt.get_num_programs y : i32
    %f = arith.uitofp %n : i32 to f32
    tt.store %out_ptr, %f : !tt.ptr<f32>
    tt.return
  }

  tt.func public @get_num_programs_axis2(%out_ptr: !tt.ptr<f32>) {
    %n = tt.get_num_programs z : i32
    %f = arith.uitofp %n : i32 to f32
    tt.store %out_ptr, %f : !tt.ptr<f32>
    tt.return
  }
}
// CHECK-LABEL: get_num_programs_axis0
// CHECK: metal.threadgroups_per_grid "x"
// CHECK: builtin.unrealized_conversion_cast {{.*}} : ui32 to i32
// CHECK-LABEL: get_num_programs_axis1
// CHECK: metal.threadgroups_per_grid "y"
// CHECK: builtin.unrealized_conversion_cast {{.*}} : ui32 to i32
// CHECK-LABEL: get_num_programs_axis2
// CHECK: metal.threadgroups_per_grid "z"
// CHECK: builtin.unrealized_conversion_cast {{.*}} : ui32 to i32

// MSL-LABEL: kernel void get_num_programs_axis0(
// MSL: uint3 tgpg {{\[\[threadgroups_per_grid\]\]}}
// MSL: tgpg.x
// MSL-LABEL: kernel void get_num_programs_axis1(
// MSL: uint3 tgpg {{\[\[threadgroups_per_grid\]\]}}
// MSL: tgpg.y
// MSL-LABEL: kernel void get_num_programs_axis2(
// MSL: uint3 tgpg {{\[\[threadgroups_per_grid\]\]}}
// MSL: tgpg.z
