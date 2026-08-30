// RUN: triton-metal-opt --convert-tritongpu-to-metal %s --split-input-file | FileCheck %s
//
// L3a-tileloop-compiler-A: scalar tt.load on bare !tt.ptr<f32>.
// Two shapes are covered: (a) bare load (offset 0, no addptr — Triton
// folds `addptr(p, 0)` away), and (b) addptr+load where the offset is a
// scalar i32 from a `tl.static_range` iter or similar.
// See the implementation notes.

// Case (a): bare scalar load (no addptr).
#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @scalar_load_bare(%kernel_ptr: !tt.ptr<f32>, %output_ptr: !tt.ptr<f32>) {
    %r = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %kv = tt.load %kernel_ptr : !tt.ptr<f32>
    %kv_t = tt.splat %kv : f32 -> tensor<1024xf32, #blocked>
    %o = tt.splat %output_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %oa = tt.addptr %o, %r : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    tt.store %oa, %kv_t : tensor<1024x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// CHECK: metal.kernel scalar_load_bare
// CHECK: arith.constant 0 : i32
// CHECK: builtin.unrealized_conversion_cast %{{.*}} : i32 to ui32
// CHECK: metal.get_element %arg0[%{{.*}}] : (!metal.memref<? x f32>, ui32) -> f32

// -----

// Case (b): addptr + scalar load (non-zero scalar offset).
#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @scalar_load_offset(%kernel_ptr: !tt.ptr<f32>, %output_ptr: !tt.ptr<f32>) {
    %c3 = arith.constant 3 : i32
    %r = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %addr = tt.addptr %kernel_ptr, %c3 : !tt.ptr<f32>, i32
    %kv = tt.load %addr : !tt.ptr<f32>
    %kv_t = tt.splat %kv : f32 -> tensor<1024xf32, #blocked>
    %o = tt.splat %output_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %oa = tt.addptr %o, %r : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    tt.store %oa, %kv_t : tensor<1024x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// CHECK: metal.kernel scalar_load_offset
// The scalar offset (3) is converted to ui32 and fed to get_element.
// CHECK: %{{.*}} = builtin.unrealized_conversion_cast %{{.*}} : i32 to ui32
// CHECK: metal.get_element %arg0[%{{.*}}] : (!metal.memref<? x f32>, ui32) -> f32
