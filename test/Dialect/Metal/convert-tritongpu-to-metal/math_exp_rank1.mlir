// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | FileCheck %s
//
// Wall 10 regression-guard: rank-1 math.exp lowering via MathExpLowering
// (TritonGPUToMetal.cpp:1307, already present). Verifies the existing
// op-generic pattern accepts rank-1 f32 tensors and lowers to
// `metal.unary_exp ..., expOp`.
//
// Architect F7: MathExpLowering has no rank precondition; this fixture is
// expected to PASS on the unmodified tree, serving as a regression-guard
// rather than a verify-first probe.
// See the implementation notes AC3.

#blocked = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @exp_kernel_spt8_unmasked(%x_ptr: !tt.ptr<f32>, %output_ptr: !tt.ptr<f32>) {
    %offsets = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %x_splat = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %x_addr = tt.addptr %x_splat, %offsets : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    %x_val = tt.load %x_addr : tensor<1024x!tt.ptr<f32>, #blocked>
    %expv = math.exp %x_val : tensor<1024xf32, #blocked>
    %o_splat = tt.splat %output_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %o_addr = tt.addptr %o_splat, %offsets : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    tt.store %o_addr, %expv : tensor<1024x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// CHECK-LABEL: metal.kernel exp_kernel_spt8_unmasked
// CHECK: metal.get_element %arg0
// CHECK: metal.unary_exp {{.*}}, expOp
// CHECK: metal.store
// CHECK: metal.return
