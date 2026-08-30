// RUN: not triton-metal-opt --convert-tritongpu-to-metal %s 2>&1 | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @clampf_f16_unsupported(%in_ptr: !tt.ptr<f16>, %out_ptr: !tt.ptr<f16>) {
    %offs = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked>
    %is = tt.splat %in_ptr : !tt.ptr<f16> -> tensor<256x!tt.ptr<f16>, #blocked>
    %ia = tt.addptr %is, %offs : tensor<256x!tt.ptr<f16>, #blocked>, tensor<256xi32, #blocked>
    %iv = tt.load %ia : tensor<256x!tt.ptr<f16>, #blocked>
    %lo = arith.constant dense<-1.000000e+00> : tensor<256xf16, #blocked>
    %hi = arith.constant dense<1.000000e+00> : tensor<256xf16, #blocked>
    %r = tt.clampf %iv, %lo, %hi, propagateNan = none : tensor<256xf16, #blocked>
    %os = tt.splat %out_ptr : !tt.ptr<f16> -> tensor<256x!tt.ptr<f16>, #blocked>
    %oa = tt.addptr %os, %offs : tensor<256x!tt.ptr<f16>, #blocked>, tensor<256xi32, #blocked>
    tt.store %oa, %r : tensor<256x!tt.ptr<f16>, #blocked>
    tt.return
  }
}

// CHECK: error: 'tt.clampf' op Metal backend: tt.clampf requires f32 operands and result
