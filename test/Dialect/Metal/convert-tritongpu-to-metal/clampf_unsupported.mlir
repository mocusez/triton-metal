// RUN: not triton-metal-opt --convert-tritongpu-to-metal %s 2>&1 | FileCheck %s
//
// `tt.clampf` covers f32, f16 and bf16 — the widths MSL's `fmin`/`fmax` can
// carry. f64 is not a Metal type at all, so it must be named before conversion
// rather than left to fail-conversion teardown. The f16 case that used to live
// here is now a positive in `elementwise_transcendental.mlir`.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @clampf_f64_unsupported(%in_ptr: !tt.ptr<f64>, %out_ptr: !tt.ptr<f64>) {
    %offs = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked>
    %is = tt.splat %in_ptr : !tt.ptr<f64> -> tensor<256x!tt.ptr<f64>, #blocked>
    %ia = tt.addptr %is, %offs : tensor<256x!tt.ptr<f64>, #blocked>, tensor<256xi32, #blocked>
    %iv = tt.load %ia : tensor<256x!tt.ptr<f64>, #blocked>
    %lo = arith.constant dense<-1.000000e+00> : tensor<256xf64, #blocked>
    %hi = arith.constant dense<1.000000e+00> : tensor<256xf64, #blocked>
    %r = tt.clampf %iv, %lo, %hi, propagateNan = none : tensor<256xf64, #blocked>
    %os = tt.splat %out_ptr : !tt.ptr<f64> -> tensor<256x!tt.ptr<f64>, #blocked>
    %oa = tt.addptr %os, %offs : tensor<256x!tt.ptr<f64>, #blocked>, tensor<256xi32, #blocked>
    tt.store %oa, %r : tensor<256x!tt.ptr<f64>, #blocked>
    tt.return
  }
}

// CHECK: error: 'tt.clampf' op Metal backend: tt.clampf requires f32, f16 or bf16 operands and result
