// RUN: not --crash triton-metal-opt --convert-tritongpu-to-metal -debug-only=dialect-conversion %s 2>&1 | FileCheck %s
//
// NOTE on `not --crash`: triton-metal-opt has a known cleanup-on-conversion-
// failure bug where the post-failure region teardown SIGSEGVs (see exit code
// 139 in the lit output). The bug is unrelated to Wall 11 — it occurs on any
// conversion failure, not specifically chain rejection. `not --crash` accepts
// the crash exit so we can FileCheck the diagnostic that fires BEFORE the
// crash.
//
// Wall 11 negative fixture: a rank-1 reduce whose producer chain contains
// `math.log` — an op outside the W11 walker whitelist (which only allows
// arith.addf/subf/mulf/divf, math.exp, tt.splat passthrough, with tt.load
// as terminator).
//
// `math.log` is chosen because it HAS a Metal lowering (MathLogLowering at
// TritonGPUToMetal.cpp:~1340 lowers it to metal.unary_exp logOp). That
// keeps the conversion driver from aborting on a totally-unconvertible op
// (which is what `arith.minnumf` did — no lowering at all, hard crash before
// the reduce walker fires). Here the driver successfully lowers math.log to
// metal.unary_exp; when ReduceLowering runs, the walker sees the post-log
// chain (metal.unary_exp or math.log depending on pattern order) and rejects.
// Either way the diagnostic substring `unsupported producer in elementwise
// chain:` is emitted.
//
// See .omc/specs/deep-interview-tutorial02-walls-9-to-13.md AC4.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @unsupported_chain(%x_ptr: !tt.ptr<f32>) {
    %offsets = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %x_splat = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %x_addr = tt.addptr %x_splat, %offsets : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    %x = tt.load %x_addr : tensor<1024x!tt.ptr<f32>, #blocked>
    // math.log: has a Metal lowering but is NOT in the W11 walker whitelist.
    %lg = math.log %x : tensor<1024xf32, #blocked>
    %r = "tt.reduce"(%lg) ({
    ^bb0(%a: f32, %b: f32):
      %s = arith.addf %a, %b : f32
      tt.reduce.return %s : f32
    }) {axis = 0 : i32} : (tensor<1024xf32, #blocked>) -> f32
    tt.return
  }
}

// CHECK: unsupported producer in elementwise chain:
