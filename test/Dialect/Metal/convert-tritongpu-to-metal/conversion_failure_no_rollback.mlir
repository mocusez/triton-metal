// RUN: env MallocScribble=1 not triton-metal-opt --allow-unregistered-dialect --split-input-file --convert-tritongpu-to-metal %s 2>&1 | FileCheck %s --implicit-check-not=error:
//
// An op that no lowering accepts, reached inside `applyFullConversion` after
// `FuncOpLowering` has moved and spliced the body by hand. The driver used to
// roll the whole conversion back, which unwound the body move into a block
// `FuncOpLowering` had already freed: the process died (SIGSEGV / SIGABRT under
// MallocScribble) instead of reporting. The conversion now never rolls back: the
// first op is reported exactly as the driver reports it, the rest of the
// conversion is drained silently, and the pass fails. Pre-conversion checks
// still name every known refusal; this is the floor under the ones not yet
// known. MallocScribble makes a read of freed memory fail every time on macOS.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK: error: failed to legalize operation 'test.unknown'
  tt.func public @unknown_tensor_op(%x: !tt.ptr<f32>, %o: !tt.ptr<f32>) {
    %r = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32, #blocked>
    %xs = tt.splat %x : !tt.ptr<f32> -> tensor<512x!tt.ptr<f32>, #blocked>
    %xa = tt.addptr %xs, %r : tensor<512x!tt.ptr<f32>, #blocked>, tensor<512xi32, #blocked>
    %v = tt.load %xa : tensor<512x!tt.ptr<f32>, #blocked>
    %w = "test.unknown"(%v) : (tensor<512xf32, #blocked>) -> tensor<512xf32, #blocked>
    %os = tt.splat %o : !tt.ptr<f32> -> tensor<512x!tt.ptr<f32>, #blocked>
    %oa = tt.addptr %os, %r : tensor<512x!tt.ptr<f32>, #blocked>, tensor<512xi32, #blocked>
    tt.store %oa, %w : tensor<512x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

// Eight elements per thread, so `FuncOpLowering` also wraps the body in its
// register-band loop before the user loop is converted.
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK: error: failed to legalize operation 'test.unknown_in_loop'
  tt.func public @unknown_in_band_loop(%x: !tt.ptr<f32>, %o: !tt.ptr<f32>, %n: i32) {
    %r = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %xs = tt.splat %x : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %xa = tt.addptr %xs, %r : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    %v = tt.load %xa : tensor<1024x!tt.ptr<f32>, #blocked>
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %acc = scf.for %i = %c0 to %n step %c1 iter_args(%a = %v) -> (tensor<1024xf32, #blocked>) : i32 {
      %w = "test.unknown_in_loop"(%a) : (tensor<1024xf32, #blocked>) -> tensor<1024xf32, #blocked>
      scf.yield %w : tensor<1024xf32, #blocked>
    }
    %os = tt.splat %o : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %oa = tt.addptr %os, %r : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    tt.store %oa, %acc : tensor<1024x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

// A scalar op feeding later tensor work, and a result-less one after the
// store: only the first is reported.
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK: error: failed to legalize operation 'test.scalar'
  tt.func public @unknown_scalar_then_effect(%x: !tt.ptr<f32>, %o: !tt.ptr<f32>, %n: i32) {
    %r = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %xs = tt.splat %x : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %xa = tt.addptr %xs, %r : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    %v = tt.load %xa : tensor<128x!tt.ptr<f32>, #blocked>
    %m = "test.scalar"(%n) : (i32) -> i32
    %ms = tt.splat %m : i32 -> tensor<128xi32, #blocked>
    %oo = arith.addi %r, %ms : tensor<128xi32, #blocked>
    %os = tt.splat %o : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %oa = tt.addptr %os, %oo : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    tt.store %oa, %v : tensor<128x!tt.ptr<f32>, #blocked>
    "test.effect"() : () -> ()
    tt.return
  }
}
