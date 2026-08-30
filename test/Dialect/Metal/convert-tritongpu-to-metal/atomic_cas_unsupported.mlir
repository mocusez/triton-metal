// RUN: triton-metal-opt --convert-tritongpu-to-metal --verify-diagnostics --split-input-file %s
//
// Adjacent unsupported CAS forms must fail with deterministic diagnostics
// rather than asserting or segfaulting during conversion cleanup.

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @atomic_cas_scalar_i64(%out_ptr: !tt.ptr<i64>, %cmp: i64, %value: i64) {
    // expected-error @+1 {{Metal backend: atomic_cas requires 32-bit integer compare, value, and storage}}
    %old = tt.atomic_cas acq_rel, gpu, %out_ptr, %cmp, %value : (!tt.ptr<i64>, i64, i64) -> i64
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @atomic_cas_scalar_f32(%out_ptr: !tt.ptr<f32>, %cmp: f32, %value: f32) {
    // expected-error @+1 {{Metal backend: atomic_cas requires 32-bit integer compare, value, and storage}}
    %old = tt.atomic_cas acq_rel, gpu, %out_ptr, %cmp, %value : (!tt.ptr<f32>, f32, f32) -> f32
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @atomic_cas_tensor_i64(%out_ptr: !tt.ptr<i64>, %old_ptr: !tt.ptr<i64>, %cmp: i64, %value: i64) {
    %r = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32, #blocked>
    %out = tt.splat %out_ptr : !tt.ptr<i64> -> tensor<16x!tt.ptr<i64>, #blocked>
    %addr = tt.addptr %out, %r : tensor<16x!tt.ptr<i64>, #blocked>, tensor<16xi32, #blocked>
    %cmps = tt.splat %cmp : i64 -> tensor<16xi64, #blocked>
    %values = tt.splat %value : i64 -> tensor<16xi64, #blocked>
    // expected-error @+1 {{Metal backend: atomic_cas requires 32-bit integer compare, value, and storage}}
    %old = tt.atomic_cas acq_rel, gpu, %addr, %cmps, %values : (tensor<16x!tt.ptr<i64>, #blocked>, tensor<16xi64, #blocked>, tensor<16xi64, #blocked>) -> tensor<16xi64, #blocked>
    %old_out = tt.splat %old_ptr : !tt.ptr<i64> -> tensor<16x!tt.ptr<i64>, #blocked>
    %old_addr = tt.addptr %old_out, %r : tensor<16x!tt.ptr<i64>, #blocked>, tensor<16xi32, #blocked>
    tt.store %old_addr, %old : tensor<16x!tt.ptr<i64>, #blocked>
    tt.return
  }
}
