// RUN: triton-metal-opt --convert-tritongpu-to-metal --verify-diagnostics --split-input-file %s
//
// Adjacent unsupported CAS forms must fail with deterministic diagnostics
// rather than asserting or segfaulting during conversion cleanup.

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @atomic_cas_scalar_i64(%out_ptr: !tt.ptr<i64>, %cmp: i64, %value: i64) {
    // expected-error @+1 {{Metal backend: 64-bit atomic_cas is unsupported because MSL has no atomic_ulong compare-exchange}}
    %old = tt.atomic_cas acq_rel, gpu, %out_ptr, %cmp, %value : (!tt.ptr<i64>, i64, i64) -> i64
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @atomic_cas_scalar_f16(%out_ptr: !tt.ptr<f16>, %cmp: f16, %value: f16) {
    // expected-error @+1 {{Metal backend: atomic_cas requires matching i32/u32 or f32 compare, value, and storage}}
    %old = tt.atomic_cas acq_rel, gpu, %out_ptr, %cmp, %value : (!tt.ptr<f16>, f16, f16) -> f16
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
    // expected-error @+1 {{Metal backend: 64-bit atomic_cas is unsupported because MSL has no atomic_ulong compare-exchange}}
    %old = tt.atomic_cas acq_rel, gpu, %addr, %cmps, %values : (tensor<16x!tt.ptr<i64>, #blocked>, tensor<16xi64, #blocked>, tensor<16xi64, #blocked>) -> tensor<16xi64, #blocked>
    %old_out = tt.splat %old_ptr : !tt.ptr<i64> -> tensor<16x!tt.ptr<i64>, #blocked>
    %old_addr = tt.addptr %old_out, %r : tensor<16x!tt.ptr<i64>, #blocked>, tensor<16xi32, #blocked>
    tt.store %old_addr, %old : tensor<16x!tt.ptr<i64>, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @atomic_cas_tensor_splat_ptr(%out_ptr: !tt.ptr<i32>, %cmp: i32, %value: i32) {
    %out = tt.splat %out_ptr : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>, #blocked>
    %cmps = tt.splat %cmp : i32 -> tensor<16xi32, #blocked>
    %values = tt.splat %value : i32 -> tensor<16xi32, #blocked>
    // expected-error @+1 {{Metal backend: atomic_cas tensor address must be a tt.addptr tile}}
    %old = tt.atomic_cas acq_rel, gpu, %out, %cmps, %values : (tensor<16x!tt.ptr<i32>, #blocked>, tensor<16xi32, #blocked>, tensor<16xi32, #blocked>) -> tensor<16xi32, #blocked>
    tt.return
  }
}
