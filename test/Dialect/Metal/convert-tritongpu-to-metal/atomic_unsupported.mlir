// RUN: triton-metal-opt --convert-tritongpu-to-metal --verify-diagnostics --split-input-file %s
//
// Unsupported atomic forms are rejected before conversion mutates the module.
// Besides keeping the diagnostics stable, this guards against a failed partial
// conversion crashing while moved regions are torn down.

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @atomic_min_scalar_i64(%out_ptr: !tt.ptr<i64>, %v: i64) {
    // expected-error @+1 {{Metal backend: signed atomic min/max requires i32 payload and 32-bit storage}}
    %old = tt.atomic_rmw min, acq_rel, gpu, %out_ptr, %v : (!tt.ptr<i64>, i64) -> i64
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @atomic_add_scalar_i64(%out_ptr: !tt.ptr<i64>, %v: i64) {
    // expected-error @+1 {{Metal backend: 64-bit integer atomic add is unsupported because MSL has no atomic_ulong fetch-add}}
    %old = tt.atomic_rmw add, acq_rel, gpu, %out_ptr, %v : (!tt.ptr<i64>, i64) -> i64
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @atomic_and_scalar_i64(%out_ptr: !tt.ptr<i64>, %v: i64) {
    // expected-error @+1 {{Metal backend: bitwise atomic and/or/xor require a 32-bit integer payload; MSL has no 64-bit bitwise atomic}}
    %old = tt.atomic_rmw and, acq_rel, gpu, %out_ptr, %v : (!tt.ptr<i64>, i64) -> i64
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @atomic_xchg_scalar_i64(%out_ptr: !tt.ptr<i64>, %v: i64) {
    // expected-error @+1 {{Metal backend: 64-bit atomic exchange is unsupported because MSL has no atomic_ulong exchange}}
    %old = tt.atomic_rmw exch, acq_rel, gpu, %out_ptr, %v : (!tt.ptr<i64>, i64) -> i64
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @atomic_max_scalar_f64(%out_ptr: !tt.ptr<f64>, %v: f64) {
    %int_ptr = tt.bitcast %out_ptr : !tt.ptr<f64> -> !tt.ptr<i64>
    %bits = arith.bitcast %v : f64 to i64
    // expected-error @+1 {{Metal backend: f64 atomic min/max is unsupported because Metal has no signed i64 atomic min/max}}
    %old = tt.atomic_rmw umax, acq_rel, gpu, %int_ptr, %bits : (!tt.ptr<i64>, i64) -> i64
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @atomic_umin_tensor_u64_old_value(%out_ptr: !tt.ptr<i64>, %old_ptr: !tt.ptr<i64>, %v: i64) {
    %range = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32, #blocked>
    %out = tt.splat %out_ptr : !tt.ptr<i64> -> tensor<16x!tt.ptr<i64>, #blocked>
    %addr = tt.addptr %out, %range : tensor<16x!tt.ptr<i64>, #blocked>, tensor<16xi32, #blocked>
    %values = tt.splat %v : i64 -> tensor<16xi64, #blocked>
    // expected-error @+1 {{Metal backend: u64 atomic min/max does not support an old-value result}}
    %old = tt.atomic_rmw umin, acq_rel, gpu, %addr, %values : (tensor<16x!tt.ptr<i64>, #blocked>, tensor<16xi64, #blocked>) -> tensor<16xi64, #blocked>
    %old_out = tt.splat %old_ptr : !tt.ptr<i64> -> tensor<16x!tt.ptr<i64>, #blocked>
    %old_addr = tt.addptr %old_out, %range : tensor<16x!tt.ptr<i64>, #blocked>, tensor<16xi32, #blocked>
    tt.store %old_addr, %old : tensor<16x!tt.ptr<i64>, #blocked>
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @atomic_add_scalar_dynamic_mask(%out_ptr: !tt.ptr<i32>, %v: i32, %mask: i1) {
    // expected-error @+1 {{Metal backend: scalar atomic add/fadd requires a constant true mask}}
    %old = tt.atomic_rmw add, acq_rel, gpu, %out_ptr, %v, %mask : (!tt.ptr<i32>, i32, i1) -> i32
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @atomic_fadd_scalar_dynamic_mask(%out_ptr: !tt.ptr<f32>, %v: f32, %mask: i1) {
    // expected-error @+1 {{Metal backend: scalar atomic add/fadd requires a constant true mask}}
    %old = tt.atomic_rmw fadd, acq_rel, gpu, %out_ptr, %v, %mask : (!tt.ptr<f32>, f32, i1) -> f32
    tt.return
  }
}
