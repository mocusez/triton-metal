// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | FileCheck %s --check-prefix=METAL
// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | triton-metal-translate --mlir-to-msl | FileCheck %s --check-prefix=MSL
//
// P0d regression coverage: supported integer Triton CAS lowers to first-class
// Metal CAS, including scalar and rank-1 blocked tensor forms.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @atomic_cas_scalar_i32(%out_ptr: !tt.ptr<i32>, %old_ptr: !tt.ptr<i32>, %cmp: i32, %value: i32) {
    %old = tt.atomic_cas acq_rel, gpu, %out_ptr, %cmp, %value : (!tt.ptr<i32>, i32, i32) -> i32
    tt.store %old_ptr, %old : !tt.ptr<i32>
    tt.return
  }

  tt.func public @atomic_cas_tensor_i32(%out_ptr: !tt.ptr<i32>, %old_ptr: !tt.ptr<i32>, %cmp: i32, %value: i32) {
    %r = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32, #blocked>
    %out = tt.splat %out_ptr : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>, #blocked>
    %addr = tt.addptr %out, %r : tensor<16x!tt.ptr<i32>, #blocked>, tensor<16xi32, #blocked>
    %cmps = tt.splat %cmp : i32 -> tensor<16xi32, #blocked>
    %values = tt.splat %value : i32 -> tensor<16xi32, #blocked>
    %old = tt.atomic_cas acq_rel, gpu, %addr, %cmps, %values : (tensor<16x!tt.ptr<i32>, #blocked>, tensor<16xi32, #blocked>, tensor<16xi32, #blocked>) -> tensor<16xi32, #blocked>
    %old_out = tt.splat %old_ptr : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>, #blocked>
    %old_addr = tt.addptr %old_out, %r : tensor<16x!tt.ptr<i32>, #blocked>, tensor<16xi32, #blocked>
    tt.store %old_addr, %old : tensor<16x!tt.ptr<i32>, #blocked>
    tt.return
  }

  tt.func public @atomic_cas_scalar_u32(%out_ptr: !tt.ptr<i32>, %cmp: i32, %value: i32) {
    %old = tt.atomic_cas acq_rel, gpu, %out_ptr, %cmp, %value : (!tt.ptr<i32>, i32, i32) -> i32
    tt.return
  }

  tt.func public @atomic_cas_scalar_f32(%out_ptr: !tt.ptr<f32>, %old_ptr: !tt.ptr<f32>, %cmp: f32, %value: f32) {
    %old = tt.atomic_cas acq_rel, gpu, %out_ptr, %cmp, %value : (!tt.ptr<f32>, f32, f32) -> f32
    tt.store %old_ptr, %old : !tt.ptr<f32>
    tt.return
  }

  tt.func public @atomic_cas_tensor_f32(%out_ptr: !tt.ptr<f32>, %old_ptr: !tt.ptr<f32>, %cmp: f32, %value: f32) {
    %r = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32, #blocked>
    %out = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>, #blocked>
    %addr = tt.addptr %out, %r : tensor<16x!tt.ptr<f32>, #blocked>, tensor<16xi32, #blocked>
    %cmps = tt.splat %cmp : f32 -> tensor<16xf32, #blocked>
    %values = tt.splat %value : f32 -> tensor<16xf32, #blocked>
    %old = tt.atomic_cas acq_rel, gpu, %addr, %cmps, %values : (tensor<16x!tt.ptr<f32>, #blocked>, tensor<16xf32, #blocked>, tensor<16xf32, #blocked>) -> tensor<16xf32, #blocked>
    %old_out = tt.splat %old_ptr : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>, #blocked>
    %old_addr = tt.addptr %old_out, %r : tensor<16x!tt.ptr<f32>, #blocked>, tensor<16xi32, #blocked>
    tt.store %old_addr, %old : tensor<16x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// METAL-LABEL: metal.kernel atomic_cas_scalar_i32
// METAL:       metal.threadgroup_alloca
// METAL:       arith.cmpi eq
// METAL:       scf.if
// METAL:       metal.atomic_cas {{.*}} : (ui32, ui32, !metal.memref<? x ui32>, ui32) -> ui32
// METAL:       metal.barrier
// METAL:       metal.tg_load_indexed
// METAL-NOT:   tt.atomic_cas

// METAL-LABEL: metal.kernel atomic_cas_tensor_i32
// METAL:       arith.cmpi slt
// METAL:       %[[OLD:.*]] = metal.atomic_cas {{.*}} : (ui32, ui32, !metal.memref<? x ui32>, ui32) -> ui32
// METAL:       scf.yield %[[OLD]] : ui32
// METAL:       builtin.unrealized_conversion_cast {{.*}} : ui32 to i32
// METAL-NOT:   tt.atomic_cas

// METAL-LABEL: metal.kernel atomic_cas_scalar_u32
// METAL:       metal.atomic_cas {{.*}} : (ui32, ui32, !metal.memref<? x ui32>, ui32) -> ui32
// METAL-NOT:   tt.atomic_cas

// METAL-LABEL: metal.kernel atomic_cas_scalar_f32
// METAL:       metal.threadgroup_alloca
// METAL:       metal.bitcast {{.*}} : (f32) -> ui32
// METAL:       metal.atomic_cas {{.*}} : (ui32, ui32, !metal.memref<? x f32>, ui32) -> ui32
// METAL:       metal.bitcast {{.*}} : (ui32) -> f32
// METAL-NOT:   tt.atomic_cas

// METAL-LABEL: metal.kernel atomic_cas_tensor_f32
// METAL:       metal.bitcast {{.*}} : (f32) -> ui32
// METAL:       %[[F32_RESULT:.*]] = scf.if {{.*}} -> (ui32)
// METAL:       metal.atomic_cas {{.*}} : (ui32, ui32, !metal.memref<? x f32>, ui32) -> ui32
// METAL:       metal.bitcast %[[F32_RESULT]] : (ui32) -> f32
// METAL-NOT:   tt.atomic_cas

// MSL-LABEL: kernel void atomic_cas_scalar_i32
// MSL:       atomic_compare_exchange

// MSL-LABEL: kernel void atomic_cas_tensor_i32
// MSL:       atomic_compare_exchange

// MSL-LABEL: kernel void atomic_cas_scalar_f32
// MSL:       atomic_compare_exchange_weak_explicit((device atomic_uint*)&

// MSL-LABEL: kernel void atomic_cas_tensor_f32
// MSL:       atomic_compare_exchange_weak_explicit((device atomic_uint*)&
