// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | FileCheck %s
// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | triton-metal-translate --mlir-to-msl | FileCheck %s --check-prefix=MSL
//
// Scalar INTEGER `tl.atomic_add(ptr, i32)` — `tt.atomic_rmw add` (RMWOp::ADD),
// not `fadd`. `AtomicRmwLowering` used to accept FADD/f32 only, so
// `tl.atomic_add(out, tl.sum(x == K))` (whose reduce result is i32) failed to
// legalize: the second wall in leet-triton/medium-count_array_element.py.
//
// The payload must enter `metal.atomic_rmw` as the memref's ui32 STORAGE type,
// not as signless i32 — `Metal_Type` (MetalOps.td:17) admits ui32 but not
// signless i32, so an i32 operand fails the op's own verifier. The emitter then
// picks `atomic_uint` over `atomic_float` from that element type.
//
// The tensor cases below additionally pin signed/unsigned kind selection,
// sub-threadgroup lane guards, old-value materialization and the native
// void-returning atomic_ulong min/max surface.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @atomic_add_scalar_i32(%out_ptr: !tt.ptr<i32>, %v: i32) {
    %r = tt.atomic_rmw add, acq_rel, gpu, %out_ptr, %v : (!tt.ptr<i32>, i32) -> i32
    tt.return
  }

  tt.func public @atomic_min_scalar_i32(%out_ptr: !tt.ptr<i32>, %v: i32) {
    %r = tt.atomic_rmw min, acq_rel, gpu, %out_ptr, %v : (!tt.ptr<i32>, i32) -> i32
    tt.return
  }

  tt.func public @atomic_minmax_tensor_i32(%out_ptr: !tt.ptr<i32>, %v: i32) {
    %r = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32, #blocked>
    %out = tt.splat %out_ptr : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>, #blocked>
    %addr = tt.addptr %out, %r : tensor<16x!tt.ptr<i32>, #blocked>, tensor<16xi32, #blocked>
    %values = tt.splat %v : i32 -> tensor<16xi32, #blocked>
    %min = tt.atomic_rmw min, acq_rel, gpu, %addr, %values : (tensor<16x!tt.ptr<i32>, #blocked>, tensor<16xi32, #blocked>) -> tensor<16xi32, #blocked>
    %max = tt.atomic_rmw max, acq_rel, gpu, %addr, %values : (tensor<16x!tt.ptr<i32>, #blocked>, tensor<16xi32, #blocked>) -> tensor<16xi32, #blocked>
    %umin = tt.atomic_rmw umin, acq_rel, gpu, %addr, %values : (tensor<16x!tt.ptr<i32>, #blocked>, tensor<16xi32, #blocked>) -> tensor<16xi32, #blocked>
    %umax = tt.atomic_rmw umax, acq_rel, gpu, %addr, %values : (tensor<16x!tt.ptr<i32>, #blocked>, tensor<16xi32, #blocked>) -> tensor<16xi32, #blocked>
    tt.return
  }

  tt.func public @atomic_min_tensor_old_i32(%out_ptr: !tt.ptr<i32>, %old_ptr: !tt.ptr<i32>, %v: i32) {
    %r = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32, #blocked>
    %out = tt.splat %out_ptr : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>, #blocked>
    %addr = tt.addptr %out, %r : tensor<128x!tt.ptr<i32>, #blocked>, tensor<128xi32, #blocked>
    %values = tt.splat %v : i32 -> tensor<128xi32, #blocked>
    %old = tt.atomic_rmw min, acq_rel, gpu, %addr, %values : (tensor<128x!tt.ptr<i32>, #blocked>, tensor<128xi32, #blocked>) -> tensor<128xi32, #blocked>
    %old_out = tt.splat %old_ptr : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>, #blocked>
    %old_addr = tt.addptr %old_out, %r : tensor<128x!tt.ptr<i32>, #blocked>, tensor<128xi32, #blocked>
    tt.store %old_addr, %old : tensor<128x!tt.ptr<i32>, #blocked>
    tt.return
  }

  tt.func public @atomic_minmax_tensor_u64(%out_ptr: !tt.ptr<i64>, %v: i64) {
    %r = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32, #blocked>
    %out = tt.splat %out_ptr : !tt.ptr<i64> -> tensor<16x!tt.ptr<i64>, #blocked>
    %addr = tt.addptr %out, %r : tensor<16x!tt.ptr<i64>, #blocked>, tensor<16xi32, #blocked>
    %values = tt.splat %v : i64 -> tensor<16xi64, #blocked>
    %umin = tt.atomic_rmw umin, acq_rel, gpu, %addr, %values : (tensor<16x!tt.ptr<i64>, #blocked>, tensor<16xi64, #blocked>) -> tensor<16xi64, #blocked>
    %umax = tt.atomic_rmw umax, acq_rel, gpu, %addr, %values : (tensor<16x!tt.ptr<i64>, #blocked>, tensor<16xi64, #blocked>) -> tensor<16xi64, #blocked>
    tt.return
  }
}

// CHECK-LABEL: metal.kernel atomic_add_scalar_i32
// Single-lane guard, then the atomic on the ui32 buffer.
// CHECK: arith.cmpi eq
// CHECK: scf.if
// CHECK: metal.atomic_rmw {{.*}} : (ui32, !metal.memref<? x ui32>, ui32) -> ui32

// CHECK-LABEL: metal.kernel atomic_min_scalar_i32
// CHECK: arith.cmpi eq
// CHECK: scf.if
// CHECK: metal.atomic_rmw {{.*}} : (si32, !metal.memref<? x ui32>, ui32) -> si32
// CHECK-NOT: tt.atomic_rmw

// MSL-LABEL: kernel void atomic_min_scalar_i32
// MSL: atomic_fetch_min_explicit((device atomic_int*)&

// CHECK-LABEL: metal.kernel atomic_minmax_tensor_i32
// Sub-TPB tensor atomics synthesize a lane-valid guard without a user mask.
// CHECK: arith.cmpi slt
// CHECK: metal.atomic_rmw Min {{.*}} : (si32, !metal.memref<? x ui32>, ui32) -> si32
// CHECK: metal.atomic_rmw Max {{.*}} : (si32, !metal.memref<? x ui32>, ui32) -> si32
// CHECK: metal.atomic_rmw UMin {{.*}} : (ui32, !metal.memref<? x ui32>, ui32) -> ui32
// CHECK: metal.atomic_rmw UMax {{.*}} : (ui32, !metal.memref<? x ui32>, ui32) -> ui32

// CHECK-LABEL: metal.kernel atomic_min_tensor_old_i32
// CHECK: %[[OLD:.*]] = metal.atomic_rmw Min {{.*}} : (si32, !metal.memref<? x ui32>, ui32) -> si32
// CHECK: builtin.unrealized_conversion_cast %[[OLD]] : si32 to i32

// CHECK-LABEL: metal.kernel atomic_minmax_tensor_u64
// CHECK: metal.atomic_rmw UMin {{.*}} : (ui64, !metal.memref<? x ui64>, ui32) -> ui64
// CHECK: metal.atomic_rmw UMax {{.*}} : (ui64, !metal.memref<? x ui64>, ui32) -> ui64

// MSL-LABEL: kernel void atomic_min_tensor_old_i32
// MSL: int32_t v{{[0-9]+}} = atomic_fetch_min_explicit
// MSL-NOT: atomic_fetch_min_explicit

// MSL-LABEL: kernel void atomic_minmax_tensor_u64
// MSL: atomic_min_explicit((device atomic_ulong*)&
// MSL: atomic_max_explicit((device atomic_ulong*)&
// MSL-NOT: atomic_fetch_{{min|max}}_explicit((device atomic_ulong*)&
