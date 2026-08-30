// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | FileCheck %s
// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | triton-metal-translate --mlir-to-msl | FileCheck %s --check-prefix=MSL
//
// `tt.atomic_rmw and/or/xor/exch` — `tl.atomic_and`, `tl.atomic_or`,
// `tl.atomic_xor`, `tl.atomic_xchg`. The pass accepted add/fadd/min/max/umin/
// umax only, so all four were rejected outright.
//
// MSL has every one of them, but only in 32-bit form: there is no
// `atomic_fetch_and_explicit` overload for atomic_ulong, while
// `atomic_exchange_explicit` additionally covers atomic_float. Each name here
// was verified by compiling it through `torch.mps.compile_shader` — a name that
// does not exist still passes through this pass and only fails at shader load.
//
// Unlike min/max, these kinds do NOT reinterpret their payload: min/max carry
// Triton's ordered-bit f32 expansion and so wrap the value in `as_type<>`,
// whereas and/or/xor/exchange take the value as it stands.
//
// CHECK-LABEL: metal.kernel atomic_bitwise_scalar_i32
// CHECK: metal.atomic_rmw And
// CHECK: metal.atomic_rmw Or
// CHECK: metal.atomic_rmw Xor
// MSL-LABEL: kernel void atomic_bitwise_scalar_i32
// MSL: atomic_fetch_and_explicit((device atomic_uint*)&v0[0], v1[0], memory_order_relaxed)
// MSL: atomic_fetch_or_explicit((device atomic_uint*)&v0[0], v1[0], memory_order_relaxed)
// MSL: atomic_fetch_xor_explicit((device atomic_uint*)&v0[0], v1[0], memory_order_relaxed)

// CHECK-LABEL: metal.kernel atomic_xchg_scalar_f32
// CHECK: metal.atomic_rmw Xchg
// MSL-LABEL: kernel void atomic_xchg_scalar_f32
// MSL: atomic_exchange_explicit((device atomic_float*)&v0[0], v1[0], memory_order_relaxed)

// CHECK-LABEL: metal.kernel atomic_xor_tensor_i32
// CHECK: metal.atomic_rmw Xor
// MSL-LABEL: kernel void atomic_xor_tensor_i32
// MSL: atomic_fetch_xor_explicit((device atomic_uint*)&v0[(id.x - (tgid.x * 128))]

// A scalar atomic is executed by lane zero alone, so a consumed old value has
// to be published to the rest of the threadgroup — the same slot-plus-barrier
// the scalar CAS already used, now hoisted for `tt.atomic_rmw` as well. Before
// this, any scalar atomic whose result was read was rejected outright.
// CHECK-LABEL: metal.kernel atomic_add_scalar_old_value
// CHECK: metal.threadgroup_alloca
// CHECK: metal.atomic_rmw Add
// CHECK: metal.tg_store_indexed
// CHECK: metal.barrier
// CHECK: metal.tg_load_indexed
// MSL-LABEL: kernel void atomic_add_scalar_old_value
// MSL: threadgroup uint32_t v[[SLOT:[0-9]+]][1];
// MSL: uint32_t v[[OLD:[0-9]+]] = atomic_fetch_add_explicit((device atomic_uint*)&v0[0]
// MSL: v[[SLOT]][0] = v[[OLD]];
// MSL: threadgroup_barrier(mem_flags::mem_threadgroup);
// MSL: = v[[SLOT]][0];

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @atomic_bitwise_scalar_i32(%out: !tt.ptr<i32>, %v: i32) {
    %a = tt.atomic_rmw and, acq_rel, gpu, %out, %v : (!tt.ptr<i32>, i32) -> i32
    %o = tt.atomic_rmw or, acq_rel, gpu, %out, %v : (!tt.ptr<i32>, i32) -> i32
    %x = tt.atomic_rmw xor, acq_rel, gpu, %out, %v : (!tt.ptr<i32>, i32) -> i32
    tt.return
  }

  tt.func public @atomic_xchg_scalar_f32(%out: !tt.ptr<f32>, %v: f32) {
    %e = tt.atomic_rmw exch, acq_rel, gpu, %out, %v : (!tt.ptr<f32>, f32) -> f32
    tt.return
  }

  tt.func public @atomic_xor_tensor_i32(%out: !tt.ptr<i32>, %v: i32) {
    %r = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32, #blocked>
    %po = tt.splat %out : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>, #blocked>
    %addr = tt.addptr %po, %r : tensor<128x!tt.ptr<i32>, #blocked>, tensor<128xi32, #blocked>
    %vals = tt.splat %v : i32 -> tensor<128xi32, #blocked>
    %x = tt.atomic_rmw xor, acq_rel, gpu, %addr, %vals : (tensor<128x!tt.ptr<i32>, #blocked>, tensor<128xi32, #blocked>) -> tensor<128xi32, #blocked>
    tt.return
  }

  tt.func public @atomic_add_scalar_old_value(%out: !tt.ptr<i32>, %seen: !tt.ptr<i32>, %v: i32) {
    %old = tt.atomic_rmw add, acq_rel, gpu, %out, %v : (!tt.ptr<i32>, i32) -> i32
    %r = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32, #blocked>
    %ps = tt.splat %seen : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>, #blocked>
    %addr = tt.addptr %ps, %r : tensor<128x!tt.ptr<i32>, #blocked>, tensor<128xi32, #blocked>
    %bcast = tt.splat %old : i32 -> tensor<128xi32, #blocked>
    tt.store %addr, %bcast : tensor<128x!tt.ptr<i32>, #blocked>
    tt.return
  }
}
