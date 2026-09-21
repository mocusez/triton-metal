// RUN: split-file %s %t
// RUN: triton-metal-translate --mlir-to-msl %t/valid.mlir 2>&1 | FileCheck %s
// RUN: triton-metal-opt --split-input-file --verify-diagnostics %t/invalid.mlir

//--- valid.mlir
//
// The staged load exists because chaining `simdgroup_multiply_accumulate` on
// values loaded straight from `const device T*` silently drops output columns
// on Apple GPU family 9; the operands have to come from threadgroup memory.
// That staging is bracketed by fences, and their SCOPE is what this pins.
//
// When every staged load in a kernel carries a warp index, `_stage_shared` is
// declared `[num_warps][elems]` and each simdgroup writes and reads only its
// own slice. No other simdgroup can observe it, so a simdgroup-scoped fence is
// sufficient and a threadgroup-wide barrier would needlessly serialize the
// other simdgroups — measured at roughly a sixth of the runtime of a large
// matmul, whose inner loop is one 8x8 multiply-accumulate per staging pass.
//
// A SINGLE staged load without a warp index takes that back: it reads
// `_stage_shared[0]`, which every simdgroup shares, so all fences in that
// kernel must stay threadgroup-wide.

module attributes {"ttg.num-warps" = 4 : i32} {
  metal.module {
    // Every staged load is warp-indexed: simdgroup-scoped fences.
    metal.kernel all_per_warp address_space_device [true, true, true] {
    ^bb0(%a: !metal.memref<? x f32>, %b: !metal.memref<? x f32>, %c: !metal.memref<? x f32>):
      %z = metal.constant 0 : ui32
      %s = metal.constant 8 : ui32
      %w = metal.simdgroup_index : ui32
      %ta = metal.simdgroup_load_device_staged %a[%z, %z], %s warp [%w] : (!metal.memref<? x f32>, ui32, ui32, ui32, ui32) -> !metal.simdgroup_matrix<8 x 8 x f32>
      %tb = metal.simdgroup_load_device_staged %b[%z, %z], %s warp [%w] : (!metal.memref<? x f32>, ui32, ui32, ui32, ui32) -> !metal.simdgroup_matrix<8 x 8 x f32>
      %acc = metal.simdgroup_matrix_zero : !metal.simdgroup_matrix<8 x 8 x f32>
      %r = metal.simdgroup_multiply_accumulate %acc, %ta, %tb : (!metal.simdgroup_matrix<8 x 8 x f32>, !metal.simdgroup_matrix<8 x 8 x f32>, !metal.simdgroup_matrix<8 x 8 x f32>) -> !metal.simdgroup_matrix<8 x 8 x f32>
      metal.simdgroup_store %r, %c[%z, %z], %s : !metal.simdgroup_matrix<8 x 8 x f32>, !metal.memref<? x f32>, ui32, ui32, ui32
      metal.return
    }
    // One staged load WITHOUT a warp index shares slice zero across every
    // simdgroup, so the fences stay threadgroup-wide for the whole kernel.
    metal.kernel mixed_scope address_space_device [true, true, true] {
    ^bb0(%a: !metal.memref<? x f32>, %b: !metal.memref<? x f32>, %c: !metal.memref<? x f32>):
      %z = metal.constant 0 : ui32
      %s = metal.constant 8 : ui32
      %w = metal.simdgroup_index : ui32
      %ta = metal.simdgroup_load_device_staged %a[%z, %z], %s warp [%w] : (!metal.memref<? x f32>, ui32, ui32, ui32, ui32) -> !metal.simdgroup_matrix<8 x 8 x f32>
      %tb = metal.simdgroup_load_device_staged %b[%z, %z], %s : (!metal.memref<? x f32>, ui32, ui32, ui32) -> !metal.simdgroup_matrix<8 x 8 x f32>
      %acc = metal.simdgroup_matrix_zero : !metal.simdgroup_matrix<8 x 8 x f32>
      %r = metal.simdgroup_multiply_accumulate %acc, %ta, %tb : (!metal.simdgroup_matrix<8 x 8 x f32>, !metal.simdgroup_matrix<8 x 8 x f32>, !metal.simdgroup_matrix<8 x 8 x f32>) -> !metal.simdgroup_matrix<8 x 8 x f32>
      metal.simdgroup_store %r, %c[%z, %z], %s : !metal.simdgroup_matrix<8 x 8 x f32>, !metal.memref<? x f32>, ui32, ui32, ui32
      metal.return
    }
    // Independent half operands share two fences, but never a scratch slot.
    // Distinct strides/origins and a transposed second tile pin their mapping.
    metal.kernel batched_half address_space_device [true, true, true] {
    ^bb0(%a: !metal.memref<? x f16>, %b: !metal.memref<? x f16>, %c: !metal.memref<? x f32>):
      %z = metal.constant 0 : ui32
      %eight = metal.constant 8 : ui32
      %sixteen = metal.constant 16 : ui32
      %thirtytwo = metal.constant 32 : ui32
      %w = metal.simdgroup_index : ui32
      %scratch = metal.threadgroup_alloca : !metal.memref<512 x f16>
      %tiles:2 = metal.simdgroup_load_device_staged_batch %scratch buffers [%a, %b] rows [%eight, %z] cols [%z, %eight] strides [%sixteen, %thirtytwo] warp %w {transposed = array<i1: false, true>} : (!metal.memref<512 x f16>, !metal.memref<? x f16>, !metal.memref<? x f16>, ui32, ui32, ui32, ui32, ui32, ui32, ui32) -> (!metal.simdgroup_matrix<8 x 8 x f16>, !metal.simdgroup_matrix<8 x 8 x f16>)
      %acc = metal.simdgroup_matrix_zero : !metal.simdgroup_matrix<8 x 8 x f32>
      %r = metal.simdgroup_multiply_accumulate %acc, %tiles#0, %tiles#1 : (!metal.simdgroup_matrix<8 x 8 x f32>, !metal.simdgroup_matrix<8 x 8 x f16>, !metal.simdgroup_matrix<8 x 8 x f16>) -> !metal.simdgroup_matrix<8 x 8 x f32>
      metal.simdgroup_store %r, %c[%z, %z], %eight : !metal.simdgroup_matrix<8 x 8 x f32>, !metal.memref<? x f32>, ui32, ui32, ui32
      metal.return
    }
    // Direct stores never touch scratch and do not widen its fence scope.
    metal.kernel stores_all_per_warp address_space_device [true] {
    ^bb0(%c: !metal.memref<? x f32>):
      %z = metal.constant 0 : ui32
      %s = metal.constant 8 : ui32
      %w = metal.simdgroup_index : ui32
      %acc = metal.simdgroup_matrix_zero : !metal.simdgroup_matrix<8 x 8 x f32>
      metal.simdgroup_store %acc, %c[%z, %z], %s : !metal.simdgroup_matrix<8 x 8 x f32>, !metal.memref<? x f32>, ui32, ui32, ui32
      metal.simdgroup_store %acc, %c[%z, %z], %s partial [%s, %s] warp [%w] : !metal.simdgroup_matrix<8 x 8 x f32>, !metal.memref<? x f32>, ui32, ui32, ui32, ui32, ui32, ui32
      metal.simdgroup_store %acc, %c[%z, %z], %s partial [%s, %s] warp [%w] : !metal.simdgroup_matrix<8 x 8 x f32>, !metal.memref<? x f32>, ui32, ui32, ui32, ui32, ui32, ui32
      metal.return
    }
    // A following kernel must not inherit the preceding kernel's narrow scope.
    metal.kernel stores_mixed_scope address_space_device [true] {
    ^bb0(%c: !metal.memref<? x f32>):
      %z = metal.constant 0 : ui32
      %s = metal.constant 8 : ui32
      %w = metal.simdgroup_index : ui32
      %acc = metal.simdgroup_matrix_zero : !metal.simdgroup_matrix<8 x 8 x f32>
      metal.simdgroup_store %acc, %c[%z, %z], %s partial [%s, %s] warp [%w] : !metal.simdgroup_matrix<8 x 8 x f32>, !metal.memref<? x f32>, ui32, ui32, ui32, ui32, ui32, ui32
      metal.simdgroup_store %acc, %c[%z, %z], %s partial [%s, %s] : !metal.simdgroup_matrix<8 x 8 x f32>, !metal.memref<? x f32>, ui32, ui32, ui32, ui32, ui32
      metal.return
    }
    // A present but uniform index still shares one scratch slice across warps.
    metal.kernel stores_uniform_index address_space_device [true] {
    ^bb0(%c: !metal.memref<? x f32>):
      %z = metal.constant 0 : ui32
      %s = metal.constant 8 : ui32
      %w = metal.simdgroup_index : ui32
      %acc = metal.simdgroup_matrix_zero : !metal.simdgroup_matrix<8 x 8 x f32>
      metal.simdgroup_store %acc, %c[%z, %z], %s partial [%s, %s] warp [%w] : !metal.simdgroup_matrix<8 x 8 x f32>, !metal.memref<? x f32>, ui32, ui32, ui32, ui32, ui32, ui32
      %lo = arith.constant 0 : i32
      %hi = arith.constant 1 : i32
      scf.for %iv = %lo to %hi step %hi : i32 {
        %uniform = builtin.unrealized_conversion_cast %iv : i32 to ui32
        metal.simdgroup_store %acc, %c[%z, %z], %s partial [%s, %s] warp [%uniform] : !metal.simdgroup_matrix<8 x 8 x f32>, !metal.memref<? x f32>, ui32, ui32, ui32, ui32, ui32, ui32
      }
      metal.return
    }
  }
}

// CHECK-LABEL: kernel void all_per_warp
// CHECK: threadgroup float _stage_shared[4][64]
// CHECK: simdgroup_barrier(mem_flags::mem_threadgroup)
// CHECK: simdgroup_load(v{{[0-9]+}}, &_stage_shared[
// CHECK-NOT: threadgroup_barrier

// CHECK-LABEL: kernel void mixed_scope
// CHECK: threadgroup_barrier(mem_flags::mem_threadgroup)
// CHECK-NOT: simdgroup_barrier

// CHECK-LABEL: kernel void batched_half
// CHECK-NOT: _stage_shared
// CHECK: threadgroup half [[SCRATCH:v[0-9]+]][512];
// CHECK-NEXT: simdgroup_barrier(mem_flags::mem_threadgroup);
// CHECK-NEXT: for (uint c = (id.x & 31u); c < 64u; c += 32u) {
// CHECK: [[SCRATCH]][(sgid * 2u + 0u) * 64u + c] = v0[(8 + si) * 16 + 0 + sj];
// CHECK-NOT: barrier
// CHECK-NOT: simdgroup_load
// CHECK: for (uint c = (id.x & 31u); c < 64u; c += 32u) {
// CHECK: [[SCRATCH]][(sgid * 2u + 1u) * 64u + sj * 8u + si] = v1[(0 + si) * 32 + 8 + sj];
// CHECK-NEXT: }
// CHECK-NEXT: simdgroup_barrier(mem_flags::mem_threadgroup);
// CHECK-NEXT: simdgroup_half8x8 [[A:v[0-9]+]];
// CHECK-NEXT: simdgroup_load([[A]], &[[SCRATCH]][(sgid * 2u + 0u) * 64u + 0u], 8);
// CHECK-NEXT: simdgroup_half8x8 [[B:v[0-9]+]];
// CHECK-NEXT: simdgroup_load([[B]], &[[SCRATCH]][(sgid * 2u + 1u) * 64u + 0u], 8);
// CHECK-NEXT: simdgroup_float8x8 [[ACC:v[0-9]+]](0.0f);
// CHECK-NEXT: simdgroup_multiply_accumulate([[ACC]], [[A]], [[B]], [[ACC]]);
// CHECK-NOT: barrier

// CHECK-LABEL: kernel void stores_all_per_warp
// CHECK: threadgroup float _sgstore_shared[4][64];
// CHECK-NOT: threadgroup_barrier
// CHECK: simdgroup_barrier(mem_flags::mem_threadgroup);
// CHECK-NEXT: simdgroup_store({{.*}}, &_sgstore_shared[sgid][0], 8);
// CHECK-NEXT: simdgroup_barrier(mem_flags::mem_threadgroup);
// CHECK-NOT: threadgroup_barrier
// CHECK: simdgroup_barrier(mem_flags::mem_threadgroup);
// CHECK-NEXT: simdgroup_store({{.*}}, &_sgstore_shared[sgid][0], 8);
// CHECK-NEXT: simdgroup_barrier(mem_flags::mem_threadgroup);
// CHECK-NOT: threadgroup_barrier

// CHECK-LABEL: kernel void stores_mixed_scope
// CHECK: threadgroup float _sgstore_shared[4][64];
// CHECK-NOT: simdgroup_barrier
// CHECK: threadgroup_barrier(mem_flags::mem_threadgroup);
// CHECK-NEXT: simdgroup_store({{.*}}, &_sgstore_shared[sgid][0], 8);
// CHECK-NEXT: threadgroup_barrier(mem_flags::mem_threadgroup);
// CHECK-NOT: simdgroup_barrier
// CHECK: threadgroup_barrier(mem_flags::mem_threadgroup);
// CHECK-NEXT: simdgroup_store({{.*}}, &_sgstore_shared[0][0], 8);
// CHECK-NEXT: threadgroup_barrier(mem_flags::mem_threadgroup);
// CHECK-NOT: simdgroup_barrier

// CHECK-LABEL: kernel void stores_uniform_index
// CHECK: threadgroup float _sgstore_shared[4][64];
// CHECK-NOT: simdgroup_barrier
// CHECK: threadgroup_barrier(mem_flags::mem_threadgroup);
// CHECK-NEXT: simdgroup_store({{.*}}, &_sgstore_shared[sgid][0], 8);
// CHECK-NEXT: threadgroup_barrier(mem_flags::mem_threadgroup);
// CHECK-NOT: simdgroup_barrier
// CHECK: threadgroup_barrier(mem_flags::mem_threadgroup);
// CHECK-NEXT: simdgroup_store({{.*}}, &_sgstore_shared[{{v[0-9]+}}][0], 8);
// CHECK-NEXT: threadgroup_barrier(mem_flags::mem_threadgroup);
// CHECK-NOT: simdgroup_barrier

//--- invalid.mlir

module attributes {"ttg.num-warps" = 4 : i32} {
  metal.module {
    metal.kernel mismatched_counts address_space_device [true, true] {
    ^bb0(%a: !metal.memref<? x f16>, %b: !metal.memref<? x f16>):
      %z = metal.constant 0 : ui32
      %s = metal.constant 8 : ui32
      %w = metal.simdgroup_index : ui32
      %scratch = metal.threadgroup_alloca : !metal.memref<512 x f16>
      // expected-error @+1 {{requires at least two tiles and equal operand, result, and transposition counts}}
      %tiles:2 = metal.simdgroup_load_device_staged_batch %scratch buffers [%a, %b] rows [%z, %z] cols [%z, %z] strides [%s, %s] warp %w {transposed = array<i1: false>} : (!metal.memref<512 x f16>, !metal.memref<? x f16>, !metal.memref<? x f16>, ui32, ui32, ui32, ui32, ui32, ui32, ui32) -> (!metal.simdgroup_matrix<8 x 8 x f16>, !metal.simdgroup_matrix<8 x 8 x f16>)
      metal.return
    }
  }
}

// -----

module attributes {"ttg.num-warps" = 4 : i32} {
  metal.module {
    metal.kernel float_matrix address_space_device [true, true] {
    ^bb0(%a: !metal.memref<? x f16>, %b: !metal.memref<? x f16>):
      %z = metal.constant 0 : ui32
      %s = metal.constant 8 : ui32
      %w = metal.simdgroup_index : ui32
      %scratch = metal.threadgroup_alloca : !metal.memref<512 x f16>
      // expected-error @+1 {{requires 8x8 f16 matrix results}}
      %tiles:2 = metal.simdgroup_load_device_staged_batch %scratch buffers [%a, %b] rows [%z, %z] cols [%z, %z] strides [%s, %s] warp %w {transposed = array<i1: false, true>} : (!metal.memref<512 x f16>, !metal.memref<? x f16>, !metal.memref<? x f16>, ui32, ui32, ui32, ui32, ui32, ui32, ui32) -> (!metal.simdgroup_matrix<8 x 8 x f32>, !metal.simdgroup_matrix<8 x 8 x f16>)
      metal.return
    }
  }
}

// -----

module attributes {"ttg.num-warps" = 4 : i32} {
  metal.module {
    metal.kernel float_input address_space_device [true, true] {
    ^bb0(%a: !metal.memref<? x f32>, %b: !metal.memref<? x f16>):
      %z = metal.constant 0 : ui32
      %s = metal.constant 8 : ui32
      %w = metal.simdgroup_index : ui32
      %scratch = metal.threadgroup_alloca : !metal.memref<512 x f16>
      // expected-error @+1 {{requires f16 input buffers}}
      %tiles:2 = metal.simdgroup_load_device_staged_batch %scratch buffers [%a, %b] rows [%z, %z] cols [%z, %z] strides [%s, %s] warp %w {transposed = array<i1: false, true>} : (!metal.memref<512 x f16>, !metal.memref<? x f32>, !metal.memref<? x f16>, ui32, ui32, ui32, ui32, ui32, ui32, ui32) -> (!metal.simdgroup_matrix<8 x 8 x f16>, !metal.simdgroup_matrix<8 x 8 x f16>)
      metal.return
    }
  }
}

// -----

module attributes {"ttg.num-warps" = 4 : i32} {
  metal.module {
    metal.kernel uniform_warp address_space_device [true, true] {
    ^bb0(%a: !metal.memref<? x f16>, %b: !metal.memref<? x f16>):
      %z = metal.constant 0 : ui32
      %s = metal.constant 8 : ui32
      %w = metal.simdgroup_index : ui32
      %scratch = metal.threadgroup_alloca : !metal.memref<512 x f16>
      // expected-error @+1 {{warp_index must be defined by metal.simdgroup_index}}
      %tiles:2 = metal.simdgroup_load_device_staged_batch %scratch buffers [%a, %b] rows [%z, %z] cols [%z, %z] strides [%s, %s] warp %z {transposed = array<i1: false, true>} : (!metal.memref<512 x f16>, !metal.memref<? x f16>, !metal.memref<? x f16>, ui32, ui32, ui32, ui32, ui32, ui32, ui32) -> (!metal.simdgroup_matrix<8 x 8 x f16>, !metal.simdgroup_matrix<8 x 8 x f16>)
      metal.return
    }
  }
}

// -----

module attributes {"ttg.num-warps" = 4 : i32} {
  metal.module {
    metal.kernel wrong_scratch_size address_space_device [true, true] {
    ^bb0(%a: !metal.memref<? x f16>, %b: !metal.memref<? x f16>):
      %z = metal.constant 0 : ui32
      %s = metal.constant 8 : ui32
      %w = metal.simdgroup_index : ui32
      %scratch = metal.threadgroup_alloca : !metal.memref<256 x f16>
      // expected-error @+1 {{scratch must contain num_warps * tile_count * 64 f16 elements}}
      %tiles:2 = metal.simdgroup_load_device_staged_batch %scratch buffers [%a, %b] rows [%z, %z] cols [%z, %z] strides [%s, %s] warp %w {transposed = array<i1: false, true>} : (!metal.memref<256 x f16>, !metal.memref<? x f16>, !metal.memref<? x f16>, ui32, ui32, ui32, ui32, ui32, ui32, ui32) -> (!metal.simdgroup_matrix<8 x 8 x f16>, !metal.simdgroup_matrix<8 x 8 x f16>)
      metal.return
    }
  }
}

// -----

module attributes {"ttg.num-warps" = 4 : i32} {
  metal.module {
    metal.kernel shared_scratch address_space_device [true, true] {
    ^bb0(%a: !metal.memref<? x f16>, %b: !metal.memref<? x f16>):
      %z = metal.constant 0 : ui32
      %s = metal.constant 8 : ui32
      %w = metal.simdgroup_index : ui32
      %scratch = metal.threadgroup_alloca : !metal.memref<512 x f16>
      // expected-error @+1 {{requires a private, single-use threadgroup allocation}}
      %tiles:2 = metal.simdgroup_load_device_staged_batch %scratch buffers [%a, %b] rows [%z, %z] cols [%z, %z] strides [%s, %s] warp %w {transposed = array<i1: false, true>} : (!metal.memref<512 x f16>, !metal.memref<? x f16>, !metal.memref<? x f16>, ui32, ui32, ui32, ui32, ui32, ui32, ui32) -> (!metal.simdgroup_matrix<8 x 8 x f16>, !metal.simdgroup_matrix<8 x 8 x f16>)
      %extra = metal.simdgroup_load %scratch[%z, %z], %s : (!metal.memref<512 x f16>, ui32, ui32, ui32) -> !metal.simdgroup_matrix<8 x 8 x f16>
      metal.return
    }
  }
}

// -----

module attributes {"ttg.num-warps" = 64 : i32} {
  metal.module {
    metal.kernel oversized_scratch address_space_device [true, true] {
    ^bb0(%a: !metal.memref<? x f16>, %b: !metal.memref<? x f16>):
      %z = metal.constant 0 : ui32
      %s = metal.constant 8 : ui32
      %w = metal.simdgroup_index : ui32
      %scratch = metal.threadgroup_alloca : !metal.memref<8192 x f16>
      // expected-error @+1 {{batch scratch exceeds 8192 bytes}}
      %tiles:2 = metal.simdgroup_load_device_staged_batch %scratch buffers [%a, %b] rows [%z, %z] cols [%z, %z] strides [%s, %s] warp %w {transposed = array<i1: false, true>} : (!metal.memref<8192 x f16>, !metal.memref<? x f16>, !metal.memref<? x f16>, ui32, ui32, ui32, ui32, ui32, ui32, ui32) -> (!metal.simdgroup_matrix<8 x 8 x f16>, !metal.simdgroup_matrix<8 x 8 x f16>)
      metal.return
    }
  }
}
