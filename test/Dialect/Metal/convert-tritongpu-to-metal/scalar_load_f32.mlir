// RUN: triton-metal-opt --convert-tritongpu-to-metal %s --split-input-file | FileCheck %s
//
// L3a-tileloop-compiler-A: scalar tt.load on bare !tt.ptr<f32>.
// Two shapes are covered: (a) bare load (offset 0, no addptr — Triton
// folds `addptr(p, 0)` away), and (b) addptr+load where the offset is a
// scalar i32 from a `tl.static_range` iter or similar.
// See the implementation notes.

// Case (a): bare scalar load (no addptr).
#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @scalar_load_bare(%kernel_ptr: !tt.ptr<f32>, %output_ptr: !tt.ptr<f32>) {
    %r = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %kv = tt.load %kernel_ptr : !tt.ptr<f32>
    %kv_t = tt.splat %kv : f32 -> tensor<1024xf32, #blocked>
    %o = tt.splat %output_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %oa = tt.addptr %o, %r : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    tt.store %oa, %kv_t : tensor<1024x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// CHECK: metal.kernel scalar_load_bare
// CHECK: arith.constant 0 : i32
// CHECK: builtin.unrealized_conversion_cast %{{.*}} : i32 to ui32
// CHECK: metal.get_element %arg0[%{{.*}}] : (!metal.memref<? x f32>, ui32) -> f32

// -----

// Case (b): addptr + scalar load (non-zero scalar offset).
#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @scalar_load_offset(%kernel_ptr: !tt.ptr<f32>, %output_ptr: !tt.ptr<f32>) {
    %c3 = arith.constant 3 : i32
    %r = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %addr = tt.addptr %kernel_ptr, %c3 : !tt.ptr<f32>, i32
    %kv = tt.load %addr : !tt.ptr<f32>
    %kv_t = tt.splat %kv : f32 -> tensor<1024xf32, #blocked>
    %o = tt.splat %output_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %oa = tt.addptr %o, %r : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    tt.store %oa, %kv_t : tensor<1024x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// CHECK: metal.kernel scalar_load_offset
// The scalar offset (3) is converted to ui32 and fed to get_element.
// CHECK: %{{.*}} = builtin.unrealized_conversion_cast %{{.*}} : i32 to ui32
// CHECK: metal.get_element %arg0[%{{.*}}] : (!metal.memref<? x f32>, ui32) -> f32

// -----

// Selecting an integer offset keeps one bound buffer and remains supported.
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @scalar_load_selected_offset(%input: !tt.ptr<f32>, %output: !tt.ptr<f32>, %cond: i1, %a: i32, %b: i32) {
    %offset = arith.select %cond, %a, %b : i32
    %ptr = tt.addptr %input, %offset : !tt.ptr<f32>, i32
    %value = tt.load %ptr : !tt.ptr<f32>
    tt.store %output, %value : !tt.ptr<f32>
    tt.return
  }
}

// CHECK: metal.kernel scalar_load_selected_offset
// CHECK: %[[OFFSET:.*]] = arith.select {{.*}} : i32
// CHECK: %[[INDEX:.*]] = builtin.unrealized_conversion_cast %[[OFFSET]] : i32 to ui32
// CHECK: metal.get_element %arg0[%[[INDEX]]] : (!metal.memref<? x f32>, ui32) -> f32

// -----

// Branch-local accesses keep their buffer origins. Returning data and an
// integer offset from a multi-result if must not trigger the pointer guard.
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @scalar_load_branch_values(%a: !tt.ptr<f32>, %b: !tt.ptr<f32>, %output: !tt.ptr<f32>, %cond: i1, %i: i32, %j: i32) {
    %value, %offset = scf.if %cond -> (f32, i32) {
      %x = tt.load %a : !tt.ptr<f32>
      scf.yield %x, %i : f32, i32
    } else {
      %y = tt.load %b : !tt.ptr<f32>
      scf.yield %y, %j : f32, i32
    }
    %ptr = tt.addptr %output, %offset : !tt.ptr<f32>, i32
    tt.store %ptr, %value : !tt.ptr<f32>
    tt.return
  }
}

// CHECK: metal.kernel scalar_load_branch_values
// CHECK: scf.if {{.*}} -> (f32, i32)
// CHECK: metal.get_element %arg0[
// CHECK: scf.yield {{.*}} : f32, i32
// CHECK: } else {
// CHECK: metal.get_element %arg1[
// CHECK: scf.yield {{.*}} : f32, i32
// CHECK: metal.store {{.*}}, %arg2[

// -----

// Carry data and offsets instead of pointers. Scalar memory accesses inside
// both while regions retain their bound buffer origins.
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @scalar_load_while_offset(%input: !tt.ptr<f32>, %output: !tt.ptr<f32>, %limit: !tt.ptr<i32>) {
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %init = arith.constant 0.0 : f32
    %result:2 = scf.while (%i = %zero, %acc = %init) : (i32, f32) -> (i32, f32) {
      %n = tt.load %limit : !tt.ptr<i32>
      %cond = arith.cmpi slt, %i, %n : i32
      scf.condition(%cond) %i, %acc : i32, f32
    } do {
    ^bb0(%i: i32, %acc: f32):
      %ptr = tt.addptr %input, %i : !tt.ptr<f32>, i32
      %value = tt.load %ptr : !tt.ptr<f32>
      %sum = arith.addf %acc, %value : f32
      %next = arith.addi %i, %one : i32
      scf.yield %next, %sum : i32, f32
    }
    tt.store %output, %result#1 : !tt.ptr<f32>
    tt.return
  }
}

// CHECK: metal.kernel scalar_load_while_offset
// CHECK: scf.while {{.*}} : (i32, f32) -> (i32, f32)
// CHECK: metal.get_element %arg2[
// CHECK: scf.condition({{.*}}) {{.*}} : i32, f32
// CHECK: } do {
// CHECK: metal.get_element %arg0[
// CHECK: scf.yield {{.*}} : i32, f32
// CHECK: metal.store {{.*}}, %arg1[

// -----

// Ordinary for loops carry integer offsets and loaded data, with each access
// rooted in a bound buffer. Matched pointer-advance matmuls have separate dot
// regression coverage and must be consumed before the pointer preflight.
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @scalar_load_for_offset(%input: !tt.ptr<f32>, %output: !tt.ptr<f32>, %n: i32) {
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %init = arith.constant 0.0 : f32
    %result:2 = scf.for %i = %zero to %n step %one iter_args(%offset = %zero, %acc = %init) -> (i32, f32) : i32 {
      %ptr = tt.addptr %input, %offset : !tt.ptr<f32>, i32
      %value = tt.load %ptr : !tt.ptr<f32>
      %sum = arith.addf %acc, %value : f32
      %next = arith.addi %offset, %one : i32
      scf.yield %next, %sum : i32, f32
    }
    tt.store %output, %result#1 : !tt.ptr<f32>
    tt.return
  }
}

// CHECK: metal.kernel scalar_load_for_offset
// CHECK: scf.for {{.*}} -> (i32, f32)
// CHECK: metal.get_element %arg0[
// CHECK: scf.yield {{.*}} : i32, f32
// CHECK: metal.store {{.*}}, %arg1[

// -----

// Scalar masks need no tensor layout. The read must stay in the true branch;
// a false mask can accompany an out-of-bounds pointer.
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @scalar_masked_load_f32(%input: !tt.ptr<f32>, %output: !tt.ptr<f32>, %mask: i1) {
    %other = arith.constant -7.0 : f32
    %value = tt.load %input, %mask, %other : !tt.ptr<f32>
    tt.store %output, %value : !tt.ptr<f32>
    tt.return
  }
}
// CHECK-LABEL: metal.kernel scalar_masked_load_f32
// CHECK: %[[MASK:.*]] = metal.get_element %arg2[
// CHECK-NOT: metal.get_element %arg0[
// CHECK: scf.if %[[MASK]] -> (f32)
// CHECK: metal.get_element %arg0[
// CHECK: scf.yield {{.*}} : f32
// CHECK: } else {
// CHECK-NOT: metal.get_element
// CHECK: scf.yield {{.*}} : f32
// CHECK: metal.store {{.*}}, %arg1[

// -----

// Keep all scalar address offsets and return the logical integer type in
// both branches. Runtime scalar `other` is block-uniform by construction.
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @scalar_masked_load_i16_dynamic(%input: !tt.ptr<i16>, %output: !tt.ptr<i16>, %i: i32, %j: i32, %mask: i1, %other: i16) {
    %base = tt.addptr %input, %i : !tt.ptr<i16>, i32
    %ptr = tt.addptr %base, %j : !tt.ptr<i16>, i32
    %value = tt.load %ptr, %mask, %other : !tt.ptr<i16>
    tt.store %output, %value : !tt.ptr<i16>
    tt.return
  }
}
// CHECK-LABEL: metal.kernel scalar_masked_load_i16_dynamic
// CHECK: %[[I:.*]] = builtin.unrealized_conversion_cast {{.*}} : ui32 to i32
// CHECK: %[[J:.*]] = builtin.unrealized_conversion_cast {{.*}} : ui32 to i32
// CHECK: %[[MASK:.*]] = metal.get_element %arg4[
// CHECK: %[[OTHER:.*]] = builtin.unrealized_conversion_cast {{.*}} : ui16 to i16
// CHECK: %[[OFFSET:.*]] = arith.addi %[[J]], %[[I]] : i32
// CHECK: %[[INDEX:.*]] = builtin.unrealized_conversion_cast %[[OFFSET]] : i32 to ui32
// CHECK-NOT: metal.get_element %arg0[
// CHECK: scf.if %[[MASK]] -> (i16)
// CHECK: %[[VALUE:.*]] = metal.get_element %arg0[%[[INDEX]]] : (!metal.memref<? x ui16>, ui32) -> ui16
// CHECK: %[[LOGICAL:.*]] = builtin.unrealized_conversion_cast %[[VALUE]] : ui16 to i16
// CHECK: scf.yield %[[LOGICAL]] : i16
// CHECK: } else {
// CHECK-NOT: metal.get_element
// CHECK: scf.yield %[[OTHER]] : i16
// CHECK: metal.store

// -----

// No `other` is also a supported scalar load. The backend chooses zero for
// the otherwise undefined masked-off value, matching its tensor path.
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @scalar_masked_load_bf16_default(%input: !tt.ptr<bf16>, %output: !tt.ptr<bf16>, %mask: i1) {
    %value = tt.load %input, %mask : !tt.ptr<bf16>
    tt.store %output, %value : !tt.ptr<bf16>
    tt.return
  }
}
// CHECK-LABEL: metal.kernel scalar_masked_load_bf16_default
// CHECK: %[[MASK:.*]] = metal.get_element %arg2[
// CHECK-NOT: metal.get_element %arg0[
// CHECK: scf.if %[[MASK]] -> (bf16)
// CHECK: metal.get_element %arg0[
// CHECK: scf.yield {{.*}} : bf16
// CHECK: } else {
// CHECK-NOT: metal.get_element
// CHECK: %[[ZERO:.*]] = arith.constant 0.000000e+00 : bf16
// CHECK: scf.yield %[[ZERO]] : bf16
// CHECK: metal.store
