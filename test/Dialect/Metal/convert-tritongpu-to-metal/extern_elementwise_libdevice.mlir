// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | FileCheck %s
//
// `triton.language.extra.libdevice` reaches the backend as
// `tt.extern_elementwise` carrying a `__metal_*` symbol declared by
// third_party/metal/language/metal/libdevice.py. TritonExternElementwiseLowering
// maps the symbol through a fixed table to an MSL intrinsic and emits
// `metal.math_intrinsic`.
//
// Two things this pins that are easy to get wrong:
//   * the callee is the MSL name from the table, NOT the `__metal_*` symbol —
//     the symbol is Triton-side naming and does not exist in MSL;
//   * an integer intrinsic is bridged through ui32. `Metal_Type` has no
//     signless integer, so passing the i32 operand straight through fails the
//     metal.math_intrinsic verifier.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @libdevice_asin(%x: !tt.ptr<f32>, %o: !tt.ptr<f32>) {
    %r = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %xp = tt.splat %x : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %xa = tt.addptr %xp, %r : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    %v = tt.load %xa : tensor<128x!tt.ptr<f32>, #blocked>
    %a = tt.extern_elementwise %v {libname = "", libpath = "", pure = true, symbol = "__metal_asin"} : (tensor<128xf32, #blocked>) -> tensor<128xf32, #blocked>
    %op = tt.splat %o : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %oa = tt.addptr %op, %r : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    tt.store %oa, %a : tensor<128x!tt.ptr<f32>, #blocked>
    tt.return
  }

  tt.func public @libdevice_atan2(%x: !tt.ptr<f32>, %o: !tt.ptr<f32>) {
    %r = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %xp = tt.splat %x : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %xa = tt.addptr %xp, %r : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    %v = tt.load %xa : tensor<128x!tt.ptr<f32>, #blocked>
    %a = tt.extern_elementwise %v, %v {libname = "", libpath = "", pure = true, symbol = "__metal_atan2"} : (tensor<128xf32, #blocked>, tensor<128xf32, #blocked>) -> tensor<128xf32, #blocked>
    %op = tt.splat %o : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %oa = tt.addptr %op, %r : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    tt.store %oa, %a : tensor<128x!tt.ptr<f32>, #blocked>
    tt.return
  }

  tt.func public @libdevice_clz(%x: !tt.ptr<i32>, %o: !tt.ptr<i32>) {
    %r = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %xp = tt.splat %x : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>, #blocked>
    %xa = tt.addptr %xp, %r : tensor<128x!tt.ptr<i32>, #blocked>, tensor<128xi32, #blocked>
    %v = tt.load %xa : tensor<128x!tt.ptr<i32>, #blocked>
    %a = tt.extern_elementwise %v {libname = "", libpath = "", pure = true, symbol = "__metal_clz"} : (tensor<128xi32, #blocked>) -> tensor<128xi32, #blocked>
    %op = tt.splat %o : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>, #blocked>
    %oa = tt.addptr %op, %r : tensor<128x!tt.ptr<i32>, #blocked>, tensor<128xi32, #blocked>
    tt.store %oa, %a : tensor<128x!tt.ptr<i32>, #blocked>
    tt.return
  }

  // erf is the one entry that does NOT become a math_intrinsic: MSL has no
  // erf, and `__triton_erff` is a polynomial the emitter writes into the module
  // preamble only when it sees a unary_exp erfOp node. A math_intrinsic naming
  // it would emit a call with no definition.
  tt.func public @libdevice_erf(%x: !tt.ptr<f32>, %o: !tt.ptr<f32>) {
    %r = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %xp = tt.splat %x : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %xa = tt.addptr %xp, %r : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    %v = tt.load %xa : tensor<128x!tt.ptr<f32>, #blocked>
    %a = tt.extern_elementwise %v {libname = "", libpath = "", pure = true, symbol = "__metal_erf"} : (tensor<128xf32, #blocked>) -> tensor<128xf32, #blocked>
    %op = tt.splat %o : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %oa = tt.addptr %op, %r : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    tt.store %oa, %a : tensor<128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// CHECK-LABEL: metal.kernel libdevice_asin
// The IEEE-precise namespace, not the unqualified name: MSL's unqualified
// spellings are allowed to use the fast approximations, and Triton's
// elementwise contract is IEEE.
// CHECK: metal.math_intrinsic "metal::precise::asin"
// CHECK: metal.return

// CHECK-LABEL: metal.kernel libdevice_atan2
// CHECK: metal.math_intrinsic "metal::precise::atan2"({{.*}}, {{.*}})
// CHECK: metal.return

// CHECK-LABEL: metal.kernel libdevice_clz
// ui32 in and back out around the call.
// CHECK: builtin.unrealized_conversion_cast {{.*}} : i32 to ui32
// CHECK: metal.math_intrinsic "metal::clz"
// CHECK: builtin.unrealized_conversion_cast {{.*}} : ui32 to i32
// CHECK: metal.return

// CHECK-LABEL: metal.kernel libdevice_erf
// CHECK-NOT: metal.math_intrinsic
// CHECK: metal.unary_exp {{.*}}, erfOp
// CHECK: metal.return
