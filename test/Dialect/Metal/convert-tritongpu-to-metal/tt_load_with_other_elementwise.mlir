// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | FileCheck %s
// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | triton-metal-translate --mlir-to-msl | FileCheck %s --check-prefix=MSL
//
// Session L1 positive fixture: tt.load with a splat-constant `other` in
// elementwise (non-tt.dot-operand) position lowers to `scf.if` whose else
// branch yields the splat-constant scalar. The MSL emitter renders the
// else value as a literal (here `0.5`) — distinct from the v1 hardcoded
// `0.0` to prove the value flows through.
// See `.omc/specs/deep-interview-leet-triton-l1-refine-and-ship.md` §3.1.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @load_with_other_elementwise(%x_ptr: !tt.ptr<f32>, %out_ptr: !tt.ptr<f32>) {
    %c100_i32 = arith.constant 100 : i32
    %offsets = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %n_splat = tt.splat %c100_i32 : i32 -> tensor<128xi32, #blocked>
    %mask = arith.cmpi slt, %offsets, %n_splat : tensor<128xi32, #blocked>
    %x_splat = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %x_addr = tt.addptr %x_splat, %offsets : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    %other_f = arith.constant 0.5 : f32
    %other_splat = tt.splat %other_f : f32 -> tensor<128xf32, #blocked>
    %x_val = tt.load %x_addr, %mask, %other_splat : tensor<128x!tt.ptr<f32>, #blocked>

    %o_splat = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %o_addr = tt.addptr %o_splat, %offsets : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    tt.store %o_addr, %x_val : tensor<128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// The masked load lowers to an scf.if whose then-branch yields the loaded
// element and whose else-branch yields the splat-constant `other` (0.5),
// not the v1 hardcoded zero. The else-branch const is metal.constant.
// CHECK: metal.kernel
// CHECK: scf.if
// CHECK: metal.constant 5.000000e-01

// MSL must contain the else-branch value `5.000000e-01` (the emitter uses
// scientific notation for floats) — distinct from the masked-no-other
// fixtures which carry `0` in their else branch.
// MSL: kernel void load_with_other_elementwise
// MSL: if (
// MSL: else
// MSL: 5.000000e-01
