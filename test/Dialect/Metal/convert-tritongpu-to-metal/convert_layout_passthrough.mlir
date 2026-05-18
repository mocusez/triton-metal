// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | FileCheck %s
//
// Positive fixture: ttg.convert_layout with structurally equal src/dst types
// (same layout attribute) is treated as an identity passthrough by
// `ConvertLayoutLowering`. The pass replaces the cvt with its source value
// and erases the op. See
// `.omc/specs/deep-interview-leet-triton-l1c-cvt-passthrough.md`.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @cvt_passthrough_kernel(%x_ptr: !tt.ptr<f32>, %output_ptr: !tt.ptr<f32>) {
    %c128_i32 = arith.constant 128 : i32
    %pid = tt.get_program_id x : i32
    %block_start = arith.muli %pid, %c128_i32 : i32
    %offsets = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %bs_splat = tt.splat %block_start : i32 -> tensor<128xi32, #blocked>
    %abs_off = arith.addi %bs_splat, %offsets : tensor<128xi32, #blocked>
    %x_splat = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %x_addr = tt.addptr %x_splat, %abs_off : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    %x_val = tt.load %x_addr : tensor<128x!tt.ptr<f32>, #blocked>
    // Hand-authored same-attr cvt: src and dst types are identical
    // (same #blocked, same shape, same element type). Must be replaced
    // by passthrough and erased.
    %x_cvt = ttg.convert_layout %x_val : tensor<128xf32, #blocked> -> tensor<128xf32, #blocked>
    %o_splat = tt.splat %output_ptr : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %o_addr = tt.addptr %o_splat, %abs_off : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    tt.store %o_addr, %x_cvt : tensor<128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// CHECK: metal.module
// CHECK: metal.kernel cvt_passthrough_kernel
// CHECK-NOT: ttg.convert_layout
// CHECK-NOT: convert_layout
// CHECK: metal.store
// CHECK: metal.return
// CHECK: metal.module_end
