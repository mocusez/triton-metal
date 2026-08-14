// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | FileCheck %s
//
// Positive fixture: ttg.convert_layout with structurally equal src/dst types
// (same layout attribute) is treated as an identity passthrough by
// `ConvertLayoutLowering`. The pass replaces the cvt with its source value
// and erases the op. See
// `.omc/specs/deep-interview-leet-triton-l1c-cvt-passthrough.md`.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked4 = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
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

  // Although these encodings declare different vector widths, this backend's
  // scalar tile loop maps both to the same rank-1 distribution: tpb=128,
  // E=1024/128=8, contiguous index = localTid*8+iv. The source has an external
  // use so producer-cone normalization cannot erase the observable cvt.
  tt.func public @cvt_same_rank1_metal_distribution(%x_ptr: !tt.ptr<f32>, %source_out_ptr: !tt.ptr<f32>, %dest_out_ptr: !tt.ptr<f32>) {
    %off2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked2>
    %x = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked2>
    %xp = tt.addptr %x, %off2 : tensor<1024x!tt.ptr<f32>, #blocked2>, tensor<1024xi32, #blocked2>
    %v = tt.load %xp : tensor<1024x!tt.ptr<f32>, #blocked2>

    // External source-layout use keeps this conversion out of the cone
    // normalizer and exercises ConvertLayoutLowering's proven identity pair.
    %source_out = tt.splat %source_out_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked2>
    %source_out_p = tt.addptr %source_out, %off2 : tensor<1024x!tt.ptr<f32>, #blocked2>, tensor<1024xi32, #blocked2>
    tt.store %source_out_p, %v : tensor<1024x!tt.ptr<f32>, #blocked2>

    %cvt = ttg.convert_layout %v : tensor<1024xf32, #blocked2> -> tensor<1024xf32, #blocked4>
    %zero4 = arith.constant dense<0.000000e+00> : tensor<1024xf32, #blocked4>
    %used = arith.addf %cvt, %zero4 : tensor<1024xf32, #blocked4>
    %off4 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked4>
    %dest_out = tt.splat %dest_out_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked4>
    %dest_out_p = tt.addptr %dest_out, %off4 : tensor<1024x!tt.ptr<f32>, #blocked4>, tensor<1024xi32, #blocked4>
    tt.store %dest_out_p, %used : tensor<1024x!tt.ptr<f32>, #blocked4>
    tt.return
  }

  // A scan result is a normalization boundary, so this variant cannot be
  // legalized by rewriting its producer cone. It specifically locks the
  // blocked2 -> blocked4 identity admitted by their equal Metal distribution.
  tt.func public @cvt_same_rank1_scan_boundary(%x_ptr: !tt.ptr<f32>, %dest_out_ptr: !tt.ptr<f32>) {
    %off2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked2>
    %x = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked2>
    %xp = tt.addptr %x, %off2 : tensor<1024x!tt.ptr<f32>, #blocked2>, tensor<1024xi32, #blocked2>
    %v = tt.load %xp : tensor<1024x!tt.ptr<f32>, #blocked2>
    %scan = "tt.scan"(%v) <{axis = 0 : i32, reverse = false}> ({
    ^bb0(%a: f32, %b: f32):
      %sum = arith.addf %a, %b : f32
      tt.scan.return %sum : f32
    }) : (tensor<1024xf32, #blocked2>) -> tensor<1024xf32, #blocked2>
    %cvt = ttg.convert_layout %scan : tensor<1024xf32, #blocked2> -> tensor<1024xf32, #blocked4>
    %zero4 = arith.constant dense<0.000000e+00> : tensor<1024xf32, #blocked4>
    %used = arith.addf %cvt, %zero4 : tensor<1024xf32, #blocked4>
    %off4 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked4>
    %dest_out = tt.splat %dest_out_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked4>
    %dest_out_p = tt.addptr %dest_out, %off4 : tensor<1024x!tt.ptr<f32>, #blocked4>, tensor<1024xi32, #blocked4>
    tt.store %dest_out_p, %used : tensor<1024x!tt.ptr<f32>, #blocked4>
    tt.return
  }
}

// CHECK: metal.module
// CHECK: metal.kernel cvt_passthrough_kernel
// CHECK-NOT: ttg.convert_layout
// CHECK-NOT: convert_layout
// CHECK: metal.store
// CHECK: metal.return
// CHECK-LABEL: metal.kernel cvt_same_rank1_metal_distribution
// CHECK-NOT: ttg.convert_layout
// CHECK: metal.store
// CHECK: metal.store
// CHECK: metal.return
// CHECK-LABEL: metal.kernel cvt_same_rank1_scan_boundary
// CHECK: metal.threadgroup_prefix_sum
// CHECK-NOT: ttg.convert_layout
// CHECK: metal.return
// CHECK: metal.module_end
