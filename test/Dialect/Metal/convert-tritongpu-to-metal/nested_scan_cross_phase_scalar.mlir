// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | FileCheck %s
//
// A nested scan whose post-scan phase reads a scalar computed in the pre-scan
// phase (`bias = row * 3 + 1`, read by a pure scalar loop after the scan). The
// value must be re-created after the first band loop, and so must the pure
// scalar producer it reads from inside that band (`row * 3`): cloning `bias`
// alone left it reading a value the band loop's scope hides.
// CHECK-LABEL: metal.kernel _rankn_scan_cross_phase_pure_region_kernel
// CHECK: metal.tg_store_indexed
// CHECK: } {metal.scan_band_loop}
// CHECK-NEXT: %[[MUL:.*]] = arith.muli %arg3, %c3_i32 : i32
// CHECK-NEXT: %[[BIAS:.*]] = arith.addi %[[MUL]], %c1_i32 : i32
// CHECK-NEXT: metal.barrier
// CHECK: metal.tg_load_indexed
// CHECK: scf.for {{.*}} iter_args
// CHECK: arith.addi %{{.*}}, %[[BIAS]] : i32
#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @_rankn_scan_cross_phase_pure_region_kernel(%x_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %out_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %rows: i32) attributes {noinline = false} {
    %c2_i32 = arith.constant 2 : i32
    %c3_i32 = arith.constant 3 : i32
    %c512_i32 = arith.constant 512 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %index = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32, #blocked>
    %index_0 = tt.reshape %index : tensor<512xi32, #blocked> -> tensor<8x64xi32, #blocked1>
    scf.for %row = %c0_i32 to %rows step %c1_i32  : i32 {
      %x = arith.muli %row, %c512_i32 : i32
      %x_1 = tt.addptr %x_ptr, %x : !tt.ptr<f32>, i32
      %x_2 = tt.splat %x_1 : !tt.ptr<f32> -> tensor<8x64x!tt.ptr<f32>, #blocked1>
      %x_3 = tt.addptr %x_2, %index_0 : tensor<8x64x!tt.ptr<f32>, #blocked1>, tensor<8x64xi32, #blocked1>
      %x_4 = tt.load %x_3 : tensor<8x64x!tt.ptr<f32>, #blocked1>
      %bias = arith.muli %row, %c3_i32 : i32
      %bias_5 = arith.addi %bias, %c1_i32 : i32
      %0 = arith.addi %row, %c1_i32 : i32
      %delta = scf.for %col = %c0_i32 to %0 step %c1_i32 iter_args(%delta_6 = %c0_i32) -> (i32)  : i32 {
        %delta_7 = arith.muli %col, %c2_i32 : i32
        %delta_8 = arith.addi %delta_7, %bias_5 : i32
        %delta_9 = arith.addi %delta_6, %delta_8 : i32
        scf.yield %delta_9 : i32
      }
      %scanned = "tt.scan"(%x_4) <{axis = 1 : i32, reverse = false}> ({
      ^bb0(%scanned_6: f32, %scanned_7: f32):
        %scanned_8 = arith.addf %scanned_6, %scanned_7 : f32
        tt.scan.return %scanned_8 : f32
      }) : (tensor<8x64xf32, #blocked1>) -> tensor<8x64xf32, #blocked1>
      %1 = tt.addptr %out_ptr, %x : !tt.ptr<f32>, i32
      %2 = tt.splat %1 : !tt.ptr<f32> -> tensor<8x64x!tt.ptr<f32>, #blocked1>
      %3 = tt.addptr %2, %index_0 : tensor<8x64x!tt.ptr<f32>, #blocked1>, tensor<8x64xi32, #blocked1>
      %4 = arith.sitofp %delta : i32 to f32
      %5 = tt.splat %4 : f32 -> tensor<8x64xf32, #blocked1>
      %6 = arith.addf %scanned, %5 : tensor<8x64xf32, #blocked1>
      tt.store %3, %6 : tensor<8x64x!tt.ptr<f32>, #blocked1>
    }
    tt.return
  }
}
