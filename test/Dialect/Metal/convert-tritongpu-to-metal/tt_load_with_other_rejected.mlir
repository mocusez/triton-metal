// RUN: triton-metal-opt --convert-tritongpu-to-metal --verify-diagnostics --split-input-file %s
//
// Negative fixture: tt.load with an `other` operand feeding a `tt.dot`
// operand position is rejected by the dot-prepass. Elementwise other-loads
// are accepted via MaskedLoadLowering (constant or runtime-uniform splat);
// this fixture validates that the dot-operand-position guardrail remains
// intact. See `.omc/specs/deep-interview-leet-triton-l1-refine-and-ship.md`
// §3.0 ADR.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#dotA = #ttg.dot_op<{opIdx = 0, parent = #blocked}>
#dotB = #ttg.dot_op<{opIdx = 1, parent = #blocked}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @dot_with_other(%a_ptr: !tt.ptr<f32>, %b_ptr: !tt.ptr<f32>, %c_ptr: !tt.ptr<f32>) {
    %offsA = arith.constant dense<0> : tensor<8x8xi32, #dotA>
    %offsB = arith.constant dense<0> : tensor<8x8xi32, #dotB>
    %offsC = arith.constant dense<0> : tensor<8x8xi32, #blocked>
    %a_splat = tt.splat %a_ptr : !tt.ptr<f32> -> tensor<8x8x!tt.ptr<f32>, #dotA>
    %a_addr = tt.addptr %a_splat, %offsA : tensor<8x8x!tt.ptr<f32>, #dotA>, tensor<8x8xi32, #dotA>
    %b_splat = tt.splat %b_ptr : !tt.ptr<f32> -> tensor<8x8x!tt.ptr<f32>, #dotB>
    %b_addr = tt.addptr %b_splat, %offsB : tensor<8x8x!tt.ptr<f32>, #dotB>, tensor<8x8xi32, #dotB>
    %c_splat = tt.splat %c_ptr : !tt.ptr<f32> -> tensor<8x8x!tt.ptr<f32>, #blocked>
    %c_addr = tt.addptr %c_splat, %offsC : tensor<8x8x!tt.ptr<f32>, #blocked>, tensor<8x8xi32, #blocked>

    // Mask + splat-constant `other` on the load feeding tt.dot operand 0.
    %true_i1 = arith.constant true
    %mask = tt.splat %true_i1 : i1 -> tensor<8x8xi1, #dotA>
    %zero_f = arith.constant 0.0 : f32
    %other = tt.splat %zero_f : f32 -> tensor<8x8xf32, #dotA>

    // expected-error @+1 {{`other` operand not supported in tt.dot operand position}}
    %a = tt.load %a_addr, %mask, %other : tensor<8x8x!tt.ptr<f32>, #dotA>
    %b = tt.load %b_addr : tensor<8x8x!tt.ptr<f32>, #dotB>
    %c0 = arith.constant dense<0.000000e+00> : tensor<8x8xf32, #blocked>
    %c = tt.dot %a, %b, %c0 : tensor<8x8xf32, #dotA> * tensor<8x8xf32, #dotB> -> tensor<8x8xf32, #blocked>
    tt.store %c_addr, %c : tensor<8x8x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
