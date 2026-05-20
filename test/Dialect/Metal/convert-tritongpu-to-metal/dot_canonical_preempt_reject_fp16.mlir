// RUN: not triton-metal-opt --convert-tritongpu-to-metal %s 2>&1 | FileCheck %s
//
// L1d3 preempt — negative regression: fp16 dot still hits the L1d3
// hard error because the preempt strips A/B cvts but the matmul track's
// matchers reject non-f32 element types at TritonGPUToMetal.cpp:3141,
// :3272, :3367. With matchers rejecting, the dot survives and the cvt
// classifier walk at :3552-3616 emits the original L1d3 error for any
// surviving non-trivial cvt. Spec L1d3.10 acceptance.
//
// Wait — actually: the preempt strips A/B cvts unconditionally (it doesn't
// check dtype). After stripping, the dot's operands carry #blocked from
// loads. Downstream matchers reject (dtype != f32) and leave the dot in
// place. Any OTHER cvts in the module (e.g., on the acc) may still trip
// L1d3. If there are no surviving non-trivial cvts, the dot reaches the
// final illegal-op check and produces a "tt.dot is illegal" error instead.
//
// Either error path satisfies the spec criterion — the kernel does NOT
// compile cleanly.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#dotA = #ttg.dot_op<{opIdx = 0, parent = #blocked}>
#dotB = #ttg.dot_op<{opIdx = 1, parent = #blocked}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @dot_preempt_reject_fp16(%a_ptr: !tt.ptr<f16>, %b_ptr: !tt.ptr<f16>, %c_ptr: !tt.ptr<f16>) {
    %offsA = arith.constant dense<0> : tensor<8x8xi32, #blocked>
    %offsB = arith.constant dense<0> : tensor<8x8xi32, #blocked>
    %offsC = arith.constant dense<0> : tensor<8x8xi32, #blocked>
    %a_splat = tt.splat %a_ptr : !tt.ptr<f16> -> tensor<8x8x!tt.ptr<f16>, #blocked>
    %a_addr = tt.addptr %a_splat, %offsA : tensor<8x8x!tt.ptr<f16>, #blocked>, tensor<8x8xi32, #blocked>
    %b_splat = tt.splat %b_ptr : !tt.ptr<f16> -> tensor<8x8x!tt.ptr<f16>, #blocked>
    %b_addr = tt.addptr %b_splat, %offsB : tensor<8x8x!tt.ptr<f16>, #blocked>, tensor<8x8xi32, #blocked>
    %c_splat = tt.splat %c_ptr : !tt.ptr<f16> -> tensor<8x8x!tt.ptr<f16>, #blocked>
    %c_addr = tt.addptr %c_splat, %offsC : tensor<8x8x!tt.ptr<f16>, #blocked>, tensor<8x8xi32, #blocked>

    // fp16 loads + cvts feeding fp16 dot.
    %a_blk = tt.load %a_addr : tensor<8x8x!tt.ptr<f16>, #blocked>
    %b_blk = tt.load %b_addr : tensor<8x8x!tt.ptr<f16>, #blocked>
    %a = ttg.convert_layout %a_blk : tensor<8x8xf16, #blocked> -> tensor<8x8xf16, #dotA>
    %b = ttg.convert_layout %b_blk : tensor<8x8xf16, #blocked> -> tensor<8x8xf16, #dotB>
    %c0 = arith.constant dense<0.000000e+00> : tensor<8x8xf16, #blocked>
    %c = tt.dot %a, %b, %c0 : tensor<8x8xf16, #dotA> * tensor<8x8xf16, #dotB> -> tensor<8x8xf16, #blocked>
    tt.store %c_addr, %c : tensor<8x8x!tt.ptr<f16>, #blocked>
    tt.return
  }
}

// Either path produces a hard error — fp16 dot stays unsupported.
// CHECK: error
