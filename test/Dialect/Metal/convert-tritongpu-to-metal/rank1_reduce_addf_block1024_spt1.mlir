// RUN: triton-opt --tritongpu-propagate-coalesced-layouts %s | FileCheck %s --check-prefix=PROP
//
// Wall 6 lit fixture: spt=[1] input, BLOCK=1024, num_warps=8.
// PropagateCoalescedLayouts (phase 1, Wall 6) inserts an spt=[4]
// ttg.convert_layout immediately before the qualifying rank-1 tt.reduce.
//
// End-to-end lowering to metal is exercised via the python tutorial
// (leet-triton/tutorials_python/02-fused-softmax.py) where the inserted cvt
// is subsequently folded into the load by tritongpu-remove-layout-conversions
// before the metal conversion pass runs. A standalone METAL CHECK would need
// to mirror the full ttgir pipeline (coalesce → propagate → remove-layout
// → optimize → canon → cse → symbol-dce) which is brittle to mirror in lit.
//
// Predecessor: rank1_reduce_addf_block1024.mlir (same kernel but starts
// from spt=[4] — no cvt insertion needed there).
// See .omc/plans/tutorial02-wall6-spt1-reduce-consensus.md AC4/AC9.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "metal:m1", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @rank1_reduce_addf_block1024_spt1(%x_ptr: !tt.ptr<f32>) {
    %arange = tt.make_range {start = 0 : i32, end = 1024 : i32} : tensor<1024xi32, #blocked>
    %x_ptrs = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %x_ptrs_off = tt.addptr %x_ptrs, %arange : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    %x = tt.load %x_ptrs_off : tensor<1024x!tt.ptr<f32>, #blocked>
    %r = "tt.reduce"(%x) ({
    ^bb0(%a: f32, %b: f32):
      %s = arith.addf %a, %b : f32
      tt.reduce.return %s : f32
    }) {axis = 0 : i32} : (tensor<1024xf32, #blocked>) -> f32
    tt.return
  }
}

// --- PROP checks (AC9): exactly one inserted cvt mapping #blocked → spt=[4].
// PROP: #[[BLOCKED1:.*]] = #ttg.blocked<{sizePerThread = [4],
// PROP-LABEL: tt.func public @rank1_reduce_addf_block1024_spt1
// PROP: ttg.convert_layout
// PROP-SAME: -> tensor<1024xf32, #[[BLOCKED1]]>
// PROP-NEXT: tt.reduce
