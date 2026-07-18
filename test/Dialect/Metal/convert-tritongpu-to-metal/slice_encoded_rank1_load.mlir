// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | FileCheck %s
//
// Rank-1 `tt.load` carrying `#ttg.slice<{dim, parent = #blocked2D}>`.
//
// A 1D vector that is expand_dims'd/broadcast into a 2D tile gets a slice
// encoding rather than a blocked one. `tileFromTensor` gates on
// BlockedEncodingAttr, so such a load used to fail to legalize outright
// ("tt.load operand missing ttg.blocked layout"). The load patterns now go
// through `tileFromLoadPtrTensor`, which reads the thread geometry off
// `slice.getParent()` — never off the slice itself, whose getThreadsPerWarp /
// getWarpsPerCTA return the dim-REMOVED vectors.
//
// The second, subtler half is the LIVE path. Under slice<dim=1,parent=B>
// thread t holds row t/N; under the rank-1 #blocked1 that the store wants it
// holds row t. Here the `tt.reduce` result (inherently slice-encoded) is added
// to the loaded vector, which forces both onto one encoding, so only the
// `ttg.convert_layout` separates them — and it was classified as a scalar
// identity, leaving every lane with element t/N (element 0 whenever N >= tpb).
// `normalizeBlockedDivergentCvt` now re-encodes that cone to #blocked1 and
// bridges the reduce result — which it may NOT re-type, since tt.reduce's
// result encoding is inferred from its 2D source — with a fresh cvt.
//
// Transcribed from a TRITON_KERNEL_DUMP of the frontend, not hand-shaped:
// hand-written IR can take a different lowering branch (see .omc/progress.txt).

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @slice_load_plus_reduce(%a_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %skip_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %y_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %d_model: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %other = arith.constant dense<0.000000e+00> : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %cN = arith.constant dense<64> : tensor<32x1xi32, #blocked>
    %d_off = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %d_off_0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #blocked1>
    %d_mask = tt.splat %d_model : i32 -> tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %d_mask_1 = tt.splat %d_model : i32 -> tensor<32xi32, #blocked1>
    %d_mask_2 = arith.cmpi slt, %d_off, %d_mask : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %d_mask_3 = arith.cmpi slt, %d_off_0, %d_mask_1 : tensor<32xi32, #blocked1>
    %a_4 = tt.expand_dims %d_off {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<32x1xi32, #blocked>
    %a_5 = arith.muli %a_4, %cN : tensor<32x1xi32, #blocked>
    %a_6 = tt.splat %a_ptr : !tt.ptr<f32> -> tensor<32x1x!tt.ptr<f32>, #blocked>
    %a_7 = tt.addptr %a_6, %a_5 : tensor<32x1x!tt.ptr<f32>, #blocked>, tensor<32x1xi32, #blocked>
    %a_8 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %a_9 = tt.expand_dims %a_8 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
    %a_10 = tt.broadcast %a_7 : tensor<32x1x!tt.ptr<f32>, #blocked> -> tensor<32x64x!tt.ptr<f32>, #blocked>
    %a_11 = tt.broadcast %a_9 : tensor<1x64xi32, #blocked> -> tensor<32x64xi32, #blocked>
    %a_12 = tt.addptr %a_10, %a_11 : tensor<32x64x!tt.ptr<f32>, #blocked>, tensor<32x64xi32, #blocked>
    %a_13 = tt.load %a_12 : tensor<32x64x!tt.ptr<f32>, #blocked>
    %s = "tt.reduce"(%a_13) <{axis = 1 : i32}> ({
    ^bb0(%s_17: f32, %s_18: f32):
      %s_19 = arith.addf %s_17, %s_18 : f32
      tt.reduce.return %s_19 : f32
    }) : (tensor<32x64xf32, #blocked>) -> tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %skip_14 = tt.splat %skip_ptr : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #blocked}>>
    %skip_15 = tt.addptr %skip_14, %d_off : tensor<32x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %skip_16 = tt.load %skip_15, %d_mask_2, %other : tensor<32x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #blocked}>>
    %0 = tt.splat %y_ptr : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>, #blocked1>
    %1 = tt.addptr %0, %d_off_0 : tensor<32x!tt.ptr<f32>, #blocked1>, tensor<32xi32, #blocked1>
    %2 = arith.addf %s, %skip_16 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %3 = ttg.convert_layout %2 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<32xf32, #blocked1>
    tt.store %1, %3, %d_mask_3 : tensor<32x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}

// The kernel legalizes at all — the slice-encoded load no longer bails.
// CHECK-LABEL: metal.kernel slice_load_plus_reduce

// localTid = tid.x - tgid.x*tpb. This is the FIRST subi emitted, and the
// re-encoded (#blocked1) cone must use it verbatim.
// CHECK: %[[LTID:[0-9a-z_]+]] = arith.subi

// The 2D tile path keeps its slice projection (row = idx / N) — normalizing the
// rank-1 cone must not disturb it.
// CHECK: arith.divsi

// CHECK: %[[MASK:[0-9a-z_]+]] = arith.cmpi slt, %[[LTID]],

// The rank-2 axis=1 reduce still stages its per-row sums in threadgroup memory.
// CHECK: metal.threadgroup_alloca : !metal.memref<32 x f32>

// The masked rank-1 load: guarded by the #blocked1 mask and addressed by
// localTid DIRECTLY. Before the fix this index came from the slice projection
// (localTid / 64), so every lane below tpb read element 0.
// CHECK: scf.if %[[MASK]] -> (f32) {
// CHECK-NEXT: %[[IDX:[0-9a-z_]+]] = builtin.unrealized_conversion_cast %[[LTID]] : i32 to ui32
// CHECK-NEXT: metal.get_element %arg1[%[[IDX]]]
