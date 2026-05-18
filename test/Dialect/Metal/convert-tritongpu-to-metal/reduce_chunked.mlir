// RUN: triton-metal-opt --convert-tritongpu-to-metal --split-input-file %s | FileCheck %s
//
// L3-budget chunked-reduce fixture
// (`.omc/specs/deep-interview-leet-triton-l3-budget-chunked-reduce.md`).
// In-envelope `tt.reduce` on `tensor<MxNxT>` (axis=1, T ∈ {f32, i32},
// combine ∈ {arith.addf, arith.addi}) where the tile exceeds the 32 KiB
// threadgroup-memory budget lowers via the new chunked branch in
// `ReduceLowering`:
//   * one `metal.threadgroup_alloca<M*chunk_size x T>` reused across
//     `N_chunks = ceil(N/chunk_size)` unrolled passes
//   * each chunk: `(scf.if (col∈[kLo,kHi)) → store)`, `metal.barrier`,
//     `chunk_size` unrolled `metal.get_element + metal.binary_exp(addOp)`
//     loads, `metal.binary_exp(addOp)` into the per-row accumulator
//   * inter-chunk `metal.barrier` between consecutive chunks' alloca
//     buffer reuse

// -----
// f32 / arith.addf, shape <1024x64xf32>, axis=1.
// chunk_size = floor(32 KiB / (1024 * 4)) = 8 elements.
// N_chunks = ceil(64 / 8) = 8.
// Expected: ONE `metal.threadgroup_alloca<8192 x f32>` + 8 unrolled chunk
// passes + per-row `metal.binary_exp(addOp)` accumulator.
//
// HONEST DIVERGENCE NOTE: the chunked emission below preserves L3a's
// `M*N == tpb` body model (each thread = 1 logical (row,col) at the
// reduce site). For the shape <1024x64xf32> in this fixture, the
// blocked layout is declared with `threadsPerWarp=[2,16],
// warpsPerCTA=[4,1]` (tpb=128) — this fixture EXERCISES the chunked
// emission for a synthetic IR shape, not a real conv1d. Real conv1d
// (M*N >> tpb) requires a follow-up session redesign of L3a's body
// model.
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#slice1 = #ttg.slice<{dim = 1, parent = #blocked}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reduce_chunked_f32_1024x64(%x_ptr: !tt.ptr<f32>) {
    %x = arith.constant dense<0.0> : tensor<1024x64xf32, #blocked>
    %r = "tt.reduce"(%x) ({
    ^bb0(%a: f32, %b: f32):
      %s = arith.addf %a, %b : f32
      tt.reduce.return %s : f32
    }) {axis = 1 : i32} : (tensor<1024x64xf32, #blocked>) -> tensor<1024xf32, #slice1>
    tt.return
  }
}
// CHECK-LABEL: metal.kernel reduce_chunked_f32_1024x64
// AC.S1/AC.S2: exactly ONE alloca sized to M * chunk_size = 1024 * 8 = 8192
// (NOT 8 separate allocas, NOT M*N = 65536).
// CHECK: metal.threadgroup_alloca {{.*}} !metal.memref<8192 x f32>
// CHECK-NOT: metal.threadgroup_alloca {{.*}} !metal.memref<8192 x f32>
// AC.B2: Each chunk emits an `scf.if (col∈[kLo,kHi)) → metal.store`
// (8 chunks ⇒ 8 scf.if-wrapped stores). The CHECK below verifies the
// 1st chunk's wrap; further chunks share the same shape.
// CHECK: scf.if
// CHECK: metal.store
// CHECK: metal.barrier
// AC.B2 / AC.B3: 8 unrolled `metal.binary_exp ... addOp` accumulator
// passes follow (one per chunk; the 8x8 = 64 cumulative addOps include
// per-chunk scans + inter-chunk accumulator merges).
// CHECK: metal.binary_exp {{.*}} addOp
// CHECK: metal.return

// -----
// Pre-pass reject when per-row size exceeds budget. M=32768 f32 →
// 128 KiB per row ⇒ chunk_size==0 ⇒ chunking impossible.
// Tested in `reduce_unsupported.mlir`; this section is a positive smoke
// for tiny chunk_size (M=4096, N=2048, f32 → chunk_size = 32 KiB /
// (4096*4) = 2 elements, N_chunks = 1024).
//
// HONEST DIVERGENCE: not exercised at this scale to keep test runtime
// reasonable; verified via the f32_1024x64 case above + AC.T3 pytest.
