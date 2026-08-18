// RUN: triton-opt %s --convert-triton-to-tritongpu='num-warps=8 threads-per-warp=32 target=metal:m1' -tritongpu-coalesce -tritongpu-propagate-coalesced-layouts | FileCheck %s --check-prefix=METAL
// RUN: triton-opt %s --convert-triton-to-tritongpu='num-warps=8 threads-per-warp=32 target=cuda:80' -tritongpu-coalesce -tritongpu-propagate-coalesced-layouts | FileCheck %s --check-prefix=CUDA

// Rank-1 sum reduce with BLOCK=1024, num_warps=8, threads_per_warp=32 →
// threads-per-block=256, BLOCK/tpb=4, f32 vector cap=4 → Coalesce promotes
// the load to sizePerThread=[4] and inserts a down-convert back to the
// default sizePerThread=[1] before the reduce.
//
// propagate-coalesced-layouts removes the down-convert and lets the reduce
// consume the sizePerThread=[4] layout directly.
//
// ⚠️ The pass is NOT target-gated, and the second RUN line is here to pin that
// down rather than to contradict it. The gate was removed deliberately: the
// Metal compiler stitches `ttg.target = "cuda:80"` via
// `_TTGPUIR_PARSER_STUB_TRIPLE` for parser compatibility, so a
// `starts_with("metal:")` check rejected the pass's own intended target.
// PIPELINE WIRING is the gate — only `third_party/metal/backend/compiler.py`
// adds this pass — so a module that reaches it behaves the same whatever its
// target string says. The CUDA checks below asserted the deleted gate and had
// been failing ever since; they now assert what the pass does.
//
// METAL: #ttg.blocked<{sizePerThread = [4]
// METAL-LABEL: tt.func public @reduce_sum_rank1
// METAL: tt.load {{.*}} : tensor<1024x!tt.ptr<f32>, #{{[a-z0-9]+}}>
// METAL-NOT: ttg.convert_layout {{.*}} : tensor<1024xf32, #{{[a-z0-9]+}}> -> tensor<1024xf32, #
// METAL: "tt.reduce"({{.*}}) <{axis = 0 : i32}>
// METAL: (tensor<1024xf32, #{{[a-z0-9]+}}>)
//
// CUDA: #ttg.blocked<{sizePerThread = [4]
// CUDA-LABEL: tt.func public @reduce_sum_rank1
// CUDA: tt.load {{.*}} : tensor<1024x!tt.ptr<f32>, #{{[a-z0-9]+}}>
// CUDA-NOT: ttg.convert_layout {{.*}} : tensor<1024xf32, #{{[a-z0-9]+}}> -> tensor<1024xf32, #
// CUDA: "tt.reduce"({{.*}}) <{axis = 0 : i32}>
// CUDA: (tensor<1024xf32, #{{[a-z0-9]+}}>)
module {
  tt.func public @reduce_sum_rank1(
      %X: !tt.ptr<f32> {tt.divisibility = 16 : i32},
      %O: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %offs = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %xp = tt.splat %X : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %xps = tt.addptr %xp, %offs : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    %x = tt.load %xps : tensor<1024x!tt.ptr<f32>>
    %s = "tt.reduce"(%x) <{axis = 0 : i32}> ({
      ^bb0(%a: f32, %b: f32):
        %r = arith.addf %a, %b : f32
        tt.reduce.return %r : f32
    }) : (tensor<1024xf32>) -> f32
    tt.store %O, %s : !tt.ptr<f32>
    tt.return
  }
}
