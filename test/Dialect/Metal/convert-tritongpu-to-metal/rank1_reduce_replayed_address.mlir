// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | FileCheck %s --implicit-check-not=tt.reduce
//
// A rank-1 reduce above one element per thread rebuilds each element's address.
// The legacy single-load walker can only spell a contiguous range plus scalar
// pointer offsets, so these three shapes used to be refused by name. They are
// handed to the general per-element cone evaluator instead, which walks the
// same pointer chain and re-derives the address level by level.
// Neighbours that stay refused: reduce_unsupported.mlir's
// @reduce_rank1_bitand_address_deferred, an index the evaluator cannot rebuild.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  // A runtime stride in the load offset: the element at logical index i lives
  // at `i * stride`, so the multiply must survive into the address.
  tt.func public @reduce_rank1_strided_address(%x_ptr: !tt.ptr<f32>, %stride: i32) {
    %offsets = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %stride_splat = tt.splat %stride : i32 -> tensor<128xi32, #blocked>
    %strided = arith.muli %offsets, %stride_splat : tensor<128xi32, #blocked>
    %x_splat = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %x_addr = tt.addptr %x_splat, %strided : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    %x = tt.load %x_addr : tensor<128x!tt.ptr<f32>, #blocked>
    %r = "tt.reduce"(%x) ({
    ^bb0(%a: f32, %b: f32):
      %s = arith.addf %a, %b : f32
      tt.reduce.return %s : f32
    }) {axis = 0 : i32} : (tensor<128xf32, #blocked>) -> f32
    tt.return
  }

  // Two pointer levels, each carrying a tensor offset. Accumulating only the
  // outer one would drop the inner stride entirely.
  tt.func public @reduce_rank1_nested_tensor_address(%x_ptr: !tt.ptr<f32>, %stride: i32) {
    %offsets = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %stride_splat = tt.splat %stride : i32 -> tensor<128xi32, #blocked>
    %inner_offsets = arith.muli %offsets, %stride_splat : tensor<128xi32, #blocked>
    %x_splat = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %inner_addr = tt.addptr %x_splat, %inner_offsets : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    %x_addr = tt.addptr %inner_addr, %offsets : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    %x = tt.load %x_addr : tensor<128x!tt.ptr<f32>, #blocked>
    %r = "tt.reduce"(%x) ({
    ^bb0(%a: f32, %b: f32):
      %s = arith.addf %a, %b : f32
      tt.reduce.return %s : f32
    }) {axis = 0 : i32} : (tensor<128xf32, #blocked>) -> f32
    tt.return
  }

  // A bare splat pointer has no per-element offset at all: every lane reads the
  // same element, which the evaluator spells directly.
  tt.func public @reduce_rank1_bare_splat_address(%x_ptr: !tt.ptr<f32>) {
    %x_splat = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %x = tt.load %x_splat : tensor<128x!tt.ptr<f32>, #blocked>
    %r = "tt.reduce"(%x) ({
    ^bb0(%a: f32, %b: f32):
      %s = arith.addf %a, %b : f32
      tt.reduce.return %s : f32
    }) {axis = 0 : i32} : (tensor<128xf32, #blocked>) -> f32
    tt.return
  }
}

// CHECK-LABEL: metal.kernel reduce_rank1_strided_address
// The stride multiply reaches the address, so the fold is not reading a
// contiguous window. `--implicit-check-not=tt.reduce` covers the whole module:
// a refused reduce would also fail the RUN line outright, since the pass exits
// non-zero and prints nothing for FileCheck to read.
// CHECK: arith.muli
// CHECK: metal.get_element
// CHECK: metal.binary_exp {{.*}}, {{.*}}, addOp

// CHECK-LABEL: metal.kernel reduce_rank1_nested_tensor_address
// Both levels contribute: the inner multiply and an add of the outer index.
// CHECK: arith.muli
// CHECK: arith.addi
// CHECK: metal.get_element

// CHECK-LABEL: metal.kernel reduce_rank1_bare_splat_address
// CHECK: metal.get_element
// CHECK: metal.binary_exp {{.*}}, {{.*}}, addOp
