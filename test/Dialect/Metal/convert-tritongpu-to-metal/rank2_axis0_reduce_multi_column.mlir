// RUN: triton-metal-opt --convert-tritongpu-to-metal --split-input-file %s | FileCheck %s
//
// A rank-2 axis=0 reduce whose columns outnumber the threadgroup's threads.
// Each thread then owns several output columns, one per tile-loop iteration,
// and WHICH one is a property of the layout of the value the result becomes --
// not something the tile-loop model may assume. This used to be refused on
// exactly that assumption.
//
// The check that matters in both fixtures below is that the column the reduce
// folds at and the index the store writes to are the SAME expression. One
// buffer under two index mappings is how this backend's silent wrong answers
// have always been spelled.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
#slice0 = #ttg.slice<{dim = 0, parent = #blocked}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @colsum_more_columns_than_threads(%in: !tt.ptr<f32>, %out: !tt.ptr<f32>) {
    %rows = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %rows2 = tt.expand_dims %rows {axis = 1 : i32} : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<8x1xi32, #blocked>
    %cN = arith.constant dense<64> : tensor<8x1xi32, #blocked>
    %rowsN = arith.muli %rows2, %cN : tensor<8x1xi32, #blocked>
    %cols = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #slice0>
    %cols2 = tt.expand_dims %cols {axis = 0 : i32} : tensor<64xi32, #slice0> -> tensor<1x64xi32, #blocked>
    %sp = tt.splat %in : !tt.ptr<f32> -> tensor<8x1x!tt.ptr<f32>, #blocked>
    %ap0 = tt.addptr %sp, %rowsN : tensor<8x1x!tt.ptr<f32>, #blocked>, tensor<8x1xi32, #blocked>
    %apb = tt.broadcast %ap0 : tensor<8x1x!tt.ptr<f32>, #blocked> -> tensor<8x64x!tt.ptr<f32>, #blocked>
    %colsb = tt.broadcast %cols2 : tensor<1x64xi32, #blocked> -> tensor<8x64xi32, #blocked>
    %ap = tt.addptr %apb, %colsb : tensor<8x64x!tt.ptr<f32>, #blocked>, tensor<8x64xi32, #blocked>
    %tile = tt.load %ap : tensor<8x64x!tt.ptr<f32>, #blocked>
    %s = "tt.reduce"(%tile) ({
    ^bb0(%a: f32, %b: f32):
      %add = arith.addf %a, %b : f32
      tt.reduce.return %add : f32
    }) {axis = 0 : i32} : (tensor<8x64xf32, #blocked>) -> tensor<64xf32, #slice0>
    %ocols = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked1>
    %osp = tt.splat %out : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>, #blocked1>
    %oap = tt.addptr %osp, %ocols : tensor<64x!tt.ptr<f32>, #blocked1>, tensor<64xi32, #blocked1>
    %sc = ttg.convert_layout %s : tensor<64xf32, #slice0> -> tensor<64xf32, #blocked1>
    tt.store %oap, %sc : tensor<64x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}

// The consuming tile interleaves the bands (sizePerThread = [1]), so the
// column is `localTid + iv * tpb`. The clamp is the first comparison in the
// loop body, which is the one anchor the dead address cone the load lowering
// leaves behind cannot be confused with.
// CHECK-LABEL: metal.kernel colsum_more_columns_than_threads
// CHECK: scf.for %[[IV:[a-z0-9_]+]] = %{{.*}} to %{{.*}} step %{{.*}}
// A column past the end of the row would address outside it. The store's own
// guard discards the value, but the read has to be clamped before it happens.
// CHECK:   %[[INROW:[0-9]+]] = arith.cmpi slt, %[[COLI:[0-9]+]], %{{.*}} : i32
// CHECK:   %[[SAFECOL:[0-9]+]] = arith.select %[[INROW]], %[[COLI]], %{{.*}} : i32
// The fold walks the rows at that one column.
// CHECK:   %[[SUM:[0-9]+]] = scf.for %[[M:[a-z0-9_]+]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ACC:[a-z0-9_]+]] = %{{.*}}) -> (f32)
// CHECK:     %[[ROWOFF:[0-9]+]] = arith.muli %[[M]], %{{.*}} : i32
// CHECK:     %[[ADDR:[0-9]+]] = arith.addi %[[SAFECOL]], %[[ROWOFF]] : i32
// CHECK:     metal.get_element %{{.*}}[%{{.*}}]
// CHECK:     arith.addf %[[ACC]]
// The store lands on the column the fold used, spelled the same way: the local
// thread id plus this iteration's band.
// CHECK:   %[[BAND:[0-9]+]] = arith.muli %[[IV]], %{{.*}} : i32
// CHECK:   %[[STORECOL:[0-9]+]] = arith.addi %{{.*}}, %[[BAND]] : i32
// CHECK:   %[[SIDX:[0-9]+]] = builtin.unrealized_conversion_cast %[[STORECOL]] : i32 to ui32
// CHECK:   metal.store %[[SUM]], %{{.*}}[%[[SIDX]]]

// -----

// The same reduce whose consumer holds a CONTIGUOUS band per thread
// (sizePerThread = [2]). Triton picks the interleaved layout for this shape
// family today, so this fixture is how the other half of the index function
// stays covered: the column becomes `localTid * E + iv`.
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
#slice0 = #ttg.slice<{dim = 0, parent = #blocked}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @colsum_contiguous_out_band(%in: !tt.ptr<f32>, %out: !tt.ptr<f32>) {
    %rows = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %rows2 = tt.expand_dims %rows {axis = 1 : i32} : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<8x1xi32, #blocked>
    %cN = arith.constant dense<64> : tensor<8x1xi32, #blocked>
    %rowsN = arith.muli %rows2, %cN : tensor<8x1xi32, #blocked>
    %cols = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #slice0>
    %cols2 = tt.expand_dims %cols {axis = 0 : i32} : tensor<64xi32, #slice0> -> tensor<1x64xi32, #blocked>
    %sp = tt.splat %in : !tt.ptr<f32> -> tensor<8x1x!tt.ptr<f32>, #blocked>
    %ap0 = tt.addptr %sp, %rowsN : tensor<8x1x!tt.ptr<f32>, #blocked>, tensor<8x1xi32, #blocked>
    %apb = tt.broadcast %ap0 : tensor<8x1x!tt.ptr<f32>, #blocked> -> tensor<8x64x!tt.ptr<f32>, #blocked>
    %colsb = tt.broadcast %cols2 : tensor<1x64xi32, #blocked> -> tensor<8x64xi32, #blocked>
    %ap = tt.addptr %apb, %colsb : tensor<8x64x!tt.ptr<f32>, #blocked>, tensor<8x64xi32, #blocked>
    %tile = tt.load %ap : tensor<8x64x!tt.ptr<f32>, #blocked>
    %s = "tt.reduce"(%tile) ({
    ^bb0(%a: f32, %b: f32):
      %add = arith.addf %a, %b : f32
      tt.reduce.return %add : f32
    }) {axis = 0 : i32} : (tensor<8x64xf32, #blocked>) -> tensor<64xf32, #slice0>
    %ocols = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked1>
    %osp = tt.splat %out : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>, #blocked1>
    %oap = tt.addptr %osp, %ocols : tensor<64x!tt.ptr<f32>, #blocked1>, tensor<64xi32, #blocked1>
    %sc = ttg.convert_layout %s : tensor<64xf32, #slice0> -> tensor<64xf32, #blocked1>
    tt.store %oap, %sc : tensor<64x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}

// CHECK-LABEL: metal.kernel colsum_contiguous_out_band
// CHECK: scf.for %[[IV2:[a-z0-9_]+]] = %{{.*}} to %{{.*}} step %{{.*}}
// CHECK:   %[[INROW2:[0-9]+]] = arith.cmpi slt, %[[COLI2:[0-9]+]], %{{.*}} : i32
// CHECK:   %[[SAFE2:[0-9]+]] = arith.select %[[INROW2]], %[[COLI2]], %{{.*}} : i32
// CHECK:   %[[SUM2:[0-9]+]] = scf.for %{{.*}} iter_args(%{{.*}} = %{{.*}}) -> (f32)
// CHECK:     arith.addi %[[SAFE2]], %{{.*}} : i32
// CHECK:     metal.get_element
// CHECK:     arith.addf
// CHECK:     scf.yield
// A contiguous band means the thread's own columns are adjacent, so the index
// is `localTid * E + iv` rather than `localTid + iv * tpb`.
// CHECK:   %[[SCOL:[0-9]+]] = arith.addi %{{.*}}, %[[IV2]] : i32
// CHECK:   %[[SIDX2:[0-9]+]] = builtin.unrealized_conversion_cast %[[SCOL]] : i32 to ui32
// CHECK:   metal.store %[[SUM2]], %{{.*}}[%[[SIDX2]]]

// -----
// A masked-off row contributes the load's `other`, not the combine identity.
// Folding the identity in instead answers a different program: a fully masked
// column came back as -inf for a max, and a partially masked one silently
// dropped those rows. Only a mask with NO `other` may take the identity — that
// value is unspecified, and the identity is what makes the result
// deterministic (the second module below pins that half of the rule).
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
#slice0 = #ttg.slice<{dim = 0, parent = #blocked}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @colmax_masked_other(%in: !tt.ptr<f32>, %out: !tt.ptr<f32>, %bound: i32, %other: f32) {
    %rows = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %rows2 = tt.expand_dims %rows {axis = 1 : i32} : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<8x1xi32, #blocked>
    %cN = arith.constant dense<32> : tensor<8x1xi32, #blocked>
    %rowoff = arith.muli %rows2, %cN : tensor<8x1xi32, #blocked>
    %cols = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #slice0>
    %cols2 = tt.expand_dims %cols {axis = 0 : i32} : tensor<32xi32, #slice0> -> tensor<1x32xi32, #blocked>
    %rowb = tt.broadcast %rowoff : tensor<8x1xi32, #blocked> -> tensor<8x32xi32, #blocked>
    %colb = tt.broadcast %cols2 : tensor<1x32xi32, #blocked> -> tensor<8x32xi32, #blocked>
    %offs = arith.addi %rowb, %colb : tensor<8x32xi32, #blocked>
    %bsplat = tt.splat %bound : i32 -> tensor<8x1xi32, #blocked>
    %rowok = arith.cmpi slt, %rows2, %bsplat : tensor<8x1xi32, #blocked>
    %mask = tt.broadcast %rowok : tensor<8x1xi1, #blocked> -> tensor<8x32xi1, #blocked>
    %osplat = tt.splat %other : f32 -> tensor<8x32xf32, #blocked>
    %base = tt.splat %in : !tt.ptr<f32> -> tensor<8x32x!tt.ptr<f32>, #blocked>
    %addr = tt.addptr %base, %offs : tensor<8x32x!tt.ptr<f32>, #blocked>, tensor<8x32xi32, #blocked>
    %tile = tt.load %addr, %mask, %osplat : tensor<8x32x!tt.ptr<f32>, #blocked>
    %red = "tt.reduce"(%tile) ({
    ^bb0(%a: f32, %b: f32):
      %mx = arith.maxnumf %a, %b : f32
      tt.reduce.return %mx : f32
    }) {axis = 0 : i32} : (tensor<8x32xf32, #blocked>) -> tensor<32xf32, #slice0>
    %ocols = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #blocked1>
    %obase = tt.splat %out : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>, #blocked1>
    %oaddr = tt.addptr %obase, %ocols : tensor<32x!tt.ptr<f32>, #blocked1>, tensor<32xi32, #blocked1>
    %cv = ttg.convert_layout %red : tensor<32xf32, #slice0> -> tensor<32xf32, #blocked1>
    tt.store %oaddr, %cv : tensor<32x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}
// CHECK-LABEL: metal.kernel colmax_masked_other
// The `other` kernel argument, read once.
// CHECK: [[OTHER:%.*]] = metal.get_element %arg3
// CHECK: scf.for {{.*}} iter_args({{.*}} = {{.*}}) -> (f32)
// CHECK: [[KEEP:%.*]] = arith.cmpi slt
// CHECK: metal.get_element %arg0
// The masked row folds in `other`, NOT an -inf identity constant.
// CHECK: arith.select [[KEEP]], %{{.*}}, [[OTHER]] : f32
// CHECK: metal.binary_exp {{.*}}, {{.*}}, maxOp
// CHECK: scf.yield

// -----
// The other half of the rule: with no `other` on the load, the masked row does
// take the combine identity, so a max folds -inf.
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
#slice0 = #ttg.slice<{dim = 0, parent = #blocked}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @colmax_masked_no_other(%in: !tt.ptr<f32>, %out: !tt.ptr<f32>, %bound: i32) {
    %rows = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %rows2 = tt.expand_dims %rows {axis = 1 : i32} : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<8x1xi32, #blocked>
    %cN = arith.constant dense<32> : tensor<8x1xi32, #blocked>
    %rowoff = arith.muli %rows2, %cN : tensor<8x1xi32, #blocked>
    %cols = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #slice0>
    %cols2 = tt.expand_dims %cols {axis = 0 : i32} : tensor<32xi32, #slice0> -> tensor<1x32xi32, #blocked>
    %rowb = tt.broadcast %rowoff : tensor<8x1xi32, #blocked> -> tensor<8x32xi32, #blocked>
    %colb = tt.broadcast %cols2 : tensor<1x32xi32, #blocked> -> tensor<8x32xi32, #blocked>
    %offs = arith.addi %rowb, %colb : tensor<8x32xi32, #blocked>
    %bsplat = tt.splat %bound : i32 -> tensor<8x1xi32, #blocked>
    %rowok = arith.cmpi slt, %rows2, %bsplat : tensor<8x1xi32, #blocked>
    %mask = tt.broadcast %rowok : tensor<8x1xi1, #blocked> -> tensor<8x32xi1, #blocked>
    %base = tt.splat %in : !tt.ptr<f32> -> tensor<8x32x!tt.ptr<f32>, #blocked>
    %addr = tt.addptr %base, %offs : tensor<8x32x!tt.ptr<f32>, #blocked>, tensor<8x32xi32, #blocked>
    %tile = tt.load %addr, %mask : tensor<8x32x!tt.ptr<f32>, #blocked>
    %red = "tt.reduce"(%tile) ({
    ^bb0(%a: f32, %b: f32):
      %mx = arith.maxnumf %a, %b : f32
      tt.reduce.return %mx : f32
    }) {axis = 0 : i32} : (tensor<8x32xf32, #blocked>) -> tensor<32xf32, #slice0>
    %ocols = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #blocked1>
    %obase = tt.splat %out : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>, #blocked1>
    %oaddr = tt.addptr %obase, %ocols : tensor<32x!tt.ptr<f32>, #blocked1>, tensor<32xi32, #blocked1>
    %cv = ttg.convert_layout %red : tensor<32xf32, #slice0> -> tensor<32xf32, #blocked1>
    tt.store %oaddr, %cv : tensor<32x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}
// CHECK-LABEL: metal.kernel colmax_masked_no_other
// CHECK: scf.for {{.*}} iter_args({{.*}} = {{.*}}) -> (f32)
// CHECK: [[KEEP2:%.*]] = arith.cmpi slt
// CHECK: metal.get_element %arg0
// CHECK: [[NEGINF:%.*]] = arith.constant 0xFF800000 : f32
// CHECK: arith.select [[KEEP2]], %{{.*}}, [[NEGINF]] : f32
// CHECK: metal.binary_exp {{.*}}, {{.*}}, maxOp
