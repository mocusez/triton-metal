// RUN: triton-metal-opt --convert-tritongpu-to-metal --split-input-file %s | FileCheck %s

// A 2-D tile address with a UNIFORM axis.
//
// Triton builds such an address at the lower rank and broadcasts the POINTER
// tensor: `tt.broadcast : tensor<64x1x!tt.ptr<f32>> -> tensor<64x64x!tt.ptr<f32>>`
// is what `tl.full((BLOCK,), k, tl.int32)[None, :]` compiles to. Every gate used
// to refuse that root, and the decline killed the process on its way out.
//
// Nothing needs re-deriving for it. The offset under the broadcast was emitted
// against the CONSUMER's layout already — the `tt.make_range` beneath carries
// `#ttg.slice<{parent = <the tile's blocked layout>}>`, so its converted scalar
// is "my row index in the parent tile" and `tt.expand_dims` / `tt.broadcast` are
// identities over it. These fixtures pin that: the surviving index is the
// coordinate of the axis that is NOT uniform, and the uniform axis contributes
// nothing.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>

// The COLUMN axis is uniform, so the read index is the ROW coordinate — the
// flat position divided by the tile width.
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: metal.kernel load_through_broadcast_ptr
  tt.func public @load_through_broadcast_ptr(%x: !tt.ptr<f32>, %o: !tt.ptr<f32>) {
    %rows = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %cols = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %row2 = tt.expand_dims %rows {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %col2 = tt.expand_dims %cols {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>

    %xsp = tt.splat %x : !tt.ptr<f32> -> tensor<64x1x!tt.ptr<f32>, #blocked>
    %xrow = tt.addptr %xsp, %row2 : tensor<64x1x!tt.ptr<f32>, #blocked>, tensor<64x1xi32, #blocked>
    %xbc = tt.broadcast %xrow : tensor<64x1x!tt.ptr<f32>, #blocked> -> tensor<64x64x!tt.ptr<f32>, #blocked>
    // The row coordinate, and nothing from the broadcast axis.
    // CHECK: %[[ROW:.*]] = arith.divsi %{{.*}}, %c64_i32
    // CHECK: %[[RIDX:.*]] = builtin.unrealized_conversion_cast %[[ROW]] : i32 to ui32
    // CHECK: metal.get_element %{{.*}}[%[[RIDX]]]
    %v = tt.load %xbc : tensor<64x64x!tt.ptr<f32>, #blocked>

    %osp = tt.splat %o : !tt.ptr<f32> -> tensor<64x1x!tt.ptr<f32>, #blocked>
    %orow = tt.addptr %osp, %row2 : tensor<64x1x!tt.ptr<f32>, #blocked>, tensor<64x1xi32, #blocked>
    %obc = tt.broadcast %orow : tensor<64x1x!tt.ptr<f32>, #blocked> -> tensor<64x64x!tt.ptr<f32>, #blocked>
    %ocol = tt.broadcast %col2 : tensor<1x64xi32, #blocked> -> tensor<64x64xi32, #blocked>
    %oaddr = tt.addptr %obc, %ocol : tensor<64x64x!tt.ptr<f32>, #blocked>, tensor<64x64xi32, #blocked>
    // CHECK: metal.store
    tt.store %oaddr, %v : tensor<64x64x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>

// The mirror on the write side: the ROW axis is uniform, so the write index is
// the COLUMN coordinate — the flat position modulo the tile width.
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: metal.kernel store_through_broadcast_ptr
  tt.func public @store_through_broadcast_ptr(%x: !tt.ptr<f32>, %o: !tt.ptr<f32>) {
    %rows = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %cols = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %row2 = tt.expand_dims %rows {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %col2 = tt.expand_dims %cols {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>

    %xsp = tt.splat %x : !tt.ptr<f32> -> tensor<64x1x!tt.ptr<f32>, #blocked>
    %xrow = tt.addptr %xsp, %row2 : tensor<64x1x!tt.ptr<f32>, #blocked>, tensor<64x1xi32, #blocked>
    %xbc = tt.broadcast %xrow : tensor<64x1x!tt.ptr<f32>, #blocked> -> tensor<64x64x!tt.ptr<f32>, #blocked>
    %xcol = tt.broadcast %col2 : tensor<1x64xi32, #blocked> -> tensor<64x64xi32, #blocked>
    %xaddr = tt.addptr %xbc, %xcol : tensor<64x64x!tt.ptr<f32>, #blocked>, tensor<64x64xi32, #blocked>
    %v = tt.load %xaddr : tensor<64x64x!tt.ptr<f32>, #blocked>

    %osp = tt.splat %o : !tt.ptr<f32> -> tensor<1x64x!tt.ptr<f32>, #blocked>
    %ocolp = tt.addptr %osp, %col2 : tensor<1x64x!tt.ptr<f32>, #blocked>, tensor<1x64xi32, #blocked>
    %obc = tt.broadcast %ocolp : tensor<1x64x!tt.ptr<f32>, #blocked> -> tensor<64x64x!tt.ptr<f32>, #blocked>
    // CHECK: %[[COL:.*]] = arith.remsi %{{.*}}, %c64_i32
    // CHECK: %[[CIDX:.*]] = builtin.unrealized_conversion_cast %[[COL]] : i32 to ui32
    // CHECK: metal.store %{{.*}}, %{{.*}}[%[[CIDX]]]
    tt.store %obc, %v : tensor<64x64x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
