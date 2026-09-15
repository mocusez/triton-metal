// RUN: triton-metal-opt --convert-tritongpu-to-metal --verify-diagnostics --split-input-file %s

// What the tensor-address envelope still refuses, and the fact that it says so
// BEFORE conversion.
//
// The four load/store patterns decline with `notifyMatchFailure` inside
// `applyFullConversion`, and this backend does not survive that: the failed
// conversion's rollback leaves the module unverifiable and the process exits on
// a signal. `validateTensorAddressSupport` asks the patterns' own predicate
// (`unsupportedTensorAddressReason`) early enough to produce a diagnostic.
//
// The shapes that LOWER are pinned next door in
// `address_root_broadcast_ptr.mlir`.

#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

// A uniform splat pointer under a WIDE store tile: every lane targets one
// address holding values that need not agree. A race, not a store — refused
// permanently, not pending an implementation. `validateUnsupportedOpsRejected`
// gets here first with the more specific message; pinned so that stays true.
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @store_through_wide_splat_ptr(%x: !tt.ptr<f32>, %o: !tt.ptr<f32>) {
    %idx = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked1>
    %xsp = tt.splat %x : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked1>
    %xaddr = tt.addptr %xsp, %idx : tensor<128x!tt.ptr<f32>, #blocked1>, tensor<128xi32, #blocked1>
    %v = tt.load %xaddr : tensor<128x!tt.ptr<f32>, #blocked1>
    %osp = tt.splat %o : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked1>
    // expected-error @+1 {{Metal backend: an unmasked store through a uniform (tt.splat) pointer sends every element of the tile to one address, which is a race, not a store; only a single-element tile is supported}}
    tt.store %osp, %v : tensor<128x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>

// A load whose address root is a `ttg.convert_layout` of a pointer tensor. The
// store side peels a rank-1 shape-preserving cvt (it re-reads the store through
// the source layout); the load side has no such path, so the root is named
// rather than left to the pattern.
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @load_through_converted_ptr(%x: !tt.ptr<f32>, %o: !tt.ptr<f32>) {
    %rows = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %row2 = tt.expand_dims %rows {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %xsp = tt.splat %x : !tt.ptr<f32> -> tensor<64x1x!tt.ptr<f32>, #blocked>
    %xrow = tt.addptr %xsp, %row2 : tensor<64x1x!tt.ptr<f32>, #blocked>, tensor<64x1xi32, #blocked>
    %xcvt = ttg.convert_layout %xrow : tensor<64x1x!tt.ptr<f32>, #blocked> -> tensor<64x1x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>>
    // expected-error @+1 {{Metal backend: tt.load expects a tt.addptr or a splat scalar ptr feeding ptr}}
    %v = tt.load %xcvt : tensor<64x1x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>>
    %osp = tt.splat %o : !tt.ptr<f32> -> tensor<64x1x!tt.ptr<f32>, #blocked>
    %oaddr = tt.addptr %osp, %row2 : tensor<64x1x!tt.ptr<f32>, #blocked>, tensor<64x1xi32, #blocked>
    %ocvt = ttg.convert_layout %v : tensor<64x1xf32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>> -> tensor<64x1xf32, #blocked>
    tt.store %oaddr, %ocvt : tensor<64x1x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
