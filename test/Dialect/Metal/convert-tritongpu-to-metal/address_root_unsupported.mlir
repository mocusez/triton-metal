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
