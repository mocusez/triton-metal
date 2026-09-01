// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | FileCheck %s
// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | triton-metal-translate --mlir-to-msl | FileCheck %s --check-prefix=MSL
//
// Wall 14 fixture (updated for Wall 15): BLOCK > tpb + Wall-11 walker
// chain (softmax 2nd-reduce shape). num_warps=8, threads_per_warp=32 ⇒
// tpb=256; BLOCK=4096 ⇒ E=16. Chain: masked tt.load → arith.subf
// splat(rmax) → math.exp → tt.reduce addf.
//
// Wall 15: the Wall-11 chain re-emission now runs ONCE PER LOOP ITER
// inside the scf.for body. Combiner-identity arith.select fires inside
// the body after the chain, before the combine step.
//
// See the implementation notes AC8.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
#blocked_spt4 = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @rank1_reduce_addf_block4096_chain(%x_ptr: !tt.ptr<f32>, %n_cols: i32, %rmax: f32) {
    %offsets = tt.make_range {end = 4096 : i32, start = 0 : i32} : tensor<4096xi32, #blocked>
    %n_splat = tt.splat %n_cols : i32 -> tensor<4096xi32, #blocked>
    %mask = arith.cmpi slt, %offsets, %n_splat : tensor<4096xi32, #blocked>
    %neg_inf = arith.constant dense<0xFF800000> : tensor<4096xf32, #blocked>
    %x_splat = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<4096x!tt.ptr<f32>, #blocked>
    %x_addr = tt.addptr %x_splat, %offsets : tensor<4096x!tt.ptr<f32>, #blocked>, tensor<4096xi32, #blocked>
    %row = tt.load %x_addr, %mask, %neg_inf : tensor<4096x!tt.ptr<f32>, #blocked>
    %rmax_t = tt.splat %rmax : f32 -> tensor<4096xf32, #blocked>
    %diff = arith.subf %row, %rmax_t : tensor<4096xf32, #blocked>
    %ex = math.exp %diff : tensor<4096xf32, #blocked>
    %sum1 = "tt.reduce"(%ex) ({
    ^bb0(%a: f32, %b: f32):
      %s = arith.addf %a, %b : f32
      tt.reduce.return %s : f32
    }) {axis = 0 : i32} : (tensor<4096xf32, #blocked>) -> f32
    tt.return
  }

  // A constexpr mask bound canonicalizes to arith.constant dense<splat>, not
  // tt.splat(scalar). This is the stats-reduce shape used by Leet GroupNorm.
  tt.func public @rank1_reduce_addf_block4096_dense_mask(%x_ptr: !tt.ptr<f32>) {
    %offsets = tt.make_range {end = 4096 : i32, start = 0 : i32} : tensor<4096xi32, #blocked_spt4>
    %bound = arith.constant dense<8> : tensor<4096xi32, #blocked_spt4>
    %mask = arith.cmpi slt, %offsets, %bound : tensor<4096xi32, #blocked_spt4>
    %zero = arith.constant dense<0.0> : tensor<4096xf32, #blocked_spt4>
    %x_splat = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<4096x!tt.ptr<f32>, #blocked_spt4>
    %x_addr = tt.addptr %x_splat, %offsets : tensor<4096x!tt.ptr<f32>, #blocked_spt4>, tensor<4096xi32, #blocked_spt4>
    %row = tt.load %x_addr, %mask, %zero : tensor<4096x!tt.ptr<f32>, #blocked_spt4>
    %sum = "tt.reduce"(%row) ({
    ^bb0(%a: f32, %b: f32):
      %s = arith.addf %a, %b : f32
      tt.reduce.return %s : f32
    }) {axis = 0 : i32} : (tensor<4096xf32, #blocked_spt4>) -> f32
    tt.return
  }

  // A computed cone may use the same tensor SSA value more than once. The
  // per-logical-index replay must rebuild the source once and share it across
  // both operands, rather than reloading the same device element for `d * d`.
  tt.func public @rank1_reduce_addf_block4096_square_reuses_load(%x_ptr: !tt.ptr<f32>) {
    %offsets = tt.make_range {end = 4096 : i32, start = 0 : i32} : tensor<4096xi32, #blocked_spt4>
    %one = arith.constant dense<1.0> : tensor<4096xf32, #blocked_spt4>
    %x_splat = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<4096x!tt.ptr<f32>, #blocked_spt4>
    %x_addr = tt.addptr %x_splat, %offsets : tensor<4096x!tt.ptr<f32>, #blocked_spt4>, tensor<4096xi32, #blocked_spt4>
    %row = tt.load %x_addr : tensor<4096x!tt.ptr<f32>, #blocked_spt4>
    %d = arith.subf %row, %one : tensor<4096xf32, #blocked_spt4>
    %sq = arith.mulf %d, %d : tensor<4096xf32, #blocked_spt4>
    %sum = "tt.reduce"(%sq) ({
    ^bb0(%a: f32, %b: f32):
      %s = arith.addf %a, %b : f32
      tt.reduce.return %s : f32
    }) {axis = 0 : i32} : (tensor<4096xf32, #blocked_spt4>) -> f32
    tt.return
  }

  // The multi-tile GroupNorm specialization keeps the tile offset in an
  // enclosing scf.for and forms absolute indices as splat(off) + make_range.
  tt.func public @rank1_reduce_addf_block4096_loop_offset_mask(%x_ptr: !tt.ptr<f32>) {
    %c0 = arith.constant 0 : i32
    %c4096 = arith.constant 4096 : i32
    %c8192 = arith.constant 8192 : i32
    %bound = arith.constant dense<8192> : tensor<4096xi32, #blocked_spt4>
    %zero = arith.constant dense<0.0> : tensor<4096xf32, #blocked_spt4>
    %offsets = tt.make_range {end = 4096 : i32, start = 0 : i32} : tensor<4096xi32, #blocked_spt4>
    %x_splat = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<4096x!tt.ptr<f32>, #blocked_spt4>
    scf.for %off = %c0 to %c8192 step %c4096 : i32 {
      %off_splat = tt.splat %off : i32 -> tensor<4096xi32, #blocked_spt4>
      %absolute = arith.addi %off_splat, %offsets : tensor<4096xi32, #blocked_spt4>
      %mask = arith.cmpi slt, %absolute, %bound : tensor<4096xi32, #blocked_spt4>
      %x_addr = tt.addptr %x_splat, %absolute : tensor<4096x!tt.ptr<f32>, #blocked_spt4>, tensor<4096xi32, #blocked_spt4>
      %row = tt.load %x_addr, %mask, %zero : tensor<4096x!tt.ptr<f32>, #blocked_spt4>
      %sum = "tt.reduce"(%row) ({
      ^bb0(%a: f32, %b: f32):
        %s = arith.addf %a, %b : f32
        tt.reduce.return %s : f32
      }) {axis = 0 : i32} : (tensor<4096xf32, #blocked_spt4>) -> f32
    }
    tt.return
  }

  // A scalar statistics recurrence followed by a multi-band output must not
  // be nested in the synthetic output tile loop. The recurrence consumes two
  // complete 4096-element tiles, returns one block-uniform scalar, and the
  // store then broadcasts that scalar over a separate 4096-element tile.
  tt.func public @rank1_reduce_loop_before_output_tile(%x_ptr: !tt.ptr<f32>, %out_ptr: !tt.ptr<f32>, %eps: f32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c16 = arith.constant 16 : i32
    %c4096 = arith.constant 4096 : i32
    %c8192 = arith.constant 8192 : i32
    %zero_scalar = arith.constant 0.0 : f32
    %one_scalar = arith.constant 1.0 : f32
    %denominator = arith.constant 8192.0 : f32
    %bound = arith.constant dense<8192> : tensor<4096xi32, #blocked_spt4>
    %zero = arith.constant dense<0.0> : tensor<4096xf32, #blocked_spt4>
    %offsets = tt.make_range {end = 4096 : i32, start = 0 : i32} : tensor<4096xi32, #blocked_spt4>
    %x_splat = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<4096x!tt.ptr<f32>, #blocked_spt4>
    %sum = scf.for %off = %c0 to %c8192 step %c4096 iter_args(%running = %zero_scalar) -> (f32) : i32 {
      %off_splat = tt.splat %off : i32 -> tensor<4096xi32, #blocked_spt4>
      %absolute = arith.addi %off_splat, %offsets : tensor<4096xi32, #blocked_spt4>
      %mask = arith.cmpi slt, %absolute, %bound : tensor<4096xi32, #blocked_spt4>
      %x_addr = tt.addptr %x_splat, %absolute : tensor<4096x!tt.ptr<f32>, #blocked_spt4>, tensor<4096xi32, #blocked_spt4>
      %row = tt.load %x_addr, %mask, %zero : tensor<4096x!tt.ptr<f32>, #blocked_spt4>
      %partial = "tt.reduce"(%row) ({
      ^bb0(%a: f32, %b: f32):
        %combined = arith.addf %a, %b : f32
        tt.reduce.return %combined : f32
      }) {axis = 0 : i32} : (tensor<4096xf32, #blocked_spt4>) -> f32
      %next = arith.addf %running, %partial : f32
      scf.yield %next : f32
    }
    %mean = arith.divf %sum, %denominator : f32
    %variance = arith.maxnumf %mean, %zero_scalar : f32
    %shifted = arith.addf %variance, %eps : f32
    %std = math.sqrt %shifted : f32
    %rstd = arith.divf %one_scalar, %std : f32
    scf.for %channel = %c0 to %c16 step %c1 : i32 {
      %gain_ptr = tt.addptr %x_ptr, %channel : !tt.ptr<f32>, i32
      %gain = tt.load %gain_ptr : !tt.ptr<f32>
      %scale = arith.mulf %gain, %rstd : f32
      %channel_offset = arith.muli %channel, %c4096 : i32
      %channel_ptr = tt.addptr %out_ptr, %channel_offset : !tt.ptr<f32>, i32
      %out_splat = tt.splat %channel_ptr : !tt.ptr<f32> -> tensor<4096x!tt.ptr<f32>, #blocked_spt4>
      %out_addr = tt.addptr %out_splat, %offsets : tensor<4096x!tt.ptr<f32>, #blocked_spt4>, tensor<4096xi32, #blocked_spt4>
      %mean_tile = tt.splat %scale : f32 -> tensor<4096xf32, #blocked_spt4>
      %out_mask = arith.cmpi slt, %offsets, %bound : tensor<4096xi32, #blocked_spt4>
      tt.store %out_addr, %mean_tile, %out_mask : tensor<4096x!tt.ptr<f32>, #blocked_spt4>
    }
    tt.return
  }
}
// CHECK-LABEL: metal.kernel rank1_reduce_addf_block4096_chain
// Multi-accumulator reduce (K=8, metal-multiacc-reduce-plan.md): scf.for + 8
// f32 iter_args, with the Wall-11 chain ops nested inside the body (per
// accumulator). The CHECKs below match the first accumulator's chain.
// CHECK: scf.for {{.*}} step {{.*}} iter_args({{.*}}) -> (f32, f32, f32, f32, f32, f32, f32, f32)
// CHECK: metal.binary_exp {{.*}}, {{.*}}, subOp
// CHECK: metal.unary_exp {{.*}}, expOp
// CHECK: arith.select
// CHECK: metal.binary_exp {{.*}}, {{.*}}, addOp
// CHECK: scf.yield
// CHECK: metal.threadgroup_alloca : !metal.memref<256 x f32>
// CHECK: metal.return
// CHECK-LABEL: metal.kernel rank1_reduce_addf_block4096_dense_mask
// CHECK: arith.constant 8 : i32
// CHECK: scf.for
// CHECK: arith.cmpi slt
// CHECK: metal.get_element
// CHECK: metal.threadgroup_alloca : !metal.memref<256 x f32>
// CHECK: metal.return
// CHECK-LABEL: metal.kernel rank1_reduce_addf_block4096_square_reuses_load
// CHECK: scf.for
// CHECK: metal.get_element
// CHECK: metal.binary_exp {{.*}}, {{.*}}, subOp
// CHECK-NOT: metal.get_element
// CHECK: metal.binary_exp {{.*}}, {{.*}}, mulOp
// CHECK: metal.threadgroup_alloca : !metal.memref<256 x f32>
// CHECK: metal.return
// CHECK-LABEL: metal.kernel rank1_reduce_addf_block4096_loop_offset_mask
// CHECK: scf.for
// CHECK: scf.for
// CHECK: arith.cmpi slt
// CHECK: metal.get_element
// CHECK: metal.threadgroup_alloca : !metal.memref<256 x f32>
// CHECK: metal.return
// CHECK-LABEL: metal.kernel rank1_reduce_loop_before_output_tile
// CHECK-NOT: scf.for
// A masked-store scratch declaration inserted by an earlier pre-pass stays in
// the function prologue and does not block selection of the recurrence.
// CHECK: metal.threadgroup_alloca : !metal.memref<256 x f32>
// CHECK-NOT: scf.for
// The scalar recurrence is the first loop and owns the reduction scratch.
// CHECK: %[[STATS:[0-9]+]] = scf.for {{.*}} iter_args({{.*}} = {{.*}}) -> (f32)
// The original masked tensor load is dead after rank-1 replay.  Its converted
// scf.if must be removed before the replay loop rather than emitted as one
// redundant device load per statistics tile.
// CHECK-NOT: scf.if
// CHECK: scf.for {{.*}} iter_args({{.*}}) -> (f32, f32, f32, f32, f32, f32, f32, f32)
// CHECK: metal.threadgroup_alloca : !metal.memref<256 x f32>
// The block-uniform scalar epilogue follows the complete recurrence, still
// outside the synthetic output loop.
// CHECK: %[[MEAN:[0-9]+]] = arith.divf %[[STATS]],
// CHECK: %[[VARIANCE:[0-9]+]] = arith.maxnumf %[[MEAN]],
// CHECK: %[[SHIFTED:[0-9]+]] = arith.addf %[[VARIANCE]],
// CHECK: %[[STD:[0-9]+]] = metal.unary_exp %[[SHIFTED]], sqrtOp
// CHECK: arith.divf {{.*}}, %[[STD]]
// The synthetic output-band loop follows the complete recurrence.
// CHECK: scf.for
// CHECK-NOT: metal.threadgroup_alloca
// CHECK: metal.store
// CHECK: metal.return
// MSL-LABEL: kernel void rank1_reduce_loop_before_output_tile(
// The expensive scalar suffix must be materialized before either output loop;
// otherwise the expression emitter inlines sqrt into the per-channel body.
// MSL: metal::precise::sqrt
// MSL: for (int
// MSL-NOT: metal::precise::sqrt
// MSL: return;
