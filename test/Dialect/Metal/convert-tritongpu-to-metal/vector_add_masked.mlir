// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | FileCheck %s --check-prefix=METAL
// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | triton-metal-translate --mlir-to-msl | FileCheck %s --check-prefix=MSL
//
// Masked vector_add (unmasked path's masked sibling) per
// the implementation notes. Lowers TTGIR
// `tt.load %addr, %mask` / `tt.store %val, %addr, %mask` (no `other`) to
// scf.if wrapping metal.get_element for the loads, and to the L1d2c
// Phase B select-on-value+address pattern for the store
// (the implementation notes).

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  // N is baked in as a constant: `metal.kernel` only accepts memref-typed
  // kernel arguments, so a dynamic scalar N would need a memref<i32, 1>
  // wrapper plus a get_element on entry. Deferred to a future session;
  // this fixture exercises the masked IR/MSL surface against a constant N.
  tt.func public @add_kernel_masked(%x_ptr: !tt.ptr<f32>, %y_ptr: !tt.ptr<f32>, %output_ptr: !tt.ptr<f32>) {
    %c128_i32 = arith.constant 128 : i32
    %c100_i32 = arith.constant 100 : i32
    %pid = tt.get_program_id x : i32
    %block_start = arith.muli %pid, %c128_i32 : i32
    %offsets = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %bs_splat = tt.splat %block_start : i32 -> tensor<128xi32, #blocked>
    %abs_off = arith.addi %bs_splat, %offsets : tensor<128xi32, #blocked>
    %n_splat = tt.splat %c100_i32 : i32 -> tensor<128xi32, #blocked>
    %mask = arith.cmpi slt, %abs_off, %n_splat : tensor<128xi32, #blocked>
    %x_splat = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %x_addr = tt.addptr %x_splat, %abs_off : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    %x_val = tt.load %x_addr, %mask : tensor<128x!tt.ptr<f32>, #blocked>
    %y_splat = tt.splat %y_ptr : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %y_addr = tt.addptr %y_splat, %abs_off : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    %y_val = tt.load %y_addr, %mask : tensor<128x!tt.ptr<f32>, #blocked>
    %sum = arith.addf %x_val, %y_val : tensor<128xf32, #blocked>
    %o_splat = tt.splat %output_ptr : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %o_addr = tt.addptr %o_splat, %abs_off : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    tt.store %o_addr, %sum, %mask : tensor<128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// Post-Lmultiload-Phase-C: 1D canonical short-circuit deleted. MakeRange
// emits the per-thread `localTid = id.x - tgid.x*tpb`; AddPtr accumulates
// `pid*BLOCK + localTid`. The mask path's `arith.cmpi` still uses a raw
// `metal.thread_id` rebuild (mask reconstruction is independent of the
// load-index chain).
// METAL: metal.module
// METAL: metal.kernel add_kernel_masked
// L1d2c Phase B: per-kernel threadgroup scratch sentinel hoisted at
// function entry, sized threadsPerBlock × sizeof(f32) = 128 × 4 B.
// METAL: metal.threadgroup_alloca : !metal.memref<128 x f32>
// Load mask = the scalarized index cone `pid*BLOCK + localtid < N` (per-thread
// correct, reconstructed from the ACTUAL mask cone via the typeconverter, not a
// global-flat emitPerIterIndex). The two loads share one mask (CSE), so only one
// cmpi + reused scf.if condition are emitted for the pair.
// METAL: arith.subi
// METAL: arith.addi
// METAL: arith.cmpi slt
// METAL: scf.if {{.*}} -> (f32)
// METAL: metal.get_element
// METAL: scf.yield
// METAL: metal.constant
// METAL: scf.yield
// METAL: scf.if {{.*}} -> (f32)
// METAL: metal.get_element
// METAL: scf.yield
// METAL: metal.constant
// METAL: scf.yield
// METAL: metal.binary_exp
// METAL-SAME: addOp
// L1d2c Phase B masked-store rewrite: arith.select on value +
// unconditional tg_store_indexed (scratch). Device store remains guarded
// by scf.if. The store REUSES the load's scalarized mask (same tt mask value,
// CSE), so no separate store-mask cmpi is emitted.
// METAL: metal.tg_load_indexed
// METAL: arith.select
// METAL: metal.tg_store_indexed
// METAL: scf.if
// METAL: metal.store
// METAL: metal.return
// METAL: metal.module_end

// MSL: kernel void add_kernel_masked(
// MSL: device float
// MSL: device float
// MSL: device float
// MSL: thread_position_in_grid
// MSL: threadgroup_position_in_grid
// MSL: threadgroup float v{{[0-9]+}}[128];
// Load mask index = pid*BLOCK + localtid; shared bool for both loads.
// MSL: int v{{[0-9]+}} = ((tgid.x * 128) + (id.x - (tgid.x * 128)));
// MSL: bool v{{[0-9]+}} = ((int32_t)(v{{[0-9]+}}) < (int32_t)(100));
// MSL: float v{{[0-9]+}};
// MSL: if (v{{[0-9]+}})
// MSL: v{{[0-9]+}} = v{{[0-9]+}}[v{{[0-9]+}}];
// MSL: else
// MSL: v{{[0-9]+}} = 0
// MSL: float v{{[0-9]+}};
// MSL: if (v{{[0-9]+}})
// MSL: v{{[0-9]+}} = v{{[0-9]+}}[v{{[0-9]+}}];
// MSL: else
// MSL: v{{[0-9]+}} = 0
// MSL: float v{{[0-9]+}} = (v{{[0-9]+}}) + (v{{[0-9]+}});
// Store reuses the shared load mask bool (CSE) for both the tg select and the
// guarded device store.
// MSL: v{{[0-9]+}}[{{.*}}] = ({{.*}} ? {{.*}} : {{.*}});
// MSL: if (v{{[0-9]+}})
// MSL: v{{[0-9]+}}[v{{[0-9]+}}] = v{{[0-9]+}};
// MSL: return;
