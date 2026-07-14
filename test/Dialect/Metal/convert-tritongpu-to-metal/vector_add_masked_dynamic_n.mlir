// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | FileCheck %s --check-prefix=METAL
// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | triton-metal-translate --mlir-to-msl | FileCheck %s --check-prefix=MSL
//
// Masked vector_add with `n_elements` as a runtime i32 kernel argument
// (the canonical Triton shape). Validates the scalar-arg wrapping
// per `.omc/specs/deep-interview-metal-dynamic-scalar-args.md`:
//   * scalar i32 kernel arg -> !metal.memref<1 x ui32> in metal.kernel sig
//   * metal.get_element prologue at function entry materializes the value
//   * unrealized_conversion_cast bridges ui32 -> signless i32 for arith
//   * downstream masked-load/store consume the materialized value via
//     the existing matchMaskShape helper

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @add_kernel_dynamic(%x_ptr: !tt.ptr<f32>, %y_ptr: !tt.ptr<f32>, %output_ptr: !tt.ptr<f32>, %n_elements: i32) {
    %c128_i32 = arith.constant 128 : i32
    %pid = tt.get_program_id x : i32
    %block_start = arith.muli %pid, %c128_i32 : i32
    %offsets = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %bs_splat = tt.splat %block_start : i32 -> tensor<128xi32, #blocked>
    %abs_off = arith.addi %bs_splat, %offsets : tensor<128xi32, #blocked>
    %n_splat = tt.splat %n_elements : i32 -> tensor<128xi32, #blocked>
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

// METAL: metal.module
// METAL: metal.kernel add_kernel_dynamic address_space_device [true, true, true, true]
// METAL: !metal.memref<1 x ui32>
// METAL: metal.constant 0 : ui32
// METAL: metal.get_element
// METAL-SAME: !metal.memref<1 x ui32>, ui32
// METAL-SAME: -> ui32
// METAL: builtin.unrealized_conversion_cast
// METAL-SAME: ui32 to i32
// L1d2c Phase B: per-kernel threadgroup scratch sentinel hoisted at
// function entry (after the wrapper-arg prologue).
// METAL: metal.threadgroup_alloca : !metal.memref<128 x f32>
// METAL: arith.cmpi slt
// METAL: scf.if {{.*}} -> (f32)
// METAL: metal.get_element
// METAL: scf.yield
// METAL: metal.constant
// METAL: scf.yield
// METAL: arith.cmpi slt
// METAL: scf.if {{.*}} -> (f32)
// METAL: metal.get_element
// METAL: scf.yield
// METAL: metal.constant
// METAL: scf.yield
// METAL: metal.binary_exp
// METAL-SAME: addOp
// METAL: arith.cmpi slt
// L1d2c Phase B masked-store rewrite: arith.select on value +
// unconditional tg_store_indexed (scratch). Device store remains guarded
// by scf.if (honest divergence — see canary lit fixture for details).
// METAL: metal.tg_load_indexed
// METAL: arith.select
// METAL: metal.tg_store_indexed
// METAL: scf.if
// METAL: metal.store
// METAL: metal.return
// METAL: metal.module_end

// Post-Lmultiload-Phase-C: 1D canonical short-circuit deleted. The
// per-thread store/load index is now the arithmetic-explicit
// `pid*BLOCK + (id.x - pid*tpb)` form. See `.omc/specs/deep-interview-
// lmultiload-phase-c-makerange.md`.
// MSL: kernel void add_kernel_dynamic(
// MSL: device float *v{{[0-9]+}}
// MSL: device float *v{{[0-9]+}}
// MSL: device float *v{{[0-9]+}}
// MSL: device uint32_t *v{{[0-9]+}}
// MSL: thread_position_in_grid
// MSL: threadgroup_position_in_grid
// MSL: threadgroup float v{{[0-9]+}}[128];
// MSL: int v{{[0-9]+}} = ((tgid.x * 128) + (id.x - (tgid.x * 128)));
// MSL: float v{{[0-9]+}};
// MSL: if ((id.x < v{{[0-9]+}}[0]))
// MSL: v{{[0-9]+}} = v{{[0-9]+}}[v{{[0-9]+}}];
// MSL: else
// MSL: v{{[0-9]+}} = 0
// MSL: float v{{[0-9]+}};
// MSL: if ((id.x < v{{[0-9]+}}[0]))
// MSL: v{{[0-9]+}} = v{{[0-9]+}}[v{{[0-9]+}}];
// MSL: else
// MSL: v{{[0-9]+}} = 0
// MSL: float v{{[0-9]+}} = (v{{[0-9]+}}) + (v{{[0-9]+}});
// MSL: bool v{{[0-9]+}} = (id.x < v{{[0-9]+}}[0]);
// MSL: if (v{{[0-9]+}})
// MSL: v{{[0-9]+}}[v{{[0-9]+}}] = v{{[0-9]+}};
// MSL: return;
