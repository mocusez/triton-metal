// RUN: triton-metal-opt --convert-tritongpu-to-metal --split-input-file %s | FileCheck %s
//
// Session L2 positive fixture: every elementwise integer arith op in the
// broad set lowers through `convert-tritongpu-to-metal`. Each split section
// is a minimal kernel that derives two i32 tensor operands from
// `tt.make_range` + `tt.splat` (the existing i32 sources that don't require
// an i32-typed load — uint8/i32 data-path is deferred to L2b), applies one
// target op, then uses the result as the offset for an f32 store. FileCheck
// verifies that the tensor-form `arith.<op>` is GONE from post-conversion
// IR and that `metal.kernel` is present.
// See `.omc/specs/deep-interview-leet-triton-l2-int-arith-broad.md`.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @int_subi_kernel(%out_ptr: !tt.ptr<f32>, %k: i32) {
    %c1 = arith.constant 1.0 : f32
    %offs = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %ksp = tt.splat %k : i32 -> tensor<128xi32, #blocked>
    %r = arith.subi %offs, %ksp : tensor<128xi32, #blocked>
    %vs = tt.splat %c1 : f32 -> tensor<128xf32, #blocked>
    %os = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %oa = tt.addptr %os, %r : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    tt.store %oa, %vs : tensor<128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
// CHECK-LABEL: metal.kernel int_subi_kernel
// CHECK-NOT:   arith.subi {{.*}} tensor

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @int_divsi_kernel(%out_ptr: !tt.ptr<f32>, %k: i32) {
    %c1 = arith.constant 1.0 : f32
    %offs = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %ksp = tt.splat %k : i32 -> tensor<128xi32, #blocked>
    %r = arith.divsi %offs, %ksp : tensor<128xi32, #blocked>
    %vs = tt.splat %c1 : f32 -> tensor<128xf32, #blocked>
    %os = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %oa = tt.addptr %os, %r : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    tt.store %oa, %vs : tensor<128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
// CHECK-LABEL: metal.kernel int_divsi_kernel
// CHECK-NOT:   arith.divsi {{.*}} tensor

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @int_remsi_kernel(%out_ptr: !tt.ptr<f32>, %k: i32) {
    %c1 = arith.constant 1.0 : f32
    %offs = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %ksp = tt.splat %k : i32 -> tensor<128xi32, #blocked>
    %r = arith.remsi %offs, %ksp : tensor<128xi32, #blocked>
    %vs = tt.splat %c1 : f32 -> tensor<128xf32, #blocked>
    %os = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %oa = tt.addptr %os, %r : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    tt.store %oa, %vs : tensor<128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
// CHECK-LABEL: metal.kernel int_remsi_kernel
// CHECK-NOT:   arith.remsi {{.*}} tensor

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @int_andi_kernel(%out_ptr: !tt.ptr<f32>, %k: i32) {
    %c1 = arith.constant 1.0 : f32
    %offs = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %ksp = tt.splat %k : i32 -> tensor<128xi32, #blocked>
    %r = arith.andi %offs, %ksp : tensor<128xi32, #blocked>
    %vs = tt.splat %c1 : f32 -> tensor<128xf32, #blocked>
    %os = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %oa = tt.addptr %os, %r : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    tt.store %oa, %vs : tensor<128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
// CHECK-LABEL: metal.kernel int_andi_kernel
// CHECK-NOT:   arith.andi {{.*}} tensor

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @int_shrsi_kernel(%out_ptr: !tt.ptr<f32>, %k: i32) {
    %c1 = arith.constant 1.0 : f32
    %offs = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %ksp = tt.splat %k : i32 -> tensor<128xi32, #blocked>
    %r = arith.shrsi %offs, %ksp : tensor<128xi32, #blocked>
    %vs = tt.splat %c1 : f32 -> tensor<128xf32, #blocked>
    %os = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %oa = tt.addptr %os, %r : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    tt.store %oa, %vs : tensor<128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
// CHECK-LABEL: metal.kernel int_shrsi_kernel
// CHECK-NOT:   arith.shrsi {{.*}} tensor

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @int_shli_kernel(%out_ptr: !tt.ptr<f32>, %k: i32) {
    %c1 = arith.constant 1.0 : f32
    %offs = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %ksp = tt.splat %k : i32 -> tensor<128xi32, #blocked>
    %r = arith.shli %offs, %ksp : tensor<128xi32, #blocked>
    %vs = tt.splat %c1 : f32 -> tensor<128xf32, #blocked>
    %os = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %oa = tt.addptr %os, %r : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    tt.store %oa, %vs : tensor<128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
// CHECK-LABEL: metal.kernel int_shli_kernel
// CHECK-NOT:   arith.shli {{.*}} tensor

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @int_ori_kernel(%out_ptr: !tt.ptr<f32>, %k: i32) {
    %c1 = arith.constant 1.0 : f32
    %offs = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %ksp = tt.splat %k : i32 -> tensor<128xi32, #blocked>
    %r = arith.ori %offs, %ksp : tensor<128xi32, #blocked>
    %vs = tt.splat %c1 : f32 -> tensor<128xf32, #blocked>
    %os = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %oa = tt.addptr %os, %r : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    tt.store %oa, %vs : tensor<128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
// CHECK-LABEL: metal.kernel int_ori_kernel
// CHECK-NOT:   arith.ori {{.*}} tensor

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @int_xori_kernel(%out_ptr: !tt.ptr<f32>, %k: i32) {
    %c1 = arith.constant 1.0 : f32
    %offs = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %ksp = tt.splat %k : i32 -> tensor<128xi32, #blocked>
    %r = arith.xori %offs, %ksp : tensor<128xi32, #blocked>
    %vs = tt.splat %c1 : f32 -> tensor<128xf32, #blocked>
    %os = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %oa = tt.addptr %os, %r : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    tt.store %oa, %vs : tensor<128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
// CHECK-LABEL: metal.kernel int_xori_kernel
// CHECK-NOT:   arith.xori {{.*}} tensor

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @int_divui_kernel(%out_ptr: !tt.ptr<f32>, %k: i32) {
    %c1 = arith.constant 1.0 : f32
    %offs = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %ksp = tt.splat %k : i32 -> tensor<128xi32, #blocked>
    %r = arith.divui %offs, %ksp : tensor<128xi32, #blocked>
    %vs = tt.splat %c1 : f32 -> tensor<128xf32, #blocked>
    %os = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %oa = tt.addptr %os, %r : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    tt.store %oa, %vs : tensor<128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
// CHECK-LABEL: metal.kernel int_divui_kernel
// CHECK-NOT:   arith.divui {{.*}} tensor

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @int_remui_kernel(%out_ptr: !tt.ptr<f32>, %k: i32) {
    %c1 = arith.constant 1.0 : f32
    %offs = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %ksp = tt.splat %k : i32 -> tensor<128xi32, #blocked>
    %r = arith.remui %offs, %ksp : tensor<128xi32, #blocked>
    %vs = tt.splat %c1 : f32 -> tensor<128xf32, #blocked>
    %os = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %oa = tt.addptr %os, %r : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    tt.store %oa, %vs : tensor<128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
// CHECK-LABEL: metal.kernel int_remui_kernel
// CHECK-NOT:   arith.remui {{.*}} tensor

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @int_shrui_kernel(%out_ptr: !tt.ptr<f32>, %k: i32) {
    %c1 = arith.constant 1.0 : f32
    %offs = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %ksp = tt.splat %k : i32 -> tensor<128xi32, #blocked>
    %r = arith.shrui %offs, %ksp : tensor<128xi32, #blocked>
    %vs = tt.splat %c1 : f32 -> tensor<128xf32, #blocked>
    %os = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %oa = tt.addptr %os, %r : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    tt.store %oa, %vs : tensor<128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
// CHECK-LABEL: metal.kernel int_shrui_kernel
// CHECK-NOT:   arith.shrui {{.*}} tensor

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @int_select_kernel(%x_ptr: !tt.ptr<f32>, %y_ptr: !tt.ptr<f32>, %out_ptr: !tt.ptr<f32>) {
    %c64_i32 = arith.constant 64 : i32
    %offs = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %thresh = tt.splat %c64_i32 : i32 -> tensor<128xi32, #blocked>
    %cond = arith.cmpi slt, %offs, %thresh : tensor<128xi32, #blocked>
    %xs = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %xa = tt.addptr %xs, %offs : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    %xv = tt.load %xa : tensor<128x!tt.ptr<f32>, #blocked>
    %ys = tt.splat %y_ptr : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %ya = tt.addptr %ys, %offs : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    %yv = tt.load %ya : tensor<128x!tt.ptr<f32>, #blocked>
    %r = arith.select %cond, %xv, %yv : tensor<128xi1, #blocked>, tensor<128xf32, #blocked>
    %os = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %oa = tt.addptr %os, %offs : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    tt.store %oa, %r : tensor<128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
// CHECK-LABEL: metal.kernel int_select_kernel
// CHECK-NOT:   arith.select {{.*}} tensor

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @umulhi_i32_kernel(%x_ptr: !tt.ptr<i32>, %y_ptr: !tt.ptr<i32>, %out_ptr: !tt.ptr<i32>) {
    %offs = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %xs = tt.splat %x_ptr : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>, #blocked>
    %xa = tt.addptr %xs, %offs : tensor<128x!tt.ptr<i32>, #blocked>, tensor<128xi32, #blocked>
    %x = tt.load %xa : tensor<128x!tt.ptr<i32>, #blocked>
    %ys = tt.splat %y_ptr : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>, #blocked>
    %ya = tt.addptr %ys, %offs : tensor<128x!tt.ptr<i32>, #blocked>, tensor<128xi32, #blocked>
    %y = tt.load %ya : tensor<128x!tt.ptr<i32>, #blocked>
    %r = tt.mulhiui %x, %y : tensor<128xi32, #blocked>
    %os = tt.splat %out_ptr : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>, #blocked>
    %oa = tt.addptr %os, %offs : tensor<128x!tt.ptr<i32>, #blocked>, tensor<128xi32, #blocked>
    tt.store %oa, %r : tensor<128x!tt.ptr<i32>, #blocked>
    tt.return
  }
}
// CHECK-LABEL: metal.kernel umulhi_i32_kernel
// CHECK-NOT:   tt.mulhiui
// CHECK:       metal.mulhi_ui

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @umulhi_i64_kernel(%x_ptr: !tt.ptr<i64>, %y_ptr: !tt.ptr<i64>, %out_ptr: !tt.ptr<i64>) {
    %offs = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %xs = tt.splat %x_ptr : !tt.ptr<i64> -> tensor<128x!tt.ptr<i64>, #blocked>
    %xa = tt.addptr %xs, %offs : tensor<128x!tt.ptr<i64>, #blocked>, tensor<128xi32, #blocked>
    %x = tt.load %xa : tensor<128x!tt.ptr<i64>, #blocked>
    %ys = tt.splat %y_ptr : !tt.ptr<i64> -> tensor<128x!tt.ptr<i64>, #blocked>
    %ya = tt.addptr %ys, %offs : tensor<128x!tt.ptr<i64>, #blocked>, tensor<128xi32, #blocked>
    %y = tt.load %ya : tensor<128x!tt.ptr<i64>, #blocked>
    %r = tt.mulhiui %x, %y : tensor<128xi64, #blocked>
    %os = tt.splat %out_ptr : !tt.ptr<i64> -> tensor<128x!tt.ptr<i64>, #blocked>
    %oa = tt.addptr %os, %offs : tensor<128x!tt.ptr<i64>, #blocked>, tensor<128xi32, #blocked>
    tt.store %oa, %r : tensor<128x!tt.ptr<i64>, #blocked>
    tt.return
  }
}
// CHECK-LABEL: metal.kernel umulhi_i64_kernel
// CHECK-NOT:   tt.mulhiui
// CHECK:       metal.mulhi_ui
