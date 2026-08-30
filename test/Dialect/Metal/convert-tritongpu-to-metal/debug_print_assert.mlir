// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | FileCheck %s
// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | triton-metal-translate --mlir-to-msl | FileCheck %s --check-prefix=MSL
//
// `tl.device_print` and `tl.device_assert`. MSL has neither printf nor a
// host-visible assert channel, so both were rejected outright — which left a
// kernel author with no way to look inside a running kernel at all.
//
// They write fixed-size records into a trailing `device uint32_t*` parameter
// the launcher binds, and the host formats them once the dispatch completes.
// Word 0 is an atomically incremented count, then six words per record:
// [kind | dtype << 8, message id, program id, lane, value bits, 0]. Only the
// INDEX of the message crosses to the device; the strings ride to the launcher
// in a module attribute the pybind forwards.
//
// The buffer parameter appears only in kernels that use it, so every other
// kernel's signature — and the launcher's argument binding — is unchanged.
//
// CHECK-LABEL: metal.kernel print_and_assert
// CHECK: metal.debug_record kind = 0, msg = 0
// The assert records only when the condition is FALSE, and it carries the
// threadgroup-LOCAL lane: id.x is global and would label every program's
// records with the wrong element.
// CHECK: metal.debug_record kind = 1, msg = 1
// MSL-LABEL: kernel void print_and_assert
// MSL: device uint32_t *__triton_dbg {{\[\[}}buffer(1)
// MSL: atomic_fetch_add_explicit((device atomic_uint*)&__triton_dbg[0], 1u
// MSL: __triton_dbg[{{.*}} + 4u] = as_type<uint32_t>(
// A one-bit constant has to print as `true`, not as the -1 an i1 APInt renders
// as: `cond ^ -1` is nonzero for BOTH values of cond, and every passing assert
// reported a failure until that was fixed.
// MSL: ^ true)
// MSL: __triton_dbg[{{.*}} + 3u] = uint32_t((id.x - (tgid.x * 32)))

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @print_and_assert(%x: !tt.ptr<f32>) {
    %r = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #blocked>
    %px = tt.splat %x : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>, #blocked>
    %pxo = tt.addptr %px, %r : tensor<32x!tt.ptr<f32>, #blocked>, tensor<32xi32, #blocked>
    %v = tt.load %pxo : tensor<32x!tt.ptr<f32>, #blocked>
    tt.print "v: " {hex = false, isSigned = array<i32: 0>} : %v : tensor<32xf32, #blocked>
    %zero = arith.constant dense<0.000000e+00> : tensor<32xf32, #blocked>
    %ok = arith.cmpf oge, %v, %zero : tensor<32xf32, #blocked>
    tt.assert %ok, "must be non-negative" : tensor<32xi1, #blocked>
    tt.store %pxo, %v : tensor<32x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
