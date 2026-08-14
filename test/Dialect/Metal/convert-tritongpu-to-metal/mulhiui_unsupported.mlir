// RUN: triton-metal-opt --convert-tritongpu-to-metal --verify-diagnostics %s

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @umulhi_i16_unsupported(%x_ptr: !tt.ptr<i16>, %y_ptr: !tt.ptr<i16>, %out_ptr: !tt.ptr<i16>) {
    %offs = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %xs = tt.splat %x_ptr : !tt.ptr<i16> -> tensor<128x!tt.ptr<i16>, #blocked>
    %xa = tt.addptr %xs, %offs : tensor<128x!tt.ptr<i16>, #blocked>, tensor<128xi32, #blocked>
    %x = tt.load %xa : tensor<128x!tt.ptr<i16>, #blocked>
    %ys = tt.splat %y_ptr : !tt.ptr<i16> -> tensor<128x!tt.ptr<i16>, #blocked>
    %ya = tt.addptr %ys, %offs : tensor<128x!tt.ptr<i16>, #blocked>, tensor<128xi32, #blocked>
    %y = tt.load %ya : tensor<128x!tt.ptr<i16>, #blocked>
    // expected-error @+1 {{Metal backend: tt.mulhiui supports 32-bit or 64-bit integers only}}
    %r = tt.mulhiui %x, %y : tensor<128xi16, #blocked>
    %os = tt.splat %out_ptr : !tt.ptr<i16> -> tensor<128x!tt.ptr<i16>, #blocked>
    %oa = tt.addptr %os, %offs : tensor<128x!tt.ptr<i16>, #blocked>, tensor<128xi32, #blocked>
    tt.store %oa, %r : tensor<128x!tt.ptr<i16>, #blocked>
    tt.return
  }
}
