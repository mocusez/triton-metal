// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | FileCheck %s
//
// A `torch.bool` tensor. Triton knows the buffer is one BYTE per element and
// re-types the pointer at every access — `tt.load(tt.bitcast(tt.addptr(...) :
// ptr<i1> -> ptr<i8>))` — while the argument itself stays `!tt.ptr<i1>`.
//
// That used to be rejected by argument type, and for a real reason: the memref
// kept the i1 pointee (so an i8 value landed on an i1-element buffer) and the
// per-element offset sat UNDERNEATH the cast, where the index paths could not
// see it — the converted bitcast is a memref view, not an integer, which
// tripped an assertion inside `emitLoadStoreIndex`.
//
// Both halves are fixed here: an i1 pointee stores as i8, which is what the
// hardware layout already is, and the cast is pushed down to the base pointer
// before conversion so the addptr chain — and its offset — is intact. The push
// is scoped to the i1 -> i8 pointee change, leaving the f32 atomic min/max
// expansion (the other producer of pointer bitcasts) byte for byte as it was.
//
// CHECK-LABEL: metal.kernel bool_copy
// Both buffers present as i8, and the index is the ordinary per-lane one.
// CHECK: metal.get_element %arg0[%{{.*}}] : (!metal.memref<? x i8>, ui32) -> i8
// CHECK: metal.store %{{.*}}, %arg1[%{{.*}}] : i8, !metal.memref<? x i8>, ui32

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @bool_copy(%x: !tt.ptr<i1>, %o: !tt.ptr<i1>) {
    %r = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %px = tt.splat %x : !tt.ptr<i1> -> tensor<128x!tt.ptr<i1>, #blocked>
    %pxo = tt.addptr %px, %r : tensor<128x!tt.ptr<i1>, #blocked>, tensor<128xi32, #blocked>
    %pxb = tt.bitcast %pxo : tensor<128x!tt.ptr<i1>, #blocked> -> tensor<128x!tt.ptr<i8>, #blocked>
    %v = tt.load %pxb : tensor<128x!tt.ptr<i8>, #blocked>
    %po = tt.splat %o : !tt.ptr<i1> -> tensor<128x!tt.ptr<i1>, #blocked>
    %poo = tt.addptr %po, %r : tensor<128x!tt.ptr<i1>, #blocked>, tensor<128xi32, #blocked>
    %pob = tt.bitcast %poo : tensor<128x!tt.ptr<i1>, #blocked> -> tensor<128x!tt.ptr<i8>, #blocked>
    tt.store %pob, %v : tensor<128x!tt.ptr<i8>, #blocked>
    tt.return
  }
}
