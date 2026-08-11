// RUN: triton-metal-opt --split-input-file --verify-diagnostics %s
//
// metal.fused_attention verifier. The score region is the whole point of the
// op — it is what replaces a per-variant emitter — so its contract has to be
// enforced structurally rather than trusted from the matcher. Each chunk below
// breaks one clause.

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  metal.module {
    metal.kernel k address_space_device [true, true, true, true, true, true, true, true, true, true, true, true] {
    ^bb0(%q: !metal.memref<? x f32>, %k: !metal.memref<? x f32>, %v: !metal.memref<? x f32>, %o: !metal.memref<? x f32>, %m: !metal.memref<? x ui32>, %n: !metal.memref<? x ui32>, %dh: !metal.memref<? x ui32>, %sq: !metal.memref<? x ui32>, %sk: !metal.memref<? x ui32>, %sv: !metal.memref<? x ui32>, %so: !metal.memref<? x ui32>, %p: !metal.memref<? x f32>):
      // A declared score_param with no matching block arg would silently read a
      // different scalar than the matcher resolved.
      // expected-error @+1 {{score region must take 3 + 1 arguments}}
      metal.fused_attention %q, %k, %v, %o, %m, %n, %dh strides %sq, %sk, %sv, %so params %p {bm = 64 : i64, bn = 64 : i64, bd = 64 : i64, norm = 0 : i32} : !metal.memref<? x f32>, !metal.memref<? x f32>, !metal.memref<? x f32>, !metal.memref<? x f32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x f32> {
      ^bb0(%score: f32, %row: si32, %key: si32):
        metal.score_yield %score : f32
      }
      metal.return
    }
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  metal.module {
    metal.kernel k address_space_device [true, true, true, true, true, true, true, true, true, true, true] {
    ^bb0(%q: !metal.memref<? x f32>, %k: !metal.memref<? x f32>, %v: !metal.memref<? x f32>, %o: !metal.memref<? x f32>, %m: !metal.memref<? x ui32>, %n: !metal.memref<? x ui32>, %dh: !metal.memref<? x ui32>, %sq: !metal.memref<? x ui32>, %sk: !metal.memref<? x ui32>, %sv: !metal.memref<? x ui32>, %so: !metal.memref<? x ui32>):
      // Signless i32 is NOT a Metal_Type, so a signless index could not feed
      // metal.binary_exp; and it must be SIGNED because `row - key` is meant to
      // go negative — that difference is the content of a causal/decay mask.
      // expected-error @+1 {{score region arg 1 (row) must be si32}}
      metal.fused_attention %q, %k, %v, %o, %m, %n, %dh strides %sq, %sk, %sv, %so {bm = 32 : i64, bn = 32 : i64, bd = 16 : i64, norm = 1 : i32} : !metal.memref<? x f32>, !metal.memref<? x f32>, !metal.memref<? x f32>, !metal.memref<? x f32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32> {
      ^bb0(%score: f32, %row: ui32, %key: si32):
        metal.score_yield %score : f32
      }
      metal.return
    }
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  metal.module {
    metal.kernel k address_space_device [true, true, true, true, true, true, true, true, true, true, true, true] {
    ^bb0(%q: !metal.memref<? x f32>, %k: !metal.memref<? x f32>, %v: !metal.memref<? x f32>, %o: !metal.memref<? x f32>, %m: !metal.memref<? x ui32>, %n: !metal.memref<? x ui32>, %dh: !metal.memref<? x ui32>, %sq: !metal.memref<? x ui32>, %sk: !metal.memref<? x ui32>, %sv: !metal.memref<? x ui32>, %so: !metal.memref<? x ui32>, %p: !metal.memref<? x f32>):
      // The emitter binds each trailing arg to `<param>[0]`, so a type mismatch
      // here is a wrong-typed read, not a compile error downstream.
      // expected-error @+1 {{score region arg 3 type ('si32') must match score_param 0 element type ('f32')}}
      metal.fused_attention %q, %k, %v, %o, %m, %n, %dh strides %sq, %sk, %sv, %so params %p {bm = 64 : i64, bn = 64 : i64, bd = 64 : i64, norm = 0 : i32} : !metal.memref<? x f32>, !metal.memref<? x f32>, !metal.memref<? x f32>, !metal.memref<? x f32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x f32> {
      ^bb0(%score: f32, %row: si32, %key: si32, %bad: si32):
        metal.score_yield %score : f32
      }
      metal.return
    }
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  metal.module {
    metal.kernel k address_space_device [true, true, true, true, true, true, true, true, true, true, true] {
    ^bb0(%q: !metal.memref<? x f32>, %k: !metal.memref<? x f32>, %v: !metal.memref<? x f32>, %o: !metal.memref<? x f32>, %m: !metal.memref<? x ui32>, %n: !metal.memref<? x ui32>, %dh: !metal.memref<? x ui32>, %sq: !metal.memref<? x ui32>, %sk: !metal.memref<? x ui32>, %sv: !metal.memref<? x ui32>, %so: !metal.memref<? x ui32>):
      // The emitter shapes simdgroup/staging tiles from these; an off-multiple
      // block would mis-shape them silently.
      // expected-error @+1 {{bm must be a multiple of 8 (got 63)}}
      metal.fused_attention %q, %k, %v, %o, %m, %n, %dh strides %sq, %sk, %sv, %so {bm = 63 : i64, bn = 32 : i64, bd = 16 : i64, norm = 1 : i32} : !metal.memref<? x f32>, !metal.memref<? x f32>, !metal.memref<? x f32>, !metal.memref<? x f32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32> {
      ^bb0(%score: f32, %row: si32, %key: si32):
        metal.score_yield %score : f32
      }
      metal.return
    }
  }
}
