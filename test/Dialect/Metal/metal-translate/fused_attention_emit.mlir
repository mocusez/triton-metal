// RUN: triton-metal-translate --mlir-to-msl %s 2>&1 | FileCheck %s
//
// metal.fused_attention: the generalized `Q@K^T -> score transform -> P@V`
// body. What used to require a NEW op per attention variant is here a REGION
// carrying the score transform, so both kernels below share one emitter.
//
//   decay_kernel   — norm = None. The region is the causal decay of
//                    leet-triton/medium-decaying_causal_attention.py:
//                    s * scale * exp2((row-key) >= 0 ? (row-key)*log2(gamma)
//                                                    : -inf)
//                    Note there is NO separate mask operand: masking is what
//                    the region computes, exactly as the source kernel writes
//                    it.
//   softmax_kernel — norm = OnlineSoftmax with an EMPTY score_params list.
//                    The scale is a constant Triton folded out of the
//                    signature, and under this op a folded constant needs no
//                    operand at all — it is just `metal.constant` in the
//                    region. That is what retires the `Optional<operand> +
//                    *_const attribute` pair the predecessor ops carried per
//                    parameter.
//
// Buffers: v0..v3 = q/k/v/out, v4 = m (query rows), v5 = n (keys),
// v6 = d_head, v7..v10 = the four row strides, v11 = log2(gamma).

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  metal.module {
    metal.kernel decay_kernel address_space_device [true, true, true, true, true, true, true, true, true, true, true, true] {
    ^bb0(%q: !metal.memref<? x f32>, %k: !metal.memref<? x f32>, %v: !metal.memref<? x f32>, %o: !metal.memref<? x f32>, %m: !metal.memref<? x ui32>, %n: !metal.memref<? x ui32>, %dh: !metal.memref<? x ui32>, %sq: !metal.memref<? x ui32>, %sk: !metal.memref<? x ui32>, %sv: !metal.memref<? x ui32>, %so: !metal.memref<? x ui32>, %lg: !metal.memref<? x f32>):
      metal.fused_attention %q, %k, %v, %o, %m, %n, %dh strides %sq, %sk, %sv, %so params %lg {bm = 64 : i64, bn = 64 : i64, bd = 64 : i64, norm = 0 : i32} : !metal.memref<? x f32>, !metal.memref<? x f32>, !metal.memref<? x f32>, !metal.memref<? x f32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x f32> {
      ^bb0(%score: f32, %row: si32, %key: si32, %phase: si32, %log2g: f32):
        %diff = metal.binary_exp %row, %key, subOp : (si32, si32) -> si32
        %zero = metal.constant 0 : si32
        %ge = metal.binary_exp %diff, %zero, geOp : (si32, si32) -> i1
        %difff = metal.cast %diff : (si32) -> f32
        %expo = metal.binary_exp %difff, %log2g, mulOp : (f32, f32) -> f32
        %ninf = metal.constant 0xFF800000 : f32
        %sel = arith.select %ge, %expo, %ninf : f32
        %w = metal.unary_exp %sel, exp2Op : (f32) -> f32
        %out = metal.binary_exp %score, %w, mulOp : (f32, f32) -> f32
        metal.score_yield %out : f32
      } bounds {
      ^bb0(%blk: si32, %ph: si32, %m2: si32, %n2: si32, %log2g: f32):
        // Causal: keys [0, min((blk+1)*bm, n)). The source's own loop bound,
        // written out -- not an attribute standing in for it.
        %zero = metal.constant 0 : si32
        %one = metal.constant 1 : si32
        %bm = metal.constant 64 : si32
        %b1 = metal.binary_exp %blk, %one, addOp : (si32, si32) -> si32
        %hi = metal.binary_exp %b1, %bm, mulOp : (si32, si32) -> si32
        %end = metal.binary_exp %hi, %n2, minOp : (si32, si32) -> si32
        metal.key_bounds_yield %zero, %end
      }
      metal.return
    }
    metal.kernel softmax_kernel address_space_device [true, true, true, true, true, true, true, true, true, true, true] {
    ^bb0(%q: !metal.memref<? x f32>, %k: !metal.memref<? x f32>, %v: !metal.memref<? x f32>, %o: !metal.memref<? x f32>, %m: !metal.memref<? x ui32>, %n: !metal.memref<? x ui32>, %dh: !metal.memref<? x ui32>, %sq: !metal.memref<? x ui32>, %sk: !metal.memref<? x ui32>, %sv: !metal.memref<? x ui32>, %so: !metal.memref<? x ui32>):
      metal.fused_attention %q, %k, %v, %o, %m, %n, %dh strides %sq, %sk, %sv, %so {bm = 32 : i64, bn = 32 : i64, bd = 16 : i64, norm = 1 : i32} : !metal.memref<? x f32>, !metal.memref<? x f32>, !metal.memref<? x f32>, !metal.memref<? x f32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32> {
      ^bb0(%score: f32, %row: si32, %key: si32, %phase: si32):
        %scale = metal.constant 2.500000e-01 : f32
        %scaled = metal.binary_exp %score, %scale, mulOp : (f32, f32) -> f32
        metal.score_yield %scaled : f32
      } bounds {
      ^bb0(%blk: si32, %ph: si32, %m2: si32, %n2: si32):
        %zero = metal.constant 0 : si32
        metal.key_bounds_yield %zero, %n2
      }
      metal.return
    }
    metal.kernel head_split_kernel address_space_device [true, true, true, true, true, true, true, true, true, true, true, true] {
    ^bb0(%q: !metal.memref<? x f32>, %k: !metal.memref<? x f32>, %v: !metal.memref<? x f32>, %o: !metal.memref<? x f32>, %m: !metal.memref<? x ui32>, %n: !metal.memref<? x ui32>, %dm: !metal.memref<? x ui32>, %sq: !metal.memref<? x ui32>, %sk: !metal.memref<? x ui32>, %sv: !metal.memref<? x ui32>, %so: !metal.memref<? x ui32>, %h: !metal.memref<? x ui32>):
      metal.fused_attention %q, %k, %v, %o, %m, %n, %dm strides %sq, %sk, %sv, %so heads %h {bm = 32 : i64, bn = 32 : i64, bd = 16 : i64, norm = 1 : i32, softmax_natural_exp} : !metal.memref<? x f32>, !metal.memref<? x f32>, !metal.memref<? x f32>, !metal.memref<? x f32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32> {
      ^bb0(%score: f32, %row: si32, %key: si32, %phase: si32):
        %scale = metal.constant 2.500000e-01 : f32
        %scaled = metal.binary_exp %score, %scale, mulOp : (f32, f32) -> f32
        metal.score_yield %scaled : f32
      } bounds {
      ^bb0(%blk: si32, %ph: si32, %m2: si32, %n2: si32):
        %zero = metal.constant 0 : si32
        metal.key_bounds_yield %zero, %n2
      }
      metal.return
    }
    metal.kernel feature_tiled_kernel address_space_device [true, true, true, true, true, true, true, true, true, true, true] {
    ^bb0(%q: !metal.memref<? x f32>, %k: !metal.memref<? x f32>, %v: !metal.memref<? x f32>, %o: !metal.memref<? x f32>, %m: !metal.memref<? x ui32>, %n: !metal.memref<? x ui32>, %dh: !metal.memref<? x ui32>, %sq: !metal.memref<? x ui32>, %sk: !metal.memref<? x ui32>, %sv: !metal.memref<? x ui32>, %so: !metal.memref<? x ui32>):
      metal.fused_attention %q, %k, %v, %o, %m, %n, %dh strides %sq, %sk, %sv, %so {bm = 16 : i64, bn = 16 : i64, bd = 64 : i64, feature_tiled, k_feature_major, norm = 1 : i32, safe_denominator_one, softmax_natural_exp} : !metal.memref<? x f32>, !metal.memref<? x f32>, !metal.memref<? x f32>, !metal.memref<? x f32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32> {
      ^bb0(%score: f32, %row: si32, %key: si32, %phase: si32):
        metal.score_yield %score : f32
      } bounds {
      ^bb0(%blk: si32, %ph: si32, %m2: si32, %n2: si32):
        %zero = metal.constant 0 : si32
        metal.key_bounds_yield %zero, %n2
      }
      metal.return
    }
    metal.kernel grouped_head_kernel address_space_device [true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true] {
    ^bb0(%q: !metal.memref<? x f32>, %k: !metal.memref<? x f32>, %v: !metal.memref<? x f32>, %o: !metal.memref<? x f32>, %m: !metal.memref<? x ui32>, %n: !metal.memref<? x ui32>, %dh: !metal.memref<? x ui32>, %sq: !metal.memref<? x ui32>, %sk: !metal.memref<? x ui32>, %sv: !metal.memref<? x ui32>, %so: !metal.memref<? x ui32>, %sqh: !metal.memref<? x ui32>, %skh: !metal.memref<? x ui32>, %svh: !metal.memref<? x ui32>, %soh: !metal.memref<? x ui32>, %groups: !metal.memref<? x ui32>):
      metal.fused_attention %q, %k, %v, %o, %m, %n, %dh strides %sq, %sk, %sv, %so head_params %sqh, %skh, %svh, %soh, %groups {bm = 32 : i64, bn = 32 : i64, bd = 16 : i64, norm = 1 : i32, softmax_natural_exp} : !metal.memref<? x f32>, !metal.memref<? x f32>, !metal.memref<? x f32>, !metal.memref<? x f32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32> {
      ^bb0(%score: f32, %row: si32, %key: si32, %phase: si32):
        metal.score_yield %score : f32
      } bounds {
      ^bb0(%blk: si32, %ph: si32, %m2: si32, %n2: si32):
        %zero = metal.constant 0 : si32
        metal.key_bounds_yield %zero, %n2
      }
      metal.return
    }
  }
}

// --- decay kernel: norm = None. At 64x64x64 the simdgroup body would need
// 3*64*64 + 2*64*64 + 2*64*64 = 28672 floats against a 8192 budget, so this one
// takes the SCALAR floor instead of the op declining — that is what keeping two
// bodies buys.
// CHECK-LABEL: kernel void decay_kernel
// CHECK: threadgroup float _fa_qbuf[2048]
// CHECK: threadgroup float _fa_obuf[2048]
// CHECK-NOT: _fa_rmax
// bm=64 with bd=64 exceeds a single 32-lane pass, so the emitter chunks the
// query rows (2 x 32) instead of declining the way flash_attention's bm<=32
// gate did.
// CHECK: for (uint _fa_c0 = 0u; _fa_c0 < 64u; _fa_c0 += 32u)
// The causal key bound comes from the op's key-bounds region, so it is ordinary
// emitted arithmetic rather than a hard-coded `min((tgid.x+1)*bm, n)`.
// CHECK: // --- key phase 0 ---
// CHECK: uint _fa_kbeg =
// CHECK: uint _fa_kend =
// CHECK: for (uint _fa_key = _fa_kbeg; _fa_key < _fa_kend; ++_fa_key)
// CHECK: _fa_a += _fa_qbuf
// --- the region, emitted inline as let-bindings with score/row/key bound ---
// CHECK: = _fa_a;
// CHECK: = (int)_fa_row;
// CHECK: = (int)_fa_key;
// CHECK: [0];
// CHECK: exp2(
// CHECK: _fa_obuf{{.*}} +=
// CHECK-NOT: _fa_den

// --- softmax kernel: same op and same region, but its working set
// (3*32*16 + 2*16*32 + 2*32*32 + 2*32 = 4672 floats) fits Apple's 8192, so the
// emitter picks the SIMDGROUP body. Both matmuls run on the matrix unit and the
// region is evaluated over the staged S tile — which is the whole point: one
// emitter, two bodies, no per-variant code.
// CHECK-LABEL: kernel void softmax_kernel
// CHECK: threadgroup float _fa_sbuf[1024]
// CHECK: threadgroup float _fa_pbuf[1024]
// CHECK: threadgroup float _fa_rmax[32]
// CHECK: simdgroup_multiply_accumulate
// the region, inlined over the staged score tile
// CHECK: = _fa_sbuf[q*32u + kk];
// CHECK: _fa_pbuf[q*32u + kk] =
// running state + rescale + epilogue divide
// CHECK: float m_new = max(m_old, m_cur)
// CHECK: _fa_rsum[q] = _fa_rsum[q]*scaler + denom
// CHECK: simdgroup_multiply_accumulate
// CHECK: / denom

// --- head-split kernel: `heads` present, so the per-head feature width is
// DERIVED (`d_model / h`) rather than read from a buffer, and the feature
// column is offset by the grid's y dimension. A head-split Triton kernel
// computes `d_head = d_model // h` itself, so there is no kernel argument for a
// d_head buffer to point at -- carrying `h` and dividing here is what let this
// family be claimed at all (it moved the parity bar 10 -> 24 on its own).
// CHECK-LABEL: kernel void head_split_kernel
// CHECK: uint3 tgid {{\[\[}}threadgroup_position_in_grid]]
// CHECK: uint3 ltid {{\[\[}}thread_position_in_threadgroup]]
// CHECK: uint _fa_dh = v6[0] / v11[0];
// CHECK: uint _fa_col = tgid.y * _fa_dh;
// The padded feature columns d >= d_head must stage as zero, or the head slice
// bleeds into its neighbour.
// CHECK: _fa_qbuf[c] = (row < _fa_M && d < _fa_dh) ? v0[row * _fa_sq + _fa_col + d] : 0.0f;
//
// `softmax_natural_exp` picks base-e. It is not cosmetic: a kernel that folds
// log2(e) into its scale uses exp2 instead, and the region has already produced
// the logit, so the emitter cannot rescale after the fact.
// CHECK: float scaler = (m_old == m_new) ? 1.0f : exp(m_old - m_new);
// CHECK-NOT: exp2(

// --- feature-tiled kernel: y selects a BD-wide output tile, while QK reduces
// over the full runtime d_head and reads already-transposed K as [feature,key].
// The MMA body must sweep full-d_head Q/K chunks independently of the output
// tile, so d_head>BD remains correct without falling back to the scalar body.
// CHECK-LABEL: kernel void feature_tiled_kernel
// CHECK: uint _fa_col = tgid.y * 64u;
// CHECK: uint _fa_out_d = (_fa_col < _fa_dh) ? min(64u, _fa_dh - _fa_col) : 0u;
// CHECK: v2[kk * _fa_sv + _fa_col + d]
// CHECK: // ---- feature-tiled QK full-dhead sweep ----
// CHECK: for (uint _fa_dc = 0u; _fa_dc < _fa_dh; _fa_dc += 64u)
// CHECK: v0[row * _fa_sq + _fa_dc + d]
// CHECK: v1[(_fa_dc + d) * _fa_sk + kk]
// CHECK: simdgroup_float8x8 acc(0.0f);
// CHECK: if (_fa_dc != 0u) simdgroup_load(acc, &_fa_sbuf
// CHECK: simdgroup_multiply_accumulate(acc, a, b, acc);
// CHECK: simdgroup_store(acc, &_fa_sbuf
// CHECK: denom = (denom == 0.0f) ? 1.0f : denom;
// CHECK: v3[row * _fa_so + _fa_col + d]

// --- grouped-head kernel: y selects an independently stored query head;
// K/V share one head per runtime-sized query group. The head offsets are
// separate from row/feature indexing and are not guessed from sequence and
// d_head.
// CHECK-LABEL: kernel void grouped_head_kernel
// CHECK: uint _fa_qhoff = tgid.y * v11[0];
// CHECK: uint _fa_khoff = (tgid.y / v15[0]) * v12[0];
// CHECK: uint _fa_vhoff = (tgid.y / v15[0]) * v13[0];
// CHECK: uint _fa_ohoff = tgid.y * v14[0];
// CHECK: v0[_fa_qhoff + row * _fa_sq + _fa_col + d]
// CHECK: v1[_fa_khoff + kk * _fa_sk + _fa_col + d]
// CHECK: v2[_fa_vhoff + kk * _fa_sv + _fa_col + d]
// CHECK: v3[_fa_ohoff + row * _fa_so + _fa_col + d]
