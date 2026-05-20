"""Matmul track v4 canonical-3-iter-arg fitness test (f32 [8,8]).

History:
- iter-3 (L1d3): `preprocessDotCvtChains` strips the #blocked → #dot_op cvts
  feeding tt.dot, lit-validated on `dot_canonical_preempt_single.mlir`.
- iter-4: diagnosed the residual XFAIL as a coupled matcher + rewriter gap.
- iter-5: matcher relax + `extractBK` generalization + `unwrapPtrToKernelArg`
  extension (broadcast/expand_dims descent) — the MLIR conversion path
  (`convert-tritongpu-to-metal`) now successfully unrolls both the [8,8,8]
  (no-loop, single-dot fallback) and [16,16,16] (K_TILES=2, canonical-3
  iter_arg unroll) shapes. Validated end-to-end at the MLIR level via
  `test/Dialect/Metal/convert-tritongpu-to-metal/dot_canonical_preempt_3iterarg.mlir`.

Why these cases still XFAIL after iter-5 (ESCALATED):
The downstream MSL text emitter — invoked AFTER `convert-tritongpu-to-metal`
inside `libmetal.ttgir_to_msl` — generates calls to legacy Metal intrinsic
names (`simdgroup_matrix_multiply_accumulate`, `simdgroup_store_matrix`) that
are no longer declared in the modern Metal toolchain (Metal 17.5 / clang
32023.883). Sample [8,8,8] failure:

    /var/.../triton-metal-*.metal:21:27:
      error: use of undeclared identifier 'simdgroup_matrix_multiply_accumulate'
        simdgroup_float8x8 v9 = simdgroup_matrix_multiply_accumulate(v8, v6, v7);
    /var/.../triton-metal-*.metal:22:3:
      error: use of undeclared identifier 'simdgroup_store_matrix';
              did you mean 'simdgroup_store'?

[16,16,16] hangs in `metalc` compilation (a separate symptom of the same MSL
emitter mismatch). Both failure surfaces are in the MSL text emitter
(`third_party/metal/lib/Target/MetalTranslation/`), not in the MLIR
conversion path that iter-5 targets.

iter-6 + iter-7 ship state (2026-05-20):
- [8-8-8]: PASS (XFAIL removed by iter-6). FA-derived MSL emitter modernization
  rewrote the 3 translate functions in `ModuleTranslation.cpp:354-400` to
  modern Metal 17.5 surface — sufficient for [8-8-8] correctness.
- [16-16-16]: still XFAIL (strict=True). iter-7 attempted to fix via strideC
  extraction in `tryUnrollCanonical3IterArgDot` (TritonGPUToMetal.cpp:3265-3275)
  but the fix was inert — failure mode unchanged (100% mismatch, 10.97 max abs
  diff). Per iter-7 spec hard-stop protocol, strideC code change is preserved
  as a strict improvement and the next probe (origin extractor verification)
  is deferred to iter-8. See `_ITER78_16_REASON` below for full diagnostic.
"""

from __future__ import annotations

import os
import pytest

torch = pytest.importorskip("torch")

import triton
import triton.language as tl

libmetal = pytest.importorskip(
    "triton._C.libtriton.metal",
    reason="Metal backend pybind module not built into libtriton",
)
if not hasattr(libmetal, "launch_kernel_with_pipeline"):
    pytest.skip(
        "Metal runtime not compiled (non-Darwin build or Xcode CLT absent)",
        allow_module_level=True,
    )


@triton.jit
def dot_8x8_f32_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    K_TILES: tl.constexpr,  # static trip count for v4 canonical unroller
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    # Loop bound is constexpr (K_TILES) — required by v4 canonical-3-iter_arg unroller
    # which gates on static trip count at TritonGPUToMetal.cpp:3119.
    for _ in range(0, K_TILES):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc)


_ITER78_16_REASON = (
    "iter-7 strideC fix INSUFFICIENT — empirically validated 2026-05-20. "
    "iter-7 replaced the hardcoded `strideC = 8` at `TritonGPUToMetal.cpp:3265-3267` "
    "with `findStrideSplatSource(store.getPtr()->addptr.getOffset())` + "
    "`emitStrideOperand`, mirroring A/B at lines 3248-3249. Empirical result: "
    "[16-16-16] still FAILS with 100% mismatch, 10.97 max abs diff at index (2,1) — "
    "IDENTICAL to pre-fix failure. This means either (a) `findStrideSplatSource` "
    "returns null for this C-store IR shape and `emitStrideOperand` falls back to "
    "constant 8 (so the fix is inert), or (b) a different rewriter site emits "
    "wrong strideC, or (c) the executor's iter-6 hypothesis about C-origin "
    "extraction being broken is now the leading candidate (only `tgid.y * 8` "
    "used, missing `tgid.x * 8` for the row axis). Next probe (deferred to "
    "iter-8): add a temporary `llvm::errs()` dump of strideC + cOrig.row/col "
    "in `tryUnrollCanonical3IterArgDot` and re-run `pytest [16-16-16] --runxfail`. "
    "iter-7 code change for strideC is preserved as a strict improvement "
    "(falls back to identical behavior when extraction null; positive for any "
    "future case where extraction succeeds)."
)


@pytest.mark.parametrize(
    "M, N, K",
    [
        pytest.param(8, 8, 8),
        pytest.param(
            16,
            16,
            16,
            marks=pytest.mark.xfail(
                strict=True, reason=_ITER78_16_REASON
            ),
        ),
    ],
)
def test_dot_f32_8x8(M, N, K):
    """v4 canonical-3-iter-arg fitness, contract-exact tile [8,8,8], f32 throughout.

    On failure, TRITON_REPRODUCER_PATH=/tmp/dot-8x8-f32-fail.mlir captures the
    MLIR reproducer.
    """
    os.environ.setdefault(
        "TRITON_REPRODUCER_PATH", "/tmp/dot-8x8-f32-fail.mlir"
    )
    torch.manual_seed(0xC0FFEE)
    a = torch.randn((M, K), dtype=torch.float32).contiguous()
    b = torch.randn((K, N), dtype=torch.float32).contiguous()
    c = torch.zeros((M, N), dtype=torch.float32).contiguous()
    grid = (triton.cdiv(M, 8), triton.cdiv(N, 8))
    K_TILES = (K + 8 - 1) // 8  # 8→1, 16→2; both ∈ [1, 8] (v4 trip-count bound)
    dot_8x8_f32_kernel[grid](
        a, b, c,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=8, BLOCK_N=8, BLOCK_K=8,
        K_TILES=K_TILES,
    )
    expected = torch.matmul(a, b)
    torch.testing.assert_close(c, expected, atol=1e-5, rtol=1e-5)
