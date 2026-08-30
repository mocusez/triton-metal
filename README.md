# Triton-Metal (Proof of Concept)

> **Status: Proof of Concept.** This fork lowers a subset of Triton kernels onto Apple Silicon GPUs via Metal. It is **not** a drop-in replacement for upstream Triton on CUDA/ROCm. Many ops are unsupported, the compiler pipeline is rough around the edges, and **performance is not yet tuned**. Expect to read source code when something breaks.

This is a research-grade experiment exploring whether the Triton compiler stack can target Apple Metal. If you are looking for production-ready Triton, use the upstream project at [triton-lang/triton](https://github.com/triton-lang/triton).

---

## What this is

A new backend (`third_party/metal/`) that takes Triton kernels through the following lowering pipeline:

```
Triton kernel (Python @triton.jit)
        │
        ▼
   TritonGPU IR  (TTGIR — the existing Triton mid-level dialect)
        │   third_party/metal/lib/Conversion/TritonGPUToMetal
        ▼
   Metal Dialect (MLIR — Triton-Metal-specific ops in third_party/metal/include/Dialect/Metal)
        │   third_party/metal/lib/Target/Metal
        ▼
   MSL (Metal Shading Language source text)
        │   third_party/metal/backend/driver.py
        ▼
   torch.mps.compile_shader → dispatch on the MPS stream
```

The backend's own pipeline stops at MSL text. Launching goes through PyTorch's MPS runtime: `torch.mps.compile_shader` compiles the emitted MSL and dispatches it directly against MPS tensors, zero-copy and ordered on the MPS stream. There is no `.metallib` stage and no native Metal runtime — `libtriton.so` links no Apple framework at all (`otool -L` shows only libz, libSystem and libc++). **An MPS-enabled PyTorch on Apple Silicon is therefore a hard runtime requirement**; without it the driver raises `Metal backend is MPS-only`.

---

## Quick start

### Prerequisites

To **run** a released wheel:

- Apple Silicon Mac (M1 or newer)
- macOS 26+ for the published wheel — it is tagged `macosx_26_0_arm64` after the build machine's SDK, see Known limitations
- Python 3.12+ and an MPS-enabled PyTorch (`torch.backends.mps.is_available()`, and `torch.mps.compile_shader`, added in PyTorch 2.7). No Xcode or Metal toolchain is needed at run time — PyTorch compiles the shader.

To **build from source**, additionally:

- Xcode 15+ with Command Line Tools (`xcode-select --install`)
- [pixi](https://pixi.sh) for environment management

### Dev environment

This repo uses **pixi** to pin Python / CMake / LLVM / PyTorch versions for development. Pixi is a temporary choice — if this backend graduates toward upstream, pixi will be dropped in favor of the upstream build flow.

```bash
# install pixi (one-time)
curl -fsSL https://pixi.sh/install.sh | bash

# from repo root
pixi install
pixi run install   # builds Triton + Metal backend in editable mode
```

### Build a release wheel

```bash
TRITON_WHEEL_BACKENDS=metal TRITON_BUILD_PROTON=0 TRITON_STABLE_ABI=1 \
    pixi run python setup.py bdist_wheel
# → dist/triton-3.8.0+git<sha>-cp312-abi3-macosx_26_0_arm64.whl  (~59 MB)
```

- `TRITON_WHEEL_BACKENDS=metal` ships only the Metal Python backend. All three
  backends are still *compiled* — core TritonGPU includes the NVIDIA NVWS dialect
  headers and `python/src/gluon_ir.cc` includes the AMD ones, so the build does not
  narrow — but the wheel drops the NVIDIA Linux toolchain payload (ptxas, cupti,
  ~281 MB unpacked) that could never run on macOS. Leaving it unset ships everything,
  which is roughly a 120 MB wheel.
- `TRITON_STABLE_ABI=1` tags the wheel `cp312-abi3`, so one artifact covers Python
  3.12+ instead of a wheel per interpreter.
- `TRITON_BUILD_PROTON=0` drops the CUDA/ROCm profiler.

Install into a throwaway venv to verify it is self-contained:

```bash
python3 -m venv /tmp/triton-test
/tmp/triton-test/bin/pip install torch
/tmp/triton-test/bin/pip install --no-deps dist/triton-*.whl
/tmp/triton-test/bin/python -c "
from triton.backends import backends
from triton.backends.metal.driver import MetalDriver
print(sorted(backends), MetalDriver().get_current_target())
"
# → ['metal'] GPUTarget(backend='metal', arch=9, warp_size=32)
```

Note that `python/test/unit/test_metal_backend_l1d2d_probe.py` cannot be part of a
wheel check: it drives the `triton-metal-opt` / `triton-metal-translate` binaries out
of the build directory, which no wheel carries.

### Run the tests

```bash
# lit-based MLIR tests (no GPU required — CPU-only compiler checks)
cd $(pixi run python -c 'from build_helpers import get_cmake_dir; print(get_cmake_dir())')
ninja triton-opt
lit -v test/Dialect/Metal

# pytest GPU smoke tests (Metal/MPS required — a real Apple Silicon machine;
# these cannot run on GitHub Actions runners, see Known limitations)
pixi run pytest python/test/unit/test_metal_backend_*.py -s --tb=short

# audit all 80 migrated LeetGPU-style Python fixtures and cover the 78
# runnable workloads through standalone, interpreter, or targeted tests
pixi run leet-all
```

---

## What works today

The 80 Python fixtures under `python/test/unit/fixtures/metal_leet/` have an
exhaustive ownership manifest enforced by
`test_metal_backend_leet_uncovered.py`. `pixi run leet-all` executes all three
ownership classes sequentially:

| Ownership | Files | Validation |
|---|---:|---|
| Standalone drivers | 23 | Execute the original script and its PyTorch/smoke assertion |
| Interpreter-backed gaps | 26 | Compare Metal with `TRITON_INTERPRET=1`; 24 run and 2 source-invalid cases skip explicitly |
| Targeted backend regressions | 31 | Run the owning `test_metal_backend_*.py` modules |

Adding a fixture without assigning it to exactly one ownership class fails the
inventory test. The standalone correctness set includes:

| Task | Pass |
|---|---|
| `easy-leaky_Relu.py` | ✅ |
| `easy-color_inversion.py` | ✅ |
| `easy-1D_convolution.py` | ✅ |
| `easy-gaussian_error_gated_linear_unit.py` | ✅ |
| `easy-interleave_arrays.py` | ✅ |
| `easy-matrix_transpose.py` | ✅ |
| `medium-2d_convolution.py` | ✅ |
| `medium-2d_fft.py` | ✅ |
| `medium-mean_squared_error.py` | ✅ |
| `medium-ordinary_least_squares.py` | ✅ |
| `medium-sparse_matrix-vector_multiplication.py` | ✅ |
| `medium-sparse_matrix-Dense_matrix_multiplication.py` | ✅ |
| `medium-stream-compaction.py` | ✅ |
| `medium-subarray_sum.py` | ✅ |

These cover element-wise ops, masked load/store, reductions, scans, stable compaction, atomic accumulation, broadcast, 2D dispatch, convolution, COO sparse matvec/matmul, a correctness-first direct 2D DFT built from supported `tl.dot` tiles, and an ordinary least-squares solve backed by `tl.dot` Gram tiles.

The following examples show the narrower commands for two fixtures owned by
targeted pytest coverage; the full `pixi run leet-all` gate includes them:

| Task | Reproduce |
|---|---|
| `hard-bfs_shortest_path.py` | `pixi run pytest python/test/unit/test_metal_backend_msl.py::test_bfs_shortest_path_exact_kernel_compiles_to_msl python/test/unit/test_metal_backend_msl.py::test_bfs_shortest_path_host_contract_rejects_invalid_metadata python/test/unit/test_metal_backend_atomic_add.py::test_bfs_shortest_path_original_solve_matches_cpu -s --tb=short` |
| `medium-fused_residual_add_and_rms_norm.py` | `pixi run pytest python/test/unit/test_metal_backend_msl.py::test_fused_residual_rmsnorm_exact_kernels_compile_to_msl python/test/unit/test_metal_backend_msl.py::test_fused_residual_rmsnorm_host_contract_rejects_invalid_metadata python/test/unit/test_metal_backend_layer_norm.py::test_fused_residual_add_rms_norm_original_solve_matches_torch python/test/unit/test_metal_backend_layer_norm.py::test_fused_residual_add_rms_norm_dynamic_fallback_large_width python/test/unit/test_metal_backend_layer_norm.py::test_fused_residual_add_rms_norm_accepts_noncontiguous_readonly_inputs python/test/unit/test_metal_backend_layer_norm.py::test_fused_residual_add_rms_norm_accepts_stride_zero_weight python/test/unit/test_metal_backend_layer_norm.py::test_fused_residual_add_rms_norm_stable_for_extreme_scales -s --tb=short` |

`hard-bfs_shortest_path.py` uses a correctness-first host-controlled BFS: each
frontier depth performs one kernel launch plus one host counter synchronization.
Very deep shortest paths therefore scale linearly in launch/sync latency even
when the total grid is small. The supported contract expects an `int32` grid
whose start and end cells are free (`0`).

The sparse examples accept a CPU `torch.sparse_coo_tensor` through their
existing `solve(A, ..., nnz)` entry point, coalesce duplicate coordinates, and
transfer only the resulting COO indices and values to the GPU. Here `nnz` is
the input tensor's stored-entry count. For repeated multiplies, call
`prepare_coo(...)` once and pass its prepacked int32 row/column indices and
float32 values to `solve_coo(...)`; SpMV core work is then `O(nnz)` and SpMM
core work is `O(nnz * K)`. A contiguous strided dense `A` remains supported as
a compatibility fallback, but retains dense complexity and is not the
sparse-optimized path.

In-tree pytest suites (`python/test/unit/test_metal_backend_*.py`) cover individual lowering features — arith constants, transcendentals, integer arithmetic, masked load with `other`, dynamic `N`, multi-program launch, 2D elementwise, and the standard `kernel[grid](...)` launch protocol.

---

## Known limitations

- **Generic tensor codegen is still scalar.** The backend has specialized SIMD-group matrix, reduction, scan, and threadgroup-memory paths, but the general ranked-tensor conversion still lowers one scalar element at a time. Contiguous vector loads/stores and broader aggregate-hoisting remain performance work.
- **Op coverage is deliberately bounded.** The common elementwise, reduction, scan, gather, atomic, dot, fp8, and control-flow paths have end-to-end coverage, while unsupported shapes fail during preflight with a named diagnostic. Notable remaining envelopes include loop-carried rank-2 reductions, some cross-lane layout changes, broader `tt.dot_scaled`, and multi-result `scf.if`.
- **`tl.dot` is not layout-general.** Proven scalar and SIMD-group-matrix paths cover the tested f32, fp16/bf16, int8, batched, and scaled-dot envelopes. Masked multi-tile canonical dots, 3-D dots with `K > min(M, N)`, and arbitrary layout combinations remain unsupported.
- **Launching requires PyTorch MPS.** `torch.mps.compile_shader` compiles the MSL once per process; nothing is cached across processes, and the compiled shader library lives and dies with the interpreter.
- **The runtime tracks PyTorch's MPS surface.** Capabilities can regress with a torch upgrade: on torch 2.13 `torch.zeros(..., dtype=torch.float8_e4m3fn, device="mps")` fails with `Undefined type Float8_e4m3fn`, so the fp8 paths need torch 2.10-era MPS support even though the backend's own fp8 casts are unchanged.
- **Wheel platform tag** reflects the build machine's macOS SDK (e.g. `macosx_26_0_arm64`), so a wheel built on macOS 26 will not install on macOS 14/15. Nothing in `libtriton.so` links an Apple framework, so building with `MACOSX_DEPLOYMENT_TARGET` set lower is the fix — untested so far.
- **Only `osx-arm64`** is supported. Intel Macs and `universal2` are out of scope.
- **Hosted CI does not validate GPU numerics.** GitHub-hosted macOS
  runners are virtualized and expose no usable Metal device to PyTorch: MPS is either
  reported unavailable, or reported available while every allocation fails, with MPS
  memory capped near 1 GB ([actions/runner-images#9918](https://github.com/actions/runner-images/issues/9918),
  [community#155306](https://github.com/orgs/community/discussions/155306)). What CI can
  do is build the backend, run compiler/lit checks, and collect Python tests that do not
  allocate on MPS. GPU-dependent tests skip rather than fail. Read a green hosted job as
  "it builds and the compiler checks hold", never as "the kernels compute the right
  answer" — end-to-end correctness remains outside hosted CI and is verified
  separately on physical Apple Silicon.

---

## Remaining work

The current dependency-ordered priorities are:

1. Finish the unsupported-op/preflight safety matrix. Axes 1/2
   `tt.get_num_programs` coverage is in place, and both correctness xfails
   found by the audit are now ordinary 30-run regressions.
2. Expand exact dot, scaled-dot, reduce/scan, layout-conversion, atomic CAS,
   descriptor-reduce, `tt.unsplat`, and `tl.map_elementwise` envelopes one
   positive/adjacent-negative slice at a time.
3. Improve generic performance with vectorized contiguous memory access,
   aggregate/cone reuse, and resource-aware tile selection, backed by fixed
   MPS benchmarks rather than blanket speedup claims.
4. Decide and implement a versioned cross-process shader-cache architecture;
   the current PyTorch MPS API recompiles MSL in each interpreter.
5. Validate lower deployment-target wheels on clean macOS installations.

PTX inline assembly, integer-to-pointer device addresses, `tl.atomic_poll`,
and bf16 atomic add are platform/runtime boundaries, not implementation backlog.

---

## Repository layout (Metal-specific)

```
third_party/metal/
├── backend/                       Python backend module (driver, compiler, target)
├── include/Dialect/Metal/IR/      Metal Dialect op/type definitions
├── include/Target/Metal/          MSL emitter headers
├── lib/Conversion/
│   └── TritonGPUToMetal/          TTGIR → Metal Dialect lowering
├── lib/Dialect/Metal/             Dialect verifiers, canonicalization
└── lib/Target/Metal/              Metal Dialect → MSL text translation

python/triton/backends/metal/      Installed Python entry point
python/test/unit/test_metal_backend_*.py   GPU smoke tests
test/Dialect/Metal/                lit-based compiler tests
python/test/unit/fixtures/metal_leet/      End-to-end LeetGPU-style fixtures
```

---

## Upstream Triton

This fork is based on [triton-lang/triton](https://github.com/triton-lang/triton). For the language reference, tutorials, and CUDA/ROCm support, see the upstream project and its [official documentation](https://triton-lang.org). If you cite Triton in academic work, please cite the original MAPL2019 paper:

> Tillet, P., Kung, H.T., Cox, D. *Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations.* MAPL 2019.

---

## License

Same as upstream Triton — see [LICENSE](LICENSE).
