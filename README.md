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
        │   third_party/metal/lib/Runtime
        ▼
   Runner (xcrun metal → .metallib → MTLComputePipelineState → dispatch)
```

The runtime shells out to `xcrun -sdk macosx metal` at launch time to compile the emitted MSL into a `.metallib`, then dispatches via the Metal API. **Receiving machines must have Xcode (or the Command Line Tools' Metal toolchain) installed.**

---

## Quick start

### Prerequisites

- Apple Silicon Mac (M1 or newer; the runtime currently targets `MTLGPUFamilyApple9` with fallback to Apple8..Apple1)
- macOS 14+ (the SDK gate for Apple9 detection is `__MAC_OS_X_VERSION_MAX_ALLOWED >= 140000`)
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

### Build a wheel for local testing

```bash
pixi run python setup.py bdist_wheel
# → dist/triton-*.whl  (~120 MB; bundles all backends + libtriton.so)
```

Install the wheel into a throwaway venv to verify it is self-contained:

```bash
python3 -m venv /tmp/triton-test
/tmp/triton-test/bin/pip install --no-deps dist/triton-*.whl
/tmp/triton-test/bin/python -c "
import triton
from triton.backends.metal.driver import MetalDriver
print(MetalDriver().get_current_target())
"
# → GPUTarget(backend='metal', arch=9, warp_size=32)
```

### Run the tests

```bash
# lit-based MLIR tests (no GPU required — CPU-only compiler checks)
cd $(pixi run python -c 'from build_helpers import get_cmake_dir; print(get_cmake_dir())')
ninja triton-opt
lit -v test/Dialect/Metal

# pytest GPU smoke tests (Metal/MPS required)
pixi run pytest python/test/unit/test_metal_backend_*.py -s --tb=short

# end-to-end LeetGPU-style kernels
pixi run leet-all
```

---

## What works today

End-to-end correctness verified against PyTorch references on Apple Silicon (M-series, macOS 26.4 in the development environment) for the following LeetGPU-style kernels under `leet-triton/`:

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
| `medium-stream-compaction.py` | ✅ |
| `medium-subarray_sum.py` | ✅ |

Reproduce with `pixi run leet-all`. These cover element-wise ops, masked load/store, reductions, scans, stable compaction, atomic accumulation, broadcast, 2D dispatch, convolution, a correctness-first direct 2D DFT built from supported `tl.dot` tiles, and an ordinary least-squares solve backed by `tl.dot` Gram tiles.

In-tree pytest suites (`python/test/unit/test_metal_backend_*.py`) cover individual lowering features — arith constants, transcendentals, integer arithmetic, masked load with `other`, dynamic `N`, multi-program launch, 2D elementwise, and the standard `kernel[grid](...)` launch protocol.

---

## Known limitations

- **No autotuning, no perf work.** Generated MSL is single-threadgroup-per-program and makes no use of SIMD-group reductions, threadgroup memory, or vectorized loads. Throughput will be **far** below what Metal-native kernels achieve.
- **Op coverage is partial.** Many `tl.*` ops are unimplemented; you will hit `NYI` errors on non-trivial kernels.
- **No general `tl.dot` / matmul lowering.** Correctness fallbacks cover selected proven f32 GEMM and Gram shapes, but arbitrary dot layouts remain unsupported. Broader SIMD-group-matrix mapping is future work.
- **Runtime shells out to `xcrun`** at every launch — no MSL caching across processes yet.
- **Wheel platform tag** reflects the build machine's macOS SDK (e.g. `macosx_26_0_arm64`). For broader distribution it should be re-tagged to the minimum supported macOS.
- **Only `osx-arm64`** is supported. Intel Macs and `universal2` are out of scope.
- **No macOS CI** — verification is currently manual on developer machines.

---

## Future work

- **Performance pass:** SIMD-group primitives, threadgroup memory, vectorized loads, and a coarser tile-to-threadgroup mapping. The current generator leaves >10× on the table for memory-bound kernels.
- **`tl.dot` lowering** onto Apple SIMD-group matrix instructions (Apple7+).
- **Broader op coverage:** atomics, more transcendentals, integer reductions, gather/scatter beyond simple masked load.
- **Compiled-kernel caching:** persist `.metallib` blobs across processes instead of re-invoking `xcrun` on every launch.
- **Automated correctness suite** covering medium/hard LeetGPU and a subset of upstream Triton tutorials.
- **Build-system convergence with upstream:** drop pixi, integrate the Metal backend behind the upstream `setup.py` backend-selection flow, and produce a properly-tagged wheel for general distribution.
- **macOS CI** to gate regressions on Apple Silicon.

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
├── lib/Target/Metal/              Metal Dialect → MSL text translation
└── lib/Runtime/                   xcrun + Metal API runner (Runtime.mm)

python/triton/backends/metal/      Installed Python entry point
python/test/unit/test_metal_backend_*.py   GPU smoke tests
test/Dialect/Metal/                lit-based compiler tests
leet-triton/                       End-to-end LeetGPU-style kernels
```

---

## Upstream Triton

This fork is based on [triton-lang/triton](https://github.com/triton-lang/triton). For the language reference, tutorials, and CUDA/ROCm support, see the upstream project and its [official documentation](https://triton-lang.org). If you cite Triton in academic work, please cite the original MAPL2019 paper:

> Tillet, P., Kung, H.T., Cox, D. *Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations.* MAPL 2019.

---

## License

Same as upstream Triton — see [LICENSE](LICENSE).
