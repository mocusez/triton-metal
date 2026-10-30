# Triton-Metal (Proof of Concept)

> **Status: Proof of Concept.** This fork lowers a subset of Triton kernels onto Apple Silicon GPUs via Metal. It is **not** a drop-in replacement for upstream Triton on CUDA/ROCm. Many ops are unsupported, the compiler pipeline is rough around the edges, and performance work has only closed the fixed benchmark line described below. Expect to read source code when something breaks.

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
- macOS 15+ for the published wheel — Metal-only builds pin and verify the
  `macosx_15_0_arm64` compatibility floor
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
pixi run --frozen install   # builds Triton + Metal backend in editable mode
```

### Build a release wheel

```bash
TRITON_WHEEL_BACKENDS=metal TRITON_BUILD_PROTON=0 TRITON_STABLE_ABI=1 \
    pixi run --frozen python setup.py bdist_wheel
# → dist/triton-3.8.0+git<sha>-cp312-abi3-macosx_15_0_arm64.whl  (~61 MB)
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
- A Metal-only build defaults `MACOSX_DEPLOYMENT_TARGET` to `15.0` and passes
  the same value to CMake. Lower values are rejected because the pinned
  LLVM/MLIR archives were built for macOS 15; an explicitly higher target is
  preserved.

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
Its D4/D5 probes omit synchronization before cross-lane reads, and D7scratch
reads scratch before initialization. These historical fixtures receive
CPU-only translation checks; numerical GPU checks cover the eight remaining
probes, including the synchronized transpose counterparts.

### Run the tests

```bash
# lit-based MLIR tests (no GPU required — CPU-only compiler checks)
cd $(pixi run --frozen python -c 'from build_helpers import get_cmake_dir; print(get_cmake_dir())')
ninja triton-opt
lit -v test/Dialect/Metal

# pytest GPU smoke tests (Metal/MPS required — a real Apple Silicon machine;
# these cannot run on GitHub Actions runners, see Known limitations)
pixi run --frozen pytest python/test/unit/test_metal_backend_*.py -s --tb=short

# audit all 90 migrated LeetGPU-style Python fixtures and cover the 88
# runnable workloads through standalone, interpreter, or targeted tests
pixi run --frozen leet-all
```

> **Sandboxed macOS processes can report a false-negative MPS status.** On a
> physical Apple Silicon Mac, a process constrained by a Seatbelt-style agent
> sandbox may return `False` from `torch.backends.mps.is_available()` and raise
> PyTorch's generic "MPS backend is supported on macOS 14.0+" error even though
> the host OS, GPU, and Pixi environment are valid. Re-running the same
> `pixi run --frozen ...` command outside that sandbox exposes the MPS device
> normally. `--clean-env` does not grant the missing service access. Local GPU
> validation and performance reports must therefore record that they used the
> project Pixi environment **and** an unsandboxed process; do not treat a
> sandbox-only `is_available() == False` result as evidence that the Mac lacks
> MPS support.

---

## What works today

The 90 Python fixtures under `python/test/unit/fixtures/metal_leet/` have an
exhaustive ownership manifest enforced by
`test_metal_backend_leet_uncovered.py`. `pixi run --frozen leet-all` executes all three
ownership classes sequentially:

| Ownership | Files | Validation |
|---|---:|---|
| Standalone drivers | 24 | Execute the original script and its PyTorch/smoke assertion |
| Interpreter-backed gaps | 34 | Compare Metal with `TRITON_INTERPRET=1`; 32 run and 2 source-invalid cases skip explicitly |
| Targeted backend regressions | 32 | Run the owning `test_metal_backend_*.py` modules |

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
targeted pytest coverage; the full `pixi run --frozen leet-all` gate includes them:

| Task | Reproduce |
|---|---|
| `hard-bfs_shortest_path.py` | `pixi run --frozen pytest python/test/unit/test_metal_backend_msl.py::test_bfs_shortest_path_exact_kernel_compiles_to_msl python/test/unit/test_metal_backend_msl.py::test_bfs_shortest_path_host_contract_rejects_invalid_metadata python/test/unit/test_metal_backend_atomic_add.py::test_bfs_shortest_path_original_solve_matches_cpu -s --tb=short` |
| `hard-fast_fourier_transform.py` | `pixi run --frozen pytest python/test/unit/test_metal_backend_msl.py::test_hard_fast_fourier_transform_exact_kernel_compiles_to_msl python/test/unit/test_metal_backend_msl.py::test_hard_fast_fourier_transform_matches_torch -s --tb=short` |
| `medium-fused_residual_add_and_rms_norm.py` | `pixi run --frozen pytest python/test/unit/test_metal_backend_msl.py::test_fused_residual_rmsnorm_exact_kernels_compile_to_msl python/test/unit/test_metal_backend_msl.py::test_fused_residual_rmsnorm_host_contract_rejects_invalid_metadata python/test/unit/test_metal_backend_layer_norm.py::test_fused_residual_add_rms_norm_original_solve_matches_torch python/test/unit/test_metal_backend_layer_norm.py::test_fused_residual_add_rms_norm_dynamic_fallback_large_width python/test/unit/test_metal_backend_layer_norm.py::test_fused_residual_add_rms_norm_accepts_noncontiguous_readonly_inputs python/test/unit/test_metal_backend_layer_norm.py::test_fused_residual_add_rms_norm_accepts_stride_zero_weight python/test/unit/test_metal_backend_layer_norm.py::test_fused_residual_add_rms_norm_stable_for_extreme_scales -s --tb=short` |

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

The current values-only `tl.topk` is supported for static `i32`, `f16`,
`bf16`, and `f32` tensors, ascending or descending, along the final dimension.
This covers rank-1/2/3 and multi-register-band inputs. The bound is the 32 KiB
threadgroup-memory budget rather than a fixed number of elements per thread:
while the phase-exchange scratch fits without reusing a buffer, every exchange
gets its own and the result is correct. Above the budget the allocator falls
back to two alternating slots per pool, which is safe only while the publishes
sharing a pool alternate strictly. Those pools are keyed by element type AND
element count: a hypercube reduce interleaves two-element direction-table
publishes between the tile publishes of the same element type, and pooling
them together flipped the round-robin parity and handed a comparator the slot
it was reading. That returned duplicated values at `N = 512` on four warps with
`k = 32` or `k = 64` until the count joined the key on 2026-09-08. A 37-cell
sweep over dtype, direction, rank and `num_warps`, fed PERMUTED inputs, gives
35 correct and two named rejections, the latter measured at thirty clean
declines out of thirty since the phase-carry rejection moved into preflight.

⚠️ Probe an ordering primitive with a PERMUTATION. A broken selection still
returns the correct prefix of already-sorted input, so an `arange`-fed sweep
reported those cells as correct while they were wrong.

The shared Triton frontend rules still apply (`k > 0`, power-of-two
`k`, and `k <= N`); indices, dynamic shapes, non-final dimensions, and cases
whose required phase scratch exceeds that budget are not part of this support
contract.

`tl.sort` and `tl.flip` are supported ascending and descending over static
power-of-two tiles, and are bounded by the same 32 KiB threadgroup budget
rather than by elements per thread. Sweeping `N / (32 * num_warps)` from one to
128 elements per thread is exact in both directions; above the budget the
compiler declines with a named diagnostic. Two defects that returned a permuted
array rather than failing were fixed on 2026-09-08. The first: the global load
and store placed a thread's element at `localTid * E + iv` while the hypercube
scratch placed it where the tile's blocked layout does, and those agree only
while the tile fits the threadgroup once. The second: Triton reuses one size-2
axis indicator for two different hypercube axes once the cube needs more
register bits than the indicator's linear layout names, so the widest merge
compared along the neighbouring axis.

Flip recognition also handles duplicate loads introduced by layout
rematerialization. Equivalent singleton broadcast address cones are normalized
before common-subexpression elimination, so rank-3 int8 flips retain their
shared source. Loads separated by a memory write remain distinct; the flip
matcher still requires identical SSA inputs.

`tt.unsplat` is supported for its verifier-valid source shape: a tensor with
exactly one logical element. The Metal type conversion maps that tensor to its
scalar element type, so the lowering is an exact passthrough rather than a
lane-exchange operation.

`tl.map_elementwise` is supported with `pack=1` when its scalar callback is a
single block ending in `tt.map_elementwise.return`. Both one and multiple
result tensors are supported. `pack>1` and callbacks with multi-block control
flow remain fail-closed with named diagnostics because they require adjacent
logical elements or control-flow restructuring beyond the scalar inliner.

Dynamic `scf.if` control flow supports one or more scalar results, including
mixed scalar types. The MSL emitter preserves result order with one temporary
per SSA result and assigns every branch yield to its corresponding temporary.

Elementwise unsigned integer min/max supports 32- and 64-bit values, including
high-bit inputs. Floating remainder supports f32 scalar and tensor values using
precise truncating remainder. `tl.fma` supports f32 scalar and tensor values;
non-f32 operands fail in preflight with a named diagnostic instead of reaching
MSL emission. `tl.abs` supports f16, bf16, and f32; the 16-bit paths clear the
sign bit directly, preserving subnormals and NaN payloads.

`tl.fdiv(x, y, approx=True)` supports f32 scalars and tensors using Metal's
ordinary division implementation. It follows the same supported reduction,
scan and other source-replay paths as ordinary floating-point division.

`tl.sqrt`, `tl.erf`, `tl.exp`, `tl.exp2`, `tl.log`, `tl.log2`, `tl.rsqrt`,
`tl.sin`, and `tl.cos` require f32 operands on Metal. Unsupported math widths
receive a named compile diagnostic before MSL emission.

Numeric casts involving f64 fail in preflight with a diagnostic naming the
conversion. Float-to-integer casts preserve the operation's signedness for
8-, 16-, 32-, and 64-bit results, including negative signed results.

Scalar `tl.load` supports an optional block-uniform mask and scalar `other`,
including runtime expressions and loaded values. A false mask prevents the
input memory access; loop-varying masks and address offsets are preserved.
Scalar `tl.store` also supports a block-uniform mask: false masks leave the
destination untouched, including inside loops with changing address offsets.

Multidimensional inclusive scans support add/multiply on f32/f16/bf16/i32 and
exact i32 XOR, forward or reverse, on blocked tiles with scan-axis extent 2–64.
Top-level scans can stage multiple elements per thread, with up to 8192 logical
elements and total scratch within 32 KiB. Source ranges and prefix reads use
the same layout mapping. Single-value and tuple scans also stage multiple
elements per thread inside one top-level uniform runtime loop without loop-carried
values, with tensor work confined to its flat body. Each iteration publishes
complete tiles before prefix reads and synchronizes scratch reuse. Deeper control
flow, external tensor snapshots, unsafe cross-phase replay, rank-1 XOR, wider
integers, and compound XOR combines retain named preflight rejection.

Tensor-descriptor reducing stores use Triton's standard
`triton-rewrite-tensor-descriptor-to-pointer` pass before Metal conversion.
The audited Metal envelope includes `i32` add/min/max/and/or/xor and `f32` add;
the rewrite preserves descriptor-view bounds with an explicit atomic mask and
supports contended updates. A raw `tt.descriptor_reduce` reaching the Metal
conversion pass is rejected as a pipeline-order violation rather than reported
as an unsupported frontend operation.

`tl.atomic_cas` supports scalar and blocked rank-1/rank-2 `i32`, `u32`, and
`f32` storage. The floating-point form compares and exchanges the exact IEEE-754
object representation through a `ui32` atomic view, so signed zero and NaN
payloads follow compare-exchange bit semantics rather than floating `==`.
Old-value results and contended multi-program scalar CAS are covered on physical
MPS. `f16`, `bf16`, and 64-bit CAS remain fail-closed with a named diagnostic.
On the audited Apple M4 toolchain, `atomic_ulong` supports the existing void
u64 min/max path but rejects fetch-add, bitwise, exchange, and compare-exchange
overloads; those operations therefore remain platform boundaries rather than
being approximated with a racy load/modify/store sequence.

---

## Known limitations

- **Generic tensor codegen is still mostly scalar.** The backend has specialized SIMD-group matrix, reduction, scan, threadgroup-memory, contiguous vector-add, and aggregate-hoist/cone-reuse paths, but the general ranked-tensor conversion still lowers one scalar element at a time outside those proven envelopes.
- **Op coverage is deliberately bounded.** The common elementwise, reduction, scan, gather, atomic, dot, fp8, and control-flow paths have end-to-end coverage, while unsupported shapes fail during preflight with a named diagnostic. Notable remaining envelopes include loop-carried rank-2 reductions, some cross-lane layout changes, and broader `tt.dot_scaled`.
- **`tl.dot` is not layout-general.** Proven scalar and SIMD-group-matrix paths
  cover the tested f32, fp16/bf16, int8, batched, and scaled-dot envelopes.
  Canonical masked multi-tile dots and direct rank-3 batched dots with
  `K > min(M, N)` are covered. Layout re-encoding preserves the original
  scalar tile origins, so mixed-layout output broadcasts do not introduce
  use-before-definition errors in short-wide or batched matmuls.
  For singleton batch tiles, combined offsets retain a flat batch base added
  outside a strided matrix-axis contribution; captured bases must dominate
  the replacement operation.
  Rank-2 `tt.dot_scaled` additionally accepts
  matched zero-filled rectangular A/B loads and a matching masked output
  store. A static rank-2 loop slice is covered for matching fp16/bf16 payloads
  with 16- or 32-element E8M0 groups and matching E2M1/E4M3/E5M2 payloads with
  32-element groups. E2M1 may pack either operand along K or its outer M/N
  dimension. The loop must walk the whole K range from zero, carry one
  accumulator, and advance both scale matrices by exactly one group per
  iteration. Direct rank-3 `tt.dot_scaled` is covered for unmasked canonical
  matrix I/O with batch-shared or statically proven contiguous per-batch E8M0
  scale matrices.
  Its matched zero-filled rectangular mask also covers simultaneous rank-3
  batch/M/N/K tails when A/B/output carry the same canonical batch bound and
  zero-based batch coordinate. The store-owned batch coordinate is preserved
  through `metal.scalar_dot`, so the tail guard remains aligned across a folded
  result layout conversion; mismatched A/B/output batch bounds fail closed.
  Mixed-format loop payloads, dynamic or masked loops, non-contiguous/dynamic
  scale batch strides, and arbitrary layout combinations remain unsupported.
  A direct, loop-free single-tile dot is exact at any number of elements per
  thread. Its SIMD-group path publishes the result tile into threadgroup
  scratch and reads each element back by absolute element number, so the
  address cone and the value must be labelled alike; a folded result layout
  conversion used to leave a same-shaped sibling encoding behind, and every
  output tile needing a second register band was wrong. An 8x8 result whose
  reduction is longer than one 8x8 matrix tile now takes the tier that carries
  a K loop instead of the single-accumulate path, which had computed only the
  first 8 of K.
  A direct 8x8 dot is covered at every warp count. At one warp a 64-element
  tile does not fit 32 threads one element each, so Triton gives the dot and
  the store different blocked encodings and converts between them; the
  single-accumulate path now looks through that conversion, which it may
  because it writes the tile by explicit (row, column, stride) origins rather
  than by a per-thread element index.
  A loop-free dot whose operands are sub-tiles of a WIDER buffer is covered:
  its reduction length is the tile's own K rather than A's row stride, and a
  compile-time row stride is read from the tile's own address arithmetic rather
  than assumed to be 8. Both had been guesses that agree with the facts only
  when each tile is exactly its own buffer, and both read out of range
  otherwise.
- **Launching requires PyTorch MPS.** Triton's normal on-disk cache stores the
  terminal MSL text, while `torch.mps.compile_shader` returns only an in-process
  shader library/callable and exposes no public `.metallib` or PSO handle. The
  host stack nevertheless has an effective source-specific cross-process cache:
  on the audited Apple M4/PyTorch 2.10 environment, fresh-process compilation of
  a repeated source had a `3.189 ms` median versus `29.318 ms` for unique
  sources (`0.109x`). Keep this behavior under the
  `metal_shader_compile_cache.py` benchmark rather than adding a competing,
  non-serializable backend cache.
- **Agent sandboxes can hide an otherwise working MPS device.** In particular,
  a Seatbelt-constrained Codex command may make PyTorch report zero MPS devices
  and emit its generic minimum-macOS error. The same project Pixi interpreter
  can allocate an MPS tensor when invoked outside the sandbox. Run GPU tests
  and benchmarks with `pixi run --frozen` in an unsandboxed process, and include
  that execution context in reported evidence.
- **The runtime tracks PyTorch's MPS surface.** Capabilities can regress with a torch upgrade: on torch 2.13 `torch.zeros(..., dtype=torch.float8_e4m3fn, device="mps")` fails with `Undefined type Float8_e4m3fn`, so the fp8 paths need torch 2.10-era MPS support even though the backend's own fp8 casts are unchanged.
- **The Metal wheel floor is macOS 15.0.** [PyTorch MPS](https://docs.pytorch.org/docs/stable/notes/mps.html)
  itself supports macOS 14+, but the pinned LLVM/MLIR archives contain Mach-O objects built for
  macOS 15. Metal-only packaging therefore pins both CMake and setuptools to
  15.0 and rejects a lower target instead of publishing a misleading tag.
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

The fixed-benchmark generic-performance line is complete: aligned contiguous
vector transactions, effect-bounded aggregate/cone reuse, and resource-aware
canonical matrix launch selection now have deterministic IR/MSL coverage and
physical Apple Silicon MPS evidence.

The current dependency-ordered priorities are:

1. Finish the remaining lowering-decline portion of the preflight safety
   matrix. The audited op categories in `validateUnsupportedOpsRejected` have
   direct CPU-only lit coverage, axes 1/2 `tt.get_num_programs` coverage is in
   place, and live non-splat tensor constants, non-f32 precise math, plus
   unsupported histogram ranks/types/layouts/producer cones and pointer-valued
   `arith.select`/`scf.if` and pointer-carrying `scf.while`/generic `scf.for` now fail in preflight
   instead of conversion teardown.
   Select loaded values or integer offsets instead of buffer pointers, or keep
   memory accesses inside the branches. Carry integer offsets through while
   and for loops and access the bound buffers inside their regions. Recognized
   pointer-advance matmul loops retain their specialized lowering.
   Rank-1 gathers with multiple elements per thread now reject source expressions
   that cannot be replayed before conversion starts. This includes fused
   multiply-add, floating remainder, precise division and tensor results from
   runtime branches. With f32 payloads, these forms still work when the source
   tile has at most one element per thread; supported loop-carried gather
   recurrences retain their materialized path. Rejection leaves the compiler
   usable for subsequent kernels instead of terminating the process during
   conversion cleanup.
   Histogram source and mask expressions also reject tensor-valued runtime
   branch results before conversion, including small tiles: histogram always
   replays its inputs. Scalar branch results and elementwise selects remain
   supported.
   Non-f32 FMA and the f32-only unary math operations also fail in preflight
   with type diagnostics; explicitly convert operands to f32 to use them.
   Numeric casts with f64 operands or results also have named diagnostics.
   `tl.clamp` now covers f32, f16 and bf16 rather than f32 alone, and names f64
   in preflight; a bf16 clamp still reaches the backend widened to f32 because
   Triton's frontend promotes it. Direct narrow-width Metal IR preserves the
   operand types of constants and inlined expressions: f16 selects the half
   overloads, while bf16 rounds operands before float selection and converts
   the result back. Both NaN policies retain these rounding boundaries.
   The earlier probe correctness work is complete: defined GPU probes use ordinary
   30-run regressions; the three probes without defined numerical behavior
   remain CPU translation checks.
   A rank >= 3 reduce whose rank-2 result is stored directly agrees with the
   layout conversion that carries it there. The staged reduce may relabel its
   result so that its fold and its addressing agree, and the exchange publishes
   by the plan the reduce actually used. Stored rank-3 permutations, transposed
   consumers and loaded dot accumulators use exact layout exchanges; their
   current numerical matrices have no remaining wrong answers. Load-rooted
   rank-3/rank-4 permute-to-reduce uses the same exact coordinate mapping;
   standalone rank-4 permutation and non-load-rooted sources remain boundaries.
   An independent wide rank-2 broadcast beside a multiband rank-N reduction is
   also refused before conversion: its tile needs more register bands than the
   reduction phase, and running the whole phase more often would repeat the
   reduction. Mask and address broadcasts are not covered by this refusal.
   The `tl.dot_scaled` matcher and the unsupported-op validator have been probed
   as whole families: every refusal they raise for a shape a kernel can express
   is named and raised before conversion, and the supported shapes — e4m3 and
   mixed e5m2 scaled dots, an f32/fp8 round trip, `tl.device_print`, a tensor
   `tl.atomic_add`, `tl.join`/`tl.split` at one element per thread — compile and
   match torch. A scaled dot needs M and N to be multiples of 8, direct matrix
   loads, and a store its result reaches directly.
   A rank-2 `axis=1` reduce over a computed cone is implemented for f32 only,
   and the preflight now asks the lowering's own question instead of listing the
   combines it covers. `tl.sum` over an i8 or i16 tile is the ordinary kernel
   that reached the gap: Triton promotes it to an i32 add over an `arith.extsi`
   cone, which the old list of i32 product/extrema/bitwise combines did not
   mention, so it declined during conversion instead of before it.
   The `tl.cumsum` / `tl.associative_scan` envelope has been probed shape by
   shape and its reachable refusals pinned: f16 and i64 scans, more than
   sixty-four elements per thread, an axis of extent above sixty-four on a
   rank >= 2 scan, and the tuple forms of the last two. A bf16 scan works
   because Triton widens it to f32 before the backend sees it. The rest of that
   validator's diagnostics cannot be reached by any kernel: the frontend refuses
   a non-power-of-two `num_warps` and a `tl.arange` that is empty or not a power
   of two, and the rank >= 2 branch answers before the axis and rank checks
   below it.
   A source cone that must be replayed per element and passes through an
   `scf.if` is refused before conversion. Such a cone reads as replayable
   before conversion and unrecognisable after it, because the `scf.if` is
   rewritten and its results stop being tensors, so the validator used to accept
   what the pattern then declined — a decline inside the conversion, which takes
   the process down. The general tuple reduce, the general tuple scan and the
   canonical rank-1 reduce now answer that question the same way at both times.
   At one element per thread nothing is replayed and such kernels still compile.
   Single-value rank-1 scans also check source replay before conversion for
   padded tiles, multiple elements per thread, and scans inside conditional
   control flow. Unsupported FMA, floating remainder and precise-division
   sources receive a named diagnostic. The full one-element-per-thread path
   outside conditionals remains supported. Direct uniform tensor branches
   already normalized to selects by the scan pre-pass remain supported too.
   A function whose full-tile exchange phases cannot carry a value across the
   split joined the preflight set on 2026-09-08. That rejection used to be
   emitted from the function lowering, long after the body had been moved into
   the new kernel op, and unwinding that move segfaulted on nine of thirty
   runs; preflight and the lowering now call one shared predicate, so they
   cannot disagree about which functions are rejected.
   Canonical blocked single-result rank-1 reductions also check their geometry
   before conversion: at most 64 elements per thread. The direct masked f32
   sum specialization retains its separate limit of 1024 elements per thread
   when its index/mask pattern matches and `other` is absent or zero. Slice
   reductions keep their separate geometry rules.
   Multi-element rank-1 reductions using the single-load-chain path also
   validate mask/`other` representations before conversion. That path rebuilds
   only its canonical `<` mask against a uniform bound with a constant `other`;
   every other predicate, compound mask, mask index read from memory or derived
   from an aggregate, and runtime or computed `other` is handed to the general
   cone evaluator rather than refused, by the same shared predicate the address
   rules use. A mask neither can rebuild — one carried out of an `scf.for` —
   still raises a named error, and `tt.load`'s own rule that a masked load's
   `other` be uniform across lanes is unchanged. Float-to-int casts are replayed
   wherever they appear in a cone: as the reduced value, as a mask bound, or as
   a gather index read out of a float buffer. If a cone ever passes the dry-run
   predicate that the evaluator cannot rebuild, the fold now aborts with a named
   compiler error instead of folding the combine identity, which read as an
   all-masked result. Single-band, computed-cone, and contiguous
   masked-sum paths retain their own support rules.
   The single-load-chain path also checks that its load address is a contiguous
   range plus scalar pointer offsets or the base from its matching mask. An
   address it cannot rebuild is no longer refused outright: strided, repeated,
   gathered, reversed, and index-space-shifted offsets, including a chunk loop
   written as `tl.load(X + i * BLOCK + tl.arange(0, BLOCK))`, are handed to the
   general per-element cone evaluator, which walks the same pointer chain level
   by level. Preflight and lowering choose between the two paths with one shared
   predicate, so a source neither can replay — bitwise index arithmetic, for
   instance — still raises a named error before conversion instead of silently
   reading contiguous elements. At one element per thread none of this runs: the
   band's converted scalar is read directly, which is why those spellings always
   worked there.
   The masked f32-sum specialization separately validates both load and mask
   index expressions, including its one-element-per-thread boundary. Index
   expressions it cannot replay now fail before conversion; supported integer
   casts retain the general evaluator's broader operand support.
   That specialization also replays every `tt.addptr` level of its load's
   pointer chain, so `rows = base + offsets` followed by `tl.load(rows + cols)`
   addresses both terms. A chain carrying a level it cannot account for raises
   a named error before conversion rather than dropping that level's offset.
   Rank-2 `axis=1` reductions over a direct load take their row stride from
   that load's own pointer chain instead of the reduce tile's width. A tile read
   out of a wider matrix — `tl.load(X + rows[:, None] * stride + cols[None, :])`
   with `stride` other than `BLOCK_N` — now reduces the rows it names; it
   previously reduced rows packed at `BLOCK_N`, matching only in row 0. Chains
   with a producer the replay cannot evaluate raise a named error before
   conversion.
   Constant pointer displacements and supported scalar arithmetic over program
   IDs, grid sizes (`tl.num_programs` on any axis), and runtime arguments are
   replayed where the row fill is emitted. Scalar signed/unsigned integer
   min/max retain their comparison semantics in address expressions, so a
   fill hoisted before the tile loop does not reference scalar definitions
   inside that loop. Loop-varying offsets retain their per-iteration fill.
   A masked row of a rank-2 `axis=0` reduction contributes the load's `other`
   value rather than the combine identity, matching Triton semantics and the
   rank-1 path: a fully masked column of a `tl.max` returns `other`, not
   `-inf`, and a partially masked column no longer drops those rows. A masked
   load with no `other` still folds the combine identity, which is what keeps
   that (unspecified) case deterministic.
   A rank-2 reduction over the `#ttg.slice` result of an inner reduction —
   `tl.sum(tl.sum(x, axis=2), axis=1)` — is refused by name before conversion.
   Both rank-2 paths need a blocked source tile and the staged rank-N path
   covers only the binary hypercube shape, so the combination previously
   surfaced as an unnamed legalization failure.
   A rank >= 3 reduction over a directly loaded tile, or over an elementwise
   computation on one, works at any number of elements per thread. Such a tile is addressed in thread-major order
   (`localTid * elemPerThread + iv`), and its staged fold now indexes the
   scratch by that same order rather than by the layout's own, so one label
   serves the load, the publish and the fold. A layout that wraps into a second
   register band previously made the two disagree and reduced a group the
   kernel never addressed. `tl.sort` and `tl.topk` keep the layout's own
   indexing — their ranges are emitted in layout space, so for them the two
   already agree.
   That fold is measured across every combine the backend recognizes, not only
   `add`: the arithmetic, extremum, bitwise and replayed-region forms over f32,
   i32 and f16, and masked loads whose `other` is deliberately away from the
   combine's identity, so a dropped `other` cannot hide inside it.
   Rank >= 4 tiles outside a planned aggregate phase are refused by name.
   Within that phase their per-axis ranges use the consumer tile's coordinates,
   including load-rooted rank-4 permute-to-reduce. A rank-3 reduction over a tile
   accumulated by an enclosing loop is likewise refused by name; it previously
   failed the pass with no diagnostic at all.
   The canonical tiled matmul is now exact at every block shape, not only at
   an 8-wide reduction. One `simdgroup_multiply_accumulate` consumes an 8-wide
   slice of K, and the unrolled pointer-advance loop emitted one per K-loop
   ITERATION, so a `BLOCK_K` above 8 silently dropped every reduction column
   but the first eight of each tile. It is exact at `BLOCK_K == 8` and exact
   when the whole reduction is one tile, which is the shape the suite and the
   perf report both used, so the accumulation the tutorials write -- `acc =
   tl.dot(a, b, acc)` over several `BLOCK_K = 16` or wider tiles -- returned
   wrong values in f32 and f16 alike. The loop now steps K by 8 inside each
   iteration, and the tier requires A's reduction axis to be contiguous rather
   than reading a tile along a strided K. The block shape is measured against
   torch across the output-tile grid, the steps inside an iteration and the
   iteration count, at one and four warps.
   A `tl.dot` accumulator that does not start from zero is now checked rather
   than assumed. The unrolled tier emits such an init as a matrix load from the
   destination at the store's own origin, which answers one shape -- a kernel
   that accumulates in place -- and it used to accept any non-zero init, so a C
   tile read from a different buffer was answered with whatever the output held.
   The init must now be an unmasked load from the store's own pointer; anything
   else, including two kernel arguments that happen to point at the same tensor,
   is refused by name. Sweeping the reduction width across the other dot forms a
   kernel can write -- a transposed B, a batched rank-3 dot, a fused
   `alpha*(A@B) + beta*C` epilogue, two dots per iteration, and an f16 store --
   found no further wrong answers: each is refused by name at every reduction
   width, so none of them was dropping reduction columns either.
   Every matmul tier now checks that the tiles it addresses are contiguous
   along their inner axis. A matrix load or store reads
   `buf[(row + i) * stride + col + j]`: the row axis carries a stride, so a
   sub-block of a wider matrix works and is covered, but the inner axis
   advances one element at a time and cannot express a stride. No tier asked,
   so an A whose reduction axis advanced by two, a B whose column axis did, and
   a result written with a non-unit column stride were each computed against a
   dense tile and answered wrongly, in the unrolled, runtime-bound and
   standalone tiers alike. They are refused by name now, except where the
   per-thread scalar reduction can still answer them, which it does for a
   strided destination because it replays the store's own arithmetic.
   Transposed operands are staged the other way round and keep their existing
   behaviour rather than being refused on an untested assumption.
   A matmul whose destination carries a scalar base offset -- a program
   writing its tile into a slice of a larger buffer, `tl.store(out + base +
   tile, ...)` -- now lands where the kernel asked. The operand sides have
   always folded such a base into the column they address, because a matrix
   access adds the column unscaled; the store's base was read and then dropped,
   so every tile went to the buffer's origin instead. The masked epilogue's
   column extent moves with the origin, so a partial tile still compares
   against the same bound.
   A K loop that recomputes its addresses from the induction variable now takes
   its reduction length from the loop's own upper bound rather than from A's row
   stride. Collapsing such a loop into one reduction means naming that length,
   and the row stride is the same number only when A is the whole matrix; for a
   tile taken from a wider one it overshoots and the reduction ran past the end
   of the tile into whatever the buffer held next. Each operand is also asked
   about its own inner-axis contiguity separately, since a transposed B used to
   excuse a strided A.
   A transposed B is asked about the right axis. The two spellings need
   opposite ones: a real `tl.trans` loads an `[N, K]` tile, so the axis its
   buffer is contiguous along is the tile's axis 1, while a transpose written
   into the address arithmetic loads a `[K, N]` tile whose dense axis is 0.
   Both look transposed to the matcher, so the axis is chosen by the load's own
   consumer -- only the first spelling has a transpose to consume it. The tier
   that claims a real transpose also addresses its operands by origins alone
   and cannot honour a scalar base offset on any of the three buffers, so it
   now declines one instead of writing, and reading, at the buffer's origin.
   The canonical fp16 matmul compiles. Writing an `f16` result puts a dtype
   convert, and the layout convert that hangs off it, between the accumulation
   loop and the store. The tier that recomputes its addresses each iteration
   stepped through that pair; the two that advance their pointers -- which is
   what the tutorial's own kernel does, and every kernel copied from it -- did
   not, so an fp16 matmul with a flat swizzled grid matched no tier at all and
   was refused by name at every block shape and warp count. Both now step
   through it, bridge the destination with the store's element type, and erase
   the convert chain with the loop it belonged to. The narrowing still requires
   a masked store, which stages through an f32 slot and casts on the element
   write; an unmasked one would need a Metal overload that does not exist. A
   64x64 block at a single warp stays refused, over the accumulator register
   budget.
   The runtime-bound matmul asks once, before its reduction loop, whether the
   tiles this program touches lie wholly inside its operands, and gives the
   interior case a loop whose staged loads carry no bounds at all. Staging an
   operand behind a runtime test costs about a third of the kernel's time: a
   predicated device read cannot be hoisted out of the staging loop, and the
   reduction loop cannot be pipelined across it. Hoisting the test to once per
   tile inside the loop changes nothing, which is how the cost was identified
   as the predication rather than the comparisons. The question is answered by
   the block origins and the extents, all loop-invariant and uniform across the
   threadgroup, so the branch is safe where a staged load brackets itself with
   a threadgroup barrier. A partial block, or a reduction that is not a
   multiple of the matrix unit's eight, takes the predicated loop as before.
   This also covers single-dot loops that rebuild their addresses from the K
   induction variable, including a transposed B loaded as `[N, K]`. The
   duplicated body is limited to sixteen accumulator tiles per warp.
   The standalone pointer-advance fp16 path batches the interior operand copies
   into disjoint per-warp slots, keeping half inputs and float accumulators.
   Two SIMD-group fences protect the whole batch instead of each 8x8 operand.
   The extra scratch is capped at 8 KiB; compound kernels, downselected
   launches, and larger tiles retain the existing path. Boundary loads keep
   their masks. Masked matrix stores also use SIMD-group fences when every
   user of their scratch has a proven private warp slice; mixed users retain
   threadgroup fences. `--canonical-report --dump-dir <path>` records CPU-checked
   PyTorch comparisons and generated MSL; `--reference-msl-dir <baseline-path>`
   adds interleaved comparisons with archived baseline shaders.
   Two paired runs on Apple M4 (macOS 26.4, PyTorch 2.10.0) measured
   1.23–1.27x acceleration for 1024³ fp16, 64x64 tiles and four warps.
   End-to-end Triton latency remains about 2.1x that of preallocated
   `torch.matmul`; this does not establish parity with PyTorch.
   The `--recompute` mode of `python/test/microbenchmark/metal_matmul_canonical.py`
   measures masked and unmasked forms against a CPU numerical reference.
   The separate fused LoRA lowering checks X/W and adapter A bounds
   independently. Full interior tiles use an unmasked K loop; padded adapter
   ranks use an interior loop with only A still masked. Partial X/W tiles or
   partial eight-element K steps use the fully masked fallback. All three
   paths retain epilogue and output masks and use threadgroup-uniform
   conditions. Duplication is capped at 100 emitted MMA operations per
   original body including the epilogue (300 across the three paths); larger
   bodies keep the original masked loop. This covers the
   64x128/BR16/four-warp fixture. `--lora` benchmarks plain/masked forms and
   the owned verbatim fixture; `--lora-padded` adds padded ranks 8/15 with a
   full-rank 16 control. Both check CPU references before and after timing.
   The floating `tl.minimum`/`tl.maximum` NaN-policy defect is fixed.
   `propagate_nan=tl.PropagateNan.ALL` now returns NaN whenever an operand is
   NaN, and orders `-0.0` below `+0.0`, through the elementwise, scalar and
   reduction-fold paths alike. The default number-preferring policy compiles
   to the MSL builtin. bf16 min/max preserve operand rounding, select through
   the float overload, and explicitly convert the result back to bf16.
   Rank-1 floating extrema use a NaN identity for number-preferring partial
   folds and padding, preserving all-NaN values and their first matching index.
   Rank-2 axis-0 and axis-1 float extrema seed from the
   first source element, preserving NaN for all-NaN columns or rows and exact
   infinities. Axis-0 f32 `tl.argmin`/`tl.argmax` support the two-pass value/index
   implementation for supported direct and computed tiles, including outputs
   wider than the threadgroup and uniform runtime loops. Mixed f32 column
   results retain their own type when used by the i32 index pass; scalar-staged
   results must still refer to the same column. Scalar-staged dependencies
   reached through a transpose or shape-changing reshape remain rejected.
   Rank-2 reductions over an axis of extent 1 preserve each input unchanged,
   including heterogeneous tuples, NaN payloads and signed zero. Flattened
   results use an exact layout exchange when register ownership changes.
   Axis-1 f32 arg extrema also support two-pass index folds over wide replayable
   tiles. Both passes run before tensor producers are scalarized; the index
   pass reads the preceding row extrema by logical row and uses a bounded i32
   loop. Other live consumers must remain row vectors, including through loop
   state. Top-level replay preserves loaded values across later stores; user
   loops with unsafe intervening writes or externally loaded values that may
   be overwritten remain rejected. General computed i32 reductions retain
   their existing direct-load or scan-publication requirements.
2. Extend aggregate combinations on the refreshed layout/tile baseline. The
   current permutation, transpose, loaded-accumulator and transposed-store
   matrices are numerically clean. Rank-3 and rank-4 permute-to-reduce now map
   thread-major load-rooted input coordinates through the permutation before
   exact publication, on every reduction axis and across register bands.
   Non-load-rooted permuted sources remain named boundaries. One top-level
   no-iter_arg uniform loop can now stage direct-load rank >= 3 reductions
   independently on each iteration, with cyclic publish addressing and barriers.
   Nested or multiple rank-N reduction loops, loop-carried sources and one-band
   loop reductions remain named boundaries. Top-level single-value rank >= 2
   scans now support multiple elements per thread. Rank-2 and rank-3 scans with
   different register-band counts can share a function: each scan's consumers
   finish in that scan's own phase before the next publication begins. Rank >= 2
   tuple scans now support f32/i32
   carried values, both directions, and axis extents 2..64. Top-level tuples
   support multiple elements per thread, consecutive scans, and mixed
   tuple/single-value exact phases. All components publish together before
   prefix reads; grouped buffers use dedicated storage within the combined
   32 KiB threadgroup budget. One-element-per-thread tuples also support
   uniform user loops. Multi-element single-value and tuple scans now support
   one top-level uniform runtime loop, including consecutive scans, memory
   updates across iterations, tensor iter_args, and snapshots of tensors
   computed before the loop. Carried tensors use threadgroup slots and must be
   integer or floating-point tiles of at least eight bits, have one owner per
   element, and use the function tile's register count; their slots count
   against the 32 KiB threadgroup budget. Scalar carries, replicated-owner or
   mismatched-register-count tiles, a tile-carrying loop containing `tl.dot`,
   multiple tile-carrying scan loops in one function, deeper control flow, and
   unsafe cross-phase replay remain rejected.
   General rank-1 tuple combines and axis-0 reductions with multiple output
   elements per thread are implemented. General rank-1 tuple scans support
   both directions with up to 64 elements per thread, within the existing
   f32/i32 and threadgroup-memory limits. Single-value blocked and flattened
   linear-layout rank-1 reductions replay up to 256 elements per thread;
   larger bands remain rejected without changing tuple or scan envelopes.
   Single-value f32/i32 rank-1 scan
   results wider than the threadgroup can now broadcast along either axis of
   a rank-2 tile, including elementwise combinations of scans and subsequent
   reductions. Broadcasts use their own consumer tile coordinates and retain
   prefix buffers through deferred readers. A direct f32/i32 row sum, product,
   min/max or i32 AND/OR/XOR of a staged scan's result, with either NaN policy for
   f32 extrema and signed or unsigned comparisons for i32, publishes that result
   after the scan and reduces the second buffer in the next phase. Each of these
   reductions can also expand back into the original tile, including alongside a
   direct row store. An independent single-band f32 column sum before
   such a scan publishes its own small tile and finishes its consumer in that
   tile's own phase. A single top-level multi-band rank-2 scan can also
   share its input publication with subsequent reductions of the exact same
   source. Axis-1 reductions support direct row stores or
   same-axis broadcasts into the original tile. Shared-source f32 row sums also
   support cross-axis broadcasts into the original tile and same-axis broadcasts
   into a wider tile, evaluated at the consumer's projection. Scan-input f32 row
   sum/product/min/max support both consumers together for tiles with 2..64 rows
   and columns, including both NaN-propagating and number-preferring extrema,
   using separate vector and broadcast folds over one input publication. Row sums
   also accept a matching distinct reduction source; other combines require the
   exact scan input. For row sums, the mixed direct-row/broadcast form accepts a
   same-axis broadcast into a wider tile for either source, including the
   consumer's layout conversion. Shared-source row products and extrema also
   accept this wider form. Shared-source row sums, products and extrema accept a
   cross-axis broadcast into a square original tile.
   Shared-source i32 row sum/product/AND/OR/XOR and signed or unsigned min/max
   also support both consumers together under the same multiband tile limits,
   preserving comparison signedness, 32-bit values and wrapping arithmetic for
   masked or computed inputs. Axis-0 reductions support
   direct column-vector stores, including outputs wider than the threadgroup. The
   shared path preserves input snapshots across later device stores and
   supports multiple reductions. f32 row and column-vector sum/product/min/max
   accept supported masked or computed sources.
   i32 row sum/product, signed or unsigned min/max and AND/OR/XOR, and
   column-vector sum/product, signed or unsigned min/max and AND/OR/XOR also
   accept supported masked or computed sources, preserving mask fill values and
   computed results. Shared-path unsigned min/max use the scan publication for
   plain sources too, with unsigned comparisons over unchanged 32-bit values.
   Independently of scans, axis-0 i32 signed/unsigned min/max and AND/OR/XOR
   support direct and same-kind loop-carried column reductions, including
   masked or computed sources and outputs wider than the threadgroup. Wide
   re-expansion remains limited to f32 results; i32 extrema or bitwise
   re-expansion is rejected by name.
   A single-band rank-2 scan whose tile exactly fills its threadgroup also
   supports mixed row-vector stores and original-tile broadcasts of a same-source
   f32 row sum, product or min/max under either NaN policy, with 2..64 rows and
   columns. Separate row folds preserve each consumer's result mapping. Sums and
   extrema accept replayable masked/computed sources; products
   require a direct unmasked load. These reductions
   require no intervening device writes between source loads and
   the reduction; writes after its outputs are supported. Its complete emitted
   threadgroup scratch, including both row buffers, must fit the 32 KiB budget.
   Axis-1 reductions and axis-0 column-vector stores may also use distinct
   inputs with the same shape, layout, and dtype, published together with the
   scan input. Their complete input computations must precede the scan,
   without input exchanges or intervening
   device writes; repeated inputs share a publication, within the existing
   threadgroup-memory budget. The same source rules apply to distinct i32
   reduction inputs; independent scan inputs may be masked or computed.
   Shared or distinct-source f32 column sum/product/min/max and i32 column
   sum/product/min/max/AND/OR/XOR can also broadcast directly back into the
   original tile after the scan, with 2..64 rows and
   2..512 columns, including more columns than threads. The scan-axis extent
   remains at most 64, and all input buffers must fit the 32 KiB budget.
   This column-broadcast path accepts supported masked or computed f32/i32
   inputs, preserving mask fill values and computed results. i32 supports both
   signed and unsigned extrema; independent scan inputs may also be masked or
   computed.
   This path requires the original layout and
   only elementwise consumers followed by stores. Distinct sources obey the same
   pre-scan computation and write-isolation rules above; repeated sources share
   a buffer. The fold reads its own complete input buffer without another column
   publication. Other independent column broadcasts, other reductions of the
   scan result, and other mixed row-store/broadcast or consumer projections
   remain rejected in this combination. Mixed device-load
   cones for
   wide rank-1 scan broadcasts,
   wide tuple scan broadcasts, and ambiguous
   consumer layouts also remain rejected.
   Broader descriptor-reduce type/rank and
   `tl.map_elementwise` pack/control-flow support remain adjacent extensions.
   The planned scaled-dot slices are complete, including mixed E4M3/E5M2,
   canonical dynamic/masked loops, and masked rank-3 batch tails; more general
   combinations and non-canonical scale addresses remain outside that scope.
   A scaled dot must also reach its store directly: an epilogue in between,
   including the implicit truncation of an f16 output buffer, is named in
   preflight rather than half-rewritten. Accumulate and store f32, or apply the
   epilogue in a second kernel. A bias add is fine because Triton folds it into
   the dot's accumulator.
   Each extension needs a positive case and an adjacent rejection regression.
3. Run the MPS numerical wheel smoke on a physical macOS 15 machine. CI now
   builds/audits on macOS 26, then installs and compile-smokes the
   `macosx_15_0_arm64` artifact on the hosted macOS 15 runner, while physical
   Apple Silicon remains necessary for GPU numerical validation.

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
