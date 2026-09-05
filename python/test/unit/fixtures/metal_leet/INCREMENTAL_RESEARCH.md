# Triton Metal × `leetgpu-triton-solve` incremental research

Date: 2026-09-01

## Decision

The corpus needed **targeted incremental research**, not a full restart. The P0
reverse correctness gap, P1 generic GroupNorm reduction replay gap, P1 source
fidelity work, P2 fixture/matcher generalization work, P2.4 atomics, and the
P3 dot-geometry/scaled-dot slices identified below have now been
implemented and regression tested. The cache question was resolved by measurement: the
audited PyTorch/Metal stack already provides effective source-specific reuse
across fresh processes, so no competing backend cache is warranted.

The expanded Metal fixture suite is now a stronger regression baseline: the
final 2026-09-01 physical-Apple-Silicon run completed with 1,498 pytest passes,
three skips, all 24 standalone scripts passing, and all 90 fixtures owned. The
audited baseline had exposed one Metal correctness gap, one generic performance
gap, and several source/fixture fidelity issues that the old aggregate pass
count could not express. The correctness tables below retain the pre-fix
evidence deliberately, followed by the implementation result.

## Fixed baseline and execution context

- Triton: `ece17d9fe6c0111f56e66abf7e9d7118c36812eb`
- `leetgpu-triton-solve`: `4dc4e6931f5f534981b327522464b3bebcefead1`
- Host: Apple M4, macOS 26.4
- Python runtime: project Pixi environment, PyTorch 2.10.0 arm64
- GPU commands: `pixi run --frozen ...`, run serially outside the Codex
  Seatbelt sandbox

The sandbox distinction is required evidence, not incidental setup. Five
repeated checks in the sandbox reported `MPS=False` and zero devices; the same
Pixi command outside it reported `MPS=True`, one device, and successfully
allocated an MPS tensor in all five runs. The repository-level operational
warning is documented in the [root README](../../../../../README.md).

## Incremental recheck on the fixed baseline

A second, targeted check on 2026-09-01 found that neither pinned commit had
changed. Before implementation, the full `leet-all` suite was therefore not
rerun: its 1,469 passes, three skips, 24 standalone-script passes, and
`54.97 ms / iter` LLaMA result were carried forward from the earlier run of
these exact commits. After the compiler fix, `leet-all` was rerun in full; those
fresh results are reported in the implementation section below.

All new GPU commands in this recheck ran serially as `pixi run --frozen ...`
outside Seatbelt. Three independent processes each reported `MPS=True` and one
device. At that point, the seven non-reverse, non-empty source-only workloads
again matched their PyTorch references. Those eight non-empty source-only files
have since been added to the fixture suite, including reverse after the P0 fix.
The reverse probe and GroupNorm benchmark were rerun because they directly test
the two outstanding compiler findings.

The recheck strengthens, but does not reorder, the recommendations below:

- Reverse remains P0. In addition to repeatable wrong values, a 60-run
  multi-program probe at `N=2053` produced two distinct wrong outputs, confirming
  that the lowering defect can introduce nondeterministic execution rather than
  only a stable permutation.
- GroupNorm was confirmed as P1 before implementation. The split path again
  took about 55% of PyTorch MPS time, while the raw eight-warp fused path again
  took about 2.5 times PyTorch MPS time. The later compiler implementation and
  post-fix measurements are recorded below.

## Corpus fidelity

The current source tree has 88 top-level Python workloads. Mapping the causal
attention filename's case-only rename to the fixture name gives this structural
comparison:

| Class | Files | Meaning |
|---|---:|---|
| Whole file identical | 35 | Current source and fixture are byte-identical |
| JIT kernels identical | 44 | Kernel AST is identical; only text, driver, validation, or return behavior differs |
| JIT kernels changed | 6 | Fixture does not execute exactly the current source kernel set |
| Fixture only | 3 | Repository smoke/tutorial fixtures with no mapped leet source |
| Current source only | 1 | Explicitly excluded zero-byte source file |
| Current source unparsable | 2 | Mapped fixture exists, but the current source file is not valid Python |

The comparison used Python 3.12 `ast.parse` and location-free `ast.dump` for
top-level functions decorated with `triton.jit`; comments and formatting are
therefore ignored in the kernel-identical class. Whole-file identity was checked
separately before AST classification.

The eight non-empty source-only files were added as byte-identical fixtures and
are now covered by the interpreter-backed gap suite:

- `easy-relu.py`
- `easy-reverse_array.py`
- `easy-rgb_to_grayscale.py`
- `easy-sigmod_linear_layout.py`
- `easy-softmax_activation.py`
- `easy-swish-gated_linear_unit.py`
- `easy-value_clipping.py`
- `easy-vector_addition.py`

The only remaining current-source-only file is `easy-matrx_copy.py`, which is
zero bytes and is explicitly recorded as excluded in
[`source_fidelity.json`](source_fidelity.json).

The two unparsable current files are not Metal failures:

- `medium-2d_convolution.py` ends with an un-commented Chinese question.
- `medium-subarray_sum.py` contains a stray `a` after `tl.atomic_add(...)`.

The fixture inventory is exhaustive for its own 90 files, as enforced by
[`test_metal_backend_leet_uncovered.py`](../../test_metal_backend_leet_uncovered.py),
and the source-fidelity manifest is checked by
[`test_metal_leet_source_fidelity.py`](../../test_metal_leet_source_fidelity.py). The manifest
pins the 88-file current source corpus, explicitly excludes the zero-byte
`easy-matrx_copy.py`, maps the case-only causal-attention filename difference,
and classifies every non-empty source-backed fixture as `verbatim_kernel`,
`host_only_adaptation`, `source_repair`, or
`metal_specific_kernel_rewrite`.

## New correctness finding: multi-band reverse indexing

On the audited pre-fix baseline, the unmodified current
`easy-reverse_array.py` was wrong on Metal with its default
`BLOCK_SIZE=1024` and default four warps:

| Input length | Mismatched elements |
|---:|---:|
| 127 | 62 / 127 |
| 128 | 63 / 128 |
| 129 | 64 / 129 |
| 2,048 | 1,022 / 2,048 |
| 2,053 | 1,023 / 2,053 |

The failure was tied to the scalarized tile loop. With `N=127`, launches using
1, 2, 4, 8, or 16 warps all failed; 32 warps passed because 1,024 physical
threads removed the multi-register-band loop.

Fresh repeated runs made the severity clearer. For 20 launches at each tested
length, `N=127`, `128`, `129`, and `2,048` each produced one stable wrong
output. At `N=2,053`, 60 launches produced two distinct wrong outputs with the
same mismatch count and maximum error but different values at 24 positions
(`148-159`, `416-419`, `560-563`, and `704-707`). The exact 32-warp control and
the variability only in the multi-program case are consistent with overlapping
physical accesses—and therefore a race—introduced by the inconsistent lowering.

Pre-fix generated MSL confirmed inconsistent lowering of the same logical
`tl.arange(0, 1024)` value. One address cone uses a strided index equivalent to
`local_tid + iv * 128`, while the reverse-address cone uses a contiguous index
equivalent to `local_tid * 8 + iv`. A load from `N - 1 - contiguous_index` is
then stored at `strided_index`, producing the observed periodic wrong-address
pattern.

This is a valid block program: active logical elements cover disjoint mirrored
pairs, with no source-level lane IDs or divergent scalar backdoors. Inspection
of the runtime TTGIR identified the exact split. Aligned input metadata produced
a contiguous rank-1 blocked layout (`sizePerThread=[4]`) for the loads and first
store, while the second store used a strided blocked layout
(`sizePerThread=[1]`) and bridged only its value with `ttg.convert_layout`. The
generic divergent-conversion normalization rewrote a producer/address cone
shared with the first store, leaving that store's address mapping inconsistent
with its value. This is direct IR/MSL evidence, not an inference from output
values alone.

## Implemented P0 fix: rank-1 store-side normalization

The Metal conversion now admits same-shape rank-1 blocked-to-blocked pairs in
the existing store-side divergent-layout normalizer. For a conversion used only
as a store value, it clones the destination-layout pointer and mask cones into
the value's source layout, stores the original source value, and removes the
conversion. Rank-1 store-side normalization runs before the generic producer
rewrite so shared source-layout users are not mutated.

The precedence change is deliberately rank-1-only. An intermediate version ran
store-side normalization first for rank 2 as well; the full MSL test file then
failed three `test_gather_rank2_both_axes` cases while 422 tests passed. Restoring
the established generic-first rank-2 path made all 425 tests pass and retained
the reverse fix. This failed intermediate result bounds the final change rather
than broadening it speculatively.

Regression coverage now includes:

- a runtime copy of the raw reverse shape in
  [`test_metal_backend_multiload.py`](../../test_metal_backend_multiload.py),
  covering partial, exact-band, and multi-program lengths;
- a TTGIR/lit case in
  [`convert_layout_rank1_divergent_spt.mlir`](../../../../../test/Dialect/Metal/convert-tritongpu-to-metal/convert_layout_rank1_divergent_spt.mlir)
  that rejects an `iv * 128` strided address mapping in the converted kernel;
- fresh generated Metal source in which both stores and both loads derive from
  the contiguous `tgid * 1024 + local_tid * 8 + iv` logical index.

The unmodified `leetgpu-triton-solve/easy-reverse_array.py` was then run 20
times each at `N=127`, `128`, `129`, and `2,048`, plus 60 times at `N=2,053`.
All 140 launches had zero mismatches and exactly one (correct) output per size.

Final verification on the physical Apple M4, always through
`pixi run --frozen` outside Seatbelt, produced:

| Verification | Result |
|---|---:|
| Target lit regression | 1 passed |
| Full Metal lit directory | 157 passed |
| Full `test_metal_backend_msl.py` | 425 passed |
| Multiload + masked-store + reverse-scan + MPS zero-copy | 54 passed |
| Raw reverse stability launches | 140 / 140 correct |
| Full `leet-all` pytest ownership suite | 1,469 passed, 3 skipped |
| `leet-all` standalone scripts | 24 / 24 passed |

## Changed-kernel classification

Seven high-information mapped fixtures differ from their current source files
in ways that require explicit classification. They should not be treated as one
category:

| Workload | Classification | Incremental evidence |
|---|---|---|
| `easy-matrix_transpose.py` | Host-only adaptation | Raw current kernel and fixture have identical JIT AST; the remaining diff is the standalone driver and formatting |
| `medium-count_3d_array_element.py` | Source correctness repair | For a seven-element input padded to a 1,024-lane tile, raw source returns 1,019 instead of 2 when `P=-1`: all 1,017 masked lanes use `other=-1` and are counted; adding `mask &` makes the fixture correct |
| `medium-fused_residual_add_and_rms_norm.py` | Source repair and numerical stabilization | Raw source passes normal `C=1024`, fails a `1e20` scale case, and cannot compile `C=9000` because its loop kernel contains frontend errors; fixture passes all three |
| `medium-group-normalization.py` | Metal-specific tuned path | The compiler no longer replays fused statistics under the output loop; the fixture retains its split MPS path because it remains materially faster on the measured Apple M4 |
| `medium-ordinary_least_squares.py` | Source shape repair | The Gram matrix now launches over both feature axes and `X^T y` is computed by a separate one-dimensional kernel, avoiding duplicated/racy writes; standalone coverage passes at 8, 32, and 48 features |
| `medium-sparse_matrix-vector_multiplication.py` | Semantic/API expansion | Fixture adds a true COO path plus dense compatibility fallback; it is broader than the nominal raw dense kernel |
| `medium-sparse_matrix-Dense_matrix_multiplication.py` | Semantic/API expansion | Fixture adds a true COO path plus dense compatibility fallback; it is not a verbatim-source compiler result |

This classification matters: only GroupNorm currently demonstrates a required
Metal-specific kernel restructuring. Several other diffs fix source bugs or
expand the workload contract and should be labelled that way rather than counted
as backend support for the current raw program.

## Performance evidence

All values below are steady-state medians from the fixed Apple M4 baseline.
They are directional single-machine evidence, not claims about all Apple GPUs.

### GroupNorm: generic replay fixed

The repository benchmark
[`metal_group_norm.py`](../../../microbenchmark/metal_group_norm.py) uses shape
`(8, 512, 64, 64)`, 32 groups, float32, five warmups, 20 iterations per sample,
seven samples, and two order-reversed rounds.

| Path | Triton median | PyTorch MPS median | Triton / PyTorch |
|---|---:|---:|---:|
| Metal split fixture, round 1 | 6.978 ms | 13.177 ms | 0.530× |
| Metal split fixture, round 2 | 7.505 ms | 13.262 ms | 0.566× |
| Raw fused kernel, round 1 | 33.110 ms | 13.579 ms | 2.438× |
| Raw fused kernel, round 2 | 31.636 ms | 13.546 ms | 2.336× |

Both paths have maximum absolute error `1.90735e-06`. The 4.2-4.7× gap between
the raw fused kernel and the split fixture confirms that aggregate/reduction
replay under generic tensor scalarization is still a high-value compiler target.
The split fixture is already 1.77-1.89× faster than PyTorch MPS, so the
workload algorithm is viable once the replay is avoided.

The same-commit incremental recheck used identical benchmark parameters and
reported:

| Path | Triton median | PyTorch MPS median | Triton / PyTorch |
|---|---:|---:|---:|
| Metal split fixture, round 1 | 7.440 ms | 13.479 ms | 0.552× |
| Metal split fixture, round 2 | 7.365 ms | 13.467 ms | 0.547× |
| Raw fused kernel, round 1 | 33.450 ms | 13.475 ms | 2.482× |
| Raw fused kernel, round 2 | 33.273 ms | 13.531 ms | 2.459× |

Maximum absolute error again remained `1.90735e-06`. These measurements preserve
the original qualitative result: the split workaround is fast and correct,
whereas generic replay makes the raw fused form materially slower than PyTorch.

The post-reverse-fix guard produced the same conclusion:

| Path | Triton median | PyTorch MPS median | Triton / PyTorch |
|---|---:|---:|---:|
| Metal split fixture, round 1 | 6.787 ms | 12.998 ms | 0.522× |
| Metal split fixture, round 2 | 7.012 ms | 12.978 ms | 0.540× |
| Raw fused kernel, round 1 | 30.307 ms | 13.248 ms | 2.288× |
| Raw fused kernel, round 2 | 31.429 ms | 13.011 ms | 2.416× |

Maximum absolute error remained `1.90735e-06`. The split path also passed a
`Triton/PyTorch <= 0.8` assertion in both rounds. The P0 layout fix neither
removes nor worsens the separate generic reduction-replay opportunity.

After the follow-up rank-1 cone memoization subfix, the duplicate-leaf replay
case is structurally fixed but the raw fused GroupNorm path is still dominated
by aggregate replay under the output tile loop:

| Path | Triton median | PyTorch MPS median | Triton / PyTorch |
|---|---:|---:|---:|
| Metal split fixture, round 1 | 6.634 ms | 12.973 ms | 0.511× |
| Metal split fixture, round 2 | 6.740 ms | 12.959 ms | 0.520× |
| Raw fused kernel, round 1 | 30.427 ms | 13.111 ms | 2.321× |
| Raw fused kernel, round 2 | 28.944 ms | 13.111 ms | 2.208× |

Maximum absolute error remained `1.90735e-06`. The raw fused run used a shorter
diagnostic protocol (`warmup=3`, `iterations=10`, `repeats=5`) to compare the
compiler subfix quickly; the split path used the established guard protocol and
again passed `Triton/PyTorch <= 0.8`. This confirms that memoizing replayed
leaves removes redundant loads inside a computed cone, but does not replace the
larger aggregate-prelude/output-loop scheduling work.

A direct attempt to complete the larger scheduling step by cloning external
ranked tensor cones into the newly split output tile loop was not retained. It
made raw fused GroupNorm reach Metal translation with an unconverted
`tt.make_range` and abort in `ModuleTranslation.cpp`. After reverting only that
attempt, raw fused GroupNorm again compiled and produced the same `1.90735e-06`
maximum absolute error. That failure motivated the retained pre-conversion
selection plus `FuncOpLowering` materialization described next, where cloned
Triton ops are guaranteed to re-enter dialect conversion.

The retained compiler implementation selects a deliberately narrow top-level
scalar recurrence: scalar loop-carried state, at least one replayable f32
rank-1 sum, no writes or nested control flow, and a result that reaches device
output. `FuncOpLowering` clones only the recurrence's pure SSA dependency cone
and the loop into the kernel prologue before constructing the synthetic output
tile loop. Function-scope threadgroup allocations inserted by earlier
pre-passes remain ahead of the recurrence; arbitrary reads and writes are still
hard hoist boundaries. This is structural matching and does not depend on the
GroupNorm kernel name.

Generated Metal IR now places the 65,536-element statistics loop and its pure
scalar mean/variance/rstd chain before the 16-trip output-band loop. Three
independent invocations of the exact full raw-fused benchmark protocol produced:

| Invocation / round | Triton median | PyTorch MPS median | Triton / PyTorch |
|---|---:|---:|---:|
| Raw fused 1 / 1 | 15.228 ms | 12.952 ms | 1.176× |
| Raw fused 1 / 2 | 15.183 ms | 12.949 ms | 1.172× |
| Raw fused 2 / 1 | 15.383 ms | 12.950 ms | 1.188× |
| Raw fused 2 / 2 | 15.288 ms | 12.930 ms | 1.182× |
| Raw fused 3 / 1 | 15.371 ms | 12.955 ms | 1.186× |
| Raw fused 3 / 2 | 15.440 ms | 12.944 ms | 1.193× |
| Metal split fixture, round 1 | 6.553 ms | 12.971 ms | 0.505× |
| Metal split fixture, round 2 | 6.572 ms | 12.928 ms | 0.508× |

Both paths retained maximum absolute error `1.90735e-06`. The raw fused path
passed the `Triton/PyTorch <= 1.25` acceptance gate in all six rounds, versus
2.288× and 2.416× immediately before the scheduling fix. The split path passed
its existing `<= 0.8` guard and remains the preferred M4 specialization on
performance grounds even though the fused source no longer requires it.

Final MSL inspection found two smaller code-placement issues after the main
scheduling change. First, the translator inlined the single-use reciprocal
standard deviation at its nested consumer, which moved `sqrt` back into the
output loop despite the Metal IR order. The final scalar suffix is now marked
for materialization, so MSL names it once before both output loops. Second,
rank-1 replay left the original masked tensor-load cone dead but printable,
adding one unused device read per statistics tile. Post-conversion DCE now
cleans the complete pure cone only inside the newly materialized recurrence.
A broader module-wide version was rejected after it removed intentionally dead
lowering anchors in five lit fixtures; the scoped version preserves them and
passes the full suite.

The exact final-tree acceptance run produced:

| Path / round | Triton median | PyTorch MPS median | Triton / PyTorch |
|---|---:|---:|---:|
| Raw fused, round 1 | 15.795 ms | 13.014 ms | 1.214× |
| Raw fused, round 2 | 16.039 ms | 13.037 ms | 1.230× |
| Metal split fixture, round 1 | 6.649 ms | 12.951 ms | 0.513× |
| Metal split fixture, round 2 | 6.990 ms | 13.075 ms | 0.535× |

Maximum absolute error remained `1.90735e-06`. Generated MSL contains no
abandoned tile-head load, and the sole `metal::precise::sqrt` is before the two
output loops.

The first complete fixture run after the initial selector exposed a false
positive in `medium-softmax`: its second (sum) recurrence depends on an earlier
max recurrence and must not move ahead of it. The selector now treats every
unselected preceding loop as an ordering boundary and preflights the complete
top-level dependency cone. The softmax case then passed directly, and the final
serial `leet-all` run completed with 1,469 pytest passes, three skips, and all
24 standalone scripts passing. The full Metal conversion lit directory also
passed all 157 tests. The newest LLaMA regression anchor was `53.63 ms / iter`;
as above, that anchor has no same-shape PyTorch comparator.

### Dispatch-bound rank-1 reduction

[`test_metal_perf_rank1_reduce.py`](../../test_metal_perf_rank1_reduce.py)
passed all three tests:

| Block | Latency | Informational throughput |
|---:|---:|---:|
| 512 | 6.30 µs / launch | 0.325 GB/s |
| 1,024 | 6.26 µs / launch | 0.654 GB/s |

These 2-4 KiB reductions are dispatch-bound. The test's prose still describes
“hundreds of microseconds” and `0.003-0.01 GB/s`, so its explanatory range is
stale even though the assertion and current measurement are valid.

### End-to-end signal

The full fixture audit's warmed LLaMA block reported `54.97 ms / iter` at
`seq_len=2048`. There is no same-shape PyTorch baseline in that driver, so this
is a regression anchor, not evidence of relative speed.

The final post-fix full `leet-all` run reported `53.63 ms / iter` for the same
anchor. This is a fresh regression signal after the compiler change, still not
a relative-performance claim.

### Cross-process shader compilation cache

[`metal_shader_compile_cache.py`](../../../microbenchmark/metal_shader_compile_cache.py)
measures `torch.mps.compile_shader` in a fresh Python process for every sample.
One exact source is primed and repeated, while control samples change both the
kernel name and source constant so they receive distinct source keys. On the
audited Apple M4/PyTorch 2.10 environment, five samples produced:

| Measurement | Result |
|---|---:|
| First compile of repeated source | 41.287 ms |
| Repeated-source fresh-process median | 3.189 ms |
| Unique-source fresh-process median | 29.318 ms |
| Repeated / unique median | 0.109x |

The `<= 0.5` regression gate passed. This is direct evidence that the host
stack avoids most source compilation work across processes even though the
returned `ShaderLibrary` object itself is process-local and non-serializable.
It is host/version-specific behavior rather than a public PyTorch cache API
guarantee, so the correct action is to keep the benchmark and recheck on
PyTorch/macOS upgrades, not to add a parallel Triton binary-cache format.

## Generalization debt in the compiler

The lowering formerly contained structurally guarded, whole-kernel replacements
that also required exact Python/JIT function names:

- `int8_quant_matmul_kernel`
- `matmulKV_kernel`
- `linear_attn_kernel`
- `_attn_bwd_pre`
- `_attn_bwd_dq`
- `_attn_bwd_dkdv`

The int8, linear-attention, and softmax-attention-backward paths now rely on
their structural matchers rather than exact symbol names. Renamed-positive tests
cover all six former name gates, while adjacent negative tests keep nearby
non-equivalent shapes from being claimed by the replacement. The DKDV positive
also uses two row chunks, exercising both `accumulate=0` and `accumulate=1`.
The remaining debt is not symbol spelling; it is the broader fact that these
are still whole-kernel replacements rather than layout-general lowering.

## Completed work and ranked next work

1. **Completed P0 — multi-band address-cone agreement.** The rank-1 store-only
   rewrite, runtime regression, structural lit check, full MSL suite, and raw
   source stability probe are complete.
2. **Completed P1 subfix — reduce generic cone replay duplication.** A new
   lit case in
   [`rank1_reduce_addf_block4096_chain.mlir`](../../../../../test/Dialect/Metal/convert-tritongpu-to-metal/rank1_reduce_addf_block4096_chain.mlir)
   proves that `d * d` reuses one replayed `metal.get_element` per logical
   index. Fresh verification: the new lit was RED before the patch and GREEN
   after it; six related rank-1 reduction lit files passed; the full
   `test_metal_backend_reduce_rank1.py` MPS suite passed with 157 passes and
   one skip.
3. **Completed P1 — remove aggregate reduction replay.** The compiler hoists a
   conservatively selected scalar/rank-1-reduce recurrence before the output
   tile loop. The structural lit regression includes a pre-existing masked-store
   threadgroup allocation, and the raw fused MPS benchmark passes the two-round
   `<= 1.25` ratio gate with unchanged numerical accuracy.
4. **Completed P1 — make source fidelity machine-checkable.** The fixture
   manifest pins `leetgpu-triton-solve` commit
   `4dc4e6931f5f534981b327522464b3bebcefead1`, classifies every leet-mapped
   workload as verbatim kernel, host-only adaptation, source repair, or
   Metal-specific kernel rewrite, and explicitly excludes only the zero-byte
   `easy-matrx_copy.py`. The eight non-empty missing workloads were added.
5. **Completed P2 — reduce symbol-name coupling.** The remaining exact symbol
   gates were removed from the int8, linear-attention, and
   softmax-attention-backward whole-kernel replacements. Renamed-positive and
   adjacent-negative tests now guard the int8 path, and the softmax-backward
   renamed-positive coverage passes for all three stages.
6. **Completed P2 — refresh fixture patches and shape coverage.** The transpose
   fixture is now classified as host-only adaptation, not a Metal-specific
   rewrite. OLS now launches the Gram matrix over both feature axes, separates
   `X^T y` to avoid duplicate writes, and passes explicit `n_features=48`
   standalone coverage above the old 32-feature boundary.
7. **Completed P3 investigation — retain and monitor the existing cache.**
   Triton's normal on-disk cache stores the terminal MSL text artifact, and the
   audited PyTorch/Metal stack already reuses compilation work by exact source
   across fresh processes. `torch.mps.compile_shader` still exposes no public
   serialized `.metallib` or PSO handle, so a second backend-managed compiled
   artifact cache would add ownership and invalidation risk without addressing
   a measured gap. The new benchmark is the regression monitor for future
   PyTorch/macOS changes.
8. **Completed P2.3 slice — exact `tt.unsplat` support.** Triton's verifier
   admits only a source tensor with one logical element, and the Metal type
   converter maps that tensor to the same scalar element type. The new lowering
   therefore replaces the op with its source value without a lane exchange or
   memory round trip. The upstream MPS `test_unsplat` changed from the named
   preflight rejection to a numerical pass, and the existing expand-dims lit
   file now includes the structural one-element case. Fresh post-change
   verification passed all 157 Metal conversion lit tests, all 425 MSL tests,
   1,488 fixture pytest cases with three skips, and all 24 standalone scripts.
9. **Completed P2.3 slice — multi-result `scf.if` emission.** The MSL emitter
   now allocates one temporary per SSA result, records each result separately,
   and assigns every branch yield operand to the corresponding temporary. The
   translator regression uses unlike `f32/ui32` results so aliasing both fields
   to the last temporary cannot pass accidentally. A frontend MPS kernel covers
   `f32/i32` results on both true and false branches. Fresh validation passed
   the complete nine-test scalar-store file, all 157 Metal lit tests, all 425
   MSL tests, 1,488 fixture pytest cases with three skips, and all 24 standalone
   scripts; the LLaMA regression anchor was `51.31 ms / iter`.
10. **Completed P2.3 slice — exact `tl.map_elementwise` scalar callbacks.**
    The Metal lowering now inlines `pack=1`, single-block scalar regions ending
    in `tt.map_elementwise.return`, including callbacks that produce multiple
    results. Structural lit coverage proves the map op is removed, physical MPS
    tests cover one and two `int32` results, and adjacent negative fixtures keep
    `pack>1` plus multi-block callback CFG fail-closed with named diagnostics.
    The upstream comparison callback is intentionally still rejected because
    its Python control flow lowers to a three-block CFG. The upstream unsigned
    multi-output case currently stops earlier in the MPS driver because its
    `TensorWrapper` lacks `is_contiguous`; the native `int32` multi-output test
    isolates and validates the compiler path. Fresh verification passed the
    Pixi `clang-format --dry-run --Werror` gate, all 157 Metal lit tests, the
    complete 18-test integer-arithmetic file, all 425 MSL tests, 1,488 fixture
    pytest cases with three skips, and all 24 standalone scripts; the LLaMA
    regression anchor was `47.80 ms / iter`.
11. **Closed P2.3 — descriptor reducing stores were a pipeline-audit gap, not
    a missing frontend lowering.** Metal's TTIR pipeline already runs Triton's
    `triton-rewrite-tensor-descriptor-to-pointer`, which converts
    `tt.descriptor_reduce` into a masked `tt.atomic_rmw` before Metal
    conversion. A new structural regression locks the pointer/mask/atomic
    rewrite, while the Metal TensorDescriptor suite covers i32
    add/min/max/and/or/xor, a partial descriptor view, and four-program
    contention. Upstream physical-MPS probes also passed device-created i32 add
    and host-backed f32 add. The direct-TTGIR diagnostic now identifies a
    pipeline-order violation instead of claiming that reducing descriptor
    stores are unimplemented. Fresh validation passed the 19-test Metal
    TensorDescriptor suite, the descriptor rewrite lit test, all 157 Metal lit
    tests, all 425 MSL tests, 1,488 fixture pytest cases with three skips, and
    all 24 standalone scripts; the LLaMA regression anchor was
    `48.53 ms / iter`.
12. **Closed P2.4 — exact f32 compare-and-swap and measured 64-bit bounds.** The Metal
    lowering bitcasts f32 compare/value operands to `ui32`, performs
    compare-exchange through an `atomic_uint` view of f32 device storage, and
    bitcasts the fetched old value back to f32. This preserves object-
    representation semantics: `+0` and `-0` are distinct, while an identical
    NaN payload can match. Runtime coverage includes blocked tensor old values,
    sub-threadgroup lane masking, four-program scalar contention, and broadcast
    of the winning old value to every lane. Structural tests cover scalar and
    tensor lowering, the MSL weak-to-strong retry path, and adjacent `f16`/`i64`
    capability rejection. A persistent real-MPS compiler probe additionally
    proves that `atomic_ulong` void min compiles while fetch-add,
    fetch-and/or/xor, exchange, and compare-exchange are absent; the backend
    keeps those 64-bit forms fail-closed instead of synthesizing racy updates.
    Fresh validation
    passed the Pixi build, all 157 Metal lit tests, the complete 97-test atomic
    MPS file, all 425 MSL tests, 1,491 fixture pytest cases with three skips, and
    all 24 standalone scripts; the LLaMA regression anchor was
    `54.79 ms / iter`. The generic upstream
    `test_tensor_atomic_cas` nodes were not counted because their shared
    `check_type_supported` helper unconditionally queries CUDA capability before
    reaching the backend on an MPS-only PyTorch build.
13. **Advanced P3 — close stale dot gaps and add masked direct-load
    `dot_scaled`.** Audit showed that canonical masked multi-tile dot was already
    implemented and structurally covered, so its stale source comment and plan
    status were corrected instead of duplicating the lowering. Direct rank-3
    batched dot now keeps the BxMxN result geometry when K makes an operand
    strictly larger, while preserving equal-size ties required by the existing
    loop-carried SIMD path; both K-dominant orientations pass on MPS. The
    rank-2 scaled-dot matcher now accepts matched zero-filled rectangular A/B
    loads plus a matching masked output store, caps runtime K at the static tile
    payload, and rejects nonzero fill values by name. Fresh verification passed
    the Pixi build, all 157 Metal lit tests, all 177 dot tests, 70 general-matmul
    and attention tests, all 425 MSL tests, 1,494 fixture pytest cases with three
    skips, and all 24 standalone scripts; the LLaMA regression anchor was
    `47.97 ms / iter`. At that point, loop-carried and rank-3 `tt.dot_scaled`
    remained explicit future slices.
14. **Advanced P3 — add the first exact rank-3 `tt.dot_scaled` slice.** Direct,
    unmasked E4M3 inputs now lower when A/B/output use canonical row-major
    batch addressing and the rank-3 E8M0 scale tensors reference one physical
    scale matrix shared by every batch. The pointer proof accepts the frontend's
    nested positive constexpr batch-stride chain without weakening the matrix
    row/column proof, and reduction K is taken from A's final dimension rather
    than the rank-2-only dimension index. The positive `(B,M,N,K)=(2,16,16,64)`
    regression uses distinct payloads and returns 64/256 in its two batch
    slices, so both A and B batch offsets are observable; a real per-batch scale
    pointer remains a named rejection. Fresh verification passed the Pixi
    build, all 157 Metal lit tests, all 179 dot tests, an expanded 167-test
    general-matmul/attention set, all 425 MSL tests, 1,496 fixture pytest cases
    with three skips, and all 24 standalone scripts; the LLaMA regression
    anchor was `49.47 ms / iter`. Loop-carried, masked rank-3, and per-batch
    scale-addressed `tt.dot_scaled` remain future slices.
15. **Advanced P3 — add the first exact loop-carried `tt.dot_scaled` slice.** A
    static rank-2 E4M3 loop now lowers only when it starts at zero, covers the
    full K range in divisible 32-element steps for at least two iterations,
    carries one accumulator, and contains exactly the two payload and two scale
    loads. The pointer proof requires A/B reduction addresses to use the same
    loop IV, A's physical row stride to equal full K, and each canonical scale
    matrix to advance by exactly `iv / 32` with a `fullK / 32` row stride. The
    rewrite collapses the source loop to one full-reduction `metal.scalar_dot`.
    The `K=64` MPS regression deliberately changes both the second payload tile
    and its A-scale group and returns 160, proving that neither iteration nor
    scale group is dropped; a `K=96`, start-32 neighbor is rejected by the
    `zero-based full-K loop` diagnostic. Fresh verification passed the Pixi
    build, all 157 Metal lit tests, all 181 dot tests, the 167-test
    general-matmul/attention set, all 425 MSL tests, 1,498 fixture pytest cases
    with three skips, and all 24 standalone scripts; the LLaMA smoke anchor was
    `52.11 ms / iter`. Broader loop-carried payload/scale/control-flow forms,
    masked rank-3, and per-batch scale-addressed `tt.dot_scaled` remain future
    slices.
16. **Advanced P3 — accept canonical contiguous per-batch E8M0 scales for
    rank-3 `tt.dot_scaled`.** The rank-3 pointer proof now distinguishes a
    batch-shared scale matrix from a canonical per-batch tensor whose static
    batch stride is exactly `rows * scale_groups`. The rewrite records each
    proven nonzero stride as a `metal.scalar_dot` attribute, and the lowering
    adds `logical_batch * batch_stride` to that side's scale base without
    widening the five-parameter scaled-dot operand protocol. The positive
    `(B,M,N,K)=(2,16,16,64)` MPS regression uses all-one E4M3 payloads, scales
    the two batches by 1/1 and 2/4 on A/B, and obtains 64/512; a neighboring A
    scale tensor with one padding row per batch is rejected by the named
    `contiguous per-batch scale matrices` diagnostic. Fresh verification passed
    the Pixi build, all 157 Metal lit tests, all 182 dot tests, the 167-test
    general-matmul/attention set through the integrated gate, all 425 MSL tests,
    1,499 fixture pytest cases with three skips, and all 24 standalone scripts;
    the LLaMA smoke anchor was `50.88 ms / iter`. Masked rank-3,
    non-contiguous/dynamic scale batch strides, and broader loop-carried
    payload/scale/control-flow forms remain future slices.
17. **Advanced P3 — add matched M/N/K masks to rank-3 `tt.dot_scaled`.** The
    exact new envelope keeps the complete static batch tile while accepting
    zero-filled rectangular A/B masks and a matching output mask. When no
    batch predicate is present, `metal.scalar_dot` now carries the static batch
    extent alongside the proven runtime M/N and capped K extents; this retains
    the existing batch guard without inventing a dynamic batch origin. The
    `(B,BM,BN,BK)=(2,16,16,32)` MPS regression uses runtime
    `(M,N,K)=(13,11,27)`, obtains 27 in every valid element of both batches, and
    preserves sentinels in both output tails. A neighboring true batch-tail
    mask remains rejected by the named `full static batch tile` diagnostic.
    Fresh verification passed the Pixi build, all 157 Metal lit tests, all 184
    dot tests, the 167-test general-matmul/attention set through the integrated
    gate, all 425 MSL tests, 1,501 fixture pytest cases with three skips, and
    all 24 standalone scripts; the LLaMA smoke anchor was `54.38 ms / iter`.
    Rank-3 batch tails, non-contiguous/dynamic scale batch strides, and broader
    loop-carried payload/scale/control-flow forms remain future slices.
18. **Advanced P3 — extend the exact static loop slice from E4M3 to E5M2.**
    E5M2 uses the same 32-element E8M0 scale grouping and the existing
    byte-backed decoder, so the loop matcher now accepts either matching
    E4M3/E4M3 or E5M2/E5M2 payloads without changing its structural proof. The
    parameterized `K=64` MPS test changes the second payload tile and A-scale
    group and obtains 160 for both formats. The neighboring E5M2 loop starting
    at K=32 still hits the `zero-based full-K loop` diagnostic, proving that
    adding the format does not widen loop control flow. Fresh verification
    passed the Pixi build, all 157 Metal lit tests, all 186 dot tests, the
    167-test general-matmul/attention set and all 425 MSL tests through the
    integrated gate, 1,503 fixture pytest cases with three skips, and all 24
    standalone scripts; the LLaMA smoke anchor was `53.59 ms / iter`. E2M1,
    fp16/bf16, dynamic/masked loops, rank-3 batch tails, and non-canonical scale
    batch strides remain future slices.
19. **Advanced P3 — extend the static loop slice to matching fp16/bf16
    payloads.** The structural proof is unchanged, but its scale-group step is
    now parameterized: matching fp16/fp16 and bf16/bf16 loops accept either
    legal E8M0 scale factor, 16 or 32, only when each loop iteration consumes
    exactly one group. E4M3/E5M2 remain deliberately restricted to their
    previously verified 32-element group. The six positive `K=64` MPS cases
    cover both floating formats at both scale factors plus the two FP8 formats;
    each changes the second payload half and A-scale range and obtains 160.
    All six neighboring nonzero-start cases retain the `zero-based full-K loop`
    diagnostic. Fresh verification passed the Pixi build, all 157 Metal lit
    tests, all 194 dot tests, the 167-test general-matmul/attention set and all
    425 MSL tests through the integrated gate, 1,511 fixture pytest cases with
    three skips, and all 24 standalone scripts; the LLaMA smoke anchor was
    `50.02 ms / iter`. E2M1/mixed payloads, dynamic/masked loops, rank-3 batch
    tails, and non-canonical scale batch strides remain future slices.
20. **Advanced P3 — extend the static loop slice to matching E2M1 payloads
    under every packing combination.** Either operand may pack its two E2M1
    nibbles along logical K or along its outer M/N dimension. The loop pointer
    proof now relates a K-packed physical tile to the logical loop through the
    exact `iv / 2` origin and verifies the corresponding half-width physical
    row stride; outer-packed operands retain the direct logical-K IV while
    proving their half-height/half-width storage geometry. Four `K=64` MPS
    cases cover K/K, K/outer, outer/K, and outer/outer packing. Each changes the
    second A payload and scale group and obtains 160. Four nonzero-start cases
    retain the `zero-based full-K loop` diagnostic, while two adjacent kernels
    using an incorrect `iv / 4` packed origin are rejected by the canonical
    full-K pointer diagnostic. Fresh verification passed the Pixi build, all
    157 Metal lit tests, all 204 dot tests, the 167-test
    general-matmul/attention set and all 425 MSL tests through the integrated
    gate, 1,521 fixture pytest cases with three skips, and all 24 standalone
    scripts; the LLaMA smoke anchor was `47.76 ms / iter`. Mixed-format and
    dynamic/masked loops, rank-3 batch tails, and non-canonical scale batch
    strides remain future slices.
21. **Advanced P3 — close the exact rank-3 batch-tail slice and bind its guard
    to the store coordinate.** The A/B zero-filled load masks and output store
    mask may now add a batch predicate when all three use equal bounds and
    canonical zero-based batch ranges that are present in their corresponding
    addresses. The proof treats layout-cloned `make_range` values as canonical
    coordinates and equal specialized dense splats as the same bound, without
    accepting arbitrary equivalent SSA cones. A combined
    `(B,BM,BN,BK)=(2,16,16,32)`, `(actual_B,M,N,K)=(1,13,11,27)` MPS case
    initially exposed a second issue: deriving the batch guard from the
    function-wide flat index can disagree with the final store after folding a
    rank-3 result layout conversion. `metal.scalar_dot` now carries the
    store-owned batch coordinate in `output_indices`, so the valid batch and
    M/N rectangle obtain 27 while every batch/M/N tail retains its sentinel. A
    neighboring B mask with a biased batch bound is rejected by the named
    `matching batch-tail bounds` diagnostic. Fresh verification passed the
    Pixi build, all 157 Metal lit tests, all 205 dot tests, 1,522 fixture pytest
    cases with three skips, and all 24 standalone scripts; the LLaMA smoke
    anchor was `52.19 ms / iter`. Mixed-format and dynamic/masked loops plus
    non-canonical scale batch strides remain future slices.
22. **Advanced P3 — add mixed E4M3/E5M2 payloads to the exact scaled-dot
    envelope.** The matcher now accepts both E4M3/E5M2 operand orientations at
    the FP8-mandated 32-element E8M0 scale factor. `metal.scalar_dot` records
    each operand's payload kind independently and applies the corresponding
    decoder before bf16 scaling, while retaining the legacy shared markers for
    same-format IR. The `K=64` loop-carried MPS cases change both payloads and
    both scale groups and obtain 320 in either orientation. Direct structural
    IR checks pin the E5M2 decoder on A and E4M3 decoder on B; adjacent E2M1/FP8
    source mixtures remain rejected. Malformed internal scalar-dot IR is also
    rejected when payload markers conflict or identify only one raw-i8
    operand, rather than silently selecting a decoder. Fresh verification
    passed the Pixi build, all 157 Metal lit tests, the 14 focused mixed,
    same-format, and nonzero-start MPS tests, the 275-test dot/general-matmul/
    attention gate, all 425 MSL tests, and the complete Metal backend suite
    with 2,094 passes and three skips. Dynamic/masked loops and non-canonical
    scale batch strides remain future slices; aligned nonzero starts are
    addressed by item 23 below. The `leet-all`
    pytest phase separately reached 1,523 passes and three skips, but its sole
    source-fidelity check failed on a pre-existing byte mismatch: the pinned
    checkout's `easy-swish-gated_linear_unit.py` contains one trailing space
    that the clean, nominally verbatim fixture does not. Because both files
    match their respective HEADs, this slice leaves that unrelated fidelity
    policy/data issue explicit instead of adding trailing whitespace; the gate
    stopped before its standalone phase.

23. **Scale-group-aligned nonzero loop starts are represented explicitly.**
    The exact static rank-2 loop-carried `tt.dot_scaled` envelope now accepts a
    nonnegative lower bound aligned to the 16- or 32-element scale factor while
    still requiring the upper bound to prove the physical full-K row stride.
    `metal.scalar_dot` carries separate `reduction_start` and
    `reduction_extent` operands; its scalar and SIMD-group lowerings form
    `logical_k = reduction_start + local_k`, so native FP16/BF16, byte-backed
    E4M3/E5M2, E8M0 scale groups, and E2M1 byte/nibble coordinates all retain
    the source suffix origin. Physical MPS tests cover every same-type format,
    both mixed E4M3/E5M2 orientations, and all four E2M1 packing combinations.
    Misaligned and negative starts fail with a named diagnostic, closing the
    unsigned-wrap/OOB edge. Fresh validation passed the Pixi rebuild, 28 focused
    loop tests, all 425 MSL tests, the full build-tree lit suite with 454 passes
    and two unsupported tests (including Metal 157/157), and the complete Metal
    backend suite with 2,098 passes and three skips. Masked and dynamic-upper
    loop-carried forms remain separate future envelopes.

24. **Zero-based static-upper loops now preserve matched M/N/K tails.** The
    exact rank-2 loop-carried `tt.dot_scaled` envelope accepts matched
    zero-filled rectangular A/B masks plus the corresponding rectangular output
    store mask. Its runtime K extent is clamped to `[0, static_upper]` before it
    becomes the folded `metal.scalar_dot` reduction extent, so an oversized or
    negative runtime bound cannot expose memory outside the static payload.
    Physical MPS tests cover E4M3/SF32 and FP16/SF16 at static
    `(BM,BN,BK)=(16,16,64)` with runtime `(M,N)=(13,11)` and K values `-1`, 48,
    and 80: they respectively produce 0, 96, and 160 inside the valid rectangle,
    proving both clamp boundaries, and retain the output sentinel on M/N tails.
    Nonzero load fills, mismatched A/B K bounds, and masked loops with a nonzero
    lower bound fail with named diagnostics. Fresh validation passed all 28
    focused loop tests, all 425 MSL tests, the full build-tree lit suite with 454
    passes and two unsupported tests (including Metal 157/157), and the complete
    Metal backend suite with 2,107 passes and three skips. Dynamic-upper loops
    and non-canonical scale batch strides remain future envelopes.

25. **Canonical clamped dynamic-upper loops now preserve the same masked
    tails.** The exact rank-2 loop-carried `tt.dot_scaled` envelope accepts a
    runtime upper bound only when it has the structural form
    `min(max(runtime_k, 0), static_full_k)`. The physical A row stride proves
    `static_full_k`, and both scale row strides must equal
    `static_full_k / scale_factor`; matched zero-filled rectangular A/B loads
    plus the corresponding output mask prove the runtime M/N/K extent. This
    keeps the folded reduction inside both payload and scale capacity without
    cross-operation inference. Physical MPS tests cover E4M3/SF32 and
    FP16/SF16 at static `(BM,BN,BK)=(16,16,64)` with runtime
    `(M,N)=(13,11)` and K values `-1`, 0, 27, 48, 64, and 80. They respectively
    produce 0, 0, 27, 96, 160, and 160 inside the valid rectangle while keeping
    the M/N sentinel tails unchanged. Unmasked dynamic loops, a biased
    non-canonical upper bound, and a padded scale row fail with named
    diagnostics. Fresh validation passed the Pixi rebuild, all 43 focused
    static/dynamic loop tests, the complete 153-test dot-universal file, all
    425 MSL tests, the full build-tree lit suite with 454 passes and two
    unsupported tests (including Metal 157/157), and the complete Metal backend
    suite with 2,122 passes and three skips. Non-contiguous or dynamic scale
    batch strides remain the next P3 envelope.

26. **Exact static-padded and runtime scale batch strides close P3.** Rank-3
    `tt.dot_scaled` now carries both A/B scale batch strides as `ui32`
    `metal.scalar_dot` operands instead of static contiguous-stride
    attributes. A zero operand retains batch-shared scales; an exact static or
    runtime operand preserves padding when the source address is structurally
    `zero_based_batch_index * stride`. The matrix row/group proof remains
    unchanged, and an added batch origin or slice stays outside the envelope
    because scalar-dot has no operand that could preserve it. Physical MPS
    tests use E4M3 payloads with `K=64`, one padding row filled with `0xff`
    between batches, and distinct batch-1 A/B scales of 2 and 4. Both constexpr
    and runtime strides produce 64 for batch 0 and 512 for batch 1 without
    reading the padding sentinel; a runtime `(batch + 1) * stride` neighbor
    fails with the `canonical zero-based per-batch scale stride` diagnostic.
    Fresh validation passed the Pixi rebuild, all eight rank-3 scaled-dot
    tests, the complete 155-test dot-universal file, all 425 MSL tests, the
    full build-tree lit suite with 454 passes and two unsupported tests
    (including Metal 157/157), and the complete Metal backend suite with 2,124
    passes and three skips. The planned P3 correctness/capability envelopes are
    now closed.

27. **The report-only P4 performance contract now has durable evidence.**
    `test_metal_perf_report.py` exposes a CLI that validates and times fixed
    vector-add `(4194304,)`, fused-softmax `(4096,1024)`, fp16 matmul
    `(512,512,64)`, and split GroupNorm `(8,512,64,64)` workloads. Every run
    uses five warmups, seven samples, and 20 launches per sample with explicit
    MPS synchronization around each sample; compile-plus-first-launch time is
    kept outside the steady-state samples. The report records commit/dirty
    state, Apple GPU family, macOS, PyTorch, Triton target, dtype, shape, and
    `num_warps`, then writes both JSON and CSV. On Apple M4/macOS 26.4/PyTorch
    2.10.0 at commit `4817e44e72c7985a8b88db5f97afe289040c49ab`
    (dirty worktree), the independent round medians were 0.742675/0.722198 ms
    for add, 0.757033/0.739096 ms for softmax, 0.267775/0.323517 ms for matmul,
    and 7.869725/8.163690 ms for GroupNorm. A proposed `512x512x512` matmul was
    correctly rejected because its 64 K tiles exceed the canonical Metal dot
    loop envelope; the accepted K=64 shape exercises the documented eight-tile
    maximum. Fresh validation passed the evaluator and all 10 tests in the
    performance-report file. These numbers are report-only; P4.1 and later
    optimization slices must independently apply the 1.20x target and 5%
    non-target canary limits.

28. **P4.1 vectorizes only the fully proven contiguous masked-add envelope.**
    A pre-conversion whole-function matcher retains tensor layout and pointer
    facts long enough to recognize exactly one rank-1 f32 store of two masked
    loads added together. It requires a shared canonical
    `pid * BLOCK + arange` address, the same signed `offsets < n` mask, a
    contiguous blocked tile with a per-thread extent divisible by four, and
    `tt.divisibility >= 16` on all three entry pointers. Extra operations,
    load `other` values, strided layouts, noncanonical masks, and missing
    alignment proofs stay on the existing scalar path. The matched cone becomes
    `metal.contiguous_vector_add`; its MSL full-range arm issues two `float4`
    loads from each input before two `float4` stores, while a guarded scalar
    loop preserves the partial final program and negative/zero `n` behavior.
    Deterministic tests pin `elements_per_thread = 8`, `vector_width = 4`, the
    vector transactions, and the scalar tail, with explicit no-`float4`
    neighbors for strided and unproven-alignment cases.

    The fresh pre-change P4.1 reference measured vector add at
    0.978915/1.013733 ms. Two independent post-change runs, each using a
    different empty Triton cache and forced recompilation, measured
    0.599396/0.598627 ms (1.633x/1.693x) and 0.595846/0.628479 ms
    (1.643x/1.613x). Fused softmax, matmul, and GroupNorm stayed below the 5%
    regression limit in every round. The isolated-cache requirement is
    material: an exploratory run through the old disk cache reused scalar MSL
    because native translator changes are not represented in that kernel cache
    key, so it is not accepted as evidence. Fresh validation passed the Pixi
    native build, all ten vector-add lit tests, two compiler/report contract
    tests, all 11 MPS zero-copy tests, all 13 vector-add matrix/tile GPU tests,
    and a read-only correctness review. Durable reports are under
    `.omx/goals/performance/metal-p4-vector-load-store/`.

29. **P4.2 materializes provably tile-invariant rank-1 aggregates before the
    synthetic output loop.** The selector accepts only straight-line,
    top-level, static rank-1 f32 axis-0 add/max reductions whose device-rooted
    replay cone reaches an output. Dependencies are cloned in source order, so
    softmax's sum consumes the already-materialized max, and replay memoization
    reuses shared cone values for each logical index. Observable stores,
    unrelated device reads, barriers, user loops, and region-bearing control
    flow close the hoistable prefix; dedicated lit neighbors pin every one of
    those boundaries. The final aggregate scratch read is named before the
    elementwise output loop, so generated MSL contains no aggregate scratch or
    threadgroup barrier in that loop.

    A fresh two-round evaluator run measured fused softmax at
    0.433792/0.436871 ms against the pre-change 0.784415/0.720817 ms reference,
    for 1.808x/1.650x speedups. Vector add measured 0.610971/0.612685 ms and
    GroupNorm 7.436592/7.403694 ms, both below their reference medians. The
    first candidate runs also exposed a phase-order-sensitive matmul canary:
    its MSL was byte-identical to the reference, but a structurally
    single-SIMD-group 8x8 kernel still launched four warps and repeated the
    same matrix pipeline four times. Post-conversion analysis now emits a
    32-thread launch override only for straight-line kernels with no thread or
    SIMD-group index, at least two unpartitioned staged loads, one or more MMA
    operations, and exactly one unpartitioned matrix store. The evaluator's
    matmul then measured 0.106965/0.082910 ms; genuine multi-warp kernels retain
    their source geometry.

    Fresh validation passed the Pixi native build, all 125 Metal conversion
    lit tests, 12 single-/multi-SIMD-group matmul GPU cases, five multi-warp
    loop-reduction determinism cases, numerical checks embedded in the P4
    evaluator, and the two-round performance gate. Durable reports are under
    `.omx/goals/performance/metal-p42-aggregate-hoist/`.

30. **P4.3 selects a bounded physical SIMD-group schedule for canonical
    matrix tiles.** A single-function, single-dot canonical kernel now chooses
    the largest exact tile partition within a preferred 256-thread matrix
    group, while also enforcing the backend's 1024-thread and 32 KiB
    threadgroup limits and the existing 32-accumulator-matrix ceiling. The
    source geometry remains the fallback when no lower legal schedule satisfies
    those constraints, and source requests above the physical threadgroup limit
    retain their named rejection instead of being hidden by downselection.
    Post-conversion proof rejects unrelated thread-indexed work before updating
    `ttg.num-warps` and `metal.threads_per_group`; temporary scheduling markers
    do not survive in the final Metal IR.

    The deterministic 64x64 fixture requests 16 SIMD-groups and lowers to an
    8-group/256-thread launch. Each group owns eight 8x8 accumulator tiles, the
    static body contains 32 MMA operations across four K tiles, and generated
    MSL allocates exactly eight 64-element staging slices. On the fixed
    `(2048,2048,64)` fp16 matmul with a `64x64x8` program tile, the three-round
    median was 0.982496 ms against the separately captured 1.350519 ms
    single-group reference, a 1.3746x speedup. Median canary changes were
    -1.11% for vector add, -0.12% for fused softmax, and -0.11% for GroupNorm.
    Numeric validation, launch metadata, JSON/CSV reporting, the full 125-test
    Metal conversion directory, the evaluator contract test, and Ruff all
    passed. Durable reports are under
    `.omx/goals/performance/metal-p43-resource-aware-matrix-schedule/`.

31. **P5.2 now publishes an honest macOS 15 Metal wheel contract.** An initial
    macOS 14 target produced a superficially correct wheel tag, but the linker
    warned that objects in the pinned LLVM/MLIR archives were built for macOS
    15. That candidate was rejected: forcing only the final link target would
    advertise compatibility the dependency graph does not provide. Metal-only
    builds now default both setuptools and CMake to 15.0, reject lower targets,
    preserve explicit higher targets, and fail packaging if the final arm64
    platform tag does not match.

    The replacement build on macOS 26 produced the 61 MB
    `cp312-abi3-macosx_15_0_arm64` wheel with SHA256
    `e4d17623beb4a5a2ad5aef01bf4937a279156002a7b82958bb32bdced0c1f17c`.
    Ten artifact/policy tests verified every bundled Mach-O file has `minos=15.0`,
    only the Metal backend and entry point are present, and NVIDIA/profiler
    payloads are absent. Pip also accepted the artifact for an explicit
    macOS 15 arm64 / CPython 3.12 / abi3 target. A clean Python 3.12 environment
    with PyTorch 2.10 installed the wheel, compiled through the installed Metal
    backend, and ran the vector-add numerical smoke successfully on the physical
    Apple M4/MPS host. CI now separates macOS 26 build/audit from a hosted
    macOS 15 clean install/compile smoke; the hosted runner cannot supply the
    remaining physical macOS 15 MPS numerical evidence.

The latest post-P3-slice acceptance run collected 1,525 tests and completed with
1,522 passes and three skips. This total includes the two source-fidelity checks
in the `leet-all` entry point. All 24 standalone scripts passed, the
exhaustive inventory reported 90 owned fixtures and 88 runnable workloads, and
the LLaMA smoke anchor was `52.19 ms / iter`. The CPU-only source-fidelity
check separately passed both of its manifest/checkout tests.

## What does not need to be redone

- Do not discard the existing 90-fixture ownership and regression suite.
- Do not rerun all historical lowering investigations solely because MPS was
  falsely unavailable in a Seatbelt process; the unsandboxed Pixi run restores
  valid device evidence.
- Do not count raw source defects as Metal regressions.

Repeat this incremental audit when either pinned commit changes, or when generic
tile-loop/address lowering or a name-gated whole-kernel matcher changes.

## Reproduction commands

The persistent suite and performance evidence can be reproduced from the
repository root with the following commands. On agent-hosted macOS, run them
outside Seatbelt while retaining the Pixi environment.

```bash
pixi run --frozen leet-all
pixi run --frozen lit -v build/cmake.macosx-11.0-arm64-cpython-3.12/test \
  --filter=convert_layout_rank1_divergent_spt
pixi run --frozen pytest python/test/unit/test_metal_backend_multiload.py \
  python/test/unit/test_metal_backend_masked_store_sweep.py \
  python/test/unit/test_metal_backend_parallel_reverse_scan_gae.py \
  python/test/unit/test_metal_backend_mps_zero_copy.py -s --tb=short
pixi run --frozen pytest python/test/unit/test_metal_perf_rank1_reduce.py -s --tb=short
pixi run --frozen python python/test/microbenchmark/metal_group_norm.py \
  --warmup 5 --iterations 20 --repeats 7 --rounds 2 --assert-max-ratio 0.8
pixi run --frozen python python/test/microbenchmark/metal_group_norm.py \
  --warmup 5 --iterations 20 --repeats 7 --rounds 2 --raw-fused \
  --num-warps 8 --assert-max-ratio 1.25
pixi run --frozen pytest python/test/unit/test_metal_leet_source_fidelity.py \
  -s --tb=short
pixi run --frozen pytest python/test/unit/language/test_core.py::test_unsplat \
  --device=mps -s --tb=short
pixi run --frozen pytest \
  python/test/unit/test_metal_backend_scalar_store.py::test_scalar_multi_result_if_preserves_each_result \
  -s --tb=short
pixi run --frozen pytest \
  python/test/unit/test_metal_backend_int_arith.py::test_map_elementwise_pack1_runtime \
  -s --tb=short
pixi run --frozen pytest python/test/unit/test_metal_backend_tensor_descriptor.py \
  -s --tb=short
pixi run --frozen pytest \
  python/test/unit/test_metal_backend_atomic_add.py::test_atomic_cas_f32_uses_bitwise_compare_and_returns_old_value \
  python/test/unit/test_metal_backend_atomic_add.py::test_atomic_cas_f32_scalar_contention_and_old_value_broadcast \
  python/test/unit/test_metal_backend_atomic_add.py::test_atomic_ulong_capability_surface_matches_backend_gate \
  -s --tb=short
pixi run --frozen python python/test/microbenchmark/metal_shader_compile_cache.py \
  --repeats 5 --assert-max-warm-ratio 0.5
pixi run --frozen python python/test/unit/test_metal_perf_report.py \
  --p4-report-dir .omx/goals/performance/metal-p4-baseline-contract/baseline \
  --rounds 2
pixi run --frozen python python/test/unit/test_metal_perf_report.py \
  --p4-report-dir .omx/goals/performance/metal-p43-resource-aware-matrix-schedule/candidate \
  --rounds 3 --p4-matmul-shape 2048 2048 64 \
  --p4-matmul-block 64 64 8 --p4-matmul-num-warps 16 \
  --p4-matmul-expected-threads-per-group 256 \
  --p4-reference-json .omx/goals/performance/metal-p43-resource-aware-matrix-schedule/reference/metal-p4-baseline.json \
  --p4-primary-workload matmul --p4-min-speedup 1.20 \
  --p4-max-canary-regression 0.05 --p4-allow-primary-num-warps-change
```

## Remaining unknowns

- Performance was measured on one Apple M4 only.
- Cross-process compile reuse is measured behavior of PyTorch 2.10/macOS 26.4,
  not a public serialization or cache-compatibility guarantee.
- The earlier same-commit recheck carried the fixture-suite and LLaMA results
  forward. Once the compiler changed, the full suite was rerun: the compiler
  build, 157-test Metal lit directory, 425-test MSL suite, 54 focused tests,
  reverse stability probes, and GroupNorm measurements are fresh post-fix
  evidence. After the P1/P2 fixture and matcher work, the latest integrated
  suite is 1,503-pass/3-skip with all 24 standalone scripts passing.
- The current raw source corpus has not been repaired and replayed wholesale;
  the suite result applies to fixtures, while raw probes deliberately targeted
  the missing and changed high-information workloads.
