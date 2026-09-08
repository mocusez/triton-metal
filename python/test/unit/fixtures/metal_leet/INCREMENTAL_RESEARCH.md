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

32. **P0 directly locks every op category in the unsupported-op preflight.**
    The existing CPU-only split-input lit already exercised join, split,
    map-elementwise, descriptor-reduce, extern-elementwise, integer-to-pointer,
    and pointer-to-integer rejection. Five adjacent fixtures now cover the
    remaining `validateUnsupportedOpsRejected` TypeSwitch categories: tensor
    `device_print` with a non-Blocked/Slice layout, PTX inline asm,
    `atomic_poll`, a call to a multi-block callee that preprocessing cannot
    inline, and scalar fp8 conversion. Each fixture pins the named diagnostic
    emitted before dialect conversion, so these platform/runtime boundaries
    fail normally instead of reaching failed-conversion teardown. The focused
    file and all 125 Metal conversion lit tests pass. P0 remains open only for
    the separate audit of lowering patterns whose matchers can still decline.

33. **P0 moves non-splat tensor constants ahead of full conversion.** A live
    `arith.constant dense<[...]>` reached `ArithConstantDenseLowering`, whose
    scalar model can represent only a splat. The resulting match failure first
    printed a generic legalization error and then crashed split-input test
    processing in `Region::dropAllReferences` with SIGSEGV. A shared
    `tensorConstantSplatValue` predicate now drives both the lowering and an
    early validator, which names the unsupported per-element constant while
    the source module is intact. The reproducer now exits through the expected
    diagnostic, the existing splat fixture still lowers, the Pixi native build
    succeeds, and all 125 Metal conversion lit tests pass.

34. **P0 preflights the f32 envelope of Triton's precise math ops.** TTIR
    permits `tt.precise_sqrt` and `tt.precise_divf` on adjacent float widths,
    while the Metal expression lowerings currently accept only f32. A live f16
    result therefore produced a generic legalization error followed by the
    same `Region::dropAllReferences` SIGSEGV. Both lowerings and the new early
    validator now use one `hasF32ElementType` predicate. Direct f16 sqrt/div
    fixtures pin their named diagnostics, the existing f32 tensor fixtures
    still lower to Metal expression ops, the Pixi native build succeeds, and
    all 125 Metal conversion lit tests pass.

35. **P0 preflights every reachable `tt.histogram` decline.** A rank-2
    histogram reached the conversion pattern, printed a generic legalization
    error, and then aborted in failed-conversion teardown. The lowering and a
    new early validator now share `unsupportedHistogramReason`, including the
    same dry-run `rank1ConeSupported` predicate used by the scalar evaluator.
    Five fixtures pin rank, i32 element type, blocked result layout, source
    cone, and mask cone diagnostics. Dynamic/empty tensor and non-power-of-two
    thread-geometry guards were removed because TTIR tensor-size and TritonGPU
    layout verification reject those shapes before this pass. The existing
    masked i32 histogram still lowers, the Pixi native build succeeds, and all
    125 Metal conversion lit tests pass.

36. **P0 rejects pointer selection before buffer resolution declines.** A
    scalar `arith.select` between two bound buffer pointers followed by
    `tt.load` reproduced a legalization failure and SIGSEGV in
    `Region::dropAllReferences` (lit exit -11). Buffer resolution cannot trace
    a selected pointer to one bound argument. The unsupported-op validator now
    diagnoses scalar and tensor pointer-valued selects before conversion.
    Three negative fixtures cover scalar loads, scalar stores, and tensor
    pointer loads; an adjacent positive fixture checks that selecting integer
    offsets still feeds the load index. Existing value-select tests also pass.
    `pixi run --frozen make` succeeded, and all 157 tests under the build
    directory's `test/Dialect/Metal` passed. No MPS numerical suite was rerun
    for this source-rejection change. The remaining decline audit stays open.

37. **P0 also preflights pointer-valued `scf.if` results.** Selecting a
    buffer through structured control flow bypassed the `arith.select` guard:
    a following scalar load still failed legalization and crashed in
    `Region::dropAllReferences` (lit exit -11). The validator now checks every
    if result for scalar or tensor pointers and emits one named diagnostic.
    Negative fixtures cover scalar loads and a tensor store whose pointer is
    the second result. A positive multi-result if keeps loads inside the
    branches and yields f32 data plus an i32 output offset. Both focused lit
    files and all 157 Metal lit tests passed after `pixi run --frozen make`;
    `git diff --check` passed. No MPS numerical suite was rerun for this
    source-rejection change. Other lowering declines remain under audit.

38. **P0 preflights pointer-carrying `scf.while`.** A loop returning an
    advanced scalar pointer to a following load reproduced a legalization
    failure and SIGSEGV in `Region::dropAllReferences` (exit 139). The validator
    now checks both initial operand and result types, including tensor pointer
    elements: `scf.condition` can forward different types to the after region.
    Four negative fixtures cover scalar result loads, tensor stores in the
    loop body, pointer-only operands, and pointer-only results. An adjacent
    positive fixture carries i32 offsets and f32 data while accessing bound
    buffers in both loop regions. After `pixi run --frozen make`, both focused
    lit files and all 157 Metal lit tests passed; the additional result-only
    case passed a focused rerun. The original reproducer now exits normally
    with the named diagnostic (exit 1). The existing while-loop MSL compile
    test and all eight physical-MPS while numerical cases passed, including
    zero-trip and masked multi-block loops. `git diff --check` passed. This
    closes the while-pointer slice; other lowering declines remain under audit.

39. **P0 preflights generic pointer-carrying `scf.for`.** The check runs
    after recognized pointer-advance matmuls have been replaced and before
    partial/full conversion rewrites the function. Scalar pointer results and
    tensor pointer loop-body stores have negative coverage; integer-offset
    and data-carrying loops retain positive coverage. This completes the
    previously unrecorded for-loop slice alongside the select/if/while checks.

40. **Elementwise unsigned min/max and f32 remainder are implemented.**
    Tensor u32/u64 min/max reuse the signedness-preserving scalar arithmetic
    emitter, with high-bit, mixed-sign-bit, and masked-tail numerical coverage.
    Scalar and tensor `arith.remf` use `metal::precise::fmod`; tests distinguish
    truncating remainder from floor/IEEE remainder, including signed zero,
    large quotients, infinities, NaNs and zero divisors. Non-f32 remainder
    remains preflighted. The fresh-cache integer/MSL suite passed all 453
    cases on physical MPS before the abs extension below.

41. **f16/bf16 absolute value preserves every magnitude bit.** The original
    minimal `tl.abs(tl.load(fp16_ptr))` frontend program aborted in MSL
    emission with `translateValue: unexpected op math.absf` (exit 134).
    The new scalar/tensor path bitcasts to ui16, clears bit 15, and bitcasts
    back, preserving subnormals, signed-zero semantics, and every NaN payload.
    Conversion now requires these abs operations to lower; f64/fp8 retain
    named preflight rejection. Four IR cases and two neighboring type
    rejections extend existing lit files. Runtime tests use integer storage
    around floating bitcasts to preserve signaling/quiet NaN representations,
    exhaust all 65,536 bit patterns for each format, add a 37-element masked
    tail, and cover scalar/multi-program execution with one and four warps.

Fresh 2026-09-05 verification after all three follow-up slices above used
macOS 26.4 / PyTorch 2.10.0 on the physical Apple M4. Native rebuild passed
(`CCACHE_DIR=/tmp/triton-metal-followup-ccache pixi run --frozen make`), all 157
Metal lit tests passed, and the full `test_metal_backend_*.py` suite completed
with **2,146 passes, three skips, and no failures** in 152.39 seconds. GPU tests
ran outside Seatbelt with a fresh `TRITON_CACHE_DIR`; the original f16 frontend
reproducer now exits successfully. Ruff, Python syntax compilation, and
`git diff --check` passed. Independent review of the abs implementation and
regressions found no actionable issues. The two warnings came from the existing dropout
fixture's deprecated non-boolean `tl.where` condition. This is a compiler
correctness/coverage run, not a new performance benchmark or macOS 15 device
validation.

42. **Scalar masked loads compose pointer, mask and fallback support.**
    The i16 frontend form `tl.load(ptr + pid, pid < N, other=-7)` previously
    failed the uniform-other preflight, and the scalar load pattern itself
    declined all masks. The shared predicate now recognizes integer/float
    scalars as block-uniform. `ScalarLoadLowering` retains the original buffer
    and complete scalar offset chain, reads memory only inside the true
    `scf.if` region, and yields the supplied scalar other in the false region.
    Both branches use the logical result type, with the existing storage-type
    bridge applied to the loaded value. The tensor masked pattern explicitly
    defers scalar pointers; its non-uniform tensor-other rejection is retained.

    Targeted physical MPS verification passed all 115 scalar-load tests. New
    coverage spans i8/i16/i32/i64/f16/bf16/f32, constant and runtime fallback,
    fallback from a scalar load, omitted other, zero/partial/full active
    programs, loop-varying masks and offset chains, and exact floating bits
    including negative zero, subnormals, infinities and NaNs at one/four warps.
    Tests do not assert a masked-off value when no other was specified.
    Twelve CPU-only frontend compile cases pin the input read inside the
    emitted MSL true branch. Three adjacent lit files verify guarded reads,
    negative integer fallback typing, bf16 default handling, and unchanged
    tensor positives/negatives. Independent read-only review found no emitter
    constraint requiring additional storage-type conversions around the if.

    Final native rebuild and all 157 Metal lit tests passed. A fresh-cache full
    `test_metal_backend_*.py` run on physical MPS completed with **2,254 passes,
    three skips, and no failures** in 151.78 seconds. The two warnings remain
    the existing dropout fixture's deprecated non-boolean `tl.where` condition.
    The original i16 frontend reproducer now compiles successfully. Ruff,
    Python syntax compilation, and `git diff --check` passed. Environment:
    Apple M4, macOS 26.4, PyTorch 2.10.0; GPU commands used `pixi run --frozen`
    outside Seatbelt with `TRITON_CACHE_DIR=/tmp/triton-metal-scalar-mask-full`.

43. **Staged multidimensional i32 XOR scans reuse the existing fold.**
    A frontend `tl.associative_scan` using `a ^ b` on a 4x8 i32 tile previously
    failed the staged-scan preflight. The exact single-op combine matcher now
    accepts XOR only when called by the rank >= 2 staged plan. The rank-1
    distributed template retains its add/mul restriction. Existing blocked
    layout, at-most-one-element-per-thread, axis extent 2..64, and control-flow
    checks are unchanged. The existing publication barriers, ui32 staging,
    zero identity, and generic binary fold require no new lowering path.

    All 24 new physical MPS cases passed bit-exact CPU differential checks:
    rank-2 both axes, rank-3 all axes, forward/reverse, one/four warps, extent
    64, three independent programs, random full-range i32 values and explicit
    zero/all-ones/sign-bit/max-positive inputs. Three frontend negative cases
    retain named rejection for rank-1 XOR, an overfull tile, and extent 128.
    Lit pins publication/read ordering, zero identity, ui32 storage, swapped
    combine arguments, and rejects compound XOR, wrong return, and i64 XOR.
    Independent architecture review found no blocking issues.

    Native rebuild, all 157 Metal lit tests, Ruff, Python syntax compilation,
    and whitespace checks passed. The first fresh-cache full GPU run had
    **2,280 passes, three skips, and one failure**: the existing raw Metal D5
    probe matched 29/30 dispatches. Its hand-written IR intentionally omits
    the barrier before a cross-warp transposed shared-memory read
    (`test/Dialect/Metal/l1d2d_probe/cell_D5.mlir`); it does not invoke the
    TritonGPU-to-Metal scan conversion. Isolated recheck matched 30/30. This
    remains a limitation of the historical probe's unconditional PASS
    expectation; a passing retry does not establish that it is race-free.

    The second fresh-cache full GPU run completed with **2,281 passes, three
    skips, and no failures** in 148.77 seconds; the two warnings remain the
    existing dropout fixture's deprecated non-boolean `tl.where` condition.
    Environment: Apple M4, macOS 26.4, PyTorch 2.10.0, frozen Pixi; GPU runs
    outside Seatbelt. Logs: `/tmp/triton-metal-xor-full.log` (first run) and
    `/tmp/triton-metal-xor-full-repeat.log` (second run), with the latter using
    `TRITON_CACHE_DIR=/tmp/triton-metal-xor-full-repeat`.

44. **Numerical probe expectations now require defined memory reads.**
    The D5 failure recorded above came from a hand-written Metal probe that
    intentionally omitted the publication barrier before a transposed
    cross-warp shared-memory read. D4 has the same missing synchronization
    within one SIMD group; SIMD execution alone does not justify its previous
    unconditional PASS expectation. The neighboring D7scratch probe also
    loads scratch before initialization, even though its always-true mask
    selects the other value. Their original IR remains available as historical
    diagnostics, but none of these three shaders is dispatched by the routine
    GPU correctness matrix.

    Three CPU-only tests preserve the diagnostic translation evidence: paired
    D4/D6 and D5/D7 checks pin the store/barrier/read ordering and the trailing
    barrier, while a D7scratch check preserves the original read-before-store
    form. The eight defined-memory GPU probes retain all 30 bit-exact dispatch
    checks, including synchronized in-warp/cross-warp transpose, masked output,
    and multi-threadgroup coverage. No dynamic xfail or success-on-mismatch
    behavior was introduced. MPS availability now gates only GPU tests, so the
    CPU translation checks also run inside Seatbelt.

    Targeted verification passed three CPU checks (eight GPU skips inside
    Seatbelt), followed by **136 passes** for physical MPS probes plus the
    existing valid Triton transpose suite. This is a Python test/documentation
    change; no native rebuild was needed.

    Final full-suite verification passed **2,281 tests with three skips and
    no failures** in 149.03 seconds, using the physical Apple M4, macOS 26.4,
    PyTorch 2.10.0 and frozen Pixi outside Seatbelt with
    `TRITON_CACHE_DIR=/tmp/triton-metal-probe-safety-full`. The total is
    unchanged because three CPU checks replace three invalid GPU cases.
    All 157 Metal lit tests, Ruff, Python syntax compilation and
    `git diff --check` passed. In-memory removal of the D6/D7 publication
    barriers made both new structural assertions fail as intended.
    Independent read-only review agreed with the split; old plan language
    and diagnostic-source comments were corrected without changing their IR
    operations. The two warnings remain the existing dropout fixture's
    deprecated non-boolean `tl.where`. Full log:
    `/tmp/triton-metal-probe-safety-full.log`.

45. **Reject unsupported FMA widths before MSL emission.**
    The valid frontend `tl.fma` on f16 loads aborted with SIGABRT in
    `ModuleTranslation::translateValue` (`unexpected op math.fma`). The
    scalar/tensor cases at one/four warps all reproduced the failure in
    CPU-only subprocesses. Non-f32 math was legal during conversion, while
    `MathFmaLowering` only supported f32, so the unsupported operation survived
    into the emitter.

    Preflight now shares `hasF32ElementType` with the FMA lowering and reports
    `math.fma requires f32 operands and result`. The compiler does not silently
    promote half FMA or change its rounding contract. Explicit operand casts
    to f32 use the existing fused operation. Lit rejects f16/bf16/f64 scalar
    FMA and multi-band f16 tensor FMA, while retaining scalar and tensor f32
    positive cases. Four frontend subprocess regressions changed from signal
    termination to ordinary exit 1 with the named diagnostic. Eight physical
    MPS cases cover scalar/tensor, one/four warps, f16 explicit promotion,
    f32 fused cancellation (`-2**-46` rather than a separately rounded zero),
    multiple programs, masked tails, and untouched output sentinels.

    Native rebuild passed with a writable temporary ccache. All 157 Metal
    lit tests, four CPU rejection cases, and eight targeted MPS cases passed.
    The initial lit invocation failed tool discovery with exit 127; after
    checking the rebuilt executable, the complete rerun passed. The first
    MPS run passed numerical checks but exposed a bytes/string mismatch in
    the new MSL assertion; decoding the artifact fixed the test, and all
    eight cases passed with another fresh cache. Ruff, Python syntax checks,
    and `git diff --check` passed.

    Fresh-cache full backend acceptance outside Seatbelt completed with
    **2,293 passed, three skipped, zero failed** in 189.01 seconds:
    `TRITON_CACHE_DIR=/tmp/triton-metal-p0-fma-full-cache pixi run --frozen pytest
    python/test/unit/test_metal_backend_*.py -s --tb=short`.
    The two warnings remain the existing dropout fixture's non-boolean
    `tl.where` deprecation. Independent read-only review returned GO with no
    correctness, block-model, or test defects. Full-suite log:
    `/tmp/triton-metal-p0-fma-full.log`; lit log:
    `/tmp/triton-metal-p0-fma-lit.log`. This is correctness/safety evidence,
    not a new performance measurement or physical macOS 15 validation.

    README and the current remaining-work plan were reconciled first: P3/P4
    planned scopes are complete, P5.2 awaits only physical macOS 15 numerical
    validation, and the three undefined numerical probes remain CPU checks.
    P0's remaining safety audit and P2.1/P2.2 breadth remain open.

46. **Complete the f32-only unary math preflight family.**
    `sqrt`, `erf`, `exp`, `exp2`, `log`, `log2`, `rsqrt`, `sin`, and `cos`
    shared the same unsupported-width escape as FMA: non-f32 operations stayed
    legal in conversion and aborted in the MSL emitter. Eighteen valid
    frontend programs (nine operations, scalar/tensor) load f32 buffers,
    explicitly compute in f64, and store f32 results; every pre-fix subprocess
    exited on SIGABRT. All now exit normally with a named compile error.
    Preflight and each lowering use the same f32 element-type predicate.
    Direct lit covers all nine operations on f16/bf16 scalars and f64 tensors.

47. **Support block-uniform scalar masked stores.**
    The new f32 unary positive matrix exposed another existing failure:
    even a constant-true scalar store mask was sent through the tensor-only
    masked-store path, producing a failed conversion and sometimes SIGABRT.
    `MaskedStoreLowering` now recognizes scalar pointers (including converted
    kernel memrefs), creates an `scf.if` from its adapted mask, clones its own
    store without that mask into the true region, and lets `StoreLowering`
    perform its existing address and storage conversion. The rewrite preserves
    store attributes and introduces no producer inspection or new address
    analysis. False masks perform no destination access.

    Forty-two new physical MPS cases cover i8/i16/i32/i64/f16/bf16/f32,
    one/four warps, four programs, loop-varying scalar offset chains, and
    all-disabled/partial/all-enabled masks; inactive elements and row padding
    remain sentinels. Direct f32 and offset i16 lit cases pin the only device
    store inside the conditional. The nine-op f32 scalar/tensor matrix adds
    36 cases with masked tails and multiple programs. Together with the
    existing scalar-store module, the targeted MPS run passed **87/87**.
    The scalar-store test module's obsolete f32-only scope note was corrected.

    Native rebuild, all **157 Metal lit** cases, **18 CPU subprocess** cases,
    Ruff, syntax checks, and diff checks passed. The first expanded lit run
    caught an incorrect type-spelling assertion in the new test; matching the
    actual `metal.store` syntax fixed it without changing the implementation.
    Logs: `/tmp/triton-metal-p0-unary-red-pytest.log`,
    `/tmp/triton-metal-p0-unary-green-pytest.log`,
    `/tmp/triton-metal-p0-unary-store-mps.log`, and
    `/tmp/triton-metal-p0-unary-store-lit-final.log`.

    Final fresh-cache full backend validation passed **2,389 tests**, with
    **3 skips, 0 failures**, and the two existing integer-condition `tl.where`
    deprecation warnings in **169.03s**. Command:
    `TRITON_CACHE_DIR=/tmp/triton-metal-p0-unary-store-full-cache pixi run --frozen pytest python/test/unit/test_metal_backend_*.py -s --tb=short`.
    Full log: `/tmp/triton-metal-p0-unary-store-full.log`.
    Independent read-only reviews returned GO for both the unary preflight
    and scalar masked-store changes, including conversion remapping,
    block-uniform semantics, and regression coverage. The reviewer flagged
    the stale scalar-store scope note; it was corrected before finalization.
    These close two bounded P0 slices; the remaining P0 audit and P2.1/P2.2
    extensions remain open.

48. **Diagnose f64 numeric casts and preserve float-to-integer signedness.**
    The remaining P0 type audit reached `arith.extf`, `arith.truncf`,
    `arith.sitofp`, `arith.uitofp`, `arith.fptosi`, and `arith.fptoui`.
    Twelve valid frontend scalar/tensor cases showed six widening SIGABRTs
    and six source-f64 conversion failures without a useful type diagnostic.
    Widening cases retain their f64 intermediate through an integer consumer;
    source-f64 cases compile without allocating an MPS f64 buffer. An earlier
    bitcast-based probe failed verification before reaching the emitter, so
    it was replaced with these direct numeric cases. Preflight now checks the
    input and result element types of each of the six operations and names
    f64 as unsupported. It does not reject the supported float/integer widths.

    Adjacent physical MPS tests exposed a separate numerical defect:
    converting f32 `-1.75` to i64 returned `0` instead of `-1`. The emitter
    used the unsigned storage spelling of signless i64 for a signed numeric
    conversion. It now reuses `signednessCast` to select the result cast from
    `fptosi`/`fptoui` semantics. This also handles signless i16 and direct
    scalar unsigned conversion consistently, without adding conversion-graph
    inspection or changing buffer representations.

    Sixty-four new MPS cases cover f16/bf16/f32 casts, i32/u32/i64/u64 inputs,
    signed and unsigned 8/16/32/64-bit results, high-bit unsigned values,
    negative fractional inputs, signed zeros, scalar/tensor forms, one/four
    warps, multiple programs, and masked-tail sentinels. All float-to-integer
    inputs are in range. PyTorch cannot fill a uint64 MPS tensor directly;
    the test initializes its sentinel on CPU and transfers the buffer.
    The full float-cast module passed **85/85** cases. The new twelve CPU
    rejection cases plus the existing unary rejection family passed **30/30**.
    Two direct rejection modules pin all six operations, while the MSL lit
    fixture checks all eight signed/unsigned integer result spellings.
    Native rebuild, all **157 Metal lit** cases, Ruff, syntax checks, and
    diff checks passed.

    Evidence: `/tmp/triton-metal-p0-cast-red-direct.log`,
    `/tmp/triton-metal-p0-cast-green.log`,
    `/tmp/triton-metal-p0-cast-mps.log` (pre-fix numerical failure),
    `/tmp/triton-metal-p0-cast-signed-mps.log`, and
    `/tmp/triton-metal-p0-cast-signed-lit.log`. Validation uses the dirty
    checkout based on `bb4b5978a1`, macOS 26.4, and the pinned Pixi environment.

    Final fresh-cache full backend validation passed **2,465 tests**, with
    **3 skips, 0 failures**, and the two existing integer-condition `tl.where`
    deprecation warnings in **185.46s**. Command:
    `TRITON_CACHE_DIR=/tmp/triton-metal-p0-cast-full-cache pixi run --frozen pytest python/test/unit/test_metal_backend_*.py -s --tb=short`.
    Full log: `/tmp/triton-metal-p0-cast-full.log`.
    Independent read-only review of the final source and tests returned GO
    with no blocking findings. The review confirmed both preflight directions,
    the signedness fix, and coverage of supported adjacent types.

    **New open correctness finding:** floating `tl.minimum`/`tl.maximum`
    with `propagate_nan=tl.PropagateNan.ALL` return `[2, 1]` for inputs
    `[NaN, 1]` and `[2, NaN]`, rather than `[NaN, NaN]`. Independent MPS
    reproduction is saved at `/tmp/triton_metal_minmax_repro.py`.
    `ArithMinMaxFLowering`, scalar `ModuleTranslation` cases, and the
    reduction/cone evaluators currently merge NaN-propagating and
    number-preferring operations. A future fix must preserve both policies
    across all three paths and test NaNs in either/both operands, finite
    values, signed zeros, and reduction consumers. This is recorded in the
    current plan and README as unfinished; the cast fixes do not address it.

49. **Propagate NaN through `tl.minimum`/`tl.maximum` under `PropagateNan.ALL`.**
    The finding recorded at the end of item 48 is now fixed. Root cause: MSL
    has no NaN-propagating extremum at all. A direct
    `torch.mps.compile_shader` probe on Apple M4 / macOS 26.4 returned `2.0f`
    for `max(NAN, 2.0f)`, for `max(2.0f, NAN)` and for `fmax` of either, so
    every spelling available to the emitter is IEEE-754 `maxNum`. That is what
    the default `propagate_nan=NONE` (`arith.maxnumf`/`arith.minnumf`) wants,
    and all four arith extrema shared the single `maxOp`/`minOp` operator, so
    the ALL policy silently got the number-preferring answer.

    The fix adds `maxNanOp`/`minNanOp` to `BinaryExpOperator` and a pair of
    templated preamble helpers, `__triton_maximumf`/`__triton_minimumf`, which
    return a NaN operand as-is (preserving payload and sign, like
    `metal.clampf` already does) and otherwise order `-0.0` below `+0.0`.
    Operator selection goes through ONE pair of functions,
    `floatExtremumOperator` and `isNanPropagatingExtremum`, used by the
    elementwise lowering, both cone evaluators, and all four reduce planners
    (rank-1, rank-2 axis-0, rank-N staged, rank-2), so the three paths named in
    the plan cannot drift apart. The scalar emitter calls the same helper by
    the same name. `-inf`/`+inf` remain the reduce identities under either
    policy: they lose to every real element, a NaN included, so masked-out
    lanes stay inert.

    Measured RED, then GREEN, on the same three paths: elementwise tensor
    `[NaN, 1]` vs `[2, NaN]` went `[2, 1]` -> `[NaN, NaN]`, the scalar form
    went `2.0` -> `NaN`, and a `tl.reduce` with an ALL-policy maximum combine
    over `[1, NaN, 3, 2]` went `3.0` -> `NaN`. Signed zeros were A/B'd through
    the real pipeline rather than argued: the default policy, which still emits
    the builtin, returns `-0` for `max(+0, -0)` and `+0` for `max(-0, +0)`,
    while the fixed ALL policy returns `+0` for both.

    The default policy is untouched by construction. A kernel that uses no ALL
    extremum emits no helper, and its call site still reads
    `max(float(a), float(b))`, so existing MSL is byte-for-byte unchanged.

    Coverage: 66 new physical MPS cases over both policies, NaN on either side
    and on both, the +/-inf bounds, ordinary and equal finite pairs, bit-exact
    signed zeros, scalar and tensor forms, one and four warps, three programs
    with a masked tail, and a reduce consumer with the NaN first, in the middle,
    last, and absent, at BLOCK 32 and 256. f16 and bf16 elementwise cases pin
    that the templated helper instantiates at the narrow widths too. Upstream
    `test_core.py::test_propagate_nan` on `--device=mps` passes 10 of 12; the
    two failures are `clamp` on `float16`, which is the pre-existing named
    `tt.clampf requires f32 operands and result` boundary and is not touched by
    this change. Two conversion lit cases pin both operator spellings and one
    translate case pins the helper definition plus the call sites.

    One existing fixture had encoded the defect as intended behaviour:
    `reduce_maxf_rank2_axis1.mlir` fed `arith.maximumf` and asserted `maxOp`.
    Its `maxnumf` positive still asserts `maxOp`; the `maximumf` case now
    asserts `maxNanOp` and that no `maxOp` appears. That is also the A/B for
    the lit additions, which reference an operator that did not previously
    exist.

    Full verification: 157 of 157 Metal lit cases pass, and the fresh-cache
    complete Metal backend suite is **2,531 passed / 3 skipped / 0 failed**
    in 197.81s, up from the 2,465 of item 48 by exactly the 66 added cases, so
    the change carries no regression. Ruff, `py_compile` and `git diff --check`
    pass. Command:
    `TRITON_CACHE_DIR=<fresh> pixi run --frozen pytest python/test/unit/test_metal_backend_*.py -s --tb=short`.

    The `linalg.reduce` -> `metal.reduce` matchers in `LinalgToMetal.cpp` and
    `ConvertLinalgToMetal.cpp` also name `arith.maximumf`, but that is the
    MLX-native reduce family and is not reachable from the Triton TTGIR
    pipeline, so it is deliberately unchanged.

50. **Widen `tt.clampf` from f32 to f32/f16/bf16.** Entry 49 left upstream
    `test_core.py::test_propagate_nan` at 10 of 12 on `--device=mps`, with both
    `clamp-float16` cases failing on `tt.clampf requires f32 operands and
    result`. That was not a platform limit. MSL's `fmin`/`fmax` carry `half`
    natively, and a direct shader probe confirmed an f16 clamp is exact.

    The ODS operands and result now share one `Metal_ClampFloat` constraint,
    the verifier requires a single float type across all four (ODS can restrict
    each operand but cannot make them agree, and a mixed-width clamp would let
    the emitted `fmin`/`fmax` promote silently), and the lowering pattern and
    `validateClampFSupport` share `metalClampFElementTypeSupported`. f64 and fp8
    keep a named pre-conversion rejection; the negative fixture moved from f16,
    which is now a positive, to f64.

    Two findings worth carrying. First, `bfloat` has no `fmin`/`fmax` overload
    at all: both operands promote to `float` and MSL refuses to convert the
    result back, a hard "assigning to 'bfloat' from incompatible type 'float'"
    error rather than a silent narrowing. The emitter converts explicitly, which
    is exact because the result is always one of the three operands. Second,
    that branch is currently unreachable from Python: dumping TTIR shows Triton
    widening a bf16 `tl.clamp` to f32 with `arith.extf` and truncating the
    result, so `tt.clampf` never appears in bf16. The f16 TTIR, by contrast, is
    a genuine `tt.clampf ... tensor<16xf16>`, and its MSL is the unwrapped
    native half call. The bf16 support is kept for hand-written Metal IR and for
    the day the frontend stops promoting, and it is pinned by the
    metal-translate lit rather than by a runtime test that would not exercise
    it.

    Beware the reverse trap when probing this: `tl.clamp(x_f16, -1.0, 1.0)` with
    Python float literals promotes x to f32, so it always worked and proves
    nothing. The broken form is same-dtype bounds, which is what upstream's test
    uses.

    Verification: upstream `test_propagate_nan` goes 10/12 -> **12/12**, and
    `test_clamp` plus `test_clamp_symmetric` pass 5/5. Six new physical MPS
    cases cover f32/f16/bf16 against both NaN policies with both bounds hit
    exactly, signed zeros, both infinities and NaN, at one and four warps. f32
    and f16 MSL is byte-identical to before for the shapes that already worked.
    157 of 157 Metal lit and a fresh-cache **2,537 passed / 3 skipped / 0
    failed** in 194.06s, up from entry 49's 2,531 by exactly the six added
    cases.

51. **Measure the remaining P0 decline audit instead of describing it.** The
    plan had carried "the rest of the pattern-decline list still needs pinning"
    as prose since the audit began. Three sweeps under `.omc/fuzz/` turned that
    into a number, and found no new gap on any axis they cover.

    `gapprobe11.py` runs 164 legal frontend kernels over the arithmetic and
    math element-type envelope: floor, ceil, remf, divf, fdiv, mul, sub,
    sigmoid, compare-plus-where and absf at f32/f16/bf16, and add, mul, floor
    division, remainder, both shifts, the three bitwise ops, abs, min, max and
    compare-plus-where at i32/i8/i16/i64, each in tensor and scalar form.
    Result **152 OK / 12 FRONT / 0 KILLER / 0 UNNAMED**. Every FRONT is Triton
    refusing the kernel itself with `Expected dtype ['fp32', 'fp64']` for
    floor/ceil/sigmoid at the narrow widths, in `ast_to_ttir`, before the
    backend is involved. Worth keeping as a classifier trap: a harness that
    greps for "error" files those as backend gaps.

    `gapprobe11b.py` targets the decline surfaces that have no pre-conversion
    validator at all. Twelve validators run today and none covers `gather`
    (8 declines), `join` (5) or `split` (4), so their reason strings were read
    straight out of the source and turned into 16 probes. Result **6 OK /
    10 NAMED / 0 KILLER**: all four families already fail closed with a named
    diagnostic, and each message already had a lit fixture.

    `gapprobe11c.py` does the same for atomics, where the coverage audit listed
    13 unpinned diagnostics. Result **5 OK / 4 FRONT / 8 NAMED / 0 KILLER**.
    Six of the eight reachable rejections were already pinned. The two that were
    not, rank-3 `tt.atomic_rmw` and rank-3 `tt.atomic_cas`, are now negatives in
    `atomic_unsupported.mlir` and `atomic_cas_unsupported.mlir`. The rank check
    runs before the address-form check, so a splat pointer tile reaches it
    without any rank-3 index arithmetic.

    **The metric that matters.** Counting `notifyMatchFailure` sites is
    meaningless: the pass has 260 and most are "this pattern does not apply,
    let the next one try". The countable quantity is the fail-closed named
    diagnostic. There are 158; 58 are pinned by a lit or pytest fixture and 100
    are not, and the unpinned set is dominated by `dot_scaled` (~20 messages)
    and the tile-invariant / full-tile-exchange staging (~8). That is the
    remaining checklist.

    **Do not pin blindly.** "Unpinned" is not "reachable": 4 of the 17 atomic
    probes were legal-looking Python that Triton refused before the backend saw
    them. Probing first is what keeps the suite from filling with guards for
    shapes no user can express, which is the same "pseudo-reachable guard" the
    plan already rejects for TTIR-verifier-rejected shapes.

    No compiler source changed in this round; the deliverable is two lit
    negatives plus the measurement. 157 of 157 Metal lit pass.

52. **A real P0 gap in `tl.dot_scaled`: an epilogue between the dot and the
    store left invalid IR, not a diagnostic.** Entry 51 measured the remaining
    checklist and named `dot_scaled` (~26 unpinned messages) as the largest
    cluster. Probing it found the first genuine gap of the round.

    An `@triton.jit` scaled-dot matmul that stores into an **f16 output buffer**
    failed with `operand #0 does not dominate this use` on
    `arith.truncf`, followed only by the generic
    `Metal backend: convert-tritongpu-to-metal failed`. Deterministic, 30 of 30
    runs. That is the half-applied-rewrite state from the in-conversion-decline
    doctrine: invalid IR after the pass mutated the module, which is what turns
    into an intermittent process kill, not an unsupported-shape diagnostic. An
    f32 accumulator with an f16 output is an ordinary thing to write.

    Root cause: `sdFindStore` walks transitively, so it finds the store even
    when real work sits between it and the dot. The rewrite materializes the
    result at the store, so the intervening op keeps its original position and
    ends up consuming a value now defined after it. The loop-carried form
    already required the result to reach the store directly, at most through a
    layout conversion; the **loop-free form had no such check**. Hoisting it to
    cover both forms is the fix, with its own message for the loop-free case.

    🔑 The discriminator that makes the tightening safe: `result + bias` still
    compiles, because Triton folds the add into the dot's **accumulator
    operand**, leaving the TTGIR chain direct. Confirmed in the dump: no
    `arith.addf` survives. `result * 2.0` and `tl.maximum(result, 0.0)` do
    survive as their own ops, and both were already producing the dominance
    error, so the new rejection only reclassifies shapes that never worked.
    Plain `tl.dot` into an f16 buffer is unaffected and still compiles.

    Pinned by two compile-only cases in `test_metal_backend_msl.py`, the f16
    output buffer and a multiply epilogue. ⚠️ They read the DIAGNOSTIC from
    stderr via `capfd` and also assert `does not dominate this use` is absent.
    The neighbouring rejection tests match only on the generic
    `convert-tritongpu-to-metal failed`, which cannot tell a named rejection
    from invalid IR and would have passed throughout this bug.

    The same mechanism was then checked at the other four `sdFindStore` call
    sites, since a transitive store walk feeding a rewrite that materializes at
    the store is the shape of the bug. Both remaining matchers with a real
    workload came back clean: the OLS gram dot and the complex real/imag pair
    from the 2D FFT each compile with a `* 0.5` epilogue and with an f16 output
    buffer. The loop-free scaled dot was the only broken one.

    157 of 157 Metal lit and **2,539 passed / 3 skipped / 0 failed**, up from
    entry 51's 2,537 by exactly the two added cases. The measured checklist is
    now 159 fail-closed named diagnostics, 60 pinned, 99 not.

53. **The loop-carried scaled-dot matcher is fail-closed; pin its boundary.**
    Entry 52 fixed the loop-free form and left the loop-carried conditions as
    the largest unpinned cluster. Nine near-misses on the working kernel from
    `test_metal_backend_dot_universal.py` (M=N=16, K=64, BLOCK_K=SCALE_FACTOR=32,
    e4m3 in u8, two static iterations) give **3 OK / 6 NAMED / 0 KILLER /
    0 UNNAMED**. Unlike the loop-free form, this one already validated every
    condition it relies on.

    Four distinct reachable diagnostics came out, all previously unpinned, and
    all now have a compile-only regression alongside a positive control: a K
    loop that is not divisible by the block or starts below zero, a second
    accumulator carried through the loop, a second output store, and a
    non-contiguous A row stride.

    Two probes compiled and are worth recording so they are not misread as
    coverage. `K == BLOCK_K` compiles because Triton unrolls the single
    iteration away and the kernel becomes the loop-free form, which is
    supported — the "at least two static iterations" clause is unreachable from
    that direction. And an extra fifth load in the body was folded out before
    the matcher saw it, because the probe combined it with `& 0`; an
    ineffective probe, not evidence that a fifth load is accepted.

    Checklist after this round: 159 fail-closed named diagnostics, **62 pinned,
    97 not**. 157 of 157 Metal lit and **2,545 passed / 3 skipped / 0 failed**,
    up from entry 52's 2,539 by exactly the six added cases. No compiler source
    changed.

54. **The full-tile exchange cluster: nothing to fix, and one correction to
    the audit.** The two exchange diagnostics were the next target because both
    are emitted from `FuncOpLowering::matchAndRewrite`, and that pattern is the
    specific generator of the rollback corruption in the decline doctrine: it
    rewrites the function signature, so a failure returned from it leaves the
    entry block mismatched. A named `emitOpError` there would still be a
    process kill.

    It is not reachable that way. Thirteen shapes — standalone rank-2
    transposes from 32x32 to 256x64 at one and eight warps, a rank-3 permute, a
    two-transpose kernel, and a symmetric sum `tl.trans(a) + a` up to 256x256
    that keeps the source live so both layouts must coexist — all compile. A
    standalone transpose never allocates a pool at all, because the L1d3 work
    re-encodes the layout instead of moving the data.

    The audit had flagged the scratch-budget message as unpinned. That is a
    false positive of the guard kind: the same check runs BEFORE conversion in
    `validateTileExchangeScratchBudget`, whose comment says why in as many
    words, that conversion failure is not a safe diagnostic path here and an
    over-budget function must stop while its `tt.func` is still intact. The
    in-conversion copy is a deliberate backstop and pinning it would pin a
    shape no user can produce.

    🔑 One result that refines the doctrine. The rank-3 permute lands on
    `ttg.convert_layout: broader staged-transpose deferred to L1d3`, an
    in-conversion decline, and it is **stable: 30 of 30 runs exit 1, no signal**.
    §1a-bis measured a different shape at 8 of 30 crashes. So an in-conversion
    decline is not uniformly fatal; whether rollback corrupts depends on what
    the pattern already mutated, and a decline that happens before any
    signature rewrite is survivable. Exit-code repetition remains the only way
    to tell the two apart.

    No compiler source changed. This round is a bounded negative: 230 probes
    across the whole audit so far, one real gap, and that one is fixed.

55. **Checking the README's CLAIMS instead of its rejections found a silent
    wrong answer in `tl.topk`.** The decline audit had saturated at 230 probes
    and one gap, so this round took the other half of the plan's stop
    condition: the README, the diagnostics and the test matrix must agree with
    what the backend does. The largest claim surface is topk, so it was checked
    NUMERICALLY against `torch.topk` rather than merely compiled.

    The axis that matters is **elements per thread**, `N / (32 * num_warps)`,
    and the measured envelope is far narrower than the claim:

        E <= 2                 correct for all four dtypes, both directions,
                               ranks 1-3
        E == 4, k == 2         correct
        E == 4, k >= 4         **WRONG VALUES**, both directions
        E >= 8                 rejects, or aborts the process

    A representative failure: `tl.topk(x, 8, descending=False)` over a
    permutation of 0..511 at four warps returned `[1, 4, 1, 4, 6, 7, 6, 10]`,
    with duplicates, where the answer is `[0..7]`. Eight such cells across
    N=128/256/512 and k=4..64.

    The E >= 8 case is the other half. Its non-crashing runs print
    `'tt.func' op Metal backend cannot carry a value across full-tile exchange
    phases` — the message entry 54 concluded was unreachable. **That conclusion
    was wrong**, and worth keeping as the correction it is: entry 54 probed the
    exchange through transposes and permutes, where the layout is re-encoded
    rather than moved, so no pool is ever allocated. topk reaches it because
    its bitonic phases genuinely republish a slot before the phase anchor. It
    is emitted from `FuncOpLowering::matchAndRewrite`, after the signature
    rewrite, and it crashes **6 of 20 runs** with the rest exiting 1.

    ⚠️ Two methodology failures produced this, both worth carrying. First, the
    initial probe fed topk an ALREADY SORTED input, so every cell passed: a
    broken selection still returns the right prefix of sorted data. Permuted
    input is mandatory for any ordering primitive. Second, the in-tree matrix
    has exactly one cell at four elements per thread and it uses `k=2`, the
    single working corner — the same shape of blind spot as the phantom-encoding
    and NaN bugs, where the suite's cells happen to sit outside the failing
    regime.

    ✅ Not caused by this session. The failing kernel's MSL contains none of the
    new spellings: no `__triton_maximumf`, no `__triton_minimumf`, no NaN
    operator, no clamp. topk decomposes to `arith.maxnumf` plus compare and
    select, and the number-preferring path was left byte-identical.

    This round corrects the README to the measured envelope and names both
    defects as defects rather than as rejections. The two failing shapes are
    NOT added to the suite: a test that returns wrong values would have to be
    an xfail, which would legitimize a silent wrong answer, and a test that
    aborts the process would break the run. Fixing the phase machinery is the
    next slice and is a design change, not a patch.

56. **A second silent wrong answer, in `tl.sort`, and the axis both share.**
    Chasing the topk root cause split the search space with one experiment:
    topk is sort plus a slice, so is `tl.sort` wrong too? It is, and worse.

    `tl.sort` is correct only at **one element per thread**. From two up it
    writes only the first `E/2` entries of each `E`-sized per-thread band and
    leaves the remainder at whatever the output buffer already held:

        E=1                       correct
        E=2   got [0, 0, 2, 0, 4, 0, 6, 0]   want [0..7]
        E=4   got [0, 1, 0, 0, 4, 5, 0, 0]   want [0..7]
        E=8   got [0, 1, 2, 3, 0, 0, 0, 0]   want [0..7]

    The same kernel under `TRITON_INTERPRET=1` matches `torch.sort`, so the
    kernel is right and the backend is wrong. The MSL contains none of this
    branch's new spellings, so it predates this work.

    🔑 **Elements per thread, `N / (32 * num_warps)`, is the axis the whole
    sort/hypercube family is broken along, and no in-tree test sweeps it.**
    sort breaks at E>=2, topk at E==4, the exchange carry at E>=8.
    `test_hypercube_sort_and_flip` sweeps N in {2,4,8,16,32} at one and four
    warps, so every cell is at most E==1; `test_mixed_sort_and_topk` uses N=128
    at four warps, also E==1. Parameterize E directly when covering anything on
    this path, or the cells land at E==1 by accident.

    ⚠️ They are not one bug: topk is correct at E==2 where sort is already
    wrong, so topk's slice compensates there and stops at E==4. Fix sort first;
    its signature is simpler and topk is built on it.

    Three hypotheses for the topk root cause were tried and discarded first, per
    the stop rule: a gather factor fixed at 2 (the final publish and read
    indices agree), a missing or repeated bitonic stage (both kernels alternate
    distances 1 and 2, the larger one merely has an extra level), and pool
    reuse in the final phase (the last three stages are individually correct).
    The sort result supersedes all three as the place to start.

    No test cells were added for the failing regimes, for the same reason as
    entry 55. Both matrices now carry a comment naming the boundary and why
    raising it will produce wrong answers rather than failures.

57. **Fixed: the sort band write. One wrong quantity in the final phase loop.**
    Entry 56 left `tl.sort` writing only the first half of each per-thread
    band. The cause is a single trip count.

    `FuncOpLowering` splits the tile loop into one phase per exchange publish
    plus a final segment holding everything after the last publish. Each phase
    loop takes its trip count from that publish's `src.elemPerThread`; the
    final one took `g_tileExchange[exchangePublishes.back()].dst.elemPerThread`.
    But the final segment holds the exchange read AND everything after it,
    including the output store, and the store is addressed
    `localTid * tile.elemPerThread + iv`. When the last exchange's destination
    has fewer elements per thread than the tile, the loop stops early and the
    tail of every band is never written. Two elements per thread ran one trip,
    four ran two, eight ran four — exactly the observed `E/2`.

    🔑 The read index was never the problem. It already carried an `iv` term;
    with a trip count of one that term was simply always zero. I had misread a
    truncated MSL line as localTid-only and briefly suspected the index. Print
    the whole expression before concluding a mapping is thread-only.

    The fix takes the larger of the two, so a destination that legitimately
    needs more trips keeps them and the tile's band is always covered.

    Result: `tl.sort` goes from correct only at ONE element per thread to
    correct at **four**, both directions, and `tl.topk` from two to two — the
    topk defect at four is a different site and is untouched, which is itself
    useful evidence that the two were never one bug. Above four the exchange
    still cannot carry a value across phases; sort now returns wrong values
    there rather than half-written ones.

    The sort matrix now sweeps N up to 128 at one and four warps, so it spans
    one, two and four elements per thread instead of sitting entirely at one.
    Those twelve new cells were RED before this change.

    157 of 157 Metal lit, fresh-cache Metal backend **2,557 passed / 3 skipped /
    0 failed**, and `leet-all` at 1,664 passed with only the pre-existing
    source-fidelity byte mismatch.

58. **The topk defect, isolated without topk, and an A/B that clears entry 57.**
    Chasing the remaining wrong answer produced a far smaller reproducer and a
    structural diagnosis, plus the A/B that entry 57's fix deserved.

    **Minimal case: reduce one hypercube axis, then sort the result.** No topk
    involved. It fails with topk's adjacent-duplicate signature,
    `[10, 10, 13, 13, 19, 19]` for `[10, 13, 15, 19, 21, 22]`. That is exactly
    what topk does internally, since its second loop alternates a hypercube-axis
    reduce with a bitonic merge. The reduce alone is fine: every axis passes at
    four elements per thread.

    **Cause, from a perfect minimal pair.** Same N=64, same one warp, same
    tile: axis 5 is correct and axis 4 is wrong. Their phase loop trip counts
    are `2 1 1 1 ...` and `2 2 1 2 1 2 1 ...`. The reduce halves the tensor, so
    the kernel holds two tiles, and the stray twos are segments belonging to the
    REDUCED tile that were handed the ORIGINAL tile's count, processing every
    element twice. `FuncOpLowering` takes `findTileInfo` over the whole
    `tt.func`, one tile per function, and the per-phase counts come from the
    exchange plan rather than from the ops actually in each segment. The
    principled fix derives each segment's count from its own ops; that is a
    change to the phase splitter, not another constant swap.

    ✅ **A/B on entry 57.** Both arms were built and measured over seven
    reduce-then-sort cells plus plain sort. The pre-fix arm gives wrong answers
    for plain sort and four-ok/three-wrong on reduce-then-sort; the post-fix arm
    gives correct plain sort and the identical four-ok/three-wrong. The
    band-write fix turned plain sort from wrong to right and moved nothing else,
    so the duplication is entirely pre-existing. ⚠️ Worth repeating before
    blaming any phase-loop change for a duplicate-valued result: the two failure
    modes look alike and live in the same few lines.

    Reproducers are now in `.omc/fuzz/`: `red_then_sort.py`, `hyper_reduce.py`,
    `sort_cell.py`, `topk_cell.py`. No compiler source changed in this entry.

59. **Correction: the remaining defect is not a trip count.** Entry 58 read the
    differing phase loop counts as segments of a reduced tile receiving the
    original tile's count, and proposed deriving each segment's count from its
    own operations. Checking that before building it showed the premise is
    wrong on two grounds.

    First, those counts are each exchange's own `src.elemPerThread`, and the
    exchange plans take source and destination from the convert layout's actual
    operand and result tensor types. They are per-exchange correct and should
    differ once a reduce halves the tensor.

    Second, and decisively: in the failing kernel the final store is
    `v1[localTid]` at ONE element per thread, and the output reads
    `[10, 10, 13, 13, 19, 19]`. Adjacent outputs being equal means two different
    THREADS produced the same value, not one thread processing an element
    twice. A trip count governs only the latter. The defect is a cross-lane
    index collapse — a read or a compare-exchange partner index dropping a
    `localTid` bit — and a per-segment trip count would have added machinery to
    the phase splitter while leaving the wrong answer untouched.

    Recorded rather than built. The minimal pair for whoever continues is
    `.omc/fuzz/red_then_sort.py 64 1 5`, which is correct, against `... 64 1 4`,
    which is wrong: same size, same warp count, same tile, only the reduced axis
    differs. A bit-count sweep over the emitted reads separates data reads,
    which carry all five `localTid` bits, from direction-flag reads, which carry
    one and clamp to slot 1; the defect should show as a data read with too few.

60. **Root cause named, first fix attempt reverted.** A read-only review lane,
    which the plan's own stop rule calls for after repeated failed hypotheses,
    found the site: the exchange double-buffer slot allocator in
    `FuncOpLowering` hands out slots by blind round-robin keyed only on element
    type. Its comment assumes two alternating buffers suffice because a phase
    may still be reading the preceding same-type exchange while publishing the
    next, which holds only while same-type publishes alternate strictly. A
    hypercube reduce along a thread axis interleaves small direction-table
    publishes of the same element type, flipping the parity, so some
    compare-exchange stages receive the slot they read from. With destination
    equal to source and no barrier inside the loop body, a distance-1 stage
    reads the partner after overwriting it and both lanes of the pair take the
    smaller value.

    ✅ Proven by simulation rather than argued: replaying the emitted
    fifteen-stage network with snapshot reads reproduces the reference answer,
    and replaying it with live-buffer reads reproduces the measured wrong output
    exactly, `[10,10,13,13,19,19,21,21]`. The correct axis is safe only
    incidentally, since at one element per thread there are no table publishes.

    ⚠️ **The first fix made it worse and was reverted.** Replacing round-robin
    with "a publish may not take a slot a read in its own segment still needs",
    clearing the live set at each publish because a publish closes its segment
    and a barrier follows, turned wrong answers into `[0, 0, ...]` — reads of a
    buffer nobody had written. The liveness model is too short: exchange reads
    are CLONED INTO LATER PHASES, which the rematerialization code states
    outright, so a published buffer stays live beyond its own following segment
    and clearing at every publish frees slots still in use.

    The tree was restored to the committed state and re-verified: sort correct
    through four elements per thread, reduce-then-sort back to its original
    signature. Any real fix must derive liveness from the actual reads including
    their clones, or add a third slot, and must keep
    `validateTileExchangeScratchBudget` on the same assignment so the preflight
    and the emission cannot disagree.

61. **Fixed: stop reusing exchange buffers when the budget allows.** Entry 60
    named the root cause and recorded a failed fix. The working fix removes the
    question instead of answering it.

    Liveness is not cheaply knowable here, because exchange reads are cloned
    into later phases, so any segment-scoped model frees buffers a later clone
    still reads. Instead, compute what one buffer per publish would cost. If it
    fits the 32 KiB threadgroup budget, give every publish its own and no reuse
    hazard can exist. Only when it does not fit fall back to the historical
    two-slot round-robin, leaving large-tile kernels exactly as they were. These
    tiles are tiny, a 32-element f32 comparator being 128 bytes, so the no-reuse
    total fits for essentially everything that reaches this path. The allocator
    and `validateTileExchangeScratchBudget` call one `assignExchangeBuffers`, so
    the preflight and the emission cannot drift.

    Measured on the 72-cell topk sweep over dtype, direction, `k` and warps:

        before   52 ok / 8 WRONG / 2 CRASH / 10 reject
        after    68 ok / 2 WRONG / 0 CRASH /  2 reject

    🔑 It also removed the crashes for free. "Cannot carry a value across
    full-tile exchange phases" fires when a read's slot is republished before
    the anchor, and with no reuse that condition cannot arise. The
    exchange-carry defect and the duplicate-value defect were one root cause
    seen from two sides, which is why entry 54 could not reach the message
    through transposes: they never allocate a pool at all.

    Still open, and separate: `tl.sort` is unchanged at four elements per thread
    correct and eight wrong, so that is a different defect; and `N = 512`,
    `k = 32` at four warps is large enough to take the budget fallback and keeps
    the old wrong answer. The README now states the envelope as the budget
    rather than as a fixed element count.

    Pinned: three topk cells at four elements per thread with `k >= 4`, all
    failing before this change, and a new `test_reduce_then_sort_hypercube_axis`
    with seven cells — the isolated reproducer, which guards the path without
    depending on topk. 157 of 157 Metal lit, fresh-cache Metal backend
    **2,567 passed / 3 skipped / 0 failed**, and `leet-all` at 1,664 passed with
    only the pre-existing source-fidelity byte mismatch.

62. **`tl.sort` at eight or more elements per thread is not the reuse bug.**
    The natural hypothesis after entry 61 was that sort's remaining failure is
    the same reuse hazard landing in the budget fallback: a 256-element bitonic
    sort has 36 stages at a kilobyte each, which does not fit the 32 KiB budget,
    so it takes two alternating slots.

    Tested rather than assumed. Scaling the fallback from two slots to as many
    as the budget affords was verified to take effect — the emitted kernel went
    from two threadgroup buffers to thirty-two, so only four of the thirty-six
    publishes reuse, at a distance of thirty-two phases — and **the wrong answer
    did not change at all**, same values, same signature. Meanwhile four
    elements per thread, with twenty-eight fully distinct buffers, is correct.

    So this is a different defect and the change was reverted: it does not fix
    the target and it raises threadgroup scratch for those kernels from two
    kilobytes to thirty-two, exactly at Metal's limit, which costs occupancy for
    no benefit.

    ⚠️ Recorded so it is not retried. The signature left to explain is that
    outputs zero through three are correct while outputs four through seven hold
    values starting at `tpb * 4`: measured at N=256 and N=512 at one warp, both
    starting from 128, and N=1024 and N=2048 at four warps, both from 512.
    Something is capped at four elements while the store addresses
    `localTid * E + iv` with E of eight or more.

63. **`tl.sort` above four elements per thread: diagnosis closed.** Two
    linearizations meet in one loop. The final phase reads the exchange buffer
    with the producer's index and stores with the consumer's, and the two agree
    only while the register bits sit contiguously at the bottom.

    With `LT` the lane, `b_k` its bits and `iv` the loop variable, the emitted
    expressions are:

        E=4   read  buf[64b4+32b3+16b2+8b1+4b0+2iv1+iv0] == LT*4 + iv
              store v1[LT*4 + iv]                            agree
        E=8   read  buf[128*iv2 + 64b4+32b3+16b2+8b1+4b0 + 2iv1+iv0]
                    == 128*iv2 + LT*4 + (iv & 3)
              store v1[LT*8 + iv]                            DISAGREE

    The read index is the hypercube's own linearization: `tl.sort` reshapes to
    `[2]*log2(N)` and that rank-N blocked layout places two register bits at the
    bottom and the third at the top, weight 128, which is `tpb * 4` and so the
    constant in the signature. `planLayoutIndex` renders the layout faithfully
    and caps nothing, so there is no stray four to find in it.

    The mapping is a bijection onto `0..N-1`, so the buffer is indexed by
    logical linear position. That closes the arithmetic: output slot `LT*8+iv`
    receives logical element `128*iv2 + LT*4 + (iv&3)`, giving thread 0 the
    elements 0,1,2,3,128,129,130,131 — exactly the measured output, for every
    failing shape.

    So the fix is to read the exchange buffer with the CONSUMER's linearization
    rather than the producer's; the buffer already holds logical element j at
    slot j, so it is a re-index and not a data movement. Rejecting a reshape
    that changes the linearization, instead of passing it through, would be the
    conservative alternative.

    ⚠️ Two isolations that do NOT reproduce it, recorded so they are not
    retried: a bare reshape round trip, whose pair folds away, and reshape plus
    elementwise plus reshape back, since elementwise work is insensitive to the
    mapping. Only an operation that moves data across lanes exposes it.

64. **`tl.sort` fixed for every elements-per-thread cell: two more silent
    permutations.** Entry 63 named the disagreement correctly but proposed the
    wrong side to change. The exchange slot AND the hypercube's per-axis
    compare direction are both defined in the tile's blocked linearization; the
    odd one out is the global load and store, so re-indexing the exchange read
    would have been the wrong repair.

    The first defect is thread-major versus layout order. Everything outside the
    layout-indexed phase paths places the element a thread holds at tile
    iteration `iv` at flat index `localTid * E + iv`, register bits low and
    thread bits above. A blocked layout agrees with that only while the tile
    fits the threadgroup exactly once; a second register band interleaves the
    bands at the CTA-tile stride instead. Both are bijections, so a purely
    elementwise kernel cancels the difference and stays correct, and only a
    layout-indexed scratch exposes it: the phase sorted a permuted array and the
    store permuted it again. A rank-1 blocked `tt.make_range` inside a rank-N
    phase now lowers through `emitLayoutElementIndex` whenever its plan is not
    thread-major. The new `layoutFlatIndexIsThreadMajor` predicate generalizes
    `layoutFlatIndexIsLocalTid` from one element per thread to any number and
    states the condition exactly: register bit `b` must weigh `2^b`, lane bit
    `b` must weigh `E << b`, and warp bit `b` must weigh `E << (laneBits + b)`.
    That alone moved `tl.sort` from four to eight elements per thread.

    The second defect is one size-2 indicator standing for two hypercube axes.
    Triton mints a distinct `#ttg.linear` basis per axis only while the cube
    fits the register bits its indicator layouts name. At `N = 512` and one warp
    the rank-9 cube needs four register bits and the indicators only ever name
    three, so a single value is reshaped into both the stride-128 and the
    stride-256 axis and the widest merge compares along the neighbour.
    `splitMixedHypercubeMakeRanges` now also clones a range whose reshape
    consumers name different axes — it previously skipped linear layouts on the
    belief that their basis is authoritative for mixed consumers, which is not
    true once the basis is short — and `planRangeFromHypercubeConsumer` resolves
    each clone from its consumer.

    ⚠️ The trap inside that second fix: the reshape's own destination type is
    the degenerate `1x2x1...` broadcast source, not the cube. Its layout
    collapses to the same basis for every axis, so reading the physical bit from
    it agrees with the stale indicator and changes nothing. Take the axis from
    the reshape's non-unit dimension and the BIT from the full hypercube tile;
    `linearRangeHypercubeAxis` already returns exactly that pair.

    The method that made this tractable was simulating the emitted MSL. The
    bitonic body is regular enough to parse: per stage, extract the flat index
    expression, the partner stride and the direction predicate, inline the local
    definitions, and replay the network in Python. Validated against a working
    shape and a failing one, then fitted each stage's direction to a single bit
    of the flat index, it printed `S=256 dir=~f7` where the reference needs
    `~f8` and named the defect in one line.

    ⚠️ Do not read a `gapprobe12` KILLER/NAMED delta as a regression without
    re-running: `REPEAT` defaults to one, and topk at `N = 1024` sits on a
    decline path that segfaults intermittently. Measured thirty times per arm,
    nine of thirty crashed with the fix and six of thirty without, so it is the
    same pre-existing flake.

    Measured after: `tl.sort` exact at 1, 2, 4, 8, 16, 32, 64 and 128 elements
    per thread, both directions, at one, two, four and eight warps; `N = 8192`
    at four warps is a clean named budget rejection. Metal lit 157/157 and the
    backend pytest suite 2,568 passed with three skips. The in-tree matrix
    `test_hypercube_sort_and_flip` now reaches `N = 1024`.

65. **The last `tl.topk` silent wrong answer: pool the budget fallback by
    element COUNT as well as element type.** The fallback was never inherently
    unsafe — plain `tl.sort` at 256 elements and above uses it and is exact.
    Round-robin over two slots is safe exactly while the publishes sharing a
    pool alternate strictly, and the pool key was the element type alone. A
    hypercube reduce interleaves two-element direction-table publishes of the
    same element type between the tile publishes; those flip the parity and
    hand a comparator the slot it is reading. Adding the element count to the
    key gives the tables their own tiny pool and restores strict alternation
    among the tiles.

    The allocator is now a three-tier ladder, each tier allocating no more than
    the one above it: one buffer per publish, then two per (type, size), then
    the historical two per type as a last resort, so a kernel too large even
    for the size-keyed pools behaves exactly as it did before. Preflight and
    emission still share the one `assignExchangeBuffers`.

    Measured on the shape that motivated it, `tl.topk` at `N = 512`, `k = 32`,
    four warps: six buffers and 11,272 bytes returning
    `[509, 508, 509, 508, ...]` before, twenty-one buffers and 11,932 bytes
    returning the reference after. Six hundred and sixty bytes more scratch. It
    also pulled three cells off the decline path at `N = 1024`, because the
    size-keyed pools fit the budget where the no-reuse scheme did not. The
    permuted 37-cell sweep went from 32 correct, four named rejections and one
    crash to 35 correct and two crashes, and the two remaining are `N = 2048`
    on the intermittent decline path, measured at nine crashes and twenty-one
    clean declines out of thirty — never a wrong answer.

    🔴 The only defect left in this family is that decline crash.
    `'tt.func' op Metal backend cannot carry a value across full-tile exchange
    phases` is emitted from inside `FuncOpLowering::matchAndRewrite`, well after
    `inlineRegionBefore` has moved the body into the new kernel op, which is the
    canonical rollback-corruption site. The feasibility walk is pure and the
    buffer assignment is already available before conversion, so it can be
    hoisted beside `validateTileExchangeScratchBudget`; the work is replaying
    the phase partition without the lowering's cloned phase ops, behind one
    shared predicate so the two cannot drift.

66. **The exchange-carry rejection moved into preflight; the last measured
    process kill is closed.** A function whose full-tile exchange phases cannot
    rematerialize a value crossing a phase boundary must be rejected, and that
    rejection was emitted from the function lowering roughly two hundred and
    thirty lines after the body had already been moved into the new kernel op.
    Unwinding that move segfaulted on nine of thirty runs, measured on a
    2048-element topk at one warp; the same shape at four warps behaved the
    same. After the hoist both are thirty clean declines out of thirty, with the
    same named diagnostic.

    The hoist was cheap because the walk is pure. It only asks whether a value
    crossing a phase boundary can be replayed: no regions, memory-effect-free or
    a load, and — the condition that actually fires here — not an exchange read
    whose buffer is published into again before the phase anchor. None of that
    needs the lowering's clones or its materialized buffer values, so buffer
    identity enters the predicate as an opaque integer and the preflight passes
    assignment indices where the lowering passes its own. Both sides call the
    one predicate, so they cannot disagree about which functions are rejected.

    ⚠️ The risk worth measuring here is a FALSE rejection, not the crash. The
    preflight sees the original body, while the lowering sees it after the
    loop-carried gather, tile-invariant reduce and tile-invariant rank-one
    reduce materializers have hoisted work into the prologue. A hoisted
    definition is outside the loop body at lowering time and therefore trivially
    carryable, but is still inside it before conversion, so the preflight
    recurses into it — and a reduce carries a region, which the walk treats as
    unreplayable. Nothing in this suite triggers that. If it ever does, narrow
    the PREFLIGHT to the republish condition alone rather than weakening the
    lowering: that condition involves only exchange operations, which are never
    hoisted.

    The regression test asserts the preventing mechanism rather than the crash,
    because the crash is intermittent and when it fires it takes the whole
    pytest process with it, which is why the shape had been deliberately kept
    out of the suite. It rejects three times and then compiles and runs an
    ordinary sort in the same interpreter and checks the answer: a rejection
    raised before conversion leaves the context healthy, one raised during
    teardown does not.

The historical post-P3-slice acceptance run collected 1,525 tests and completed with
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
  evidence. After the P1/P2 fixture and matcher work, that integrated
  suite recorded 1,503 passes/3 skips with all 24 standalone scripts passing.
  Later per-slice acceptance records above supersede those counts for their
  respective suites; backend pytest and `leet-all` are distinct test scopes.
- The current raw source corpus has not been repaired and replayed wholesale;
  the suite result applies to fixtures, while raw probes deliberately targeted
  the missing and changed high-information workloads.
