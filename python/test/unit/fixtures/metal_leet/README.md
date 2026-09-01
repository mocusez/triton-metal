# Metal Leet fixtures

This directory contains the LeetGPU-style Triton workloads used by the Metal
backend's Python tests. The workload files retain their original names and
entry points so tests can exercise the exact kernels through
`importlib.util.spec_from_file_location`.

These files are test inputs, not standalone pytest modules. Assertions and
parameterization belong in `python/test/unit/test_metal_backend_*.py`; compiler
IR regressions belong in `test/Dialect/Metal/`.

Run `pixi run --frozen leet-all` from the repository root to audit all 90 Python
fixtures and cover the 88 runnable workloads through their owned execution
paths. `test_metal_backend_leet_uncovered.py` owns the exhaustive inventory: 24
standalone scripts, 34 interpreter-backed cases, and 32 targeted backend
regressions. Two interpreter cases are explicit skips because their recovered
source is invalid on every backend.

`source_fidelity.json` pins the compared `leetgpu-triton-solve` commit,
explicitly excludes the zero-byte `easy-matrx_copy.py` source file, and
classifies every source-backed fixture as a verbatim kernel, host-only
adaptation, source repair, or Metal-specific kernel rewrite. The CPU-only
`test_metal_leet_source_fidelity.py` test checks that manifest against both the
fixture inventory and the pinned source checkout when it is present.

Run this audit from the project Pixi environment. When an agent or automation
host applies a macOS Seatbelt sandbox, run the GPU-bearing command outside that
sandbox while keeping `pixi run --frozen`: PyTorch can otherwise report
`torch.backends.mps.is_available() == False` on a physical Apple Silicon Mac
whose MPS device works normally in the same Pixi environment. A sandbox-only
failure is an execution-context limitation, not a Leet workload result.

The corpus was recovered from the pre-history-rewrite `metal-develop` snapshot.
Three previously untracked workloads (`easy-matrix-addition.py`,
`easy-matrix_multiplication.py`, and `medium-layer_normalization.py`) were
recovered from the rewrite backup created alongside that snapshot.

This directory is an executable regression corpus and now has a commit-pinned
source-fidelity manifest. It is still not a raw mirror: source repairs,
host-only harness adaptations, fixture-only tutorials, and Metal-specific kernel
rewrites are intentionally labelled. See
[INCREMENTAL_RESEARCH.md](INCREMENTAL_RESEARCH.md) for the fixed commits,
raw-source probes, performance evidence, and remaining follow-up work.
