# Metal Leet fixtures

This directory contains the LeetGPU-style Triton workloads used by the Metal
backend's Python tests. The workload files retain their original names and
entry points so tests can exercise the exact kernels through
`importlib.util.spec_from_file_location`.

These files are test inputs, not standalone pytest modules. Assertions and
parameterization belong in `python/test/unit/test_metal_backend_*.py`; compiler
IR regressions belong in `test/Dialect/Metal/`.

Run `pixi run leet-all` from the repository root to audit all 80 Python
fixtures and cover the 78 runnable workloads through their owned execution
paths. `test_metal_backend_leet_uncovered.py` owns the exhaustive manifest: 23
standalone scripts, 26 interpreter-backed cases, and 31 targeted backend
regressions. Two interpreter cases are explicit skips because their recovered
source is invalid on every backend.

The corpus was recovered from the pre-history-rewrite `metal-develop` snapshot.
Three previously untracked workloads (`easy-matrix-addition.py`,
`easy-matrix_multiplication.py`, and `medium-layer_normalization.py`) were
recovered from the rewrite backup created alongside that snapshot.
