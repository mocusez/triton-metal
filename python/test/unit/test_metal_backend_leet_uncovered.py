"""The Metal Leet fixture inventory and interpreter-backed gap coverage.

Thirty-four of the corpus files have no standalone driver or narrower backend
test. Thirty-two of them run against an interpreter oracle; two are retained
as explicit skips because the source itself is invalid on every backend.

The remaining fixtures are inventoried as standalone or targeted tests. The
inventory test makes ownership exhaustive: adding a fixture without assigning
it to one of those execution paths fails collection in this module.

The oracle is `TRITON_INTERPRET=1`, not a hand-written torch reference. What is
under test here is the *backend*, so the comparison that matters is "the same
Triton program, lowered by us, versus the same Triton program interpreted" —
a torch reference would instead be re-testing whether the leet author's kernel
is a correct implementation of its problem, which is not this suite's job.

Each of the two runs is its own subprocess. A construct this backend declines
inside `applyFullConversion` takes the process down rather than raising (see
the pre-pass validators in TritonGPUToMetal.cpp), so an in-process run would
lose the whole pytest session rather than fail one test.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

_AS_SCRIPT = __name__ == "__main__"

import torch

if not _AS_SCRIPT:
    import pytest

LEET = Path(__file__).resolve().parent / "fixtures" / "metal_leet"

STANDALONE_FIXTURES = {
    "easy-1D_convolution.py",
    "easy-color_inversion.py",
    "easy-gaussian_error_gated_linear_unit.py",
    "easy-interleave_arrays.py",
    "easy-leaky_Relu.py",
    "easy-matrix_transpose.py",
    "hard-llama_transformer_block.py",
    "medium-2D_subarray_sum.py",
    "medium-2d_convolution.py",
    "medium-2d_fft.py",
    "medium-2d_jacobi_stencil.py",
    "medium-3D_convolution.py",
    "medium-3D_subarray_sum.py",
    "medium-adder_transformer_inference.py",
    "medium-grouped-query-attention.py",
    "medium-mean_squared_error.py",
    "medium-ordinary_least_squares.py",
    "medium-sparse_matrix-Dense_matrix_multiplication.py",
    "medium-sparse_matrix-vector_multiplication.py",
    "medium-stream-compaction.py",
    "medium-subarray_sum.py",
    "test-1.py",
    "tutorials_python/01-vector-add.py",
    "tutorials_python/02-fused-softmax.py",
}

TARGETED_FIXTURES = {
    "hard-bfs_shortest_path.py",
    "hard-fast_fourier_transform.py",
    "hard-mult_head_attention.py",
    "hard-sliding_window_self_attention.py",
    "medium-attention_with_linear_biases.py",
    "medium-attention_with_sinks.py",
    "medium-batch_normalization.py",
    "medium-batched_matrix_multiplication.py",
    "medium-categorical_cross_entropy_loss.py",
    "medium-causal_depthwise_conv1d.py",
    "medium-count_2d_array_element.py",
    "medium-count_3d_array_element.py",
    "medium-count_array_element.py",
    "medium-decaying_causal_attention.py",
    "medium-fp16_batched_matrix_multiplication.py",
    "medium-fused_residual_add_and_rms_norm.py",
    "medium-general_matrix_multiplication.py",
    "medium-group-normalization.py",
    "medium-grpo_surrogate_loss.py",
    "medium-int8_quantized_matmul.py",
    "medium-linear_recurrence.py",
    "medium-logistic_regression.py",
    "medium-lora_linear.py",
    "medium-matrix_power.py",
    "medium-max_subarray_sum.py",
    "medium-monte_carlo_integration.py",
    "medium-parallel_reverse_scan_gae.py",
    "medium-segmented_exclusive_prefix_sum.py",
    "medium-softmax_attention.py",
    "medium-softmax_attention_backward.py",
    "medium-speculative_decoding_verification.py",
    "medium-token_embedding_layer.py",
}

TARGETED_TEST_MODULES = (
    "test_metal_backend_atomic_add.py",
    "test_metal_backend_attention_with_sinks.py",
    "test_metal_backend_batch_normalization.py",
    "test_metal_backend_causal_depthwise_conv1d.py",
    "test_metal_backend_count_array_element.py",
    "test_metal_backend_decaying_causal_attention.py",
    "test_metal_backend_dot_dynamic_k.py",
    "test_metal_backend_dot_universal.py",
    "test_metal_backend_flash_attention.py",
    "test_metal_backend_general_matmul.py",
    "test_metal_backend_grpo_surrogate_loss.py",
    "test_metal_backend_layer_norm.py",
    "test_metal_backend_matrix_power.py",
    "test_metal_backend_mps_zero_copy.py",
    "test_metal_backend_msl.py",
    "test_metal_backend_parallel_reverse_scan_gae.py",
    "test_metal_backend_reduce_rank1.py",
    "test_metal_backend_reduce_rank2_axis0.py",
    "test_metal_backend_reduce_rank2_computed.py",
    "test_metal_backend_segmented_scan.py",
    "test_metal_backend_sliding_window_attention.py",
    "test_metal_backend_speculative_decoding.py",
)

if not _AS_SCRIPT:
    pytest.importorskip(
        "triton._C.libtriton.metal",
        reason="Metal backend pybind module not built into libtriton",
    )
    if not torch.backends.mps.is_available():
        pytest.skip(
            "Metal backend requires an MPS-enabled PyTorch (Apple Silicon)",
            allow_module_level=True,
        )
    assert LEET.is_dir(), f"Metal Leet fixtures not present: {LEET}"


def _load(name):
    path = LEET / f"{name}.py"
    spec = importlib.util.spec_from_file_location(
        "leet_uncovered_" + name.replace("-", "_"), path
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# name -> [builder(device) -> (solve args, indices of the args that are outputs)]
#
# A file may register more than one builder; each becomes its own test. Sizes
# are deliberately small: the interpreter run is the wall-clock cost here, and
# every one of these shapes already crosses the tile/threadgroup boundaries the
# backend actually branches on.
CASES = {}


def _case(name):
    def deco(fn):
        CASES.setdefault(name, []).append(fn)
        return fn

    return deco


def _rnd(*shape):
    return torch.randn(*shape, dtype=torch.float32)


@_case("easy-relu")
def _(dev):
    N = 2053
    return (_rnd(N).to(dev), torch.zeros(N, device=dev), N), [1]


@_case("easy-reverse_array")
def _(dev):
    N = 2053
    return (_rnd(N).to(dev), N), [0]


@_case("easy-rgb_to_grayscale")
def _(dev):
    width, height = 37, 29
    return (_rnd(width * height * 3).to(dev),
            torch.zeros(width * height, device=dev), width, height), [1]


@_case("easy-sigmod_linear_layout")
def _(dev):
    N = 2053
    return (_rnd(N).to(dev), torch.zeros(N, device=dev), N), [1]


@_case("easy-softmax_activation")
def _(dev):
    N = 2053
    return (_rnd(N).to(dev), torch.zeros(N, device=dev), N), [1]


@_case("easy-swish-gated_linear_unit")
def _(dev):
    N = 2050
    return (_rnd(N).to(dev), torch.zeros(N // 2, device=dev), N), [1]


@_case("easy-value_clipping")
def _(dev):
    N = 2053
    return (_rnd(N).to(dev), torch.zeros(N, device=dev), -0.25, 0.5, N), [1]


@_case("easy-vector_addition")
def _(dev):
    N = 2053
    return (_rnd(N).to(dev), _rnd(N).to(dev), torch.zeros(N, device=dev), N), [2]


@_case("easy-matrix-addition")
def _(dev):
    N = 64
    return (_rnd(N, N).to(dev), _rnd(N, N).to(dev), torch.zeros(N, N, device=dev), N), [2]


@_case("easy-matrix_multiplication")
def _(dev):
    M, N, K = 64, 32, 48
    return (_rnd(M, N).to(dev), _rnd(N, K).to(dev),
            torch.zeros(M, K, device=dev), M, N, K), [2]


@_case("hard-causal_self-Attention")
def _(dev):
    M, d = 64, 32
    return (_rnd(M, d).to(dev), _rnd(M, d).to(dev), _rnd(M, d).to(dev),
            torch.zeros(M, d, device=dev), M, d), [3]


@_case("hard-linear_self_attention")
def _(dev):
    M, D = 128, 64
    return (_rnd(M, D).to(dev), _rnd(M, D).to(dev), _rnd(M, D).to(dev),
            torch.zeros(M, D, device=dev), M, D), [3]


@_case("hard-multi_agent_simulation")
def _(dev):
    N = 64
    return (_rnd(N, 4).to(dev), torch.zeros(N, 4, device=dev), N), [1]


@_case("hard-radix_sort")
def _(dev):
    N = 300
    x = torch.randint(0, 1 << 20, (N,), dtype=torch.int32).to(dev)
    return (x, torch.zeros(N, dtype=torch.int32, device=dev), N), [1]


@_case("medium-dot_prodict")
def _(dev):
    n = 2048
    return (_rnd(n).to(dev), _rnd(n).to(dev), torch.zeros(1, device=dev), n), [2]


@_case("medium-gaussian_blur")
def _(dev):
    R, C, kr, kc = 64, 48, 3, 3
    return (_rnd(R, C).to(dev), _rnd(kr, kc).to(dev),
            torch.zeros(R, C, device=dev), R, C, kr, kc), [2]


@_case("medium-histograming")
def _(dev):
    N, bins = 2000, 10
    x = torch.randint(0, bins, (N,), dtype=torch.int32).to(dev)
    return (x, torch.zeros(bins, dtype=torch.int32, device=dev), N, bins), [1]


@_case("medium-int8_kv_cache_attnetion")
def _(dev):
    H, S, D = 4, 64, 32
    k = torch.randint(-127, 127, (H, S, D), dtype=torch.int8).to(dev)
    v = torch.randint(-127, 127, (H, S, D), dtype=torch.int8).to(dev)
    ks = (_rnd(H, S).abs() * 0.01 + 0.01).to(dev)
    vs = (_rnd(H, S).abs() * 0.01 + 0.01).to(dev)
    return (_rnd(H, D).to(dev), k, v, ks, vs,
            torch.zeros(H, D, device=dev), H, S, D), [5]


@_case("medium-layer_normalization")
def _(dev):
    N, C = 4, 257
    return (_rnd(N, C).to(dev), _rnd(C).to(dev), _rnd(C).to(dev),
            torch.zeros(N, C, device=dev), N, C, 1e-5), [3]


@_case("medium-moe_top_k_gating")
def _(dev):
    M, E, k = 32, 8, 3
    return (_rnd(M, E).to(dev), torch.zeros(M, k, device=dev),
            torch.zeros(M, k, dtype=torch.int32, device=dev), M, E, k), [1, 2]


@_case("medium-nearest_neighbor")
def _(dev):
    N = 64
    return (_rnd(N, 3).to(dev),
            torch.zeros(N, dtype=torch.int32, device=dev), N), [1]


@_case("medium-parallel_merge")
def _(dev):
    M, N = 100, 80
    return (_rnd(M).sort().values.to(dev), _rnd(N).sort().values.to(dev),
            torch.zeros(M + N, device=dev), M, N), [2]


@_case("medium-ppo_clipped_surrogate_loss")
def _(dev):
    B, S = 4, 32
    return (_rnd(B * S).to(dev), _rnd(B * S).to(dev), _rnd(B * S).to(dev),
            torch.zeros(1, device=dev), 0.2, B, S), [3]


@_case("medium-prefix_sum")
def _(dev):
    n = 3000
    return (_rnd(n).to(dev), torch.zeros(n, device=dev), n), [1]


@_case("medium-rms_normalization")
def _(dev):
    N = 1024
    return (_rnd(N).to(dev), 1.5, 0.25, torch.zeros(N, device=dev), N, 1e-5), [3]


@_case("medium-rotary-positional-embedding")
def _(dev):
    M, D = 16, 64
    return (_rnd(M, D).to(dev), _rnd(M, D).to(dev), _rnd(M, D).to(dev),
            torch.zeros(M, D, device=dev), M, D), [3]


@_case("medium-rotary-positional-embedding")
def _(dev):
    # D=2 makes BLOCK_SIZE_D = next_pow2(D/2) = 1, so the tile is 1x1 and every
    # offset in the address folds to zero — the store's pointer is then a bare
    # `tt.splat` with no `tt.addptr` under it. That had no lowering, and an
    # unmatched op here kills the process (abort or SIGSEGV, varying run to run)
    # rather than raising.
    M, D = 1, 2
    return (_rnd(M, D).to(dev), _rnd(M, D).to(dev), _rnd(M, D).to(dev),
            torch.zeros(M, D, device=dev), M, D), [3]


@_case("medium-softmax")
def _(dev):
    N = 1024
    return (_rnd(N).to(dev), torch.zeros(N, device=dev), N), [1]


def _ssm(dev):
    b, t, d, n = 2, 16, 32, 8
    return (_rnd(b, t, d).to(dev), _rnd(b, t, d).abs().to(dev),
            (-_rnd(d, n).abs()).to(dev), _rnd(b, t, n).to(dev),
            _rnd(b, t, n).to(dev), _rnd(d).to(dev),
            torch.zeros(b, t, d, device=dev), b, t, d, n), [6]


CASES["medium-ssm_selective_scan_1"] = [_ssm]
CASES["medium-ssm_selective_scan_2"] = [_ssm]


@_case("medium-top_k_selection")
def _(dev):
    N, k = 100, 5
    return (_rnd(N).to(dev), torch.zeros(k, device=dev), N, k), [1]


@_case("medium-top_p_sampling")
def _(dev):
    V = 512
    return (_rnd(V).to(dev), torch.full((1,), 0.9, device=dev),
            torch.full((1,), 1234, dtype=torch.int32, device=dev),
            torch.zeros(1, dtype=torch.int32, device=dev), V), [3]


@_case("medium-weight_dequantization")
def _(dev):
    M, N, T = 64, 64, 32
    return (_rnd(M, N).to(dev), _rnd(M // T, N // T).to(dev),
            torch.zeros(M, N, device=dev), M, N, T), [2]


# Two of the twenty-six cannot run on any backend, for reasons in the file
# rather than in this one. Listed rather than dropped so the count stays honest
# and nobody re-derives the diagnosis.
UNRUNNABLE = {
    "medium-fp16_dot_product":
        "the file hardcodes device='cuda' for its f32 accumulator",
    "medium-int4_weight_only_quantized_matmul":
        "the file masks an int8 tensor with the literal 0xF0, which Triton's "
        "frontend rejects as out of range for int8 (fails identically under "
        "TRITON_INTERPRET=1)",
}


def _run_child(name, variant, mode, out_path):
    """One `solve()` call, its outputs saved as .npz. Runs as a subprocess."""
    import numpy as np

    dev = "cpu" if mode == "interp" else "mps"
    torch.manual_seed(0)
    args, out_idx = CASES[name][int(variant)](dev)
    extra = {}
    if name == "medium-top_p_sampling":
        # The file's last step is torch.multinomial, whose RNG stream differs
        # between CPU and MPS for the same seed; comparing the sampled token
        # would measure only that. Swap in a deterministic pick and capture the
        # renormalized distribution, which is what the kernels actually produce.
        real = torch.multinomial

        def deterministic(probs, n, *a, **kw):
            extra["probs"] = probs.detach().cpu().to(torch.float32).numpy()
            return probs.argmax().reshape(1)

        torch.multinomial = deterministic
        try:
            _load(name).solve(*args)
        finally:
            torch.multinomial = real
    else:
        _load(name).solve(*args)
    if dev == "mps":
        torch.mps.synchronize()
    saved = {str(i): args[i].cpu().to(torch.float32).numpy() for i in out_idx}
    saved.update(extra)
    np.savez(out_path, **saved)


def _spawn(name, variant, mode, out_path):
    env = dict(os.environ)
    env.pop("TRITON_INTERPRET", None)
    if mode == "interp":
        env["TRITON_INTERPRET"] = "1"
    return subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), name, str(variant), mode,
         str(out_path)],
        capture_output=True, text=True, env=env,
    )


if not _AS_SCRIPT:

    _PARAMS = [(n, i) for n in sorted(CASES) for i in range(len(CASES[n]))]
    _PARAMS += [(n, 0) for n in sorted(UNRUNNABLE)]

    @pytest.mark.parametrize("name,variant", _PARAMS)
    def test_leet_uncovered_matches_interpreter(name, variant, tmp_path):
        if name in UNRUNNABLE:
            pytest.skip(f"{name}: {UNRUNNABLE[name]}")
        assert (LEET / f"{name}.py").is_file(), (
            f"{name}.py not present in Metal Leet fixtures"
        )
        import numpy as np

        results = {}
        for mode in ("mps", "interp"):
            out = tmp_path / f"{mode}.npz"
            proc = _spawn(name, variant, mode, out)
            assert proc.returncode == 0, (
                f"{name} [{mode}] exited {proc.returncode}\n"
                f"--- stderr ---\n{proc.stderr[-3000:]}"
            )
            results[mode] = np.load(out)

        got, want = results["mps"], results["interp"]
        assert got.files == want.files
        for key in want.files:
            g, w = got[key], want[key]
            assert g.shape == w.shape, f"{name}: arg {key} shape {g.shape} != {w.shape}"
            scale = np.maximum(np.abs(w), 1.0)
            err = float((np.abs(g - w) / scale).max()) if w.size else 0.0
            assert err < 2e-3, f"{name}: arg {key} relerr {err:.3e} vs interpreter"

    def test_leet_fixture_inventory_is_exhaustive():
        interpreter = {f"{name}.py" for name in CASES | UNRUNNABLE}
        categories = (STANDALONE_FIXTURES, interpreter, TARGETED_FIXTURES)
        for index, category in enumerate(categories):
            others = set().union(*(c for i, c in enumerate(categories) if i != index))
            assert category.isdisjoint(others)

        actual = {path.relative_to(LEET).as_posix() for path in LEET.rglob("*.py")}
        declared = set().union(*categories)
        assert actual == declared, (
            f"unowned={sorted(actual - declared)}, missing={sorted(declared - actual)}"
        )

        for relative in STANDALONE_FIXTURES:
            assert "__main__" in (LEET / relative).read_text(), relative

        test_root = Path(__file__).resolve().parent
        targeted_sources = "\n".join(
            (test_root / name).read_text() for name in TARGETED_TEST_MODULES
        )
        for relative in TARGETED_FIXTURES:
            assert Path(relative).name in targeted_sources, relative


def _run_all():
    test_root = Path(__file__).resolve().parent
    pytest_args = [
        sys.executable,
        "-m",
        "pytest",
        str(Path(__file__).resolve()),
        str(test_root / "test_metal_leet_source_fidelity.py"),
        *(str(test_root / name) for name in TARGETED_TEST_MODULES),
        "-s",
        "--tb=short",
    ]
    print(
        f"[metal-leet] pytest ownership: {len(CASES) + len(UNRUNNABLE)} "
        f"interpreter-backed + {len(TARGETED_FIXTURES)} targeted fixtures",
        flush=True,
    )
    subprocess.run(pytest_args, check=True)

    for index, relative in enumerate(sorted(STANDALONE_FIXTURES), start=1):
        print(
            f"[metal-leet] standalone {index}/{len(STANDALONE_FIXTURES)}: {relative}",
            flush=True,
        )
        subprocess.run([sys.executable, str(LEET / relative)], check=True)

    total = len(CASES) + len(UNRUNNABLE) + len(TARGETED_FIXTURES) + len(STANDALONE_FIXTURES)
    runnable = total - len(UNRUNNABLE)
    print(
        f"[metal-leet] PASS: all {total} fixtures are owned; "
        f"{runnable} runnable workloads covered, {len(UNRUNNABLE)} source-invalid skips",
        flush=True,
    )


if _AS_SCRIPT:
    if sys.argv[1:] == ["--all"]:
        _run_all()
    else:
        _run_child(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
