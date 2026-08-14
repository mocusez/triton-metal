"""Tuple associative scans on the Metal backend.

These tests pin the unmodified LeetTriton medium segmented-prefix-sum kernels:
each pass must compile to Metal, and the original three-pass ``solve`` entry
must match a CPU segmented-exclusive reference on MPS.

They also pin the two-state affine composition used by the medium linear
recurrence.  That algebra has a dedicated narrow lowering; unrelated tuple
scans must remain rejected instead of being miscompiled as one of the supported
monoids.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import triton
import triton.language as tl

torch = pytest.importorskip("torch")
pytest.importorskip(
    "triton._C.libtriton.metal",
    reason="Metal backend pybind module not built into libtriton",
)


REPO_ROOT = Path(__file__).resolve().parents[3]
SEGMENTED_SCAN_PATH = REPO_ROOT / "leet-triton" / "medium-segmented_exclusive_prefix_sum.py"
LINEAR_RECURRENCE_PATH = REPO_ROOT / "leet-triton" / "medium-linear_recurrence.py"
STREAM_COMPACTION_PATH = REPO_ROOT / "leet-triton" / "medium-stream-compaction.py"
TILE_SIZE = 65536
BLOCK_SIZE = 4096


@triton.jit
def _unsupported_combine(lhs_v, lhs_f, rhs_v, rhs_f):
    return lhs_v + rhs_v, lhs_f & rhs_f


@triton.jit
def _unsupported_tuple_scan(
    values_ptr,
    flags_ptr,
    output_ptr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < N
    values = tl.load(values_ptr + offsets, mask=mask, other=0.0)
    flags = tl.load(flags_ptr + offsets, mask=mask, other=0) == 1
    scanned, _ = tl.associative_scan(
        (values, flags), axis=0, combine_fn=_unsupported_combine
    )
    tl.store(output_ptr + offsets, scanned, mask=mask)


def _run_child(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def _compile_kernel_in_child(kernel_name, signature, constexprs, *, num_warps):
    script = textwrap.dedent(
        f"""
        import importlib.util
        from pathlib import Path

        import triton
        from triton.backends.compiler import GPUTarget
        from triton.compiler import ASTSource

        path = Path({str(SEGMENTED_SCAN_PATH)!r})
        spec = importlib.util.spec_from_file_location("segmented_scan_child", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        src = ASTSource(
            fn=getattr(module, {kernel_name!r}),
            signature={signature!r},
            constexprs={constexprs!r},
        )
        target = GPUTarget(backend="metal", arch=80, warp_size=32)
        compiled = triton.compile(src, target=target, options={{"num_warps": {num_warps}}})
        raw = compiled.asm["metal"]
        msl = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        assert msl, "{kernel_name} produced empty Metal output"
        assert "kernel void {kernel_name}" in msl, msl
        assert "metal.threadgroup_segmented_prefix_sum" in msl, msl
        assert "_sgps_carry_v" in msl, msl
        assert "threadgroup_barrier(mem_flags::mem_threadgroup)" in msl, msl
        """
    )
    result = _run_child(script)
    assert result.returncode == 0, (
        f"{kernel_name} failed to compile to Metal in a child process "
        f"(returncode={result.returncode}).\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )


def test_pass1_reduce_original_kernel_compiles_to_metal():
    _compile_kernel_in_child(
        "pass1_reduce",
        {
            "values_ptr": "*fp32",
            "flags_ptr": "*i32",
            "tile_sums_ptr": "*fp32",
            "tile_flags_ptr": "*i32",
            "N": "i32",
            "TILE_SIZE": "constexpr",
            "BLOCK_SIZE": "constexpr",
        },
        {"TILE_SIZE": TILE_SIZE, "BLOCK_SIZE": BLOCK_SIZE},
        num_warps=8,
    )


def test_pass2_scan_original_kernel_compiles_to_metal():
    _compile_kernel_in_child(
        "pass2_scan",
        {
            "tile_sums_ptr": "*fp32",
            "tile_flags_ptr": "*i32",
            "NUM_TILES": "i32",
            "BLOCK_SIZE": "constexpr",
        },
        {"BLOCK_SIZE": 16},
        num_warps=4,
    )


def test_pass3_downsweep_original_kernel_compiles_to_metal():
    _compile_kernel_in_child(
        "pass3_downsweep",
        {
            "values_ptr": "*fp32",
            "flags_ptr": "*i32",
            "output_ptr": "*fp32",
            "tile_sums_ptr": "*fp32",
            "tile_flags_ptr": "*i32",
            "N": "i32",
            "TILE_SIZE": "constexpr",
            "BLOCK_SIZE": "constexpr",
        },
        {"TILE_SIZE": TILE_SIZE, "BLOCK_SIZE": BLOCK_SIZE},
        num_warps=8,
    )


def _compile_linear_recurrence_kernel_in_child(
    kernel_name, signature, constexprs, *, num_warps=4
):
    script = textwrap.dedent(
        f"""
        import importlib.util
        from pathlib import Path

        import triton
        from triton.backends.compiler import GPUTarget
        from triton.compiler import ASTSource

        path = Path({str(LINEAR_RECURRENCE_PATH)!r})
        spec = importlib.util.spec_from_file_location("linear_recurrence_child", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        src = ASTSource(
            fn=getattr(module, {kernel_name!r}),
            signature={signature!r},
            constexprs={constexprs!r},
        )
        target = GPUTarget(backend="metal", arch=80, warp_size=32)
        compiled = triton.compile(src, target=target, options={{"num_warps": {num_warps}}})
        raw = compiled.asm["metal"]
        msl = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        assert msl, "{kernel_name} produced empty Metal output"
        assert "kernel void {kernel_name}" in msl, msl
        assert "metal.threadgroup_affine_prefix_scan" in msl, msl
        assert "_aps_carry_a" in msl, msl
        assert "threadgroup_barrier(mem_flags::mem_threadgroup)" in msl, msl
        """
    )
    result = _run_child(script)
    assert result.returncode == 0, (
        f"{kernel_name} failed to compile to Metal in a child process "
        f"(returncode={result.returncode}).\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )


def test_linear_recurrence_local_scan_original_kernel_compiles_to_metal():
    _compile_linear_recurrence_kernel_in_child(
        "local_scan_kernel",
        {
            "a_ptr": "*fp32",
            "x_ptr": "*fp32",
            "h_ptr": "*fp32",
            "a_prefix_ptr": "*fp32",
            "chunk_a_ptr": "*fp32",
            "chunk_x_ptr": "*fp32",
            "L": "constexpr",
            "N_CHUNKS": "constexpr",
            "BLOCK": "constexpr",
        },
        {"L": 520, "N_CHUNKS": 3, "BLOCK": 256},
    )


@pytest.mark.parametrize(
    ("n_chunks", "scan_block"),
    [(1, 1), (3, 4), (256, 256)],
)
def test_linear_recurrence_chunk_scan_original_kernel_compiles_to_metal(
    n_chunks, scan_block
):
    _compile_linear_recurrence_kernel_in_child(
        "chunk_scan_kernel",
        {
            "chunk_a_ptr": "*fp32",
            "chunk_x_ptr": "*fp32",
            "N_CHUNKS": "constexpr",
            "CHUNK_SCAN_BLOCK": "constexpr",
        },
        {"N_CHUNKS": n_chunks, "CHUNK_SCAN_BLOCK": scan_block},
    )


def test_noncanonical_tuple_scan_remains_rejected():
    # Conversion failures can crash while cleaning up after the useful
    # diagnostic. Keep this intentionally unsupported case in a child process.
    script = textwrap.dedent(
        f"""
        import importlib.util
        from pathlib import Path

        import triton
        from triton.backends.compiler import GPUTarget
        from triton.compiler import ASTSource

        path = Path({str(Path(__file__).resolve())!r})
        spec = importlib.util.spec_from_file_location("segmented_scan_test_child", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        src = ASTSource(
            fn=module._unsupported_tuple_scan,
            signature={{
                "values_ptr": "*fp32",
                "flags_ptr": "*i32",
                "output_ptr": "*fp32",
                "N": "constexpr",
                "BLOCK": "constexpr",
            }},
            constexprs={{"N": 16, "BLOCK": 16}},
        )
        target = GPUTarget(backend="metal", arch=80, warp_size=32)
        triton.compile(src, target=target, options={{"num_warps": 4}})
        """
    )
    result = _run_child(script)
    assert result.returncode != 0, "unsupported tuple scan unexpectedly compiled"
    diagnostics = result.stdout + result.stderr
    assert "failed to legalize operation 'tt.scan'" in diagnostics, diagnostics


@pytest.mark.parametrize(
    ("N", "reset_positions"),
    [
        (1024, []),
        (1024, [0]),
        (5000, [7, 128, 4095, 4096, 4999]),
        (
            TILE_SIZE + 257,
            [
                0,
                31,
                BLOCK_SIZE - 1,
                BLOCK_SIZE,
                TILE_SIZE - 1,
                TILE_SIZE,
                TILE_SIZE + 17,
            ],
        ),
    ],
)
@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="Metal backend requires an MPS-enabled PyTorch (Apple Silicon)",
)
def test_original_solve_matches_segmented_exclusive_reference(N, reset_positions):
    script = textwrap.dedent(
        f"""
        import importlib.util
        from pathlib import Path

        import torch

        path = Path({str(SEGMENTED_SCAN_PATH)!r})
        spec = importlib.util.spec_from_file_location("segmented_scan_child", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        N = {N}
        reset_positions = {reset_positions!r}
        values_cpu = (torch.arange(N, dtype=torch.float32) % 17) * 0.25 + 1.0
        flags_cpu = torch.zeros(N, dtype=torch.int32)
        for pos in reset_positions:
            if 0 <= pos < N:
                flags_cpu[pos] = 1

        expected = torch.empty_like(values_cpu)
        running = 0.0
        for i, (value, flag) in enumerate(zip(values_cpu.tolist(), flags_cpu.tolist())):
            if flag == 1:
                running = 0.0
            expected[i] = running
            running += value

        values = values_cpu.to("mps")
        flags = flags_cpu.to("mps")
        output = torch.empty_like(values)
        module.solve(values, flags, output, N)
        torch.mps.synchronize()
        torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
        """
    )
    result = _run_child(script)
    assert result.returncode == 0, (
        f"original solve failed for N={N}, reset_positions={reset_positions} "
        f"(returncode={result.returncode}).\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )


@pytest.mark.parametrize(
    ("B", "L"),
    [(1, 17), (2, 520), (1, 16384), (1, 65536)],
)
@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="Metal backend requires an MPS-enabled PyTorch (Apple Silicon)",
)
def test_original_linear_recurrence_solve_matches_reference(B, L):
    script = textwrap.dedent(
        f"""
        import importlib.util
        from pathlib import Path

        import torch

        path = Path({str(LINEAR_RECURRENCE_PATH)!r})
        spec = importlib.util.spec_from_file_location("linear_recurrence_child", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        torch.manual_seed(0xC0FFEE + {L})
        B, L = {B}, {L}
        # Keep coefficients close to one so scan carry propagation remains
        # numerically significant across chunk and internal threadgroup-tile
        # boundaries; small random coefficients would rapidly erase such bugs.
        a_cpu = torch.empty((B, L), dtype=torch.float32).uniform_(0.90, 0.99)
        x_cpu = torch.randn((B, L), dtype=torch.float32)

        expected = torch.empty_like(x_cpu)
        expected[:, 0] = x_cpu[:, 0]
        for t in range(1, L):
            expected[:, t] = a_cpu[:, t] * expected[:, t - 1] + x_cpu[:, t]

        a = a_cpu.to("mps")
        x = x_cpu.to("mps")
        output = torch.empty_like(x)
        module.solve(a, x, output, B, L)
        torch.mps.synchronize()
        torch.testing.assert_close(output.cpu(), expected, rtol=1e-5, atol=1e-5)
        """
    )
    result = _run_child(script)
    assert result.returncode == 0, (
        f"linear recurrence solve failed for B={B}, L={L} "
        f"(returncode={result.returncode}).\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="Metal backend requires an MPS-enabled PyTorch (Apple Silicon)",
)
def test_original_stream_compaction_solve_matches_reference():
    # Compile and launch the exact Leet file in a child process: before the
    # rank-1 shared-cone normalization fix, aligned MPS tensors fail conversion
    # and compiler cleanup may abort the process after the diagnostic.
    script = textwrap.dedent(
        f"""
        import importlib.util
        from pathlib import Path

        import torch

        path = Path({str(STREAM_COMPACTION_PATH)!r})
        spec = importlib.util.spec_from_file_location("stream_compaction_child", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        cases = [
            torch.empty(0, dtype=torch.float32),
            torch.tensor([3.0], dtype=torch.float32),
            torch.arange(100, dtype=torch.float32) - 50,
            (torch.arange(1024, dtype=torch.float32) % 11) - 5,
            (torch.arange(4113, dtype=torch.float32) % 17) - 8,
            torch.arange(1, 1501, dtype=torch.float32),
            -torch.arange(1500, dtype=torch.float32),
        ]
        for values_cpu in cases:
            values = values_cpu.to("mps")
            output = torch.empty_like(values)
            module.solve(values, values.numel(), output)
            torch.mps.synchronize()

            expected = torch.zeros_like(values_cpu)
            kept = values_cpu[values_cpu > 0]
            expected[: kept.numel()] = kept
            torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
        """
    )
    result = _run_child(script)
    assert result.returncode == 0, (
        "original stream-compaction solve failed "
        f"(returncode={result.returncode}).\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
