"""Tuple associative scans on the Metal backend.

These tests pin the unmodified LeetTriton medium segmented-prefix-sum kernels:
each pass must compile to Metal, and the original three-pass ``solve`` entry
must match a CPU segmented-exclusive reference on MPS.

They also pin the two-state affine composition used by the medium linear
recurrence.  That algebra has a dedicated narrow lowering.

A general tuple scan no longer has to be one of those two shapes: an inclusive
Hillis-Steele prefix over one threadgroup buffer per carried value replays the
scan's own combine region, so any associative combine over rank-1 blocked
f32/i32 sources holding up to sixty-four elements per thread is lowered.
Multidimensional tuples use an ordered prefix fold over f32/i32 slots and
scan-axis extent 2..64. Top-level multiband tuples publish every component in
one exact-layout phase before folding; one-band tuples also support user loops.
Unsupported types, over-budget tiles and longer axes are refused before conversion
by the same predicate the pattern dispatches on.
"""

from __future__ import annotations

import re
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
LEET_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "metal_leet"
SEGMENTED_SCAN_PATH = LEET_FIXTURES / "medium-segmented_exclusive_prefix_sum.py"
LINEAR_RECURRENCE_PATH = LEET_FIXTURES / "medium-linear_recurrence.py"
STREAM_COMPACTION_PATH = LEET_FIXTURES / "medium-stream-compaction.py"
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
    # Still refused, but no longer for its combine: a general tuple combine is
    # lowered now. This one carries an i1 flag, and a lockstep buffer slot is
    # f32 or i32 -- the two types with a spelling on both sides of the ui32
    # storage bridge.
    #
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
    assert result.returncode > 0, (
        f"expected a clean non-zero exit, got returncode {result.returncode} "
        "(negative means the process died on a signal)")
    diagnostics = result.stdout + result.stderr
    # Named by the scan pre-pass rather than "failed to legalize". The old
    # message came from a decline inside applyFullConversion, which in this
    # backend segfaults the caller instead of raising; the pre-pass unwinds
    # cleanly, hence the returncode assertion above.
    assert "carrying more than one value" in diagnostics, diagnostics


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


# ---------------------------------------------------------------------------
# The general multi-value scan.
#
# Only the segmented-sum and affine forms had a lowering, both matched
# structurally. Everything else was refused. The prefix is now built from the
# scan's own combine region: read both slots, replay the region with the LOWER
# position as the left operand, barrier, write, barrier.
#
# The barrier between a step's reads and its writes is NOT observable from a
# numeric test -- with it deleted, a hundred and forty differential runs over
# one, two, four and eight warps all still passed. It is pinned structurally by
# `rank1_scan_tuple.mlir` instead.
# ---------------------------------------------------------------------------


mps_only = pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="Metal backend requires an MPS-enabled PyTorch (Apple Silicon)",
)


@triton.jit
def _tuple_sum_and_count_combine(a_sum, a_count, b_sum, b_count):
    return a_sum + b_sum, a_count + b_count


@triton.jit
def _tuple_sum_and_count_kernel(x_ptr, sum_ptr, count_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    row = tl.load(x_ptr + offs)
    ones = tl.full((BLOCK,), 1.0, tl.float32)
    prefix, count = tl.associative_scan((row, ones), 0,
                                        _tuple_sum_and_count_combine)
    tl.store(sum_ptr + offs, prefix)
    tl.store(count_ptr + offs, count)


@mps_only
@pytest.mark.parametrize("num_warps", [1, 2, 4, 8])
@pytest.mark.parametrize("block", [16, 32, 64, 128, 256])
def test_tuple_scan_sum_and_count(block, num_warps):
    if block > 32 * num_warps:
        pytest.skip("a tuple scan needs at most one element per thread")
    torch.manual_seed(block * 7 + num_warps)
    x = torch.randn(block)
    prefix = torch.zeros(block, device="mps")
    count = torch.zeros(block, device="mps")
    _tuple_sum_and_count_kernel[(1,)](x.to("mps"), prefix, count, BLOCK=block,
                                      num_warps=num_warps)
    torch.mps.synchronize()
    torch.testing.assert_close(prefix.cpu().double(), x.double().cumsum(0),
                               atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(count.cpu(),
                               torch.arange(1, block + 1, dtype=torch.float32))


@triton.jit
def _tuple_segmented_combine(a_value, a_flag, b_value, b_flag):
    return tl.where(b_flag != 0, b_value, a_value + b_value), a_flag | b_flag


@triton.jit
def _tuple_segmented_kernel(x_ptr, flag_ptr, out_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    value, _ = tl.associative_scan(
        (tl.load(x_ptr + offs), tl.load(flag_ptr + offs)), 0,
        _tuple_segmented_combine)
    tl.store(out_ptr + offs, value)


@mps_only
@pytest.mark.parametrize("block,num_warps", [(32, 1), (128, 4), (64, 8)])
def test_tuple_scan_hand_written_segmented(block, num_warps):
    """A segmented scan written by hand rather than matched structurally. The
    combine is associative and NOT commutative -- swap the operands and a reset
    keeps the wrong side -- so it also pins the merge order."""
    torch.manual_seed(block + num_warps)
    x = torch.randn(block)
    flags = (torch.rand(block) < 0.25).to(torch.int32)
    out = torch.zeros(block, device="mps")
    _tuple_segmented_kernel[(1,)](x.to("mps"), flags.to("mps"), out,
                                  BLOCK=block, num_warps=num_warps)
    torch.mps.synchronize()
    want = torch.zeros(block, dtype=torch.float64)
    acc = 0.0
    for i in range(block):
        acc = float(x[i]) if int(flags[i]) else acc + float(x[i])
        want[i] = acc
    torch.testing.assert_close(out.cpu().double(), want, atol=1e-4, rtol=1e-4)


@triton.jit
def _tuple_running_max_and_sum_combine(a_max, a_sum, b_max, b_sum):
    return tl.maximum(a_max, b_max), a_sum + b_sum


@triton.jit
def _tuple_running_max_and_sum_kernel(x_ptr, max_ptr, sum_ptr,
                                      BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    row = tl.load(x_ptr + offs)
    running_max, running_sum = tl.associative_scan(
        (row, row), 0, _tuple_running_max_and_sum_combine)
    tl.store(max_ptr + offs, running_max)
    tl.store(sum_ptr + offs, running_sum)


@mps_only
@pytest.mark.parametrize("block,num_warps", [(64, 2), (128, 4)])
def test_tuple_scan_i32_running_max_and_sum(block, num_warps):
    """Both slots are i32, so both ride the ui32 storage bridge, and the
    comparison has to stay signed across it."""
    torch.manual_seed(block)
    x = torch.randint(-1000, 1000, (block,), dtype=torch.int32)
    running_max = torch.zeros(block, dtype=torch.int32, device="mps")
    running_sum = torch.zeros(block, dtype=torch.int32, device="mps")
    _tuple_running_max_and_sum_kernel[(1,)](x.to("mps"), running_max,
                                            running_sum, BLOCK=block,
                                            num_warps=num_warps)
    torch.mps.synchronize()
    assert torch.equal(running_max.cpu(), torch.cummax(x, 0).values)
    assert torch.equal(running_sum.cpu(), x.cumsum(0, dtype=torch.int32))


@triton.jit
def _tuple_first_nonzero_combine(a_value, a_index, b_value, b_index):
    keep = a_value != 0.0
    return tl.where(keep, a_value, b_value), tl.where(keep, a_index, b_index)


@triton.jit
def _tuple_first_nonzero_kernel(x_ptr, idx_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    _, index = tl.associative_scan(
        (tl.load(x_ptr + offs), offs.to(tl.int32)), 0,
        _tuple_first_nonzero_combine)
    tl.store(idx_ptr + offs, index)


@mps_only
@pytest.mark.parametrize("block,num_warps", [(32, 1), (64, 2), (128, 4)])
def test_tuple_scan_keeps_the_prefix_order(block, num_warps):
    """`first non-zero so far` is associative, not commutative, and its answer
    depends on which side of the merge each range is on. Every position at or
    after the first non-zero must name that first one."""
    torch.manual_seed(block * 3 + num_warps)
    x = torch.zeros(block)
    hits = sorted(torch.randperm(block)[:4].tolist())
    for h in hits:
        x[h] = float(h) + 1.0
    index = torch.zeros(block, dtype=torch.int32, device="mps")
    _tuple_first_nonzero_kernel[(1,)](x.to("mps"), index, BLOCK=block,
                                      num_warps=num_warps)
    torch.mps.synchronize()
    got = index.cpu().tolist()
    assert all(g == hits[0] for g in got[hits[0]:]), got[hits[0]:hits[0] + 8]


@triton.jit
def _tuple_sum_and_count_rows_kernel(x_ptr, sum_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    row = tl.load(x_ptr + pid * BLOCK + offs)
    ones = tl.full((BLOCK,), 1.0, tl.float32)
    prefix, _ = tl.associative_scan((row, ones), 0,
                                    _tuple_sum_and_count_combine)
    tl.store(sum_ptr + pid * BLOCK + offs, prefix)


@mps_only
def test_tuple_scan_multiprogram():
    """Each threadgroup addresses its own slice of the buffers by a LOCAL
    thread id, and the leading barrier keeps one trip's fill off the previous
    trip's reads."""
    torch.manual_seed(21)
    rows, block = 4, 64
    x = torch.randn(rows, block)
    out = torch.zeros(rows, block, device="mps")
    _tuple_sum_and_count_rows_kernel[(rows,)](x.to("mps").contiguous(), out,
                                              BLOCK=block, num_warps=2)
    torch.mps.synchronize()
    torch.testing.assert_close(out.cpu().double(), x.double().cumsum(1),
                               atol=1e-4, rtol=1e-4)


@triton.jit
def _tuple_scan_in_loop_kernel(x_ptr, out_ptr, rows, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    for r in range(0, rows):
        row = tl.load(x_ptr + r * BLOCK + offs)
        ones = tl.full((BLOCK,), 1.0, tl.float32)
        prefix, _ = tl.associative_scan((row, ones), 0,
                                        _tuple_sum_and_count_combine)
        tl.store(out_ptr + r * BLOCK + offs, prefix)


@mps_only
def test_tuple_scan_inside_a_user_loop():
    """One static threadgroup allocation is reused every trip, so this trip's
    fill would race the previous trip's final reads without the leading
    barrier."""
    torch.manual_seed(22)
    rows, block = 3, 64
    x = torch.randn(rows, block)
    out = torch.zeros(rows, block, device="mps")
    _tuple_scan_in_loop_kernel[(1,)](x.to("mps").contiguous(), out, rows,
                                     BLOCK=block, num_warps=2)
    torch.mps.synchronize()
    torch.testing.assert_close(out.cpu().double(), x.double().cumsum(1),
                               atol=1e-4, rtol=1e-4)


@mps_only
def test_tuple_scan_beyond_the_band_envelope_is_named(capfd):
    """Outside the envelope the refusal has to come from the pre-conversion
    validator: a pattern that declines inside `applyFullConversion` does not
    raise, it takes the process down.

    More than one element per thread is no longer outside it -- a thread scans
    its own band first -- so the boundary this pins is the band LENGTH."""
    x = torch.randn(4096, device="mps")
    prefix = torch.zeros(4096, device="mps")
    count = torch.zeros(4096, device="mps")
    with pytest.raises(Exception) as error:
        _tuple_sum_and_count_kernel[(1,)](x, prefix, count, BLOCK=4096,
                                          num_warps=1)
    assert "convert-tritongpu-to-metal failed" in str(error.value)
    reason = capfd.readouterr()
    assert "sixty-four elements per thread" in reason.err + reason.out

    survivor = torch.randn(32, device="mps")
    got = torch.zeros(32, device="mps")
    other = torch.zeros(32, device="mps")
    _tuple_sum_and_count_kernel[(1,)](survivor, got, other, BLOCK=32,
                                      num_warps=1)
    torch.mps.synchronize()
    torch.testing.assert_close(got.cpu().double(),
                               survivor.cpu().double().cumsum(0), atol=1e-4,
                               rtol=1e-4)


# ---------------------------------------------------------------------------
# The reverse direction.
#
# Triton defines a reverse scan as a forward scan of the REVERSED sequence, so
# `out[i]` folds N-1 down to i and the higher positions are applied first. The
# prefix mirrors accordingly -- partner at `tid + stride`, live while inside the
# block -- and the partner's range stays the LEFT operand.
#
# ⚠️ Nothing commutative can see that last part. A reversed running sum and
# count come out identical under either operand order; it took an asymmetric
# combine and `TRITON_INTERPRET=1` as the oracle to catch it, and with the
# operands swapped fifty-four of a hundred and eight differential cells
# disagreed -- every reverse one.
# ---------------------------------------------------------------------------


@triton.jit
def _tuple_reverse_sum_and_count_kernel(x_ptr, sum_ptr, count_ptr,
                                        BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    row = tl.load(x_ptr + offs)
    ones = tl.full((BLOCK,), 1.0, tl.float32)
    suffix, count = tl.associative_scan((row, ones), 0,
                                        _tuple_sum_and_count_combine,
                                        reverse=True)
    tl.store(sum_ptr + offs, suffix)
    tl.store(count_ptr + offs, count)


@triton.jit
def _tuple_reverse_sum_and_count_rows_kernel(x_ptr, sum_ptr,
                                             BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    row = tl.load(x_ptr + pid * BLOCK + offs)
    ones = tl.full((BLOCK,), 1.0, tl.float32)
    suffix, _ = tl.associative_scan((row, ones), 0,
                                    _tuple_sum_and_count_combine,
                                    reverse=True)
    tl.store(sum_ptr + pid * BLOCK + offs, suffix)


@triton.jit
def _tuple_reverse_scan_in_loop_kernel(x_ptr, out_ptr, rows,
                                       BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    for r in range(0, rows):
        row = tl.load(x_ptr + r * BLOCK + offs)
        ones = tl.full((BLOCK,), 1.0, tl.float32)
        suffix, _ = tl.associative_scan((row, ones), 0,
                                        _tuple_sum_and_count_combine,
                                        reverse=True)
        tl.store(out_ptr + r * BLOCK + offs, suffix)


@mps_only
@pytest.mark.parametrize("num_warps", [1, 2, 4, 8])
@pytest.mark.parametrize("block", [16, 32, 64, 128, 256])
def test_tuple_scan_reverse_sum_and_count(block, num_warps):
    if block > 32 * num_warps:
        pytest.skip("a tuple scan needs at most one element per thread")
    torch.manual_seed(block * 5 + num_warps)
    x = torch.randn(block)
    suffix = torch.zeros(block, device="mps")
    count = torch.zeros(block, device="mps")
    _tuple_reverse_sum_and_count_kernel[(1,)](x.to("mps"), suffix, count,
                                              BLOCK=block,
                                              num_warps=num_warps)
    torch.mps.synchronize()
    torch.testing.assert_close(suffix.cpu().double(),
                               x.double().flip(0).cumsum(0).flip(0),
                               atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(
        count.cpu(), torch.arange(block, 0, -1, dtype=torch.float32))


@triton.jit
def _tuple_take_right_nonzero_combine(a_value, a_index, b_value, b_index):
    keep = b_value != 0.0
    return tl.where(keep, b_value, a_value), tl.where(keep, b_index, a_index)


@triton.jit
def _tuple_take_right_nonzero_kernel(x_ptr, idx_ptr, BLOCK: tl.constexpr,
                                     REVERSE: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    _, index = tl.associative_scan(
        (tl.load(x_ptr + offs), offs.to(tl.int32)), 0,
        _tuple_take_right_nonzero_combine, reverse=REVERSE)
    tl.store(idx_ptr + offs, index)


@mps_only
@pytest.mark.parametrize("block,num_warps", [(32, 1), (64, 2), (128, 4)])
def test_tuple_scan_reverse_applies_the_higher_positions_first(block,
                                                               num_warps):
    """`take the right operand when it is non-zero` is associative and not
    commutative. Forward it names the LAST non-zero at or before each position;
    reversed -- a forward scan of the reversed sequence -- it names the FIRST
    non-zero at or after it. Swap the operands in the reverse merge and the
    second answer becomes the last non-zero instead."""
    torch.manual_seed(block * 9 + num_warps)
    x = torch.zeros(block)
    hits = sorted(torch.randperm(block)[:4].tolist())
    for h in hits:
        x[h] = float(h) + 1.0

    forward = torch.zeros(block, dtype=torch.int32, device="mps")
    _tuple_take_right_nonzero_kernel[(1,)](x.to("mps"), forward, BLOCK=block,
                                           REVERSE=False,
                                           num_warps=num_warps)
    reverse = torch.zeros(block, dtype=torch.int32, device="mps")
    _tuple_take_right_nonzero_kernel[(1,)](x.to("mps"), reverse, BLOCK=block,
                                           REVERSE=True, num_warps=num_warps)
    torch.mps.synchronize()

    want_forward, last = [], 0
    for i in range(block):
        if x[i] != 0:
            last = i
        want_forward.append(last)
    want_reverse, nxt = [0] * block, 0
    for i in range(block - 1, -1, -1):
        if x[i] != 0:
            nxt = i
        want_reverse[i] = nxt

    assert forward.cpu().tolist()[hits[0]:] == want_forward[hits[0]:]
    assert reverse.cpu().tolist()[:hits[-1] + 1] == want_reverse[:hits[-1] + 1]


# ---------------------------------------------------------------------------
# More than one element per thread.
#
# Three phases: each thread scans its OWN contiguous band [t*E, (t+1)*E), the
# band totals are scanned across threads by the same tree the one-element case
# uses, and each thread then folds the exclusive prefix of the bands before it
# into every element of its own. Thread zero has no such prefix and there is no
# identity to stand in for one, so it keeps what its band holds.
#
# The band has to be CONTIGUOUS for the cross-thread scan of totals to stand
# for a scan of positions, and the result buffer needs one slot per position
# because the consumer reads a different position on every tile-loop iteration
# -- positions that need not be in this thread's band, which is what the
# barrier before the read is for.
# ---------------------------------------------------------------------------


@mps_only
@pytest.mark.parametrize("num_warps", [1, 2, 4, 8])
@pytest.mark.parametrize("block", [64, 128, 512, 1024, 2048])
def test_tuple_scan_above_one_element_per_thread(block, num_warps):
    tpb = 32 * num_warps
    if block > tpb and (block % tpb or block // tpb > 64):
        pytest.skip("outside the band envelope")
    torch.manual_seed(block * 11 + num_warps)
    x = torch.randn(block)
    prefix = torch.zeros(block, device="mps")
    count = torch.zeros(block, device="mps")
    _tuple_sum_and_count_kernel[(1,)](x.to("mps"), prefix, count, BLOCK=block,
                                      num_warps=num_warps)
    torch.mps.synchronize()
    torch.testing.assert_close(prefix.cpu().double(), x.double().cumsum(0),
                               atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(count.cpu(),
                               torch.arange(1, block + 1, dtype=torch.float32))


@mps_only
@pytest.mark.parametrize("block,num_warps", [(128, 1), (1024, 4), (2048, 8)])
def test_tuple_scan_hand_written_segmented_above_one_element_per_thread(
        block, num_warps):
    """The order-sensitive case at more than one element per thread: a reset
    inside a band, and one that lands exactly on a band boundary, both have to
    cut the prefix."""
    torch.manual_seed(block + num_warps * 3)
    x = torch.randn(block)
    flags = (torch.rand(block) < 0.25).to(torch.int32)
    e = block // (32 * num_warps)
    flags[e] = 1          # first element of the second band
    flags[e + 1] = 1      # inside the second band
    out = torch.zeros(block, device="mps")
    _tuple_segmented_kernel[(1,)](x.to("mps"), flags.to("mps"), out,
                                  BLOCK=block, num_warps=num_warps)
    torch.mps.synchronize()
    want = torch.zeros(block, dtype=torch.float64)
    acc = 0.0
    for i in range(block):
        acc = float(x[i]) if int(flags[i]) else acc + float(x[i])
        want[i] = acc
    torch.testing.assert_close(out.cpu().double(), want, atol=1e-3, rtol=1e-3)


@mps_only
def test_tuple_scan_above_one_element_per_thread_multiprogram():
    torch.manual_seed(41)
    rows, block = 3, 512
    x = torch.randn(rows, block)
    out = torch.zeros(rows, block, device="mps")
    _tuple_sum_and_count_rows_kernel[(rows,)](x.to("mps").contiguous(), out,
                                              BLOCK=block, num_warps=4)
    torch.mps.synchronize()
    torch.testing.assert_close(out.cpu().double(), x.double().cumsum(1),
                               atol=1e-3, rtol=1e-3)


@mps_only
@pytest.mark.parametrize("elements_per_thread,num_warps", [
    (2, 1), (4, 1), (16, 1), (64, 1),
    (2, 2), (4, 2), (16, 2), (32, 2),
    (2, 4), (4, 4), (16, 4),
    (2, 8), (4, 8), (8, 8),
])
def test_tuple_scan_reverse_above_one_element_per_thread_sum_and_count(
        elements_per_thread, num_warps):
    block = 32 * num_warps * elements_per_thread
    torch.manual_seed(block * 13 + num_warps)
    x = torch.randn(block)
    suffix = torch.zeros(block, device="mps")
    count = torch.zeros(block, device="mps")
    _tuple_reverse_sum_and_count_kernel[(1,)](x.to("mps"), suffix, count,
                                              BLOCK=block,
                                              num_warps=num_warps)
    torch.mps.synchronize()
    torch.testing.assert_close(suffix.cpu().double(),
                               x.double().flip(0).cumsum(0).flip(0),
                               atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(
        count.cpu(), torch.arange(block, 0, -1, dtype=torch.float32))


@mps_only
@pytest.mark.parametrize("elements_per_thread,num_warps",
                         [(2, 1), (4, 2), (8, 4), (8, 8)])
def test_tuple_scan_reverse_above_one_element_per_thread_keeps_order(
        elements_per_thread, num_warps):
    """Boundary hits catch the reverse carry order between contiguous bands."""
    block = 32 * num_warps * elements_per_thread
    x = torch.zeros(block)
    hits = [
        elements_per_thread - 1,
        elements_per_thread,
        7 * elements_per_thread - 1,
        7 * elements_per_thread,
        block - elements_per_thread,
        block - 1,
    ]
    for h in hits:
        x[h] = float(h) + 1.0

    reverse = torch.zeros(block, dtype=torch.int32, device="mps")
    _tuple_take_right_nonzero_kernel[(1,)](x.to("mps"), reverse, BLOCK=block,
                                           REVERSE=True, num_warps=num_warps)
    torch.mps.synchronize()

    want, nxt = [0] * block, 0
    for i in range(block - 1, -1, -1):
        if x[i] != 0:
            nxt = i
        want[i] = nxt
    got = reverse.cpu().tolist()
    assert [got[h - 1] for h in hits[:-1] if h > 0] == [
        want[h - 1] for h in hits[:-1] if h > 0
    ]
    assert got[:hits[-1] + 1] == want[:hits[-1] + 1]


@mps_only
def test_tuple_scan_reverse_above_one_element_per_thread_multiprogram():
    torch.manual_seed(51)
    rows, block = 3, 512
    x = torch.randn(rows, block)
    out = torch.zeros(rows, block, device="mps")
    _tuple_reverse_sum_and_count_rows_kernel[(rows,)](
        x.to("mps").contiguous(), out, BLOCK=block, num_warps=4)
    torch.mps.synchronize()
    torch.testing.assert_close(out.cpu().double(),
                               x.double().flip(1).cumsum(1).flip(1),
                               atol=1e-3, rtol=1e-3)


@mps_only
def test_tuple_scan_reverse_above_one_element_per_thread_inside_a_user_loop():
    torch.manual_seed(52)
    rows, block = 3, 512
    x = torch.randn(rows, block)
    out = torch.zeros(rows, block, device="mps")
    _tuple_reverse_scan_in_loop_kernel[(1,)](x.to("mps").contiguous(), out,
                                             rows, BLOCK=block, num_warps=4)
    torch.mps.synchronize()
    torch.testing.assert_close(out.cpu().double(),
                               x.double().flip(1).cumsum(1).flip(1),
                               atol=1e-3, rtol=1e-3)


@mps_only
def test_tuple_scan_reverse_beyond_the_band_envelope_is_named(capfd):
    x = torch.randn(128 * 32, device="mps")
    prefix = torch.zeros(128 * 32, device="mps")
    count = torch.zeros(128 * 32, device="mps")
    with pytest.raises(Exception) as error:
        _tuple_reverse_sum_and_count_kernel[(1,)](x, prefix, count,
                                                  BLOCK=128 * 32, num_warps=1)
    assert "convert-tritongpu-to-metal failed" in str(error.value)
    reason = capfd.readouterr()
    assert "sixty-four elements per thread" in reason.err + reason.out

    survivor = torch.randn(64 * 32, device="mps")
    got = torch.zeros(64 * 32, device="mps")
    other = torch.zeros(64 * 32, device="mps")
    _tuple_reverse_sum_and_count_kernel[(1,)](survivor, got, other,
                                              BLOCK=64 * 32, num_warps=1)
    torch.mps.synchronize()
    torch.testing.assert_close(got.cpu().double(),
                               survivor.cpu().double().flip(0).cumsum(0).flip(0),
                               atol=1e-3, rtol=1e-3)


@triton.jit
def _rankn_tuple_scan_kernel(x_ptr, value_ptr, other_ptr, SHAPE: tl.constexpr,
                            BLOCK: tl.constexpr, AXIS: tl.constexpr,
                            REVERSE: tl.constexpr, ORDERED: tl.constexpr):
    index = tl.reshape(tl.arange(0, BLOCK), SHAPE)
    offset = tl.program_id(0) * BLOCK + index
    x = tl.load(x_ptr + offset)
    if ORDERED:
        value, other = tl.associative_scan((x, index), AXIS,
                                           _tuple_take_right_nonzero_combine,
                                           reverse=REVERSE)
    else:
        count = tl.full(SHAPE, 1, tl.int32)
        value, other = tl.associative_scan((x, count), AXIS,
                                           _tuple_sum_and_count_combine,
                                           reverse=REVERSE)
    tl.store(value_ptr + offset, value)
    tl.store(other_ptr + offset, other)


RANKN_TUPLE_CASES = [
    # Original one-band multidimensional coverage.
    ((4, 8), 0, 1),
    ((4, 8), 1, 1),
    ((4, 8), 0, 4),
    ((2, 4, 4), 0, 1),
    ((2, 4, 4), 1, 1),
    ((2, 4, 4), 2, 1),
    ((2, 64), 1, 4),
    ((64, 2), 0, 4),
    # Top-level multiband exact layouts.
    ((8, 8), 0, 1),
    ((8, 8), 1, 1),
    ((8, 16), 0, 1),
    ((8, 16), 1, 1),
    ((8, 16), 0, 4),
    ((8, 16), 1, 4),
    ((4, 4, 8), 0, 1),
    ((4, 4, 8), 1, 1),
    ((4, 4, 8), 2, 1),
    ((4, 4, 8), 2, 4),
    ((64, 8), 0, 1),
    ((64, 8), 1, 1),
    ((64, 8), 0, 4),
    ((64, 8), 1, 4),
]


@mps_only
@pytest.mark.parametrize("shape,axis,num_warps", RANKN_TUPLE_CASES)
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
def test_tuple_scan_rankn_sum_and_count(shape, axis, num_warps, reverse, dtype):
    torch.manual_seed(61)
    rows, block = 3, int(torch.tensor(shape).prod())
    if dtype == torch.int32:
        # Exercise wrapping i32 arithmetic in both tuple components' bridges.
        x = torch.randint(2**30, 2**31 - 1, (rows, *shape), dtype=dtype)
    else:
        x = torch.randn(rows, *shape)
    value = torch.empty_like(x, device="mps")
    count = torch.empty(x.shape, dtype=torch.int32, device="mps")
    _rankn_tuple_scan_kernel[(rows,)](
        x.to("mps"), value, count, SHAPE=shape, BLOCK=block, AXIS=axis,
        REVERSE=reverse, ORDERED=False, num_warps=num_warps)
    torch.mps.synchronize()
    dim = axis + 1
    source = x.flip(dim) if reverse else x
    expected = source.cumsum(dim, dtype=dtype)
    expected_count = torch.ones_like(x, dtype=torch.int32).cumsum(dim, dtype=torch.int32)
    if reverse:
        expected = expected.flip(dim)
        expected_count = expected_count.flip(dim)
    tolerance = 0 if dtype == torch.int32 else 1e-4
    torch.testing.assert_close(value.cpu(), expected, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(count.cpu(), expected_count, atol=0, rtol=0)


@mps_only
@pytest.mark.parametrize("shape,axis,num_warps", RANKN_TUPLE_CASES)
@pytest.mark.parametrize("reverse", [False, True])
def test_tuple_scan_rankn_noncommutative_order(shape, axis, num_warps, reverse):
    torch.manual_seed(62)
    rows, block = 3, int(torch.tensor(shape).prod())
    x = torch.randint(0, 3, (rows, *shape)).float()
    x[0].zero_()  # No identity: an all-zero line must retain its seed index.
    value = torch.empty_like(x, device="mps")
    index = torch.empty(x.shape, dtype=torch.int32, device="mps")
    _rankn_tuple_scan_kernel[(rows,)](
        x.to("mps"), value, index, SHAPE=shape, BLOCK=block, AXIS=axis,
        REVERSE=reverse, ORDERED=True, num_warps=num_warps)
    torch.mps.synchronize()
    expected_value = x.clone()
    expected_index = torch.arange(block, dtype=torch.int32).reshape(shape).expand(rows, *shape).clone()
    dim = axis + 1
    positions = list(range(shape[axis]))
    if reverse:
        positions.reverse()
    for previous, current in zip(positions, positions[1:]):
        keep = x.select(dim, current) != 0
        expected_value.select(dim, current).copy_(torch.where(
            keep, x.select(dim, current), expected_value.select(dim, previous)))
        expected_index.select(dim, current).copy_(torch.where(
            keep, expected_index.select(dim, current), expected_index.select(dim, previous)))
    torch.testing.assert_close(value.cpu(), expected_value, atol=0, rtol=0)
    torch.testing.assert_close(index.cpu(), expected_index, atol=0, rtol=0)


@triton.jit
def _rankn_tuple_scan_loop_kernel(x_ptr, value_ptr, count_ptr, rows,
                                 REVERSE: tl.constexpr):
    index = tl.reshape(tl.arange(0, 32), (4, 8))
    for row in range(rows):
        x = tl.load(x_ptr + row * 32 + index)
        count = tl.full((4, 8), 1, tl.int32)
        value, count = tl.associative_scan((x, count), 0,
                                           _tuple_sum_and_count_combine,
                                           reverse=REVERSE)
        tl.store(value_ptr + row * 32 + index, value)
        tl.store(count_ptr + row * 32 + index, count)


@triton.jit
def _rankn_tuple_scan_param_loop_kernel(x_ptr, value_ptr, count_ptr, rows,
                                        SHAPE: tl.constexpr,
                                        BLOCK: tl.constexpr,
                                        AXIS: tl.constexpr,
                                        REVERSE: tl.constexpr,
                                        START: tl.constexpr = 0,
                                        STEP: tl.constexpr = 1):
    index = tl.reshape(tl.arange(0, BLOCK), SHAPE)
    base = tl.program_id(0) * rows * BLOCK
    for row in range(START, rows, STEP):
        x = tl.load(x_ptr + base + row * BLOCK + index)
        count = tl.full(SHAPE, 1, tl.int32)
        value, count = tl.associative_scan((x, count), AXIS,
                                           _tuple_sum_and_count_combine,
                                           reverse=REVERSE)
        tl.store(value_ptr + base + row * BLOCK + index, value)
        tl.store(count_ptr + base + row * BLOCK + index, count)


@triton.jit
def _rankn_tuple_ordered_scan_param_loop_kernel(x_ptr, value_ptr, index_ptr,
                                                rows, SHAPE: tl.constexpr,
                                                BLOCK: tl.constexpr,
                                                AXIS: tl.constexpr,
                                                REVERSE: tl.constexpr):
    index = tl.reshape(tl.arange(0, BLOCK), SHAPE)
    base = tl.program_id(0) * rows * BLOCK
    for row in range(rows):
        offset = base + row * BLOCK + index
        x = tl.load(x_ptr + offset)
        value, found_index = tl.associative_scan(
            (x, index), AXIS, _tuple_take_right_nonzero_combine,
            reverse=REVERSE)
        tl.store(value_ptr + offset, value)
        tl.store(index_ptr + offset, found_index)


@triton.jit
def _rankn_tuple_recurrent_loop_kernel(state_ptr, count_ptr, iterations,
                                       SHAPE: tl.constexpr,
                                       BLOCK: tl.constexpr,
                                       AXIS: tl.constexpr):
    index = tl.reshape(tl.arange(0, BLOCK), SHAPE)
    offset = tl.program_id(0) * BLOCK + index
    for _ in range(iterations):
        x = tl.load(state_ptr + offset)
        count = tl.full(SHAPE, 1, tl.int32)
        value, count = tl.associative_scan((x, count), AXIS,
                                           _tuple_sum_and_count_combine)
        tl.store(state_ptr + offset, value)
        tl.store(count_ptr + offset, count)


@triton.jit
def _rankn_tuple_masked_loop_kernel(x_ptr, value_ptr, count_ptr, rows,
                                    n_elements, SHAPE: tl.constexpr,
                                    BLOCK: tl.constexpr,
                                    AXIS: tl.constexpr):
    index = tl.reshape(tl.arange(0, BLOCK), SHAPE)
    base = tl.program_id(0) * rows * BLOCK
    for row in range(rows):
        offset = base + row * BLOCK + index
        mask = row * BLOCK + index < n_elements
        x = tl.load(x_ptr + offset, mask=mask, other=0.0)
        count = tl.full(SHAPE, 1, tl.int32)
        value, count = tl.associative_scan((x, count), AXIS,
                                           _tuple_sum_and_count_combine)
        tl.store(value_ptr + offset, value, mask=mask)
        tl.store(count_ptr + offset, count, mask=mask)


@triton.jit
def _rankn_tuple_chained_loop_kernel(x_ptr, value_ptr, count_ptr, rows,
                                     SHAPE: tl.constexpr,
                                     BLOCK: tl.constexpr,
                                     AXIS: tl.constexpr):
    index = tl.reshape(tl.arange(0, BLOCK), SHAPE)
    base = tl.program_id(0) * rows * BLOCK
    for row in range(rows):
        offset = base + row * BLOCK + index
        x = tl.load(x_ptr + offset)
        count = tl.full(SHAPE, 1, tl.int32)
        value, count = tl.associative_scan((x, count), AXIS,
                                           _tuple_sum_and_count_combine)
        value, count = tl.associative_scan((value, count), AXIS,
                                           _tuple_sum_and_count_combine)
        tl.store(value_ptr + offset, value)
        tl.store(count_ptr + offset, count)


@triton.jit
def _rankn_tuple_scan_carried_loop_kernel(x_ptr, value_ptr, count_ptr, rows):
    index = tl.reshape(tl.arange(0, 128), (8, 16))
    value = tl.full((8, 16), 0.0, tl.float32)
    count = tl.full((8, 16), 0, tl.int32)
    for row in range(rows):
        x = tl.load(x_ptr + row * 128 + index)
        value = value + x
        value, count = tl.associative_scan((value, count + 1), 1,
                                           _tuple_sum_and_count_combine)
        tl.store(value_ptr + row * 128 + index, value)
        tl.store(count_ptr + row * 128 + index, count)


@triton.jit
def _rankn_tuple_scan_nested_loop_kernel(x_ptr, value_ptr, count_ptr, rows):
    index = tl.reshape(tl.arange(0, 128), (8, 16))
    for row in range(rows):
        for _ in range(rows):
            x = tl.load(x_ptr + row * 128 + index)
            count = tl.full((8, 16), 1, tl.int32)
            value, count = tl.associative_scan((x, count), 1,
                                               _tuple_sum_and_count_combine)
            tl.store(value_ptr + row * 128 + index, value)
            tl.store(count_ptr + row * 128 + index, count)


@triton.jit
def _rankn_tuple_scan_snapshot_loop_kernel(x_ptr, value_ptr, count_ptr, rows):
    index = tl.reshape(tl.arange(0, 128), (8, 16))
    snapshot = tl.load(x_ptr + index)
    for row in range(rows):
        count = tl.full((8, 16), 1, tl.int32) + row
        value, count = tl.associative_scan((snapshot, count), 1,
                                           _tuple_sum_and_count_combine)
        tl.store(value_ptr + row * 128 + index, value)
        tl.store(count_ptr + row * 128 + index, count)


@triton.jit
def _rankn_tuple_scan_hoisted_snapshot_loop_kernel(x_ptr, value_ptr, count_ptr, rows):
    index = tl.reshape(tl.arange(0, 128), (8, 16))
    snapshot = tl.load(x_ptr + index)
    for row in range(rows):
        count = tl.full((8, 16), 1, tl.int32)
        value, count = tl.associative_scan((snapshot, count), 1,
                                           _tuple_sum_and_count_combine)
        tl.store(value_ptr + row * 128 + index, value)
        tl.store(count_ptr + row * 128 + index, count)


@triton.jit
def _rankn_tuple_scan_scalar_carry_loop_kernel(x_ptr, value_ptr, count_ptr, rows):
    index = tl.reshape(tl.arange(0, 128), (8, 16))
    total = 0.0
    for row in range(rows):
        x = tl.load(x_ptr + row * 128 + index) + total
        count = tl.full((8, 16), 1, tl.int32)
        value, count = tl.associative_scan((x, count), 1,
                                           _tuple_sum_and_count_combine)
        tl.store(value_ptr + row * 128 + index, value)
        tl.store(count_ptr + row * 128 + index, count)
        total = total + 1.0


@triton.jit
def _rankn_tuple_scan_dot_carry_loop_kernel(x_ptr, value_ptr, count_ptr, rows):
    index = tl.reshape(tl.arange(0, 256), (16, 16))
    acc = tl.zeros((16, 16), tl.float32)
    for row in range(rows):
        x = tl.load(x_ptr + row * 256 + index)
        acc = tl.dot(x, x, acc)
        tl.store(value_ptr + row * 256 + index, tl.cumsum(acc, 1))


# A multi-band scan inside a runtime loop runs the body as register-band
# phases; a tile the loop carries has no register to live in between them and
# goes through a threadgroup slot instead.
@triton.jit
def _scan_loop_accumulate_kernel(x_ptr, out_ptr, trips, R: tl.constexpr,
                                 C: tl.constexpr, AXIS: tl.constexpr):
    off = tl.arange(0, R)[:, None] * C + tl.arange(0, C)[None, :]
    base = tl.load(x_ptr + off)
    acc = base * 0.0
    for t in range(trips):
        x = tl.load(x_ptr + t * R * C + off)
        acc += tl.cumsum(x, AXIS)
    tl.store(out_ptr + off, acc + base)


@mps_only
@pytest.mark.parametrize("trips", [0, 3])
@pytest.mark.parametrize("shape,axis,num_warps", [
    ((4, 32), 0, 1), ((4, 32), 1, 1), ((16, 16), 0, 4), ((32, 32), 1, 4)])
def test_scan_in_loop_carries_accumulator(trips, shape, axis, num_warps):
    rows, cols = shape
    x = (torch.arange(max(trips, 1) * rows * cols) % 7 - 3).float().reshape(
        max(trips, 1), rows, cols)
    out = torch.zeros(rows, cols, device="mps")
    _scan_loop_accumulate_kernel[(1,)](x.to("mps"), out, trips, rows, cols, axis,
                                       num_warps=num_warps)
    torch.mps.synchronize()
    expected = x[:trips].cumsum(axis + 1).sum(0) + x[0]
    torch.testing.assert_close(out.cpu(), expected, atol=0, rtol=0)


@triton.jit
def _scan_loop_recurrence_kernel(x_ptr, out_ptr, trips, R: tl.constexpr,
                                 C: tl.constexpr, AXIS: tl.constexpr):
    off = tl.arange(0, R)[:, None] * C + tl.arange(0, C)[None, :]
    s = tl.zeros((R, C), tl.int32)
    for t in range(trips):
        s = tl.cumsum(s + tl.load(x_ptr + t * R * C + off), AXIS)
        tl.store(out_ptr + t * R * C + off, s)


@mps_only
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("num_warps", [1, 4])
def test_scan_in_loop_reads_its_carried_result(axis, num_warps):
    trips, rows, cols = 3, 8, 64
    x = (torch.arange(trips * rows * cols) % 5 - 2).int().reshape(trips, rows, cols)
    out = torch.zeros_like(x, device="mps")
    _scan_loop_recurrence_kernel[(1,)](x.to("mps"), out, trips, rows, cols, axis,
                                       num_warps=num_warps)
    torch.mps.synchronize()
    s = torch.zeros(rows, cols, dtype=torch.int32)
    for t in range(trips):
        s = (s + x[t]).cumsum(axis, dtype=torch.int32)
        assert torch.equal(out[t].cpu(), s), t


@mps_only
def test_tuple_scan_rankn_carried_loop():
    rows = 5
    x = torch.randn(rows, 8, 16)
    value = torch.empty_like(x, device="mps")
    count = torch.empty(x.shape, dtype=torch.int32, device="mps")
    _rankn_tuple_scan_carried_loop_kernel[(1,)](x.to("mps"), value, count, rows,
                                                num_warps=1)
    torch.mps.synchronize()
    v = torch.zeros(8, 16)
    c = torch.zeros(8, 16, dtype=torch.int32)
    for row in range(rows):
        v = (v + x[row]).cumsum(1)
        c = (c + 1).cumsum(1, dtype=torch.int32)
        torch.testing.assert_close(value[row].cpu(), v, atol=1e-4, rtol=1e-4)
        assert torch.equal(count[row].cpu(), c), row


@mps_only
def test_tuple_scan_rankn_loop_reads_snapshot():
    rows = 4
    x = torch.randn(rows, 8, 16)
    value = torch.empty_like(x, device="mps")
    count = torch.empty(x.shape, dtype=torch.int32, device="mps")
    _rankn_tuple_scan_snapshot_loop_kernel[(1,)](x.to("mps"), value, count, rows,
                                                 num_warps=1)
    torch.mps.synchronize()
    for row in range(rows):
        torch.testing.assert_close(value[row].cpu(), x[0].cumsum(1),
                                   atol=1e-4, rtol=1e-4)
        assert torch.equal(count[row].cpu(),
                           torch.full((8, 16), 1 + row, dtype=torch.int32).cumsum(
                               1, dtype=torch.int32)), row


@mps_only
@pytest.mark.parametrize("reverse", [False, True])
def test_tuple_scan_rankn_inside_user_loop(reverse):
    torch.manual_seed(63)
    x = torch.randn(17, 4, 8)
    value = torch.empty_like(x, device="mps")
    count = torch.empty(x.shape, dtype=torch.int32, device="mps")
    _rankn_tuple_scan_loop_kernel[(1,)](
        x.to("mps"), value, count, x.shape[0], REVERSE=reverse, num_warps=1)
    torch.mps.synchronize()
    expected = (x.flip(1) if reverse else x).cumsum(1)
    expected_count = torch.ones_like(x, dtype=torch.int32).cumsum(1, dtype=torch.int32)
    if reverse:
        expected, expected_count = expected.flip(1), expected_count.flip(1)
    torch.testing.assert_close(value.cpu(), expected, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(count.cpu(), expected_count, atol=0, rtol=0)


@mps_only
@pytest.mark.parametrize("shape,axis,reverse,dtype,num_warps,rows,start,step", [
    ((8, 16), 1, False, torch.float32, 1, 0, 0, 1),
    ((4, 8, 8), 0, True, torch.int32, 4, 17, 2, 3),
    ((4, 4, 8), 2, True, torch.float32, 1, 17, 0, 1),
])
def test_tuple_scan_rankn_nested_loop_sum_and_count_multiband(
        shape, axis, reverse, dtype, num_warps, rows, start, step):
    torch.manual_seed(70)
    programs = 3
    block = int(torch.tensor(shape).prod())
    storage_rows = max(rows, 1)
    if dtype == torch.int32:
        x = torch.randint(-8, 8, (programs, storage_rows, *shape), dtype=dtype)
    else:
        x = torch.randn(programs, storage_rows, *shape)
    value = torch.full(x.shape, -123, dtype=dtype, device="mps")
    count = torch.full(x.shape, -123, dtype=torch.int32, device="mps")
    _rankn_tuple_scan_param_loop_kernel[(programs,)](
        x.to("mps").contiguous(), value, count, rows, SHAPE=shape, BLOCK=block,
        AXIS=axis, REVERSE=reverse, START=start, STEP=step,
        num_warps=num_warps)
    torch.mps.synchronize()

    expected_value = torch.full_like(x, -123)
    expected_count = torch.full(x.shape, -123, dtype=torch.int32)
    for row in range(start, rows, step):
        source = x[:, row].flip(axis + 1) if reverse else x[:, row]
        row_value = source.cumsum(axis + 1, dtype=dtype)
        row_count = torch.ones_like(x[:, row], dtype=torch.int32).cumsum(
            axis + 1, dtype=torch.int32)
        if reverse:
            row_value = row_value.flip(axis + 1)
            row_count = row_count.flip(axis + 1)
        expected_value[:, row] = row_value
        expected_count[:, row] = row_count
    tolerance = 0 if dtype == torch.int32 else 1e-4
    torch.testing.assert_close(value.cpu(), expected_value,
                               atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(count.cpu(), expected_count, atol=0, rtol=0)


@mps_only
def test_tuple_scan_rankn_nested_loop_noncommutative_order_multiband():
    torch.manual_seed(71)
    programs, rows, shape, axis, block = 3, 2, (8, 32), 1, 256
    x = torch.randint(0, 3, (programs, rows, *shape)).float()
    x[1, 0].zero_()
    value = torch.empty_like(x, device="mps")
    found_index = torch.empty(x.shape, dtype=torch.int32, device="mps")
    _rankn_tuple_ordered_scan_param_loop_kernel[(programs,)](
        x.to("mps").contiguous(), value, found_index, rows, SHAPE=shape,
        BLOCK=block, AXIS=axis, REVERSE=True, num_warps=4)
    torch.mps.synchronize()

    expected_value = x.clone()
    base_index = torch.arange(block, dtype=torch.int32).reshape(shape)
    expected_index = base_index.expand(programs, rows, *shape).clone()
    positions = list(range(shape[axis]))
    positions.reverse()
    dim = axis + 2
    for previous, current in zip(positions, positions[1:]):
        keep = x.select(dim, current) != 0
        expected_value.select(dim, current).copy_(torch.where(
            keep, x.select(dim, current),
            expected_value.select(dim, previous)))
        expected_index.select(dim, current).copy_(torch.where(
            keep, expected_index.select(dim, current),
            expected_index.select(dim, previous)))
    torch.testing.assert_close(value.cpu(), expected_value, atol=0, rtol=0)
    torch.testing.assert_close(found_index.cpu(), expected_index, atol=0, rtol=0)


@mps_only
@pytest.mark.parametrize("iterations", [0, 1, 17])
def test_tuple_scan_rankn_nested_loop_replays_each_iteration_from_memory(
        iterations):
    torch.manual_seed(72)
    programs, shape, axis, block = 3, (8, 16), 1, 128
    x = torch.randn(programs, *shape)
    state = x.to("mps").contiguous()
    count = torch.full((programs, *shape), -123, dtype=torch.int32, device="mps")
    _rankn_tuple_recurrent_loop_kernel[(programs,)](
        state, count, iterations, SHAPE=shape, BLOCK=block, AXIS=axis,
        num_warps=1)
    torch.mps.synchronize()

    expected = x.clone()
    for _ in range(iterations):
        expected = expected.cumsum(axis + 1)
    torch.testing.assert_close(state.cpu(), expected, atol=1e-4, rtol=1e-4)
    if iterations == 0:
        torch.testing.assert_close(
            count.cpu(), torch.full_like(count.cpu(), -123), atol=0, rtol=0)
    else:
        expected_count = torch.ones((programs, *shape), dtype=torch.int32)
        expected_count = expected_count.cumsum(axis + 1, dtype=torch.int32)
        torch.testing.assert_close(count.cpu(), expected_count, atol=0, rtol=0)


@mps_only
def test_tuple_scan_rankn_nested_loop_masked_output_multiband():
    torch.manual_seed(73)
    programs, rows, shape, axis, block = 3, 2, (8, 16), 1, 128
    n_elements = rows * block - 7
    x = torch.randn(programs, rows, *shape)
    value = torch.full_like(x, -123, device="mps")
    count = torch.full(x.shape, -123, dtype=torch.int32, device="mps")
    _rankn_tuple_masked_loop_kernel[(programs,)](
        x.to("mps").contiguous(), value, count, rows, n_elements,
        SHAPE=shape, BLOCK=block, AXIS=axis, num_warps=1)
    torch.mps.synchronize()

    expected = x.clone()
    expected.reshape(programs, rows * block)[:, n_elements:] = 0
    expected_value = expected.cumsum(axis + 2)
    expected_count = torch.ones_like(x, dtype=torch.int32).cumsum(
        axis + 2, dtype=torch.int32)
    got_value = value.cpu()
    got_count = count.cpu()
    valid = torch.arange(rows * block).reshape(rows, *shape) < n_elements
    torch.testing.assert_close(got_value[:, valid],
                               expected_value[:, valid], atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(got_count[:, valid],
                               expected_count[:, valid], atol=0, rtol=0)
    torch.testing.assert_close(got_value[:, ~valid],
                               torch.full((programs, 7), -123.0), atol=0, rtol=0)
    torch.testing.assert_close(got_count[:, ~valid],
                               torch.full((programs, 7), -123, dtype=torch.int32),
                               atol=0, rtol=0)


@mps_only
def test_tuple_scan_rankn_nested_loop_chained_tuple_multiband():
    torch.manual_seed(74)
    programs, rows, shape, axis, block = 3, 2, (8, 16), 1, 128
    x = torch.randn(programs, rows, *shape)
    value = torch.empty_like(x, device="mps")
    count = torch.empty(x.shape, dtype=torch.int32, device="mps")
    _rankn_tuple_chained_loop_kernel[(programs,)](
        x.to("mps").contiguous(), value, count, rows, SHAPE=shape,
        BLOCK=block, AXIS=axis, num_warps=1)
    torch.mps.synchronize()
    expected = x.cumsum(axis + 2).cumsum(axis + 2)
    expected_count = torch.ones_like(x, dtype=torch.int32).cumsum(
        axis + 2, dtype=torch.int32).cumsum(axis + 2, dtype=torch.int32)
    torch.testing.assert_close(value.cpu(), expected, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(count.cpu(), expected_count, atol=0, rtol=0)


@pytest.mark.parametrize("shape,axis,num_warps,dtype,reason", [
    ((128, 64), 1, 4, "fp32", "exceeds the threadgroup budget"),
    ((1, 128), 1, 4, "fp32", "scan-axis extent 2..64"),
    ((4, 8), 0, 1, "fp16", "carries f32 and i32 values only"),
])
def test_tuple_scan_rankn_outside_envelope_is_named(shape, axis, num_warps, dtype, reason):
    script = textwrap.dedent(f"""
        import importlib.util
        import triton
        from triton.backends.compiler import GPUTarget
        from triton.compiler import ASTSource
        spec = importlib.util.spec_from_file_location("scan_child", {str(Path(__file__).resolve())!r})
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        src = ASTSource(fn=module._rankn_tuple_scan_kernel,
            signature={{"x_ptr": "*{dtype}", "value_ptr": "*{dtype}", "other_ptr": "*i32"}},
            constexprs={{"SHAPE": {shape!r}, "BLOCK": {int(torch.tensor(shape).prod())},
                         "AXIS": {axis}, "REVERSE": False, "ORDERED": False}})
        triton.compile(src, target=GPUTarget("metal", 80, 32), options={{"num_warps": {num_warps}}})
    """)
    result = _run_child(script)
    assert result.returncode > 0, result.stdout + result.stderr
    assert reason in result.stdout + result.stderr


@pytest.mark.parametrize("kernel,reason", [
    ("_rankn_tuple_scan_nested_loop_kernel",
     "nested scan requires one top-level for loop directly around it"),
    ("_rankn_tuple_scan_scalar_carry_loop_kernel",
     "nested scan carries only tensors between iterations"),
    ("_rankn_tuple_scan_dot_carry_loop_kernel",
     "nested scan carrying a tile cannot share its loop with a tl.dot"),
    ("_rankn_tuple_scan_hoisted_snapshot_loop_kernel",
     "staged scan cannot partition surrounding loops"),
])
def test_tuple_scan_rankn_nested_invalid_loop_is_named(kernel, reason):
    script = textwrap.dedent(f"""
        import importlib.util
        import triton
        from triton.backends.compiler import GPUTarget
        from triton.compiler import ASTSource
        spec = importlib.util.spec_from_file_location("scan_child", {str(Path(__file__).resolve())!r})
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        src = ASTSource(fn=getattr(module, {kernel!r}),
            signature={{"x_ptr": "*fp32", "value_ptr": "*fp32", "count_ptr": "*i32", "rows": "i32"}},
            constexprs={{}})
        triton.compile(src, target=GPUTarget("metal", 80, 32), options={{"num_warps": 1}})
    """)
    result = _run_child(script)
    assert result.returncode > 0, result.stdout + result.stderr
    assert reason in result.stdout + result.stderr


@triton.jit
def _rankn_tuple_scan_budget_kernel(x_ptr, value_ptr, count_ptr, REPEATS: tl.constexpr, MASKED: tl.constexpr):
    index = tl.reshape(tl.arange(0, 32), (4, 8))
    value = tl.load(x_ptr + index)
    count = tl.full((4, 8), 1, tl.int32)
    for _ in tl.static_range(REPEATS):
        value, count = tl.associative_scan((value, count), 0,
                                           _tuple_sum_and_count_combine)
    if MASKED:
        tl.store(value_ptr + index, value, index < 31)
        tl.store(count_ptr + index, count, index < 31)
    else:
        tl.store(value_ptr + index, value)
        tl.store(count_ptr + index, count)


@pytest.mark.parametrize("repeats,masked", [(129, False), (128, True)])
def test_tuple_scan_rankn_combined_scratch_budget_is_named(repeats, masked):
    script = textwrap.dedent(f"""
        import importlib.util
        import triton
        from triton.backends.compiler import GPUTarget
        from triton.compiler import ASTSource
        spec = importlib.util.spec_from_file_location("scan_child", {str(Path(__file__).resolve())!r})
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        src = ASTSource(fn=module._rankn_tuple_scan_budget_kernel,
            signature={{"x_ptr": "*fp32", "value_ptr": "*fp32", "count_ptr": "*i32"}},
            constexprs={{"REPEATS": {repeats}, "MASKED": {masked}}})
        triton.compile(src, target=GPUTarget("metal", 80, 32), options={{"num_warps": 1}})
    """)
    result = _run_child(script)
    assert result.returncode > 0, result.stdout + result.stderr
    assert "multidimensional tuple scan scratch needs 33024 bytes" in result.stdout + result.stderr
    assert "exceeding the 32768-byte budget" in result.stdout + result.stderr


@triton.jit
def _single_add_combine(a_value, b_value):
    return a_value + b_value


@triton.jit
def _rankn_tuple_chained_kernel(x_ptr, value_ptr, count_ptr,
                                SHAPE: tl.constexpr, BLOCK: tl.constexpr,
                                AXIS: tl.constexpr):
    index = tl.reshape(tl.arange(0, BLOCK), SHAPE)
    offset = tl.program_id(0) * BLOCK + index
    x = tl.load(x_ptr + offset)
    count = tl.full(SHAPE, 1, tl.int32)
    value, count = tl.associative_scan((x, count), AXIS,
                                       _tuple_sum_and_count_combine)
    value, count = tl.associative_scan((value, count), AXIS,
                                       _tuple_sum_and_count_combine)
    tl.store(value_ptr + offset, value)
    tl.store(count_ptr + offset, count)


@mps_only
@pytest.mark.parametrize("shape,axis,num_warps", [
    ((8, 16), 1, 1),
    ((64, 8), 0, 4),
])
def test_tuple_scan_rankn_chained_multiband_carries_both_components(
        shape, axis, num_warps):
    torch.manual_seed(65)
    rows, block = 2, int(torch.tensor(shape).prod())
    x = torch.randn(rows, *shape)
    value = torch.empty_like(x, device="mps")
    count = torch.empty(x.shape, dtype=torch.int32, device="mps")
    _rankn_tuple_chained_kernel[(rows,)](
        x.to("mps"), value, count, SHAPE=shape, BLOCK=block, AXIS=axis,
        num_warps=num_warps)
    torch.mps.synchronize()
    dim = axis + 1
    expected = x.cumsum(dim).cumsum(dim)
    expected_count = torch.ones_like(x, dtype=torch.int32)
    expected_count = expected_count.cumsum(dim, dtype=torch.int32).cumsum(
        dim, dtype=torch.int32)
    torch.testing.assert_close(value.cpu(), expected, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(count.cpu(), expected_count, atol=0, rtol=0)


@triton.jit
def _rankn_tuple_and_single_kernel(x_ptr, tuple_ptr, single_ptr, count_ptr,
                                   SHAPE: tl.constexpr, BLOCK: tl.constexpr,
                                   AXIS: tl.constexpr):
    index = tl.reshape(tl.arange(0, BLOCK), SHAPE)
    offset = tl.program_id(0) * BLOCK + index
    x = tl.load(x_ptr + offset)
    count = tl.full(SHAPE, 1, tl.int32)
    tuple_value, count = tl.associative_scan((x, count), AXIS,
                                             _tuple_sum_and_count_combine)
    single_value = tl.associative_scan(x, AXIS, _single_add_combine)
    tl.store(tuple_ptr + offset, tuple_value)
    tl.store(single_ptr + offset, single_value)
    tl.store(count_ptr + offset, count)


@mps_only
@pytest.mark.parametrize("shape,axis,num_warps", [
    ((8, 16), 0, 1),
    ((64, 8), 1, 4),
])
def test_tuple_scan_rankn_mixed_tuple_and_single_multiband(
        shape, axis, num_warps):
    torch.manual_seed(66)
    rows, block = 2, int(torch.tensor(shape).prod())
    x = torch.randn(rows, *shape)
    tuple_value = torch.empty_like(x, device="mps")
    single_value = torch.empty_like(x, device="mps")
    count = torch.empty(x.shape, dtype=torch.int32, device="mps")
    _rankn_tuple_and_single_kernel[(rows,)](
        x.to("mps"), tuple_value, single_value, count, SHAPE=shape,
        BLOCK=block, AXIS=axis, num_warps=num_warps)
    torch.mps.synchronize()
    dim = axis + 1
    expected = x.cumsum(dim)
    expected_count = torch.ones_like(x, dtype=torch.int32).cumsum(
        dim, dtype=torch.int32)
    torch.testing.assert_close(tuple_value.cpu(), expected, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(single_value.cpu(), expected, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(count.cpu(), expected_count, atol=0, rtol=0)


@triton.jit
def _rankn_tuple_three_same_combine(a_x, a_y, a_z, b_x, b_y, b_z):
    return a_x + b_x, a_y + b_y, a_z + b_z


@triton.jit
def _rankn_tuple_three_same_kernel(x_ptr, a_ptr, b_ptr, c_ptr,
                                   SHAPE: tl.constexpr, BLOCK: tl.constexpr,
                                   AXIS: tl.constexpr, DUPLICATE: tl.constexpr = True,
                                   REPEATS: tl.constexpr = 1):
    index = tl.reshape(tl.arange(0, BLOCK), SHAPE)
    offset = tl.program_id(0) * BLOCK + index
    x = tl.load(x_ptr + offset)
    a = x
    if DUPLICATE:
        b, c = x, x
    else:
        b, c = x + 1.0, x * 2.0
    for _ in tl.static_range(REPEATS):
        a, b, c = tl.associative_scan((a, b, c), AXIS,
                                      _rankn_tuple_three_same_combine)
    tl.store(a_ptr + offset, a)
    tl.store(b_ptr + offset, b)
    tl.store(c_ptr + offset, c)


@mps_only
@pytest.mark.parametrize("shape,axis,num_warps", [
    ((8, 16), 1, 1),
    ((64, 8), 0, 4),
])
@pytest.mark.parametrize("duplicate", [False, True])
def test_tuple_scan_rankn_three_same_type_components_multiband(
        shape, axis, num_warps, duplicate):
    torch.manual_seed(67)
    rows, block = 2, int(torch.tensor(shape).prod())
    x = torch.randn(rows, *shape)
    a = torch.empty_like(x, device="mps")
    b = torch.empty_like(x, device="mps")
    c = torch.empty_like(x, device="mps")
    _rankn_tuple_three_same_kernel[(rows,)](
        x.to("mps"), a, b, c, SHAPE=shape, BLOCK=block, AXIS=axis,
        DUPLICATE=duplicate, num_warps=num_warps)
    torch.mps.synchronize()
    expected = x.cumsum(axis + 1)
    torch.testing.assert_close(a.cpu(), expected, atol=1e-4, rtol=1e-4)
    expected_b = expected if duplicate else (x + 1).cumsum(axis + 1)
    expected_c = expected if duplicate else (x * 2).cumsum(axis + 1)
    torch.testing.assert_close(b.cpu(), expected_b, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(c.cpu(), expected_c, atol=1e-4, rtol=1e-4)


@triton.jit
def _rankn_tuple_masked_kernel(x_ptr, value_ptr, count_ptr, n_elements,
                               SHAPE: tl.constexpr, BLOCK: tl.constexpr,
                               AXIS: tl.constexpr):
    index = tl.reshape(tl.arange(0, BLOCK), SHAPE)
    mask = index < n_elements
    x = tl.load(x_ptr + index, mask=mask, other=0.0)
    count = tl.full(SHAPE, 1, tl.int32)
    value, count = tl.associative_scan((x, count), AXIS,
                                       _tuple_sum_and_count_combine)
    tl.store(value_ptr + index, value, mask=mask)
    tl.store(count_ptr + index, count, mask=mask)


@mps_only
def test_tuple_scan_rankn_masked_values_and_stores_multiband():
    torch.manual_seed(68)
    shape, axis, num_warps = (8, 16), 1, 1
    block = int(torch.tensor(shape).prod())
    n_elements = block - 5
    x = torch.randn(block).reshape(shape)
    value = torch.full(shape, -123.0, device="mps")
    count = torch.full(shape, -123, dtype=torch.int32, device="mps")
    _rankn_tuple_masked_kernel[(1,)](
        x.to("mps").contiguous(), value, count, n_elements, SHAPE=shape,
        BLOCK=block, AXIS=axis, num_warps=num_warps)
    torch.mps.synchronize()
    expected_x = x.flatten()
    expected_x[n_elements:] = 0
    expected_x = expected_x.reshape(shape)
    expected_value = expected_x.cumsum(axis)
    expected_count = torch.ones(shape, dtype=torch.int32).cumsum(
        axis, dtype=torch.int32)
    got_value = value.cpu().flatten()
    got_count = count.cpu().flatten()
    torch.testing.assert_close(got_value[:n_elements],
                               expected_value.flatten()[:n_elements],
                               atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(got_count[:n_elements],
                               expected_count.flatten()[:n_elements],
                               atol=0, rtol=0)
    torch.testing.assert_close(got_value[n_elements:],
                               torch.full((5,), -123.0), atol=0, rtol=0)
    torch.testing.assert_close(got_count[n_elements:],
                               torch.full((5,), -123, dtype=torch.int32),
                               atol=0, rtol=0)


@triton.jit
def _rankn_tuple_three_combine(a_max, a_sum, a_count, b_max, b_sum, b_count):
    return tl.maximum(a_max, b_max), a_sum + b_sum, a_count + b_count


@triton.jit
def _rankn_tuple_three_kernel(x_ptr, max_ptr, sum_ptr, count_ptr,
                             AXIS: tl.constexpr, REVERSE: tl.constexpr):
    row = tl.arange(0, 4)[:, None]
    col = tl.arange(0, 8)[None, :]
    offset = tl.program_id(0) * 32 + row * 8 + col
    x = tl.load(x_ptr + offset)
    count = tl.full((4, 8), 1, tl.int32)
    maximum, total, count = tl.associative_scan((x, x, count), AXIS,
                                               _rankn_tuple_three_combine,
                                               reverse=REVERSE)
    tl.store(max_ptr + offset, maximum)
    tl.store(sum_ptr + offset, total)
    tl.store(count_ptr + offset, count)


@mps_only
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("num_warps", [1, 4])
def test_tuple_scan_rankn_three_values_from_axis_ranges(axis, reverse, num_warps):
    torch.manual_seed(64)
    x = torch.randn(3, 4, 8)
    maximum = torch.empty_like(x, device="mps")
    total = torch.empty_like(x, device="mps")
    count = torch.empty(x.shape, dtype=torch.int32, device="mps")
    _rankn_tuple_three_kernel[(3,)](
        x.to("mps"), maximum, total, count, AXIS=axis, REVERSE=reverse, num_warps=num_warps)
    torch.mps.synchronize()
    dim = axis + 1
    source = x.flip(dim) if reverse else x
    expected_max = source.cummax(dim).values
    expected_sum = source.cumsum(dim)
    expected_count = torch.ones_like(x, dtype=torch.int32).cumsum(dim, dtype=torch.int32)
    if reverse:
        expected_max, expected_sum = expected_max.flip(dim), expected_sum.flip(dim)
        expected_count = expected_count.flip(dim)
    torch.testing.assert_close(maximum.cpu(), expected_max, atol=0, rtol=0)
    torch.testing.assert_close(total.cpu(), expected_sum, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(count.cpu(), expected_count, atol=0, rtol=0)


@triton.jit
def _rankn_tuple_snapshot_kernel(x_ptr, value_ptr, count_ptr):
    index = tl.reshape(tl.arange(0, 128), (8, 16))
    x = tl.load(x_ptr + index)
    count = tl.full((8, 16), 1, tl.int32)
    value, count = tl.associative_scan((x + 1, count), 1, _tuple_sum_and_count_combine)
    tl.store(x_ptr + index, x + 5)
    tl.store(value_ptr + index, value + x)
    tl.store(count_ptr + index, count)


@pytest.mark.parametrize("budget", [False, True])
def test_tuple_scan_rankn_multiband_snapshot_and_group_budget_are_named(budget):
    if budget:
        kernel = "_rankn_tuple_three_same_kernel"
        signature = {"x_ptr": "*fp32", "a_ptr": "*fp32", "b_ptr": "*fp32", "c_ptr": "*fp32"}
        constants = {"SHAPE": (32, 64), "BLOCK": 2048, "AXIS": 1, "DUPLICATE": False, "REPEATS": 2}
        reason = "scratch needs 49152 bytes, exceeding the 32768-byte budget"
    else:
        kernel = "_rankn_tuple_snapshot_kernel"
        signature = {"x_ptr": "*i32", "value_ptr": "*i32", "count_ptr": "*i32"}
        constants = {}
        reason = "staged scan cannot safely replay a cross-phase value"
    script = textwrap.dedent(f"""
        import importlib.util
        import triton
        from triton.backends.compiler import GPUTarget
        from triton.compiler import ASTSource
        spec = importlib.util.spec_from_file_location("scan_child", {str(Path(__file__).resolve())!r})
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        src = ASTSource(fn=getattr(module, {kernel!r}), signature={signature!r}, constexprs={constants!r})
        triton.compile(src, target=GPUTarget("metal", 80, 32), options={{"num_warps": 1}})
    """)
    result = _run_child(script)
    assert result.returncode > 0, result.stdout + result.stderr
    assert reason in result.stdout + result.stderr


@triton.jit
def _rankn_tuple_rank1_budget_kernel(x_ptr, value_ptr, count_ptr):
    small = tl.arange(0, 32)
    x_small = tl.load(x_ptr + small)
    a, b = tl.associative_scan((x_small, tl.full((32,), 1, tl.int32)), 0,
                               _tuple_sum_and_count_combine)
    tl.store(value_ptr + small, a)
    tl.store(count_ptr + small, b)
    index = tl.reshape(tl.arange(0, 4096), (64, 64))
    x = tl.load(x_ptr + index)
    a, b = tl.associative_scan((x, tl.full((64, 64), 1, tl.int32)), 1,
                               _tuple_sum_and_count_combine)
    tl.store(value_ptr + index, a)
    tl.store(count_ptr + index, b)


def test_tuple_scan_rankn_multiband_counts_rank1_scratch():
    script = textwrap.dedent(f"""
        import importlib.util
        import triton
        from triton.backends.compiler import GPUTarget
        from triton.compiler import ASTSource
        spec = importlib.util.spec_from_file_location("scan_child", {str(Path(__file__).resolve())!r})
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        src = ASTSource(fn=module._rankn_tuple_rank1_budget_kernel,
            signature={{"x_ptr": "*fp32", "value_ptr": "*fp32", "count_ptr": "*i32"}})
        triton.compile(src, target=GPUTarget("metal", 80, 32), options={{"num_warps": 1}})
    """)
    result = _run_child(script)
    assert result.returncode > 0, result.stdout + result.stderr
    assert "scratch needs 33024 bytes, exceeding the 32768-byte budget" in result.stdout + result.stderr


@triton.jit
def _rankn_tuple_mixed_shape_kernel(x_ptr, tuple_ptr, count_ptr, single_ptr, REVERSE: tl.constexpr):
    row = tl.arange(0, 8)
    small_col = tl.arange(0, 4)
    small_offset = row[:, None] * 16 + small_col[None, :]
    x = tl.load(x_ptr + small_offset)
    value, count = tl.associative_scan((x, tl.full((8, 4), 1, tl.int32)), 0,
                                       _tuple_sum_and_count_combine, reverse=REVERSE)
    tl.store(tuple_ptr + row[:, None] * 4 + small_col[None, :], value)
    tl.store(count_ptr + row[:, None] * 4 + small_col[None, :], count)
    col = tl.arange(0, 16)
    offset = row[:, None] * 16 + col[None, :]
    large = tl.load(x_ptr + offset)
    tl.store(single_ptr + offset, tl.cumsum(large, 0, reverse=REVERSE))


@mps_only
@pytest.mark.parametrize("reverse", [False, True])
def test_tuple_scan_rankn_mixed_shapes_share_row_range(reverse):
    torch.manual_seed(69)
    x = torch.randn(8, 16)
    value = torch.empty((8, 4), device="mps")
    count = torch.empty((8, 4), dtype=torch.int32, device="mps")
    single = torch.empty_like(x, device="mps")
    _rankn_tuple_mixed_shape_kernel[(1,)](x.to("mps"), value, count, single, reverse, num_warps=1)
    torch.mps.synchronize()
    source = x.flip(0) if reverse else x
    expected = source.cumsum(0)
    expected_count = torch.ones((8, 4), dtype=torch.int32).cumsum(0, dtype=torch.int32)
    if reverse:
        expected, expected_count = expected.flip(0), expected_count.flip(0)
    torch.testing.assert_close(value.cpu(), expected[:, :4], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(single.cpu(), expected, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(count.cpu(), expected_count, atol=0, rtol=0)


@triton.jit
def _rankn_tuple_nested_loop_budget_kernel(x_ptr, value_ptr, count_ptr, rows,
                                           SHAPE: tl.constexpr,
                                           BLOCK: tl.constexpr,
                                           AXIS: tl.constexpr,
                                           REPEATS: tl.constexpr):
    index = tl.reshape(tl.arange(0, BLOCK), SHAPE)
    for row in range(rows):
        offset = row * BLOCK + index
        value = tl.load(x_ptr + offset)
        count = tl.full(SHAPE, 1, tl.int32) + row
        for _ in tl.static_range(REPEATS):
            value, count = tl.associative_scan((value, count), AXIS,
                                               _tuple_sum_and_count_combine)
        tl.store(value_ptr + offset, value)
        tl.store(count_ptr + offset, count)


@triton.jit
def _rankn_tuple_nested_loop_outer_store_kernel(x_ptr, value_ptr, count_ptr,
                                                rows):
    index = tl.reshape(tl.arange(0, 2048), (32, 64))
    tl.store(value_ptr + index, 0.0)
    for row in range(rows):
        offset = row * 2048 + index
        value = tl.load(x_ptr + offset)
        count = tl.full((32, 64), 1, tl.int32)
        value, count = tl.associative_scan((value, count), 1,
                                           _tuple_sum_and_count_combine)
        tl.store(value_ptr + offset, value)
        tl.store(count_ptr + offset, count)


@triton.jit
def _rankn_tuple_nested_loop_internal_if_kernel(x_ptr, value_ptr, count_ptr,
                                                rows):
    index = tl.reshape(tl.arange(0, 2048), (32, 64))
    for row in range(rows):
        offset = row * 2048 + index
        value = tl.load(x_ptr + offset)
        if row == 0:
            tl.store(value_ptr + offset, value)
        count = tl.full((32, 64), 1, tl.int32)
        value, count = tl.associative_scan((value, count), 1,
                                           _tuple_sum_and_count_combine)
        tl.store(value_ptr + offset, value)
        tl.store(count_ptr + offset, count)


def _compile_rankn_nested_negative_in_child(kernel, signature, constexprs,
                                            *, num_warps=1):
    script = textwrap.dedent(f"""
        import importlib.util
        import triton
        from triton.backends.compiler import GPUTarget
        from triton.compiler import ASTSource
        spec = importlib.util.spec_from_file_location("scan_child", {str(Path(__file__).resolve())!r})
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        src = ASTSource(fn=getattr(module, {kernel!r}),
            signature={signature!r}, constexprs={constexprs!r})
        triton.compile(src, target=GPUTarget("metal", 80, 32),
                       options={{"num_warps": {num_warps}}})
    """)
    return _run_child(script)


@pytest.mark.parametrize("shape,axis,repeats,num_warps", [
    ((64, 64), 1, 2, 1),
    ((32, 64), 1, 3, 1),
    ((64, 32), 0, 3, 4),
])
def test_tuple_scan_rankn_nested_loop_budget_is_named(
        shape, axis, repeats, num_warps):
    block = int(torch.tensor(shape).prod())
    result = _compile_rankn_nested_negative_in_child(
        "_rankn_tuple_nested_loop_budget_kernel",
        {"x_ptr": "*fp32", "value_ptr": "*fp32", "count_ptr": "*i32",
         "rows": "i32"},
        {"SHAPE": shape, "BLOCK": block, "AXIS": axis, "REPEATS": repeats},
        num_warps=num_warps)
    reason = result.stdout + result.stderr
    assert result.returncode > 0, reason
    assert "scratch needs" in reason
    assert "exceeding the 32768-byte budget" in reason


@pytest.mark.parametrize("kernel,reason", [
    ("_rankn_tuple_nested_loop_outer_store_kernel",
     "nested scan cannot replay an external tensor value or move outer effects"),
    ("_rankn_tuple_nested_loop_internal_if_kernel",
     "nested scan requires a flat load/compute/store loop body"),
])
def test_tuple_scan_rankn_nested_loop_body_restrictions_are_named(
        kernel, reason):
    result = _compile_rankn_nested_negative_in_child(
        kernel,
        {"x_ptr": "*fp32", "value_ptr": "*fp32", "count_ptr": "*i32",
         "rows": "i32"},
        {})
    output = result.stdout + result.stderr
    assert result.returncode > 0, output
    assert reason in output


@triton.jit
def _rankn_tuple_scan_unrelated_loop_kernel(x_ptr, value_ptr, count_ptr,
                                            scratch_ptr, rows):
    for row in range(rows):
        tl.store(scratch_ptr + row, row + 1)
    index = tl.reshape(tl.arange(0, 128), (8, 16))
    value = tl.load(x_ptr + index)
    count = tl.full((8, 16), 1, tl.int32)
    value, count = tl.associative_scan((value, count), 1,
                                       _tuple_sum_and_count_combine)
    tl.store(value_ptr + index, value)
    tl.store(count_ptr + index, count)


@mps_only
def test_tuple_scan_rankn_unrelated_runtime_loop_remains_supported():
    x = (torch.arange(128).reshape(8, 16) % 7 - 3).float()
    value = torch.empty_like(x, device="mps")
    count = torch.empty(x.shape, dtype=torch.int32, device="mps")
    scratch = torch.full((17,), -1, dtype=torch.int32, device="mps")
    _rankn_tuple_scan_unrelated_loop_kernel[(1,)](
        x.to("mps"), value, count, scratch, scratch.numel(), num_warps=1)
    torch.mps.synchronize()
    torch.testing.assert_close(value.cpu(), x.cumsum(1), atol=0, rtol=0)
    torch.testing.assert_close(count.cpu(), torch.arange(1, 17, dtype=torch.int32).expand(8, 16),
                               atol=0, rtol=0)
    torch.testing.assert_close(scratch.cpu(), torch.arange(1, 18, dtype=torch.int32), atol=0, rtol=0)


# ---------------------------------------------------------------------------
# Scans beside other staged work in ONE function (`probe_diag_staged.py`).
# ---------------------------------------------------------------------------


@triton.jit
def _scan_bcast_kernel(V, Y, R: tl.constexpr, C: tl.constexpr, ROWS: tl.constexpr):
    r = tl.arange(0, R)
    c = tl.arange(0, C)
    if ROWS:
        s = tl.cumsum(tl.load(V + r), 0)
        out = s[:, None] * 2.0 + c[None, :].to(tl.float32)
    else:
        s = tl.cumsum(tl.load(V + c), 0)
        out = s[None, :] + r[:, None].to(tl.float32)
    tl.store(Y + r[:, None] * C + c[None, :], out)


# A rank-1 scan result holds element `localTid`, and `expand_dims` /
# `broadcast` forward it, so `tl.cumsum(v, 0)[None, :]` added into a tile gave
# thread t the prefix at `t` instead of at its column: every row but the first
# wrong. It now takes the column-reduce republish, indexed by the axis the
# expand did not insert.
@pytest.mark.parametrize("R,C,num_warps,rows", [
    (4, 32, 1, False), (2, 16, 1, False), (16, 64, 4, False), (8, 32, 2, False),
    (32, 4, 1, True), (64, 16, 4, True), (16, 16, 1, True),
])
def test_scan_result_broadcast_into_tile_matches_torch(R, C, num_warps, rows):
    torch.manual_seed(R * 7 + C + num_warps)
    v = torch.randn(R if rows else C)
    y = torch.zeros(R, C, device="mps")
    _scan_bcast_kernel[(1,)](v.to("mps"), y, R, C, rows, num_warps=num_warps)
    torch.mps.synchronize()
    s = v.cumsum(0)
    want = (s[:, None] * 2 + torch.arange(C).float()[None, :]) if rows else \
        (s[None, :] + torch.arange(R).float()[:, None])
    torch.testing.assert_close(y.cpu(), want, atol=1e-4, rtol=1e-4)


@triton.jit
def _wide_scan_bcast_kernel(V, Y, WIDTH: tl.constexpr, ROWS: tl.constexpr,
                            PRODUCT: tl.constexpr, REVERSE: tl.constexpr):
    program = tl.program_id(0)
    index = tl.arange(0, WIDTH)
    other = tl.arange(0, 4)
    value = tl.load(V + program * WIDTH + index)
    if PRODUCT:
        scanned = tl.cumprod(value, 0, reverse=REVERSE)
    else:
        scanned = tl.cumsum(value, 0, reverse=REVERSE)
    # Replay arithmetic at the logical scan index, before expanding the axis.
    scanned = scanned * 2 + 3
    if ROWS:
        out = scanned[:, None] + other[None, :]
        offset = index[:, None] * 4 + other[None, :]
    else:
        out = scanned[None, :] + other[:, None]
        offset = other[:, None] * WIDTH + index[None, :]
    tl.store(Y + program * (4 * WIDTH) + offset, out)


@mps_only
@pytest.mark.parametrize("rows", [False, True])
@pytest.mark.parametrize("width,num_warps,dtype,product,reverse", [
    (64, 1, torch.float32, False, False),
    (64, 2, torch.float32, True, True),
    (64, 4, torch.int32, False, True),
    (128, 1, torch.float32, True, False),
    (128, 2, torch.int32, False, False),
    (128, 4, torch.float32, False, True),
    (256, 1, torch.int32, True, True),
    (256, 2, torch.float32, False, False),
    (256, 4, torch.float32, True, True),
    (64, 1, torch.int32, True, False),
    (128, 2, torch.float32, True, True),
    (256, 4, torch.int32, False, True),
])
def test_scan_result_broadcast_beyond_one_pass_matches_torch(
        width, num_warps, dtype, product, reverse, rows):
    programs = 3
    index = torch.arange(programs * width).reshape(programs, width)
    if product:
        # Products remain bounded; each program has distinct input values.
        values = (1 - 2 * ((index // width + index) % 3 == 0)).to(dtype)
        if dtype == torch.float32:
            values *= 1 + (index % 5 - 2).float() * 0.01
    else:
        values = (index % 11 - 5).to(dtype)
    shape = (programs, width, 4) if rows else (programs, 4, width)
    output = torch.empty(shape, dtype=dtype, device="mps")
    _wide_scan_bcast_kernel[(programs,)](
        values.to("mps"), output, width, rows, product, reverse,
        num_warps=num_warps)
    torch.mps.synchronize()
    source = values.flip(1) if reverse else values
    scanned = (source.cumprod(1) if product else source.cumsum(1)).to(dtype)
    if reverse:
        scanned = scanned.flip(1)
    scanned = scanned * 2 + 3
    other = torch.arange(4, dtype=dtype)
    expected = (scanned[:, :, None] + other[None, None, :]) if rows else \
        (scanned[:, None, :] + other[None, :, None])
    tolerance = 1e-4 if dtype == torch.float32 else 0
    torch.testing.assert_close(output.cpu(), expected, atol=tolerance, rtol=tolerance)


@triton.jit
def _scan_bcast_lifetime_kernel(X, Y, Z, W, steps, WIDTH: tl.constexpr,
                                 ROWS: tl.constexpr, LOOP: tl.constexpr):
    index = tl.arange(0, WIDTH)
    other = tl.arange(0, 4)
    if ROWS:
        offset = index[:, None] * 4 + other[None, :]
    else:
        offset = other[:, None] * WIDTH + index[None, :]
    if LOOP:
        for step in range(steps):
            value = tl.load(X + step * WIDTH + index)
            first = tl.cumsum(value, 0)
            second = tl.cumsum(value * 3 + 1, 0)
            if ROWS:
                second_tile = second[:, None] + other[None, :]
            else:
                second_tile = second[None, :] + other[:, None]
            tl.store(Z + step * (4 * WIDTH) + offset, second_tile)
            third = tl.cumsum(value * 5 - 2, 0)
            # The first prefix stays live while the second and third may share
            # a slot at top level; nested scans must retain conservative pooling.
            if ROWS:
                first_tile = (first * 2 + 3)[:, None] + other[None, :]
                third_tile = third[:, None] + other[None, :]
            else:
                first_tile = (first * 2 + 3)[None, :] + other[:, None]
                third_tile = third[None, :] + other[:, None]
            tl.store(Y + step * (4 * WIDTH) + offset, first_tile)
            tl.store(W + step * (4 * WIDTH) + offset, third_tile)
    else:
        value = tl.load(X + index)
        first = tl.cumsum(value, 0)
        second = tl.cumsum(value * 3 + 1, 0)
        if ROWS:
            second_tile = second[:, None] + other[None, :]
        else:
            second_tile = second[None, :] + other[:, None]
        tl.store(Z + offset, second_tile)
        third = tl.cumsum(value * 5 - 2, 0)
        if ROWS:
            first_tile = (first * 2 + 3)[:, None] + other[None, :]
            third_tile = third[:, None] + other[None, :]
        else:
            first_tile = (first * 2 + 3)[None, :] + other[:, None]
            third_tile = third[None, :] + other[:, None]
        tl.store(Y + offset, first_tile)
        tl.store(W + offset, third_tile)


@mps_only
@pytest.mark.parametrize("rows", [False, True])
@pytest.mark.parametrize("loop", [False, True])
def test_scan_result_broadcast_preserves_live_prefix_buffers(rows, loop):
    width = 128
    steps = 3 if loop else 1
    values = (torch.arange(steps * width).reshape(steps, width) % 13 - 6).float()
    shape = (steps, width, 4) if rows else (steps, 4, width)
    first = torch.empty(shape, device="mps")
    second = torch.empty_like(first)
    third = torch.empty_like(first)
    _scan_bcast_lifetime_kernel[(1,)](
        values.to("mps"), first, second, third, steps, width, rows, loop, num_warps=1)
    torch.mps.synchronize()
    other = torch.arange(4).float()
    for actual, scanned in [(first, values.cumsum(1) * 2 + 3),
                            (second, (values * 3 + 1).cumsum(1)),
                            (third, (values * 5 - 2).cumsum(1))]:
        expected = (scanned[:, :, None] + other[None, None, :]) if rows else \
            (scanned[:, None, :] + other[None, :, None])
        torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)


@triton.jit
def _scan_bcast_delayed_consumer_kernel(X, Y, Z, W, WIDTH: tl.constexpr,
                                        ROWS: tl.constexpr, REDUCE_MODE: tl.constexpr):
    index = tl.arange(0, WIDTH)
    other = tl.arange(0, 4)
    value = tl.load(X + index)
    first = tl.cumsum(value, 0)
    if ROWS:
        first_tile = (first * 2 + 3)[:, None] + other[None, :]
        offset = index[:, None] * 4 + other[None, :]
    else:
        first_tile = (first * 2 + 3)[None, :] + other[:, None]
        offset = other[:, None] * WIDTH + index[None, :]
    second = tl.cumsum(value * 3 + 1, 0)
    if ROWS:
        second_tile = second[:, None] + other[None, :]
    else:
        second_tile = second[None, :] + other[:, None]
    tl.store(Z + offset, second_tile)
    third = tl.cumsum(value * 5 - 2, 0)
    if ROWS:
        third_tile = third[:, None] + other[None, :]
    else:
        third_tile = third[None, :] + other[:, None]
    tl.store(W + offset, third_tile)
    # Plain stores use materialized broadcast values. A later reduction can
    # replay the original expression, extending the first scan's buffer life.
    if REDUCE_MODE == 1:
        if ROWS:
            result = tl.sum(first_tile, 1)
        else:
            result = tl.sum(first_tile, 0)
        tl.store(Y + index, result)
    elif REDUCE_MODE == 2:
        if ROWS:
            result = tl.sum(first_tile, 0)
        else:
            result = tl.sum(first_tile, 1)
        tl.store(Y + other, result)
    else:
        tl.store(Y + offset, first_tile)


@mps_only
@pytest.mark.parametrize("rows", [False, True])
@pytest.mark.parametrize("reduce_mode", [0, 1, 2],
                         ids=["store", "reduce_broadcast_axis", "reduce_scan_axis"])
def test_scan_result_broadcast_delayed_consumer_preserves_prefix(rows, reduce_mode):
    width = 128
    values = (torch.arange(width) % 13 - 6).float()
    shape = (width, 4) if rows else (4, width)
    first_shape = shape if reduce_mode == 0 else (width,) if reduce_mode == 1 else (4,)
    first = torch.empty(first_shape, device="mps")
    second = torch.empty(shape, device="mps")
    third = torch.empty_like(second)
    _scan_bcast_delayed_consumer_kernel[(1,)](
        values.to("mps"), first, second, third, width, rows, reduce_mode, num_warps=1)
    torch.mps.synchronize()
    other = torch.arange(4).float()
    for actual, scanned in [(first, values.cumsum(0) * 2 + 3),
                            (second, (values * 3 + 1).cumsum(0)),
                            (third, (values * 5 - 2).cumsum(0))]:
        expected = (scanned[:, None] + other[None, :]) if rows else \
            (scanned[None, :] + other[:, None])
        if actual is first and reduce_mode:
            axis = (1 if rows else 0) if reduce_mode == 1 else (0 if rows else 1)
            expected = expected.sum(axis)
        torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)


@triton.jit
def _scan_bcast_combined_cone_kernel(X, Y, ROWS: tl.constexpr):
    index = tl.arange(0, 128)
    other = tl.arange(0, 4)
    value = tl.load(X + index)
    first = tl.cumsum(value, 0)
    second = tl.cumsum(value * 3 + 1, 0)
    combined = first * 2 + second + 3
    if ROWS:
        tile = combined[:, None] + other[None, :]
        offset = index[:, None] * 4 + other[None, :]
    else:
        tile = combined[None, :] + other[:, None]
        offset = other[:, None] * 128 + index[None, :]
    tl.store(Y + offset, tile)


@mps_only
@pytest.mark.parametrize("rows", [False, True])
def test_scan_result_broadcast_combined_scan_cone_matches_torch(rows):
    values = (torch.arange(128) % 13 - 6).float()
    output = torch.empty((128, 4) if rows else (4, 128), device="mps")
    _scan_bcast_combined_cone_kernel[(1,)](
        values.to("mps"), output, rows, num_warps=1)
    torch.mps.synchronize()
    combined = values.cumsum(0) * 2 + (values * 3 + 1).cumsum(0) + 3
    other = torch.arange(4).float()
    expected = (combined[:, None] + other[None, :]) if rows else \
        (combined[None, :] + other[:, None])
    torch.testing.assert_close(output.cpu(), expected, atol=0, rtol=0)


@triton.jit
def _scan_bcast_independent_tile_kernel(V, X, Y, Z):
    column = tl.arange(0, 64)
    row = tl.arange(0, 4)
    scanned = tl.cumsum(tl.load(V + column), 0)
    tl.store(Y + row[:, None] * 64 + column[None, :],
             scanned[None, :] + row[:, None])
    other_row = tl.arange(0, 8)
    other_column = tl.arange(0, 128)
    offset = other_row[:, None] * 128 + other_column[None, :]
    tl.store(Z + offset, tl.load(X + offset) * 2 + 1)


@mps_only
def test_scan_result_broadcast_with_independent_tile_matches_torch():
    values = (torch.arange(64) % 11 - 5).float()
    independent = torch.arange(1024).reshape(8, 128).float()
    output = torch.empty((4, 64), device="mps")
    other_output = torch.empty((8, 128), device="mps")
    _scan_bcast_independent_tile_kernel[(1,)](
        values.to("mps"), independent.to("mps"), output, other_output,
        num_warps=1)
    torch.mps.synchronize()
    expected = values.cumsum(0)[None, :] + torch.arange(4).float()[:, None]
    torch.testing.assert_close(output.cpu(), expected, atol=0, rtol=0)
    torch.testing.assert_close(other_output.cpu(), independent * 2 + 1, atol=0, rtol=0)


@triton.jit
def _unsupported_wide_scan_bcast_kernel(V, W, Y, TUPLE: tl.constexpr):
    index = tl.arange(0, 128)
    row = tl.arange(0, 4)
    value = tl.load(V + index)
    if TUPLE:
        scanned, count = tl.associative_scan(
            (value, tl.full((128,), 1, tl.int32)), 0,
            _tuple_sum_and_count_combine)
        scanned = scanned + count
    else:
        scanned = tl.cumsum(value, 0) + tl.load(W + index)
    tl.store(Y + row[:, None] * 128 + index[None, :],
             scanned[None, :] + row[:, None])


@pytest.mark.parametrize("tuple_scan", [False, True])
def test_scan_result_broadcast_unsupported_cone_is_named(tuple_scan):
    result = _compile_rankn_nested_negative_in_child(
        "_unsupported_wide_scan_bcast_kernel",
        {"V": "*fp32", "W": "*fp32", "Y": "*fp32"},
        {"TUPLE": tuple_scan}, num_warps=1)
    output = result.stdout + result.stderr
    assert result.returncode > 0, output
    assert "a rank-1 tl.scan result broadcast into a rank-2 tile" in output


@triton.jit
def _two_scans_kernel(X, V, Y, Z, R: tl.constexpr, C: tl.constexpr,
                      N: tl.constexpr):
    r = tl.arange(0, R)
    c = tl.arange(0, C)
    n = tl.arange(0, N)
    tl.store(Z + n, tl.cumsum(tl.load(V + n), 0))
    x = tl.load(X + r[:, None] * C + c[None, :])
    tl.store(Y + r[:, None] * C + c[None, :], tl.cumsum(x, 1))


def _run_two_scans(R, C, N, w):
    torch.manual_seed(R + C + N + w)
    x = torch.randn(R, C)
    v = torch.randn(N)
    y = torch.zeros(R, C, device="mps")
    z = torch.zeros(N, device="mps")
    _two_scans_kernel[(1,)](x.to("mps"), v.to("mps"), y, z, R, C, N, num_warps=w)
    torch.mps.synchronize()
    return y.cpu(), z.cpu(), x.cumsum(1), v.cumsum(0)


# A rank-1 scan beside a MULTI-BAND staged scan runs inside that scan's band
# loop, with its trip count, and stores to a layout-space index while reading
# its result back thread-major: exact while its own layout is thread-major,
# silently wrong from eight elements per thread. Those are refused by name.
@pytest.mark.parametrize("R,C,N,w", [(4, 32, 128, 1), (16, 16, 128, 1),
                                     (2, 16, 512, 1), (8, 64, 128, 1)])
def test_rank1_scan_beside_multiband_scan_matches_torch(R, C, N, w):
    y, z, want_y, want_z = _run_two_scans(R, C, N, w)
    torch.testing.assert_close(y, want_y, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(z, want_z, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("R,C,N,w", [(4, 32, 256, 1), (16, 16, 256, 1),
                                     (8, 64, 1024, 4)])
def test_rank1_scan_beside_multiband_scan_is_named(capfd, R, C, N, w):
    with pytest.raises(Exception):
        _run_two_scans(R, C, N, w)
    assert "a rank-1 tl.scan whose layout is not thread-major" in \
        capfd.readouterr().err


@triton.jit
def _reduce_beside_scan_kernel(X, W, Y, Z, A: tl.constexpr, B: tl.constexpr,
                               C: tl.constexpr, RANK3: tl.constexpr):
    b = tl.arange(0, B)
    c = tl.arange(0, C)
    o = b[:, None] * C + c[None, :]
    w = tl.load(W + o)
    tl.store(Z + o, tl.cumsum(w, 1))
    if RANK3:
        a = tl.arange(0, A)
        x = tl.load(X + a[:, None, None] * (B * C) + b[None, :, None] * C
                    + c[None, None, :])
        tl.store(Y + o, tl.sum(x, 0))
    else:
        tl.store(Y + c, tl.sum(w, 0))


@triton.jit
def _shared_input_scan_row_reduce_kernel(X, Y, Row, N, R: tl.constexpr,
                                          C: tl.constexpr, SCAN_AXIS: tl.constexpr,
                                          REVERSE: tl.constexpr, MAX: tl.constexpr,
                                          BROADCAST: tl.constexpr, COMPUTED: tl.constexpr = False,
                                          MASKED: tl.constexpr = True, REDUCTION: tl.constexpr = "",
                                          OTHER: tl.constexpr = 0, ROW_MASKED: tl.constexpr = False,
                                          FULL_OUTPUT: tl.constexpr = False, SNAPSHOT: tl.constexpr = False):
    program = tl.program_id(0)
    row = tl.arange(0, R)
    column = tl.arange(0, C)
    offset = program * (R * C) + row[:, None] * C + column[None, :]
    if MASKED:
        x = tl.load(X + offset, offset < N, other=OTHER)
    else:
        x = tl.load(X + offset)
    if COMPUTED:
        x = x * 2 + 3
    scanned = tl.cumsum(x, SCAN_AXIS, reverse=REVERSE)
    if SNAPSHOT:
        tl.store(X + offset, x * 3 + 17, offset < N)
    if REDUCTION == "umin":
        reduced = tl.min(x.to(tl.uint32), 1)
    elif REDUCTION == "umax":
        reduced = tl.max(x.to(tl.uint32), 1)
    elif REDUCTION == "product":
        reduced = tl.reduce(x, 1, _i32_column_product_combine)
    elif REDUCTION == "and":
        reduced = tl.reduce(x, 1, _i32_row_and_combine)
    elif REDUCTION == "or":
        reduced = tl.reduce(x, 1, _i32_row_or_combine)
    elif REDUCTION == "xor":
        reduced = tl.reduce(x, 1, _i32_row_xor_combine)
    elif REDUCTION == "min":
        reduced = tl.min(x, 1)
    elif MAX or REDUCTION == "max":
        reduced = tl.max(x, 1)
    else:
        reduced = tl.sum(x, 1)
    if BROADCAST:
        result = x * 2 - scanned + reduced[:, None]
    else:
        result = scanned
        if ROW_MASKED:
            tl.store(Row + program * R + row, reduced, row < R - 1)
        else:
            tl.store(Row + program * R + row, reduced)
    if FULL_OUTPUT:
        tl.store(Y + offset, result)
    else:
        tl.store(Y + offset, result, offset < N)


@mps_only
@pytest.mark.parametrize("scan_axis", [0, 1])
@pytest.mark.parametrize("broadcast", [False, True], ids=["row_store", "broadcast"])
@pytest.mark.parametrize("rows,columns,num_warps,maximum,reverse,masked", [
    (4, 32, 1, False, False, True),
    (16, 16, 1, True, True, True),
    (8, 64, 1, True, False, False),
    (8, 64, 4, False, True, True),
    (32, 32, 1, False, False, False),
    (32, 32, 4, True, True, True),
])
def test_multiband_scan_shared_input_row_reduce_matches_torch(
        rows, columns, num_warps, maximum, reverse, masked, scan_axis, broadcast):
    programs = 2
    size = programs * rows * columns
    valid = size - columns - 3 if masked else size
    values = (torch.arange(valid) % 17 - 8).float()
    # Include a completely masked final row and a partially masked preceding
    # row. Both scan and reduction consume exactly the same zero-filled input.
    padded = torch.zeros(size)
    padded[:valid] = values
    tile = padded.reshape(programs, rows, columns)
    axis = scan_axis + 1
    source = tile.flip(axis) if reverse else tile
    scanned = source.cumsum(axis)
    if reverse:
        scanned = scanned.flip(axis)
    reduced = tile.amax(2) if maximum else tile.sum(2)
    expected = tile * 2 - scanned + reduced[:, :, None] if broadcast else scanned
    output = torch.full((size,), float("nan"), device="mps")
    row_output = torch.full((programs, rows), float("nan"), device="mps")
    _shared_input_scan_row_reduce_kernel[(programs,)](
        values.to("mps"), output, row_output, valid, rows, columns, scan_axis,
        reverse, maximum, broadcast, num_warps=num_warps)
    torch.mps.synchronize()
    actual = output.cpu()
    torch.testing.assert_close(actual[:valid], expected.flatten()[:valid], atol=0, rtol=0)
    assert torch.isnan(actual[valid:]).all()
    if not broadcast:
        torch.testing.assert_close(row_output.cpu(), reduced, atol=0, rtol=0)


@mps_only
@pytest.mark.parametrize("dtype,computed,scan_axis,maximum,broadcast", [
    (torch.int32, False, 0, False, False),
    (torch.int32, False, 1, False, True),
    (torch.int32, False, 0, True, True),
    (torch.int32, False, 1, True, False),
    (torch.float32, True, 0, False, True),
    (torch.float32, True, 1, True, True),
])
def test_multiband_scan_shared_input_typed_row_reduce_matches_torch(
        dtype, computed, scan_axis, maximum, broadcast):
    programs, rows, columns = 2, 8, 64
    size = programs * rows * columns
    masked = dtype == torch.float32
    valid = size - 7 if masked else size
    values = (torch.arange(valid) % 17 - 8).to(dtype)
    padded = torch.zeros(size, dtype=dtype)
    padded[:valid] = values
    tile = padded.reshape(programs, rows, columns)
    if computed:
        tile = tile * 2 + 3
    scanned = tile.cumsum(scan_axis + 1).to(dtype)
    reduced = tile.amax(2) if maximum else tile.sum(2).to(dtype)
    expected = tile * 2 - scanned + reduced[:, :, None] if broadcast else scanned
    sentinel = -999999
    output = torch.full((size,), sentinel, dtype=dtype, device="mps")
    row_output = torch.full((programs, rows), sentinel, dtype=dtype, device="mps")
    _shared_input_scan_row_reduce_kernel[(programs,)](
        values.to("mps"), output, row_output, valid, rows, columns, scan_axis,
        False, maximum, broadcast, computed, masked, num_warps=1)
    torch.mps.synchronize()
    actual = output.cpu()
    torch.testing.assert_close(actual[:valid], expected.flatten()[:valid], atol=0, rtol=0)
    assert (actual[valid:] == sentinel).all()
    if not broadcast:
        torch.testing.assert_close(row_output.cpu(), reduced, atol=0, rtol=0)


@triton.jit
def _shared_input_scan_row_reduce_snapshot_kernel(X, Y, Row,
                                                   SCAN_AXIS: tl.constexpr,
                                                   MUTATE: tl.constexpr):
    row = tl.arange(0, 8)
    column = tl.arange(0, 64)
    offset = row[:, None] * 64 + column[None, :]
    x = tl.load(X + offset)
    scanned = tl.cumsum(x, SCAN_AXIS)
    if MUTATE:
        tl.store(X + offset, x * 3 + 17)
    summed = tl.sum(x, 1)
    if MUTATE:
        tl.store(Y + offset, scanned)
        tl.store(Row + row, summed)
    else:
        maximum = tl.max(x, 1)
        tl.store(Y + offset, scanned + summed[:, None])
        tl.store(Row + row, maximum)


@mps_only
@pytest.mark.parametrize("scan_axis", [0, 1])
@pytest.mark.parametrize("mutate", [False, True], ids=["multiple_reducers", "input_snapshot"])
def test_multiband_scan_shared_input_snapshot_and_multiple_reducers(scan_axis, mutate):
    original = (torch.arange(512).reshape(8, 64) % 17 - 8).float()
    values = original.to("mps")
    output = torch.empty_like(values)
    row_output = torch.empty((8,), device="mps")
    _shared_input_scan_row_reduce_snapshot_kernel[(1,)](
        values, output, row_output, scan_axis, mutate, num_warps=1)
    torch.mps.synchronize()
    expected = original.cumsum(scan_axis)
    if mutate:
        expected_input = original * 3 + 17
        expected_row = original.sum(1)
    else:
        expected_input = original
        expected_row = original.amax(1)
        expected += original.sum(1)[:, None]
    torch.testing.assert_close(values.cpu(), expected_input, atol=0, rtol=0)
    torch.testing.assert_close(output.cpu(), expected, atol=0, rtol=0)
    torch.testing.assert_close(row_output.cpu(), expected_row, atol=0, rtol=0)


@triton.jit
def _unsupported_shared_input_row_reduce_kernel(X, Y, Z, Row, TWO_SCANS: tl.constexpr):
    row = tl.arange(0, 8)
    column = tl.arange(0, 64)
    offset = row[:, None] * 64 + column[None, :]
    x = tl.load(X + offset)
    scanned = tl.cumsum(x, 1)
    if TWO_SCANS:
        second = tl.cumsum(x * 2 + 1, 1)
        tl.store(Z + offset, second)
        reduced = tl.sum(x, 1)
    else:
        reduced = tl.sum(x + 1, 1)
    tl.store(Y + offset, scanned)
    tl.store(Row + row, reduced)


@pytest.mark.parametrize("two_scans,diagnostic", [
    (False, "distinct scan/reduce inputs require same-layout sources defined before the scan "
     "without intervening writes or input exchanges"),
    (True, "a rank-2 tl.reduce cannot share a function"),
], ids=["source_defined_after_scan", "two_scans"])
def test_multiband_scan_shared_input_outside_envelope_is_named(two_scans, diagnostic):
    result = _compile_rankn_nested_negative_in_child(
        "_unsupported_shared_input_row_reduce_kernel",
        {"X": "*fp32", "Y": "*fp32", "Z": "*fp32", "Row": "*fp32"},
        {"TWO_SCANS": two_scans}, num_warps=1)
    output = result.stdout + result.stderr
    assert result.returncode > 0, output
    assert diagnostic in output, output


@triton.jit
def _scan_result_row_reduce_kernel(X, Y, SCAN_AXIS: tl.constexpr):
    row = tl.arange(0, 8)
    column = tl.arange(0, 64)
    x = tl.load(X + row[:, None] * 64 + column[None, :])
    scanned = tl.cumsum(x, SCAN_AXIS)
    tl.store(Y + row, tl.sum(scanned, 1))


@pytest.mark.parametrize("scan_axis", [0, 1])
def test_multiband_scan_result_row_reduce_is_named(scan_axis):
    result = _compile_rankn_nested_negative_in_child(
        "_scan_result_row_reduce_kernel", {"X": "*fp32", "Y": "*fp32"},
        {"SCAN_AXIS": scan_axis}, num_warps=1)
    output = result.stdout + result.stderr
    assert result.returncode > 0, output
    assert "rank-2 reduce cannot replay its computed source cone across conversion" in output, output


# A rank-3 reduce beside a multi-band scan was wrong whenever its layout was
# not thread-major. The shared-input rank-2 column stores below now work.
@pytest.mark.parametrize("A,B,C,w,rank3,msg", [
    (4, 8, 16, 1, True, "a rank-3 tl.reduce whose layout is not thread-major"),
    (2, 16, 16, 1, True, "a rank-3 tl.reduce whose layout is not thread-major"),
])
def test_reduce_beside_multiband_scan_is_named(capfd, A, B, C, w, rank3, msg):
    with pytest.raises(Exception):
        _reduce_beside_scan_kernel[(1,)](
            torch.zeros(A * B * C, device="mps"), torch.zeros(B * C, device="mps"),
            torch.zeros(B * C, device="mps"), torch.zeros(B * C, device="mps"),
            A, B, C, rank3, num_warps=w)
    assert msg in capfd.readouterr().err


@mps_only
@pytest.mark.parametrize("B,C,w", [(16, 16, 1), (8, 64, 4)])
def test_column_reduce_beside_multiband_scan_matches_torch(B, C, w):
    values = (torch.arange(B * C).reshape(B, C) % 17 - 8).float()
    source = values.to("mps")
    columns = torch.full((C,), float("nan"), device="mps")
    scanned = torch.full((B, C), float("nan"), device="mps")
    _reduce_beside_scan_kernel[(1,)](
        source, source, columns, scanned, 1, B, C, False, num_warps=w)
    torch.mps.synchronize()
    torch.testing.assert_close(columns.cpu(), values.sum(0), atol=0, rtol=0)
    torch.testing.assert_close(scanned.cpu(), values.cumsum(1), atol=0, rtol=0)


@pytest.mark.parametrize("A,B,C,w", [(2, 8, 8, 1), (4, 8, 16, 4), (2, 16, 16, 4)])
def test_rank3_reduce_beside_scan_thread_major_matches_torch(A, B, C, w):
    torch.manual_seed(A + B + C + w)
    x = torch.randn(A, B, C)
    ww = torch.randn(B, C)
    y = torch.zeros(B * C, device="mps")
    z = torch.zeros(B * C, device="mps")
    _reduce_beside_scan_kernel[(1,)](x.to("mps"), ww.to("mps"), y, z, A, B, C,
                                     True, num_warps=w)
    torch.mps.synchronize()
    torch.testing.assert_close(y.cpu().reshape(B, C), x.sum(0), atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(z.cpu().reshape(B, C), ww.cumsum(1), atol=1e-4, rtol=1e-4)


@triton.jit
def _rank3_reduce_plus_tile_kernel(X, W, Y, A: tl.constexpr, B: tl.constexpr,
                                   C: tl.constexpr):
    a = tl.arange(0, A)
    b = tl.arange(0, B)
    c = tl.arange(0, C)
    x = tl.load(X + a[:, None, None] * (B * C) + b[None, :, None] * C
                + c[None, None, :])
    w = tl.load(W + b[:, None] * C + c[None, :])
    tl.store(Y + b[:, None] * C + c[None, :], tl.sum(x, 0) + w)


# A staged rank-3 reduce relabels its result thread-major, so the cvt that
# takes it to `w`'s layout is a relabel. Comparing it with the destination
# LAYOUT's plan instead of the label the destination is read under turned it
# into an exchange read under the layout's mapping: every element wrong.
@pytest.mark.parametrize("A,B,C,w", [(2, 16, 16, 1), (4, 8, 16, 1), (4, 8, 8, 2)])
def test_rank3_reduce_plus_loaded_tile_matches_torch(A, B, C, w):
    torch.manual_seed(A * 3 + B + C + w)
    x = torch.randn(A, B, C)
    ww = torch.randn(B, C)
    y = torch.zeros(B, C, device="mps")
    _rank3_reduce_plus_tile_kernel[(1,)](x.to("mps"), ww.to("mps"), y, A, B, C,
                                         num_warps=w)
    torch.mps.synchronize()
    torch.testing.assert_close(y.cpu(), x.sum(0) + ww, atol=1e-4, rtol=1e-4)


@triton.jit
def _shared_input_row_reduce_consumer_shapes_kernel(
        X, Y, Scanned, Row, R: tl.constexpr, C: tl.constexpr,
        SCAN_AXIS: tl.constexpr, MODE: tl.constexpr):
    row = tl.arange(0, R)
    column = tl.arange(0, C)
    offset = row[:, None] * C + column[None, :]
    x = tl.load(X + offset)
    scanned = tl.cumsum(x, SCAN_AXIS)
    reduced = tl.sum(x, 1)
    tl.store(Scanned + offset, scanned)
    if MODE == 0:
        # Mixing rank-1 and rank-2 consumers needs distinct result layouts.
        tl.store(Row + row, reduced)
        tl.store(Y + offset, scanned + reduced[:, None])
    elif MODE == 1:
        # Cross-axis broadcast is valid Triton but outside this shared path.
        tl.store(Y + offset, scanned + reduced[None, :])
    else:
        # A wider consumer cannot use the original scan tile indexing.
        wide_column = tl.arange(0, 64)
        wide_offset = row[:, None] * 64 + wide_column[None, :]
        tl.store(Y + wide_offset, reduced[:, None] + wide_column[None, :])


@pytest.mark.parametrize("scan_axis", [0, 1])
@pytest.mark.parametrize("mode", [1, 2], ids=["cross_axis", "wider_tile"])
def test_multiband_scan_shared_input_row_reduce_consumer_shapes_are_named(scan_axis, mode):
    rows, columns = (8, 32) if mode == 2 else (16, 16)
    result = _compile_rankn_nested_negative_in_child(
        "_shared_input_row_reduce_consumer_shapes_kernel",
        {"X": "*fp32", "Y": "*fp32", "Scanned": "*fp32", "Row": "*fp32"},
        {"R": rows, "C": columns, "SCAN_AXIS": scan_axis, "MODE": mode},
        num_warps=1)
    output = result.stdout + result.stderr
    assert result.returncode > 0, output
    assert ("a shared scan-input row reduction requires direct row stores or "
            "same-axis broadcasts into its original tile") in output, output


@triton.jit
def _shared_input_scan_column_reduce_kernel(
        X, Y, Column, N, VALID_COLUMNS, R: tl.constexpr, C: tl.constexpr,
        SCAN_AXIS: tl.constexpr, REVERSE: tl.constexpr, REDUCTION: tl.constexpr,
        COMPUTED: tl.constexpr, MASKED: tl.constexpr, SNAPSHOT: tl.constexpr,
        OTHER: tl.constexpr = 0, OUTPUT_MASKED: tl.constexpr = True):
    program = tl.program_id(0)
    row = tl.arange(0, R)
    column = tl.arange(0, C)
    offset = program * (R * C) + row[:, None] * C + column[None, :]
    if MASKED:
        x = tl.load(X + offset, offset < N, other=OTHER)
    else:
        x = tl.load(X + offset)
    if COMPUTED:
        x = x * 2 + 3
    scanned = tl.cumsum(x, SCAN_AXIS, reverse=REVERSE)
    if SNAPSHOT:
        tl.store(X + offset, x * 3 + 17, offset < N)
    if REDUCTION == "and":
        reduced = tl.reduce(x, 0, _i32_row_and_combine)
    elif REDUCTION == "or":
        reduced = tl.reduce(x, 0, _i32_row_or_combine)
    elif REDUCTION == "xor":
        reduced = tl.reduce(x, 0, _i32_row_xor_combine)
    elif REDUCTION == "umin":
        reduced = tl.min(x.to(tl.uint32), 0)
    elif REDUCTION == "umax":
        reduced = tl.max(x.to(tl.uint32), 0)
    elif REDUCTION == "product":
        reduced = tl.reduce(x, 0, _i32_column_product_combine)
    elif REDUCTION == "sum":
        reduced = tl.sum(x, 0)
    elif REDUCTION == "min":
        reduced = tl.min(x, 0)
    else:
        reduced = tl.max(x, 0)
    tl.store(Y + offset, scanned, offset < N)
    if MASKED and OUTPUT_MASKED:
        tl.store(Column + program * C + column, reduced, column < VALID_COLUMNS)
    else:
        tl.store(Column + program * C + column, reduced)


@mps_only
@pytest.mark.parametrize("reduction", ["sum", "min", "max"])
@pytest.mark.parametrize(
    "rows,columns,num_warps,scan_axis,reverse,dtype,masked,computed,snapshot", [
        (16, 16, 1, 0, False, torch.float32, True, False, False),
        (16, 16, 4, 1, True, torch.float32, True, True, True),
        (8, 32, 1, 1, False, torch.float32, False, True, False),
        (8, 64, 1, 1, True, torch.float32, True, True, False),
        (8, 64, 4, 0, True, torch.float32, True, False, True),
        (8, 128, 1, 0, False, torch.float32, True, True, False),
        (16, 16, 1, 1, False, torch.int32, False, False, False),
        (8, 32, 4, 0, True, torch.int32, False, False, True),
        (8, 128, 4, 0, False, torch.int32, False, False, False),
        (8, 256, 4, 0, True, torch.int32, False, False, True),
    ])
def test_multiband_scan_shared_input_column_reduce_matches_torch(
        rows, columns, num_warps, scan_axis, reverse, dtype, masked,
        computed, snapshot, reduction):
    programs = 2
    size = programs * rows * columns
    valid = size - columns - 3 if masked else size
    valid_columns = columns - 3 if masked else columns
    original = (torch.arange(valid) % 17 - 8).to(dtype)
    padded = torch.zeros(size, dtype=dtype)
    padded[:valid] = original
    tile = padded.reshape(programs, rows, columns)
    if computed:
        tile = tile * 2 + 3
    axis = scan_axis + 1
    source = tile.flip(axis) if reverse else tile
    scanned = source.cumsum(axis).to(dtype)
    if reverse:
        scanned = scanned.flip(axis)
    if reduction == "sum":
        reduced = tile.sum(1).to(dtype)
    elif reduction == "min":
        reduced = tile.amin(1)
    else:
        reduced = tile.amax(1)
    sentinel = -999999
    values = original.to("mps")
    output = torch.full((size,), sentinel, dtype=dtype, device="mps")
    column_output = torch.full((programs, columns), sentinel, dtype=dtype, device="mps")
    _shared_input_scan_column_reduce_kernel[(programs,)](
        values, output, column_output, valid, valid_columns, rows, columns,
        scan_axis, reverse, reduction, computed, masked, snapshot,
        num_warps=num_warps)
    torch.mps.synchronize()
    actual = output.cpu()
    actual_columns = column_output.cpu()
    torch.testing.assert_close(actual[:valid], scanned.flatten()[:valid], atol=0, rtol=0)
    torch.testing.assert_close(actual_columns[:, :valid_columns],
                               reduced[:, :valid_columns], atol=0, rtol=0)
    assert (actual[valid:] == sentinel).all()
    assert (actual_columns[:, valid_columns:] == sentinel).all()
    expected_input = (tile.flatten()[:valid] * 3 + 17) if snapshot else original
    torch.testing.assert_close(values.cpu(), expected_input, atol=0, rtol=0)


@triton.jit
def _unsupported_shared_input_column_reduce_kernel(X, Y, Scanned, BROADCAST: tl.constexpr):
    row = tl.arange(0, 8)
    if BROADCAST:
        column = tl.arange(0, 1024)
        offset = row[:, None] * 1024 + column[None, :]
    else:
        column = tl.arange(0, 64)
        offset = row[:, None] * 64 + column[None, :]
    x = tl.load(X + offset)
    if BROADCAST:
        scanned = tl.cumsum(x, 0)
    else:
        scanned = tl.cumsum(x, 1)
    if BROADCAST:
        reduced = tl.sum(x, 0)
        tl.store(Y + offset, scanned + reduced[None, :])
    else:
        reduced = tl.sum(x + 1, 0)
        tl.store(Y + column, reduced)
    tl.store(Scanned + offset, scanned)


@pytest.mark.parametrize("broadcast,diagnostic", [
    (True, "rank-2 axis=0 reduce with more output columns than threads needs a rank-1 consumer"),
    (False, "distinct scan/reduce inputs require same-layout sources defined before the scan "
     "without intervening writes or input exchanges"),
], ids=["column_broadcast", "different_source"])
def test_multiband_scan_shared_input_column_reduce_outside_envelope_is_named(
        broadcast, diagnostic):
    result = _compile_rankn_nested_negative_in_child(
        "_unsupported_shared_input_column_reduce_kernel",
        {"X": "*fp32", "Y": "*fp32", "Scanned": "*fp32"},
        {"BROADCAST": broadcast}, num_warps=1)
    output = result.stdout + result.stderr
    assert result.returncode > 0, output
    assert diagnostic in output, output


@triton.jit
def _different_input_scan_row_reduce_kernel(
        X, W, Y, Row, N, R: tl.constexpr, C: tl.constexpr,
        SCAN_AXIS: tl.constexpr, REVERSE: tl.constexpr, REDUCTION: tl.constexpr,
        MASKED: tl.constexpr, COMPUTED: tl.constexpr, BROADCAST: tl.constexpr,
        FROM_SCAN_INPUT: tl.constexpr, OTHER: tl.constexpr = 0,
        ROW_MASKED: tl.constexpr = False, FULL_OUTPUT: tl.constexpr = False):
    program = tl.program_id(0)
    row = tl.arange(0, R)
    column = tl.arange(0, C)
    offset = program * (R * C) + row[:, None] * C + column[None, :]
    if MASKED:
        x = tl.load(X + offset, offset < N, other=OTHER)
        w = tl.load(W + offset, offset < N, other=OTHER)
    else:
        x = tl.load(X + offset)
        w = tl.load(W + offset)
    if COMPUTED:
        x = x * 2 + 3
        w = w * 3 - 1
    if FROM_SCAN_INPUT:
        w = x + 1
    scanned = tl.cumsum(x, SCAN_AXIS, reverse=REVERSE)
    if REDUCTION == "umin":
        reduced = tl.min(w.to(tl.uint32), 1)
    elif REDUCTION == "umax":
        reduced = tl.max(w.to(tl.uint32), 1)
    elif REDUCTION == "product":
        reduced = tl.reduce(w, 1, _i32_column_product_combine)
    elif REDUCTION == "and":
        reduced = tl.reduce(w, 1, _i32_row_and_combine)
    elif REDUCTION == "or":
        reduced = tl.reduce(w, 1, _i32_row_or_combine)
    elif REDUCTION == "xor":
        reduced = tl.reduce(w, 1, _i32_row_xor_combine)
    elif REDUCTION == "sum":
        reduced = tl.sum(w, 1)
    elif REDUCTION == "min":
        reduced = tl.min(w, 1)
    else:
        reduced = tl.max(w, 1)
    if BROADCAST:
        result = x * 2 - scanned + reduced[:, None]
    else:
        result = scanned
        if ROW_MASKED:
            tl.store(Row + program * R + row, reduced, row < R - 1)
        else:
            tl.store(Row + program * R + row, reduced)
    if FULL_OUTPUT:
        tl.store(Y + offset, result)
    else:
        tl.store(Y + offset, result, offset < N)


@mps_only
@pytest.mark.parametrize("reduction", ["sum", "min", "max"])
@pytest.mark.parametrize(
    "rows,columns,num_warps,scan_axis,reverse,dtype,masked,computed,broadcast,from_scan_input", [
        (16, 16, 1, 0, False, torch.float32, True, False, False, False),
        (16, 16, 4, 1, True, torch.float32, True, True, True, False),
        (8, 64, 1, 1, False, torch.int32, False, False, False, False),
        (8, 64, 4, 0, True, torch.int32, False, False, True, False),
        (32, 32, 1, 1, True, torch.float32, True, True, False, True),
        (32, 32, 4, 0, False, torch.float32, False, False, True, False),
        (16, 16, 1, 1, True, torch.int32, False, False, True, False),
        (32, 32, 4, 1, False, torch.float32, False, True, True, True),
    ])
def test_multiband_scan_different_input_row_reduce_matches_torch(
        rows, columns, num_warps, scan_axis, reverse, dtype, masked, computed,
        broadcast, from_scan_input, reduction):
    programs = 2
    size = programs * rows * columns
    valid = size - columns - 3 if masked else size
    x_values = (torch.arange(valid) % 17 - 8).to(dtype)
    w_values = ((torch.arange(valid) * 3 + 5) % 23 - 11).to(dtype)
    x = torch.zeros(size, dtype=dtype)
    w = torch.zeros(size, dtype=dtype)
    x[:valid], w[:valid] = x_values, w_values
    x = x.reshape(programs, rows, columns)
    w = w.reshape(programs, rows, columns)
    if computed:
        x, w = x * 2 + 3, w * 3 - 1
    if from_scan_input:
        w = x + 1
    axis = scan_axis + 1
    source = x.flip(axis) if reverse else x
    scanned = source.cumsum(axis).to(dtype)
    if reverse:
        scanned = scanned.flip(axis)
    if reduction == "sum":
        reduced = w.sum(2).to(dtype)
    elif reduction == "min":
        reduced = w.amin(2)
    else:
        reduced = w.amax(2)
    expected = x * 2 - scanned + reduced[:, :, None] if broadcast else scanned
    sentinel = -999999
    output = torch.full((size,), sentinel, dtype=dtype, device="mps")
    row_output = torch.full((programs, rows), sentinel, dtype=dtype, device="mps")
    _different_input_scan_row_reduce_kernel[(programs,)](
        x_values.to("mps"), w_values.to("mps"), output, row_output, valid,
        rows, columns, scan_axis, reverse, reduction, masked, computed,
        broadcast, from_scan_input, num_warps=num_warps)
    torch.mps.synchronize()
    actual = output.cpu()
    torch.testing.assert_close(actual[:valid], expected.flatten()[:valid], atol=0, rtol=0)
    assert (actual[valid:] == sentinel).all()
    if not broadcast:
        torch.testing.assert_close(row_output.cpu(), reduced, atol=0, rtol=0)


@triton.jit
def _unsupported_different_input_scan_row_reduce_kernel(
        X, W, Y, Z, Row, MODE: tl.constexpr):
    row = tl.arange(0, 16)
    column = tl.arange(0, 16)
    offset = row[:, None] * 16 + column[None, :]
    x = tl.load(X + offset)
    if MODE == 2 or MODE == 5 or MODE == 6:
        w = tl.load(W + offset)
    elif MODE == 3:
        other_row = tl.arange(0, 8)
        other_column = tl.arange(0, 32)
        w = tl.load(W + other_row[:, None] * 32 + other_column[None, :])
    elif MODE == 4:
        w = tl.trans(tl.load(W + offset))
    if MODE == 5:
        # Re-reading this source at the scan would lose the earlier snapshot.
        tl.store(W + offset, x + 17)
    scanned = tl.cumsum(x, 1)
    if MODE == 0:
        w = tl.load(W + offset)
    elif MODE == 1:
        w = scanned * 2 + 1
    elif MODE == 2:
        second = tl.cumsum(w, 1)
        tl.store(Z + offset, second)
    elif MODE == 6:
        tl.store(W + offset, x + 17)
    reduced = tl.sum(w, 1)
    if MODE == 3:
        tl.store(Row + tl.arange(0, 8), reduced)
    else:
        tl.store(Row + row, reduced)
    tl.store(Y + offset, scanned)


@pytest.mark.parametrize("mode,diagnostic", [
    (0, "distinct scan/reduce inputs require same-layout sources defined before the scan "
     "without intervening writes or input exchanges"),
    (1, "rank-2 reduce cannot replay its computed source cone across conversion"),
    (2, "a rank-2 tl.reduce cannot share a function"),
    (3, "a rank-2 tl.reduce cannot share a function"),
    (4, "a rank-2 tl.reduce cannot share a function with a multi-band staged scan"),
    (5, "distinct scan/reduce inputs require same-layout sources defined before the scan "
     "without intervening writes or input exchanges"),
    (6, "distinct scan/reduce inputs require same-layout sources defined before the scan "
     "without intervening writes or input exchanges"),
], ids=["load_after_scan", "scan_result", "two_scans", "different_shape",
        "transpose", "write_before_scan", "write_after_scan"])
def test_multiband_scan_different_input_row_reduce_outside_envelope_is_named(mode, diagnostic):
    result = _compile_rankn_nested_negative_in_child(
        "_unsupported_different_input_scan_row_reduce_kernel",
        {"X": "*fp32", "W": "*fp32", "Y": "*fp32", "Z": "*fp32", "Row": "*fp32"},
        {"MODE": mode}, num_warps=1)
    output = result.stdout + result.stderr
    assert result.returncode > 0, output
    assert diagnostic in output, output


@triton.jit
def _different_input_scan_multiple_row_reducers_kernel(
        X, W, V, Y, Sum, Max, Min, R: tl.constexpr, C: tl.constexpr,
        SCAN_AXIS: tl.constexpr):
    row = tl.arange(0, R)
    column = tl.arange(0, C)
    offset = row[:, None] * C + column[None, :]
    x = tl.load(X + offset)
    w = tl.load(W + offset)
    v = tl.load(V + offset)
    scanned = tl.cumsum(x, SCAN_AXIS)
    summed = tl.sum(w, 1)
    maximum = tl.max(w, 1)
    minimum = tl.min(v, 1)
    # Sum and Max share exactly one source SSA value, while Min must read the
    # next distinct source's publication rather than the scan or W buffer.
    tl.store(Y + offset, scanned)
    tl.store(Sum + row, summed)
    tl.store(Max + row, maximum)
    tl.store(Min + row, minimum)


@mps_only
@pytest.mark.parametrize("rows,columns,scan_axis,num_warps", [
    (8, 64, 0, 1), (8, 64, 1, 4), (64, 32, 0, 4), (64, 32, 1, 4),
])
def test_multiband_scan_different_inputs_multiple_row_reducers_match_torch(
        rows, columns, scan_axis, num_warps):
    index = torch.arange(rows * columns).reshape(rows, columns)
    x = (index % 17 - 8).float()
    w = ((index * 3 + 5) % 23 - 11).float()
    v = ((index * 7 + 2) % 31 - 19).float()
    output = torch.empty((rows, columns), device="mps")
    summed = torch.empty((rows,), device="mps")
    maximum = torch.empty_like(summed)
    minimum = torch.empty_like(summed)
    _different_input_scan_multiple_row_reducers_kernel[(1,)](
        x.to("mps"), w.to("mps"), v.to("mps"), output, summed, maximum, minimum,
        rows, columns, scan_axis, num_warps=num_warps)
    torch.mps.synchronize()
    torch.testing.assert_close(output.cpu(), x.cumsum(scan_axis), atol=0, rtol=0)
    torch.testing.assert_close(summed.cpu(), w.sum(1), atol=0, rtol=0)
    torch.testing.assert_close(maximum.cpu(), w.amax(1), atol=0, rtol=0)
    torch.testing.assert_close(minimum.cpu(), v.amin(1), atol=0, rtol=0)


def test_multiband_scan_different_inputs_combined_scratch_budget_is_named():
    result = _compile_rankn_nested_negative_in_child(
        "_different_input_scan_multiple_row_reducers_kernel",
        {"X": "*fp32", "W": "*fp32", "V": "*fp32", "Y": "*fp32",
         "Sum": "*fp32", "Max": "*fp32", "Min": "*fp32"},
        {"R": 64, "C": 64, "SCAN_AXIS": 0}, num_warps=4)
    output = result.stdout + result.stderr
    assert result.returncode > 0, output
    assert "full-tile exchange scratch needs" in output, output
    assert "exceeding the 32768-byte budget" in output, output


@triton.jit
def _different_input_scan_column_reduce_kernel(
        X, W, Y, Column, N, VALID_COLUMNS, R: tl.constexpr, C: tl.constexpr,
        SCAN_AXIS: tl.constexpr, REVERSE: tl.constexpr, REDUCTION: tl.constexpr,
        MASKED: tl.constexpr, COMPUTED: tl.constexpr, SCAN_PLAIN: tl.constexpr = False,
        SCAN_MASKED: tl.constexpr = False, SCAN_COMPUTED: tl.constexpr = False,
        OTHER: tl.constexpr = 0, OUTPUT_MASKED: tl.constexpr = True):
    program = tl.program_id(0)
    row = tl.arange(0, R)
    column = tl.arange(0, C)
    offset = program * (R * C) + row[:, None] * C + column[None, :]
    if (MASKED and not SCAN_PLAIN) or SCAN_MASKED:
        x = tl.load(X + offset, offset < N, other=OTHER)
    else:
        x = tl.load(X + offset)
    if MASKED:
        w = tl.load(W + offset, offset < N, other=OTHER)
    else:
        w = tl.load(W + offset)
    if COMPUTED:
        w = w * 3 - 1
    if SCAN_COMPUTED:
        x = x * 2 + 3
    scanned = tl.cumsum(x, SCAN_AXIS, reverse=REVERSE)
    if REDUCTION == "and":
        reduced = tl.reduce(w, 0, _i32_row_and_combine)
    elif REDUCTION == "or":
        reduced = tl.reduce(w, 0, _i32_row_or_combine)
    elif REDUCTION == "xor":
        reduced = tl.reduce(w, 0, _i32_row_xor_combine)
    elif REDUCTION == "umin":
        reduced = tl.min(w.to(tl.uint32), 0)
    elif REDUCTION == "umax":
        reduced = tl.max(w.to(tl.uint32), 0)
    elif REDUCTION == "product":
        reduced = tl.reduce(w, 0, _i32_column_product_combine)
    elif REDUCTION == "sum":
        reduced = tl.sum(w, 0)
    elif REDUCTION == "min":
        reduced = tl.min(w, 0)
    else:
        reduced = tl.max(w, 0)
    tl.store(Y + offset, scanned, offset < N)
    if MASKED and OUTPUT_MASKED:
        tl.store(Column + program * C + column, reduced, column < VALID_COLUMNS)
    else:
        tl.store(Column + program * C + column, reduced)


@mps_only
@pytest.mark.parametrize("reduction", ["sum", "min", "max"])
@pytest.mark.parametrize(
    "rows,columns,num_warps,scan_axis,reverse,dtype,masked,computed", [
        (16, 16, 1, 0, False, torch.float32, True, False),
        (16, 16, 4, 1, True, torch.float32, True, True),
        (8, 64, 1, 1, False, torch.int32, False, False),
        (8, 64, 4, 0, True, torch.int32, False, False),
        (8, 128, 1, 0, False, torch.float32, True, True),
        (8, 256, 4, 0, True, torch.int32, False, False),
        (32, 32, 1, 1, False, torch.float32, False, True),
        (32, 32, 4, 0, False, torch.float32, False, False),
    ])
def test_multiband_scan_different_input_column_reduce_matches_torch(
        rows, columns, num_warps, scan_axis, reverse, dtype, masked, computed, reduction):
    programs = 2
    size = programs * rows * columns
    valid = size - columns - 3 if masked else size
    valid_columns = columns - 3 if masked else columns
    x_values = (torch.arange(valid) % 17 - 8).to(dtype)
    w_values = ((torch.arange(valid) * 3 + 5) % 23 - 11).to(dtype)
    x = torch.zeros(size, dtype=dtype)
    w = torch.zeros(size, dtype=dtype)
    x[:valid], w[:valid] = x_values, w_values
    x = x.reshape(programs, rows, columns)
    w = w.reshape(programs, rows, columns)
    if computed:
        w = w * 3 - 1
    axis = scan_axis + 1
    source = x.flip(axis) if reverse else x
    scanned = source.cumsum(axis).to(dtype)
    if reverse:
        scanned = scanned.flip(axis)
    if reduction == "sum":
        reduced = w.sum(1).to(dtype)
    elif reduction == "min":
        reduced = w.amin(1)
    else:
        reduced = w.amax(1)
    sentinel = -999999
    output = torch.full((size,), sentinel, dtype=dtype, device="mps")
    column_output = torch.full((programs, columns), sentinel, dtype=dtype, device="mps")
    _different_input_scan_column_reduce_kernel[(programs,)](
        x_values.to("mps"), w_values.to("mps"), output, column_output,
        valid, valid_columns, rows, columns, scan_axis, reverse, reduction,
        masked, computed, num_warps=num_warps)
    torch.mps.synchronize()
    actual = output.cpu()
    actual_columns = column_output.cpu()
    torch.testing.assert_close(actual[:valid], scanned.flatten()[:valid], atol=0, rtol=0)
    torch.testing.assert_close(actual_columns[:, :valid_columns],
                               reduced[:, :valid_columns], atol=0, rtol=0)
    assert (actual[valid:] == sentinel).all()
    assert (actual_columns[:, valid_columns:] == sentinel).all()


@mps_only
@pytest.mark.parametrize("scan_axis", [0, 1])
@pytest.mark.parametrize("computed", [False, True], ids=["scan_masked", "scan_computed"])
def test_multiband_scan_i32_nonplain_scan_with_plain_column_reduce_matches_torch(scan_axis, computed):
    rows, columns = 8, 64
    size = rows * columns
    valid = size if computed else size - 7
    x = (torch.arange(size).reshape(rows, columns) % 17 - 8).to(torch.int32)
    w = ((torch.arange(size).reshape(rows, columns) * 3 + 5) % 23 - 11).to(torch.int32)
    scan_source = x * 2 + 3 if computed else x.clone()
    if not computed:
        scan_source.flatten()[valid:] = 0
    sentinel = -999999
    output = torch.full((size,), sentinel, dtype=torch.int32, device="mps")
    reduced = torch.empty((columns,), dtype=torch.int32, device="mps")
    _different_input_scan_column_reduce_kernel[(1,)](
        x.to("mps"), w.to("mps"), output, reduced, valid, columns,
        rows, columns, scan_axis, False, "sum", False, False,
        SCAN_MASKED=not computed, SCAN_COMPUTED=computed, num_warps=1)
    torch.mps.synchronize()
    expected = scan_source.cumsum(scan_axis).to(torch.int32).flatten()
    actual = output.cpu()
    torch.testing.assert_close(actual[:valid], expected[:valid], atol=0, rtol=0)
    torch.testing.assert_close(reduced.cpu(), w.sum(0).to(torch.int32), atol=0, rtol=0)
    assert (actual[valid:] == sentinel).all()


@triton.jit
def _different_input_scan_mixed_reductions_kernel(X, W, V, Y, WRow, WCol, VCol, XRow,
                                                 SCAN_AXIS: tl.constexpr):
    row = tl.arange(0, 16)
    column = tl.arange(0, 16)
    offset = row[:, None] * 16 + column[None, :]
    x = tl.load(X + offset)
    w = tl.load(W + offset)
    v = tl.load(V + offset)
    scanned = tl.cumsum(x, SCAN_AXIS)
    w_row = tl.sum(w, 1)
    w_col = tl.max(w, 0)
    v_col = tl.min(v, 0)
    x_row = tl.max(x, 1)
    tl.store(Y + offset, scanned)
    tl.store(WRow + row, w_row)
    tl.store(WCol + column, w_col)
    tl.store(VCol + column, v_col)
    tl.store(XRow + row, x_row)


@mps_only
@pytest.mark.parametrize("scan_axis,num_warps,dtype", [
    (0, 1, torch.float32), (1, 4, torch.float32),
    (1, 1, torch.int32), (0, 4, torch.int32),
])
def test_multiband_scan_different_inputs_mixed_row_column_reductions_match_torch(
        scan_axis, num_warps, dtype):
    index = torch.arange(256).reshape(16, 16)
    x = (index % 17 - 8).to(dtype)
    w = ((index * 3 + 5) % 23 - 11).to(dtype)
    v = ((index * 7 + 2) % 31 - 19).to(dtype)
    output = torch.empty((16, 16), dtype=dtype, device="mps")
    vectors = [torch.empty((16,), dtype=dtype, device="mps") for _ in range(4)]
    _different_input_scan_mixed_reductions_kernel[(1,)](
        x.to("mps"), w.to("mps"), v.to("mps"), output, *vectors,
        scan_axis, num_warps=num_warps)
    torch.mps.synchronize()
    torch.testing.assert_close(output.cpu(), x.cumsum(scan_axis).to(dtype), atol=0, rtol=0)
    expected = [w.sum(1).to(dtype), w.amax(0), v.amin(0), x.amax(1)]
    for actual, reference in zip(vectors, expected):
        torch.testing.assert_close(actual.cpu(), reference, atol=0, rtol=0)


@triton.jit
def _unsupported_different_input_scan_column_reduce_kernel(X, W, Y, Column, MODE: tl.constexpr):
    if MODE == 3:
        row = tl.arange(0, 4)
        column = tl.arange(0, 1024)
        offset = row[:, None] * 1024 + column[None, :]
    else:
        row = tl.arange(0, 8)
        column = tl.arange(0, 64)
        offset = row[:, None] * 64 + column[None, :]
    x = tl.load(X + offset)
    if MODE != 0:
        w = tl.load(W + offset)
    if MODE == 1:
        tl.store(W + offset, x + 17)
    scanned = tl.cumsum(x, 0)
    if MODE == 0:
        w = tl.load(W + offset)
    elif MODE == 2:
        tl.store(W + offset, x + 17)
    reduced = tl.sum(w, 0)
    if MODE == 3:
        tl.store(Y + offset, scanned + reduced[None, :])
    else:
        tl.store(Y + offset, scanned)
        tl.store(Column + column, reduced)


@pytest.mark.parametrize("mode,diagnostic", [
    (0, "distinct scan/reduce inputs require same-layout sources defined before the scan "
     "without intervening writes or input exchanges"),
    (1, "distinct scan/reduce inputs require same-layout sources defined before the scan "
     "without intervening writes or input exchanges"),
    (2, "distinct scan/reduce inputs require same-layout sources defined before the scan "
     "without intervening writes or input exchanges"),
    (3, "rank-2 axis=0 reduce with more output columns than threads needs a rank-1 consumer"),
], ids=["load_after_scan", "write_before_scan", "write_after_scan", "column_broadcast"])
def test_multiband_scan_different_input_column_reduce_outside_envelope_is_named(mode, diagnostic):
    result = _compile_rankn_nested_negative_in_child(
        "_unsupported_different_input_scan_column_reduce_kernel",
        {"X": "*fp32", "W": "*fp32", "Y": "*fp32", "Column": "*fp32"},
        {"MODE": mode}, num_warps=1)
    output = result.stdout + result.stderr
    assert result.returncode > 0, output
    assert diagnostic in output, output


@triton.jit
def _shared_input_scan_column_broadcast_kernel(
        X, Y, N, R: tl.constexpr, C: tl.constexpr, SCAN_AXIS: tl.constexpr,
        REVERSE: tl.constexpr, REDUCTION: tl.constexpr, MASKED: tl.constexpr,
        COMPUTED: tl.constexpr, SNAPSHOT: tl.constexpr):
    program = tl.program_id(0)
    row = tl.arange(0, R)
    column = tl.arange(0, C)
    offset = program * (R * C) + row[:, None] * C + column[None, :]
    if MASKED:
        x = tl.load(X + offset, offset < N, other=0)
    else:
        x = tl.load(X + offset)
    if COMPUTED:
        x = x * 2 + 3
    scanned = tl.cumsum(x, SCAN_AXIS, reverse=REVERSE)
    if SNAPSHOT:
        if MASKED:
            tl.store(X + offset, x * 3 + 17, offset < N)
        else:
            tl.store(X + offset, x * 3 + 17)
    if REDUCTION == "sum":
        reduced = tl.sum(x, 0)
    elif REDUCTION == "min":
        reduced = tl.min(x, 0)
    else:
        reduced = tl.max(x, 0)
    result = scanned + reduced[None, :]
    if MASKED:
        tl.store(Y + offset, result, offset < N)
    else:
        tl.store(Y + offset, result)


@mps_only
@pytest.mark.parametrize("reduction", ["sum", "min", "max"])
@pytest.mark.parametrize("rows,columns,num_warps,scan_axis,reverse,masked,computed,snapshot", [
    (2, 32, 1, 1, False, False, False, False),
    (16, 16, 1, 0, True, True, False, True),
    (16, 32, 4, 1, True, True, True, False),
    (8, 64, 4, 0, False, False, True, True),
    (32, 32, 1, 1, False, True, True, False),
    (32, 16, 4, 0, True, False, False, False),
    (64, 2, 1, 1, True, False, True, True),
    (4, 64, 4, 1, False, True, False, False),
])
def test_multiband_scan_shared_input_column_broadcast_matches_torch(
        rows, columns, num_warps, scan_axis, reverse, masked, computed, snapshot, reduction):
    programs = 2
    size = programs * rows * columns
    valid = size - columns - 1 if masked else size
    index = torch.arange(valid)
    # Every unmasked column has a nonzero, distinct reduction contribution.
    original = (1 + (index % columns) * 3 + (index // columns) % 11).float()
    padded = torch.zeros(size)
    padded[:valid] = original
    tile = padded.reshape(programs, rows, columns)
    if computed:
        tile = tile * 2 + 3
    axis = scan_axis + 1
    source = tile.flip(axis) if reverse else tile
    scanned = source.cumsum(axis)
    if reverse:
        scanned = scanned.flip(axis)
    if reduction == "sum":
        reduced = tile.sum(1)
    elif reduction == "min":
        reduced = tile.amin(1)
    else:
        reduced = tile.amax(1)
    contribution = reduced[:, None, :].expand_as(tile)
    expected = scanned + contribution
    sentinel = -999999
    values = original.to("mps")
    output = torch.full((size,), sentinel, dtype=torch.float32, device="mps")
    compiled = _shared_input_scan_column_broadcast_kernel[(programs,)](
        values, output, valid, rows, columns, scan_axis, reverse, reduction,
        masked, computed, snapshot, num_warps=num_warps)
    raw = compiled.asm["metal"]
    msl = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    # Reuse one complete input publication. Extra barriers or tile buffers
    # would reintroduce legacy column republishing; the common per-thread
    # scratch reservation is optional and may appear only once.
    assert msl.count("threadgroup_barrier(") == 1, msl
    buffers = sorted(int(size) for size in re.findall(
        r"\bthreadgroup\s+float\s+\w+\s*\[\s*(\d+)\s*\]\s*;", msl))
    assert buffers in ([rows * columns], [32 * num_warps, rows * columns]), msl
    torch.mps.synchronize()
    actual = output.cpu()
    torch.testing.assert_close(actual[:valid], expected.flatten()[:valid], atol=0, rtol=0)
    torch.testing.assert_close(actual[:valid] - scanned.flatten()[:valid],
                               contribution.flatten()[:valid], atol=0, rtol=0)
    assert (actual[valid:] == sentinel).all()
    expected_input = tile.flatten()[:valid] * 3 + 17 if snapshot else original
    torch.testing.assert_close(values.cpu(), expected_input, atol=0, rtol=0)


@triton.jit
def _unsupported_shared_input_column_broadcast_kernel(
        X, W, Y, Scanned, Vector, steps, R: tl.constexpr, C: tl.constexpr,
        MODE: tl.constexpr, UNSIGNED: tl.constexpr = False, MONOID: tl.constexpr = ""):
    row = tl.arange(0, R)
    column = tl.arange(0, C)
    offset = row[:, None] * C + column[None, :]
    if MODE == 8:
        for step in range(steps):
            x = tl.load(X + step * (R * C) + offset)
            scanned = tl.cumsum(x, 1)
            reduced = _column_boundary_reduce(x, UNSIGNED, MONOID)
            tl.store(Y + step * (R * C) + offset, scanned + reduced[None, :])
    else:
        x = tl.load(X + offset)
        if MODE == 2:
            scanned = tl.cumsum(x, 0)
        else:
            scanned = tl.cumsum(x, 1)
        if MODE == 0:
            w = tl.load(X + offset, column[None, :] < C - 1, other=5)
            w = w * 2 + 3
        elif MODE == 1:
            w = tl.load(W + offset)
        if MODE == 7:
            scanned += tl.cumsum(x * 2 + 1, 1)
        if MODE == 0 or MODE == 1:
            reduced = _column_boundary_reduce(w, UNSIGNED, MONOID)
        else:
            reduced = _column_boundary_reduce(x, UNSIGNED, MONOID)
        if MODE == 3:
            tl.store(Vector + column, reduced)
        if MODE == 4:
            tile = scanned + reduced[:, None]
            tl.store(Y + offset, tile)
        elif MODE == 5:
            wider_row = tl.arange(0, 2 * R)
            wider_offset = wider_row[:, None] * C + column[None, :]
            tl.store(Y + wider_offset, reduced[None, :] + wider_row[:, None])
        elif MODE == 6:
            tl.store(Y + offset, scanned + (reduced * 2 + 1)[None, :])
        else:
            tile = scanned + reduced[None, :]
            if MODE == 9:
                if steps > 1:
                    tl.store(Y + offset, tile)
                else:
                    tl.store(Y + offset, tile * 2)
            elif MODE == 10:
                tl.store(Vector + row, tl.sum(tile, 1))
            else:
                tl.store(Y + offset, tile)
        tl.store(Scanned + offset, scanned)


@pytest.mark.parametrize("mode,diagnostic", [
    (0, "distinct scan/reduce inputs require same-layout sources defined before the scan "
     "without intervening writes or input exchanges"),
    (1, "distinct scan/reduce inputs require same-layout sources defined before the scan "
     "without intervening writes or input exchanges"),
    (2, "rank-2 axis=0 reduce with more output columns than threads needs a rank-1 consumer"),
    (3, "a scan-input column broadcast requires same-layout f32/i32 sources and "
     "direct same-layout consumers in the original tile"),
    (4, "a rank-2 tl.reduce cannot share a function"),
    (5, "a scan-input column broadcast requires same-layout f32/i32 sources and "
     "direct same-layout consumers in the original tile"),
    (6, "a rank-2 tl.reduce cannot share a function"),
    (7, "a scan-input column broadcast requires same-layout f32/i32 sources and "
     "direct same-layout consumers in the original tile"),
    (8, "nested scan requires a flat load/compute/store loop body"),
    (9, "a scan-input column broadcast requires same-layout f32/i32 sources and "
     "direct same-layout consumers in the original tile"),
    (10, "rank-2 reduce cannot replay its computed source cone across conversion"),
], ids=["late_nonplain_i32_source", "load_after_scan", "wide_columns", "mixed_store", "cross_axis",
        "wider_tile", "pre_expand_arithmetic", "multiple_scans", "nested_scan",
        "region_capture", "downstream_reduce"])
def test_multiband_scan_shared_input_column_broadcast_outside_envelope_is_named(mode, diagnostic):
    dtype = "*i32" if mode == 0 else "*fp32"
    result = _compile_rankn_nested_negative_in_child(
        "_unsupported_shared_input_column_broadcast_kernel",
        {"X": dtype, "W": dtype, "Y": dtype, "Scanned": dtype,
         "Vector": dtype, "steps": "i32"},
        {"R": 4 if mode == 2 else 16, "C": 1024 if mode == 2 else 16, "MODE": mode, "UNSIGNED": False, "MONOID": ""}, num_warps=1)
    output = result.stdout + result.stderr
    assert result.returncode > 0, output
    assert diagnostic in output, output


@triton.jit
def _small_column_broadcast_beside_multiband_scan_kernel(X, W, Y, Z, SCAN_AXIS: tl.constexpr):
    small_row = tl.arange(0, 2)
    small_column = tl.arange(0, 2)
    small_offset = small_row[:, None] * 2 + small_column[None, :]
    small = tl.load(W + small_offset)
    column_sum = tl.sum(small, 0)
    tl.store(Z + small_offset, small + column_sum[None, :])
    row = tl.arange(0, 16)
    column = tl.arange(0, 16)
    offset = row[:, None] * 16 + column[None, :]
    x = tl.load(X + offset)
    tl.store(Y + offset, tl.cumsum(x, SCAN_AXIS))


@pytest.mark.parametrize("scan_axis", [0, 1])
def test_small_column_broadcast_beside_independent_multiband_scan_is_named(scan_axis):
    # A one-band expand still shares the function's multi-band scan partition.
    # Letting this bypass the guard produced wrong values in all four cells of
    # [[2, 7], [3, 11]] + column_sum, even with separate scan/output pointers.
    result = _compile_rankn_nested_negative_in_child(
        "_small_column_broadcast_beside_multiband_scan_kernel",
        {"X": "*fp32", "W": "*fp32", "Y": "*fp32", "Z": "*fp32"},
        {"SCAN_AXIS": scan_axis}, num_warps=1)
    output = result.stdout + result.stderr
    assert result.returncode > 0, output
    assert ("a scan-input column broadcast requires same-layout f32/i32 sources "
            "and direct same-layout consumers in the original tile") in output, output


@triton.jit
def _different_input_scan_column_broadcast_kernel(
        X, W, Y, N, R: tl.constexpr, C: tl.constexpr, SCAN_AXIS: tl.constexpr,
        REVERSE: tl.constexpr, REDUCTION: tl.constexpr, MASKED: tl.constexpr,
        COMPUTED: tl.constexpr, FROM_SCAN_INPUT: tl.constexpr,
        SCAN_MASKED: tl.constexpr = False, SCAN_COMPUTED: tl.constexpr = False):
    program = tl.program_id(0)
    row = tl.arange(0, R)
    column = tl.arange(0, C)
    offset = program * (R * C) + row[:, None] * C + column[None, :]
    if MASKED or SCAN_MASKED:
        x = tl.load(X + offset, offset < N, other=0)
    else:
        x = tl.load(X + offset)
    if SCAN_COMPUTED:
        x = x * 2 + 3
    if FROM_SCAN_INPUT:
        w = x * 2 + 3
    elif MASKED:
        w = tl.load(W + offset, offset < N, other=0)
    else:
        w = tl.load(W + offset)
    if COMPUTED:
        w = w * 3 - 1
    scanned = tl.cumsum(x, SCAN_AXIS, reverse=REVERSE)
    if REDUCTION == "sum":
        reduced = tl.sum(w, 0)
    elif REDUCTION == "min":
        reduced = tl.min(w, 0)
    else:
        reduced = tl.max(w, 0)
    result = scanned + reduced[None, :]
    if MASKED:
        tl.store(Y + offset, result, offset < N)
    else:
        tl.store(Y + offset, result)


@mps_only
@pytest.mark.parametrize("reduction", ["sum", "min", "max"])
@pytest.mark.parametrize("rows,columns,num_warps,scan_axis,reverse,masked,computed,from_scan_input", [
    (2, 32, 1, 1, False, False, False, False),
    (16, 16, 1, 0, True, True, False, False),
    (16, 32, 4, 1, True, True, True, False),
    (8, 64, 4, 0, False, False, False, True),
    (32, 32, 1, 1, False, True, False, True),
    (32, 16, 4, 0, True, False, True, False),
    (64, 2, 1, 1, True, False, True, True),
    (4, 64, 4, 1, False, True, False, False),
    (64, 64, 4, 0, False, False, False, False),
])
def test_multiband_scan_different_input_column_broadcast_matches_torch(
        rows, columns, num_warps, scan_axis, reverse, masked, computed,
        from_scan_input, reduction):
    programs = 2
    size = programs * rows * columns
    valid = size - columns - 1 if masked else size
    index = torch.arange(valid)
    x_values = (1 + (index % columns) * 3 + (index // columns) % 11).float()
    w_values = (7 + (index % columns) * 5 + (index // columns) % 13 * 2).float()
    x = torch.zeros(size)
    w = torch.zeros(size)
    x[:valid], w[:valid] = x_values, w_values
    x = x.reshape(programs, rows, columns)
    w = w.reshape(programs, rows, columns)
    if from_scan_input:
        w = x * 2 + 3
    if computed:
        w = w * 3 - 1
    axis = scan_axis + 1
    source = x.flip(axis) if reverse else x
    scanned = source.cumsum(axis)
    if reverse:
        scanned = scanned.flip(axis)
    if reduction == "sum":
        reduced = w.sum(1)
    elif reduction == "min":
        reduced = w.amin(1)
    else:
        reduced = w.amax(1)
    contribution = reduced[:, None, :].expand_as(x)
    expected = scanned + contribution
    sentinel = -999999
    output = torch.full((size,), sentinel, dtype=torch.float32, device="mps")
    _different_input_scan_column_broadcast_kernel[(programs,)](
        x_values.to("mps"), w_values.to("mps"), output, valid,
        rows, columns, scan_axis, reverse, reduction, masked, computed,
        from_scan_input, num_warps=num_warps)
    torch.mps.synchronize()
    actual = output.cpu()
    torch.testing.assert_close(actual[:valid], expected.flatten()[:valid], atol=0, rtol=0)
    torch.testing.assert_close(actual[:valid] - scanned.flatten()[:valid],
                               contribution.flatten()[:valid], atol=0, rtol=0)
    assert (actual[valid:] == sentinel).all()


@triton.jit
def _different_inputs_scan_multiple_column_broadcasts_kernel(
        X, W, V, Sum, Max, Min, SharedMax, R: tl.constexpr, C: tl.constexpr,
        SCAN_AXIS: tl.constexpr, PRODUCT: tl.constexpr = False):
    row = tl.arange(0, R)
    column = tl.arange(0, C)
    offset = row[:, None] * C + column[None, :]
    x = tl.load(X + offset)
    w = tl.load(W + offset)
    v = tl.load(V + offset)
    scanned = tl.cumsum(x, SCAN_AXIS)
    if PRODUCT:
        summed = tl.reduce(w, 0, _i32_column_product_combine)
        minimum = tl.reduce(v, 0, _i32_column_product_combine)
        shared_maximum = tl.reduce(x, 0, _i32_column_product_combine)
    else:
        summed = tl.sum(w, 0)
        minimum = tl.min(v, 0)
        shared_maximum = tl.max(x, 0)
    maximum = tl.max(w, 0)
    # W is published once for two reducers, while X also remains available to
    # its scan and shared-source reducer. All reductions precede every store.
    tl.store(Sum + offset, scanned + summed[None, :])
    tl.store(Max + offset, scanned + maximum[None, :])
    tl.store(Min + offset, scanned + minimum[None, :])
    tl.store(SharedMax + offset, scanned + shared_maximum[None, :])


@mps_only
@pytest.mark.parametrize("rows,columns,scan_axis,num_warps", [(16, 16, 0, 1), (64, 32, 1, 4)])
def test_multiband_scan_different_inputs_multiple_column_broadcasts_match_torch(
        rows, columns, scan_axis, num_warps):
    index = torch.arange(rows * columns).reshape(rows, columns)
    x = (1 + (index % columns) * 3 + (index // columns) % 11).float()
    w = (7 + (index % columns) * 5 + (index // columns) % 13 * 2).float()
    v = ((index % columns) * 7 - (index // columns) % 17 * 3 - 29).float()
    outputs = [torch.empty((rows, columns), dtype=torch.float32, device="mps") for _ in range(4)]
    compiled = _different_inputs_scan_multiple_column_broadcasts_kernel[(1,)](
        x.to("mps"), w.to("mps"), v.to("mps"), *outputs,
        rows, columns, scan_axis, num_warps=num_warps)
    raw = compiled.asm["metal"]
    msl = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    # Three distinct SSA inputs need three complete publications, independent
    # of the number of reducers. Neither duplicate W nor republish reductions.
    assert msl.count("threadgroup_barrier(") == 1, msl
    buffers = sorted(int(size) for size in re.findall(
        r"\bthreadgroup\s+float\s+\w+\s*\[\s*(\d+)\s*\]\s*;", msl))
    full_buffers = [rows * columns] * 3
    assert buffers in (full_buffers, [32 * num_warps] + full_buffers), msl
    torch.mps.synchronize()
    scanned = x.cumsum(scan_axis)
    reductions = [w.sum(0), w.amax(0), v.amin(0), x.amax(0)]
    for actual, reduced in zip(outputs, reductions):
        contribution = reduced[None, :].expand_as(x)
        torch.testing.assert_close(actual.cpu(), scanned + contribution, atol=0, rtol=0)
        torch.testing.assert_close(actual.cpu() - scanned, contribution, atol=0, rtol=0)


def test_multiband_scan_different_inputs_column_broadcast_budget_is_named():
    result = _compile_rankn_nested_negative_in_child(
        "_different_inputs_scan_multiple_column_broadcasts_kernel",
        {"X": "*fp32", "W": "*fp32", "V": "*fp32", "Sum": "*fp32",
         "Max": "*fp32", "Min": "*fp32", "SharedMax": "*fp32"},
        {"R": 64, "C": 64, "SCAN_AXIS": 0, "PRODUCT": False}, num_warps=4)
    output = result.stdout + result.stderr
    assert result.returncode > 0, output
    assert "full-tile exchange scratch needs" in output, output
    assert "exceeding the 32768-byte budget" in output, output


@triton.jit
def _unsupported_different_input_column_broadcast_kernel(
        X, W, Y, MODE: tl.constexpr, UNSIGNED: tl.constexpr = False, MONOID: tl.constexpr = ""):
    row = tl.arange(0, 16)
    column = tl.arange(0, 16)
    offset = row[:, None] * 16 + column[None, :]
    x = tl.load(X + offset)
    if MODE != 0:
        w = tl.load(W + offset)
    if MODE == 2:
        tl.store(W + offset, x + 17)
    elif MODE == 4:
        w = tl.trans(w)
    scanned = tl.cumsum(x, 1)
    if MODE == 0:
        w = tl.load(W + offset)
    elif MODE == 1:
        w = w * 2 + 3
    elif MODE == 3:
        tl.store(W + offset, x + 17)
    elif MODE == 5:
        w = scanned * 2 + 1
    reduced = _column_boundary_reduce(w, UNSIGNED, MONOID)
    tl.store(Y + offset, scanned + reduced[None, :])


@pytest.mark.parametrize("mode,diagnostic", [
    (0, "distinct scan/reduce inputs require same-layout sources defined before the scan "
     "without intervening writes or input exchanges"),
    (1, "distinct scan/reduce inputs require same-layout sources defined before the scan "
     "without intervening writes or input exchanges"),
    (2, "distinct scan/reduce inputs require same-layout sources defined before the scan "
     "without intervening writes or input exchanges"),
    (3, "distinct scan/reduce inputs require same-layout sources defined before the scan "
     "without intervening writes or input exchanges"),
    (4, "a rank-2 tl.reduce cannot share a function with a multi-band staged scan"),
    (5, "rank-2 reduce cannot replay its computed source cone across conversion"),
], ids=["load_after_scan", "compute_after_scan", "write_before_scan", "write_after_scan",
        "transpose", "scan_result"])
def test_multiband_scan_different_input_column_broadcast_outside_envelope_is_named(mode, diagnostic):
    result = _compile_rankn_nested_negative_in_child(
        "_unsupported_different_input_column_broadcast_kernel",
        {"X": "*fp32", "W": "*fp32", "Y": "*fp32"}, {"MODE": mode, "UNSIGNED": False, "MONOID": ""}, num_warps=1)
    output = result.stdout + result.stderr
    assert result.returncode > 0, output
    assert diagnostic in output, output


@mps_only
@pytest.mark.parametrize("reduction", ["sum", "min", "max"])
@pytest.mark.parametrize("rows,columns,num_warps,scan_axis,reverse,shared", [
    (2, 32, 1, 1, False, True),
    (16, 16, 1, 0, True, True),
    (16, 32, 4, 1, True, True),
    (8, 64, 4, 0, False, True),
    (32, 32, 1, 1, False, False),
    (32, 16, 4, 0, True, False),
    (64, 2, 1, 1, True, False),
    (4, 64, 4, 1, False, False),
])
def test_multiband_scan_i32_column_broadcast_matches_torch(
        rows, columns, num_warps, scan_axis, reverse, shared, reduction):
    programs = 2
    size = programs * rows * columns
    index = torch.arange(size)
    if reduction == "sum":
        if rows == 64 and not shared:
            # Exercise modular i32 accumulation in the reduction itself.
            w = (2 ** 30 + index % 31).to(torch.int32)
        else:
            w = ((index * 7) % 31 - 15).to(torch.int32)
    else:
        # Min/max must compare signed integers exactly, including values that
        # cannot round-trip through f32 and values close to both i32 limits.
        category = (index // columns + index) % 4
        w = torch.where(category == 0, -(2 ** 31) + 7 + index % 13,
                        torch.where(category == 1, 2 ** 31 - 9 - index % 11,
                                    torch.where(category == 2, 2 ** 24 + index % 31,
                                                -(2 ** 24) - index % 37))).to(torch.int32)
    x = w.clone() if shared else (index % 17 - 8).to(torch.int32)
    x_tile = x.reshape(programs, rows, columns)
    w_tile = w.reshape(programs, rows, columns)
    axis = scan_axis + 1
    source = x_tile.flip(axis) if reverse else x_tile
    # Scan prefixes and elementwise addition intentionally wrap modulo 2**32.
    scanned = source.cumsum(axis, dtype=torch.int64).to(torch.int32)
    if reverse:
        scanned = scanned.flip(axis)
    if reduction == "sum":
        reduced = w_tile.sum(1, dtype=torch.int64).to(torch.int32)
    elif reduction == "min":
        reduced = w_tile.amin(1)
    else:
        reduced = w_tile.amax(1)
    contribution = reduced[:, None, :].expand_as(x_tile)
    expected = scanned + contribution
    sentinel = -999999
    output = torch.full((size + 7,), sentinel, dtype=torch.int32, device="mps")
    if shared:
        _shared_input_scan_column_broadcast_kernel[(programs,)](
            x.to("mps"), output, size, rows, columns, scan_axis, reverse,
            reduction, False, False, False, num_warps=num_warps)
    else:
        _different_input_scan_column_broadcast_kernel[(programs,)](
            x.to("mps"), w.to("mps"), output, size, rows, columns,
            scan_axis, reverse, reduction, False, False, False,
            num_warps=num_warps)
    torch.mps.synchronize()
    actual = output.cpu()
    torch.testing.assert_close(actual[:size], expected.flatten(), atol=0, rtol=0)
    torch.testing.assert_close(actual[:size] - scanned.flatten(), contribution.flatten(), atol=0, rtol=0)
    assert (actual[size:] == sentinel).all()


@mps_only
@pytest.mark.parametrize("scan_axis", [0, 1])
@pytest.mark.parametrize("computed", [False, True], ids=["masked_scan", "computed_scan"])
def test_multiband_scan_i32_nonplain_scan_column_broadcast_matches_torch(scan_axis, computed):
    rows, columns = 16, 16
    size = rows * columns
    valid = size if computed else size - 7
    index = torch.arange(size).reshape(rows, columns)
    x = (index % 17 - 8).to(torch.int32)
    w = ((index % 31 - 15) * 1000001).to(torch.int32)
    source = x * 2 + 3 if computed else x.clone()
    if not computed:
        source.flatten()[valid:] = 0
    scanned = source.cumsum(scan_axis).to(torch.int32)
    contribution = w.amin(0)[None, :].expand_as(w)
    sentinel = -999999
    output = torch.full((size + 7,), sentinel, dtype=torch.int32, device="mps")
    _different_input_scan_column_broadcast_kernel[(1,)](
        x.to("mps"), w.to("mps"), output, valid, rows, columns,
        scan_axis, False, "min", False, False, False,
        SCAN_MASKED=not computed, SCAN_COMPUTED=computed, num_warps=1)
    torch.mps.synchronize()
    actual = output.cpu()
    torch.testing.assert_close(actual[:size], (scanned + contribution).flatten(), atol=0, rtol=0)
    torch.testing.assert_close(actual[:size] - scanned.flatten(), contribution.flatten(), atol=0, rtol=0)
    assert (actual[size:] == sentinel).all()


@mps_only
@pytest.mark.parametrize("scan_axis", [0, 1])
def test_multiband_scan_i32_column_broadcast_retains_input_snapshot(scan_axis):
    x = (torch.arange(256).reshape(16, 16) % 17 - 8).to(torch.int32)
    values = x.to("mps")
    output = torch.empty((16, 16), dtype=torch.int32, device="mps")
    _shared_input_scan_column_broadcast_kernel[(1,)](
        values, output, x.numel(), 16, 16, scan_axis, False, "max",
        False, False, True, num_warps=1)
    torch.mps.synchronize()
    torch.testing.assert_close(values.cpu(), x * 3 + 17, atol=0, rtol=0)
    torch.testing.assert_close(output.cpu(), x.cumsum(scan_axis).to(torch.int32) + x.amax(0)[None, :],
                               atol=0, rtol=0)


@triton.jit
def _i32_scan_multiple_column_broadcasts_kernel(
        X, W, V, Sum, Min, Max, OtherMin, SharedMax,
        SCAN_AXIS: tl.constexpr, MULTIPLE_SOURCES: tl.constexpr):
    row = tl.arange(0, 16)
    column = tl.arange(0, 16)
    offset = row[:, None] * 16 + column[None, :]
    x = tl.load(X + offset)
    if MULTIPLE_SOURCES:
        w = tl.load(W + offset)
        v = tl.load(V + offset)
    else:
        w = x
        v = x
    scanned = tl.cumsum(x, SCAN_AXIS)
    summed = tl.sum(w, 0)
    minimum = tl.min(w, 0)
    maximum = tl.max(w, 0)
    other_minimum = tl.min(v, 0)
    shared_maximum = tl.max(x, 0)
    tl.store(Sum + offset, scanned + summed[None, :])
    tl.store(Min + offset, scanned + minimum[None, :])
    tl.store(Max + offset, scanned + maximum[None, :])
    tl.store(OtherMin + offset, scanned + other_minimum[None, :])
    tl.store(SharedMax + offset, scanned + shared_maximum[None, :])


@mps_only
@pytest.mark.parametrize("multiple_sources,scan_axis,num_warps", [(False, 0, 1), (True, 1, 4)])
def test_multiband_scan_i32_multiple_column_broadcasts_match_torch(multiple_sources, scan_axis, num_warps):
    index = torch.arange(256).reshape(16, 16)
    x = (index % 17 - 8).to(torch.int32)
    w = ((index * 3 + 5) % 23 - 11).to(torch.int32) if multiple_sources else x
    v = ((index * 7 + 2) % 31 - 19).to(torch.int32) if multiple_sources else x
    outputs = [torch.empty((16, 16), dtype=torch.int32, device="mps") for _ in range(5)]
    compiled = _i32_scan_multiple_column_broadcasts_kernel[(1,)](
        x.to("mps"), w.to("mps"), v.to("mps"), *outputs,
        scan_axis, multiple_sources, num_warps=num_warps)
    raw = compiled.asm["metal"]
    msl = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    assert msl.count("threadgroup_barrier(") == 1, msl
    buffers = sorted(int(size) for size in re.findall(
        r"\bthreadgroup\s+(?:u?int32_t|u?int)\s+\w+\s*\[\s*(\d+)\s*\]\s*;", msl))
    full_buffers = [256] * (3 if multiple_sources else 1)
    assert buffers in (full_buffers, [32 * num_warps] + full_buffers), msl
    torch.mps.synchronize()
    scanned = x.cumsum(scan_axis).to(torch.int32)
    reductions = [w.sum(0).to(torch.int32), w.amin(0), w.amax(0), v.amin(0), x.amax(0)]
    for actual, reduced in zip(outputs, reductions):
        contribution = reduced[None, :].expand_as(x)
        torch.testing.assert_close(actual.cpu(), scanned + contribution, atol=0, rtol=0)
        torch.testing.assert_close(actual.cpu() - scanned, contribution, atol=0, rtol=0)


@pytest.mark.parametrize("dtype,reduction,diagnostic", [
    ("*i64", "sum", "rank >= 2 scan is implemented for a blocked tile"),
])
def test_multiband_scan_integer_column_broadcast_type_is_named(dtype, reduction, diagnostic):
    result = _compile_rankn_nested_negative_in_child(
        "_shared_input_scan_column_broadcast_kernel", {"X": dtype, "Y": dtype, "N": "i32"},
        {"R": 16, "C": 16, "SCAN_AXIS": 1, "REVERSE": False,
         "REDUCTION": reduction, "MASKED": False, "COMPUTED": False, "SNAPSHOT": False},
        num_warps=1)
    output = result.stdout + result.stderr
    assert result.returncode > 0, output
    assert diagnostic in output, output


@pytest.mark.parametrize("mode,columns,diagnostic", [
    (2, 1024, "rank-2 axis=0 reduce with more output columns than threads needs a rank-1 consumer"),
    (3, 16, "a scan-input column broadcast requires same-layout f32/i32 sources"),
], ids=["wide_columns", "mixed_store"])
def test_multiband_scan_i32_column_broadcast_consumer_is_named(mode, columns, diagnostic):
    result = _compile_rankn_nested_negative_in_child(
        "_unsupported_shared_input_column_broadcast_kernel",
        {"X": "*i32", "W": "*i32", "Y": "*i32", "Scanned": "*i32",
         "Vector": "*i32", "steps": "i32"},
        {"R": 4 if mode == 2 else 16, "C": columns, "MODE": mode, "UNSIGNED": False, "MONOID": ""}, num_warps=1)
    output = result.stdout + result.stderr
    assert result.returncode > 0, output
    assert diagnostic in output, output


@triton.jit
def _i32_column_product_combine(lhs, rhs):
    return lhs * rhs


@triton.jit
def _unsupported_i32_scan_column_product_kernel(X, Y):
    row = tl.arange(0, 16)
    column = tl.arange(0, 16)
    offset = row[:, None] * 16 + column[None, :]
    x = tl.load(X + offset)
    scanned = tl.cumsum(x, 1)
    reduced = tl.reduce(x, 0, _i32_column_product_combine)
    tl.store(Y + offset, scanned + reduced[None, :])


@mps_only
@pytest.mark.parametrize("reduction", ["sum", "min", "max"])
@pytest.mark.parametrize("rows,columns,num_warps,scan_axis,reverse,dtype,shared,masked,computed", [
    (8, 64, 1, 1, False, torch.float32, True, True, True),
    (8, 64, 1, 0, True, torch.int32, False, False, False),
    (8, 128, 1, 0, False, torch.float32, False, True, True),
    (8, 128, 4, 0, True, torch.int32, True, False, False),
    (4, 256, 1, 0, False, torch.float32, True, True, False),
    (8, 256, 4, 0, True, torch.int32, False, False, False),
    (32, 256, 4, 0, False, torch.float32, True, False, False),
    (16, 256, 4, 0, True, torch.int32, False, False, False),
    (4, 64, 1, 1, True, torch.int32, True, False, False),
    (8, 256, 4, 0, True, torch.float32, False, False, True),
])
def test_multiband_scan_wide_column_broadcast_matches_torch(
        rows, columns, num_warps, scan_axis, reverse, dtype, shared, masked, computed, reduction):
    programs = 2
    size = programs * rows * columns
    valid = size - columns - 1 if masked else size
    index = torch.arange(valid)
    if dtype == torch.int32 and reduction != "sum":
        category = (index // columns + index) % 4
        w_values = torch.where(category == 0, -(2 ** 31) + 7 + index % 13,
                               torch.where(category == 1, 2 ** 31 - 9 - index % 11,
                                           torch.where(category == 2, 2 ** 24 + index % 31,
                                                       -(2 ** 24) - index % 37))).to(dtype)
    elif dtype == torch.int32 and rows == 16:
        w_values = (2 ** 30 + index % 31).to(dtype)
    else:
        w_values = (7 + (index % columns) * 5 + (index // columns) % 13 * 2).to(dtype)
    x_values = w_values.clone() if shared else (1 + index % columns * 3 + index // columns % 11).to(dtype)
    x = torch.zeros(size, dtype=dtype)
    w = torch.zeros(size, dtype=dtype)
    x[:valid], w[:valid] = x_values, w_values
    x = x.reshape(programs, rows, columns)
    w = w.reshape(programs, rows, columns)
    if computed:
        if shared:
            x = x * 2 + 3
            w = x
        else:
            w = w * 3 - 1
    axis = scan_axis + 1
    source = x.flip(axis) if reverse else x
    # Integer prefixes, reductions, and final addition use modulo-2**32 results.
    scanned = source.cumsum(axis).to(dtype)
    if reverse:
        scanned = scanned.flip(axis)
    if reduction == "sum":
        reduced = w.sum(1).to(dtype)
    elif reduction == "min":
        reduced = w.amin(1)
    else:
        reduced = w.amax(1)
    contribution = reduced[:, None, :].expand_as(x)
    expected = scanned + contribution
    sentinel = -999999
    output = torch.full((size + 7,), sentinel, dtype=dtype, device="mps")
    if shared:
        _shared_input_scan_column_broadcast_kernel[(programs,)](
            x_values.to("mps"), output, valid, rows, columns, scan_axis, reverse,
            reduction, masked, computed, False, num_warps=num_warps)
    else:
        _different_input_scan_column_broadcast_kernel[(programs,)](
            x_values.to("mps"), w_values.to("mps"), output, valid,
            rows, columns, scan_axis, reverse, reduction, masked, computed,
            False, num_warps=num_warps)
    torch.mps.synchronize()
    actual = output.cpu()
    torch.testing.assert_close(actual[:valid], expected.flatten()[:valid], atol=0, rtol=0)
    torch.testing.assert_close(actual[:valid] - scanned.flatten()[:valid],
                               contribution.flatten()[:valid], atol=0, rtol=0)
    assert (actual[valid:] == sentinel).all()


@mps_only
@pytest.mark.parametrize("columns,num_warps,dtype", [(128, 1, torch.float32), (256, 4, torch.int32)])
def test_multiband_scan_wide_multiple_column_broadcasts_match_torch(columns, num_warps, dtype):
    rows = 8
    index = torch.arange(rows * columns).reshape(rows, columns)
    x = (index % 17 - 8).to(dtype)
    w = ((index * 3 + 5) % 23 - 11).to(dtype)
    v = ((index * 7 + 2) % 31 - 19).to(dtype)
    outputs = [torch.empty((rows, columns), dtype=dtype, device="mps") for _ in range(4)]
    compiled = _different_inputs_scan_multiple_column_broadcasts_kernel[(1,)](
        x.to("mps"), w.to("mps"), v.to("mps"), *outputs,
        rows, columns, 0, num_warps=num_warps)
    raw = compiled.asm["metal"]
    msl = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    assert msl.count("threadgroup_barrier(") == 1, msl
    buffers = sorted(int(size) for size in re.findall(
        r"\bthreadgroup\s+(?:float|u?int32_t|u?int)\s+\w+\s*\[\s*(\d+)\s*\]\s*;", msl))
    full_buffers = [rows * columns] * 3
    assert buffers in (full_buffers, [32 * num_warps] + full_buffers), msl
    torch.mps.synchronize()
    scanned = x.cumsum(0).to(dtype)
    reductions = [w.sum(0).to(dtype), w.amax(0), v.amin(0), x.amax(0)]
    for actual, reduced in zip(outputs, reductions):
        contribution = reduced[None, :].expand_as(x)
        torch.testing.assert_close(actual.cpu(), scanned + contribution, atol=0, rtol=0)
        torch.testing.assert_close(actual.cpu() - scanned, contribution, atol=0, rtol=0)


@pytest.mark.parametrize("shared", [False, True], ids=["combined_inputs", "single_input"])
def test_multiband_scan_wide_column_broadcast_budget_is_named(shared):
    signature = {"X": "*fp32", "Y": "*fp32", "N": "i32"}
    constexprs = {"R": 64 if shared else 32, "C": 256, "SCAN_AXIS": 0,
                  "REVERSE": False, "REDUCTION": "sum", "MASKED": False, "COMPUTED": False}
    if shared:
        kernel = "_shared_input_scan_column_broadcast_kernel"
        constexprs["SNAPSHOT"] = False
    else:
        kernel = "_different_input_scan_column_broadcast_kernel"
        signature["W"] = "*fp32"
        constexprs.update(FROM_SCAN_INPUT=False, SCAN_MASKED=False, SCAN_COMPUTED=False)
    result = _compile_rankn_nested_negative_in_child(kernel, signature, constexprs, num_warps=4)
    output = result.stdout + result.stderr
    assert result.returncode > 0, output
    if shared:
        assert "rank >= 2 scan is implemented for a blocked tile" in output, output
        assert "threadgroup budget" in output, output
    else:
        assert "full-tile exchange scratch needs" in output, output
        assert "exceeding the 32768-byte budget" in output, output


@triton.jit
def _unsupported_wide_scan_column_broadcast_kernel(X, Y, Vector, MODE: tl.constexpr, UNSIGNED: tl.constexpr = False, MONOID: tl.constexpr = ""):
    if MODE == 4:
        row = tl.arange(0, 64)
        column = tl.arange(0, 64)
        offset = row[:, None] * 64 + column[None, :]
    else:
        row = tl.arange(0, 8)
        column = tl.arange(0, 128)
        offset = row[:, None] * 128 + column[None, :]
    x = tl.load(X + offset)
    if MODE == 0:
        reduced = _column_boundary_reduce(x, UNSIGNED, MONOID)
    if MODE == 4:
        w = tl.trans(x)
    if MODE == 3:
        reduced = _column_boundary_reduce(x, UNSIGNED, MONOID)
        tl.store(Y + offset, x + reduced[None, :])
    else:
        if MODE == 6:
            scanned = tl.cumsum(x, 1)
        else:
            scanned = tl.cumsum(x, 0)
        if MODE == 1:
            scanned += tl.cumsum(x + 1, 0)
        if MODE == 4:
            reduced = _column_boundary_reduce(w, UNSIGNED, MONOID)
        elif MODE == 5:
            reduced = _column_boundary_reduce(scanned, UNSIGNED, MONOID)
        elif MODE != 0:
            reduced = _column_boundary_reduce(x, UNSIGNED, MONOID)
        if MODE == 2:
            tl.store(Vector + column, reduced)
        tl.store(Y + offset, scanned + reduced[None, :])


@pytest.mark.parametrize("mode,diagnostic", [
    (0, "rank-2 axis=0 reduce with more output columns than threads needs a rank-1 consumer"),
    (1, "rank-2 axis=0 reduce with more output columns than threads needs a rank-1 consumer"),
    (2, "a scan-input column broadcast requires same-layout f32/i32 sources"),
    (3, "rank-2 axis=0 reduce with more output columns than threads needs a rank-1 consumer"),
    (4, "rank-2 axis=0 reduce with more output columns than threads needs a rank-1 consumer"),
    (5, "rank-2 reduce cannot replay its computed source cone across conversion"),
    (6, "rank >= 2 scan is implemented for a blocked tile"),
], ids=["reduce_before_scan", "multiple_scans", "mixed_store", "no_scan",
        "different_layout", "scan_result", "long_scan_axis"])
def test_multiband_scan_wide_column_broadcast_outside_envelope_is_named(mode, diagnostic):
    result = _compile_rankn_nested_negative_in_child(
        "_unsupported_wide_scan_column_broadcast_kernel",
        {"X": "*fp32", "Y": "*fp32", "Vector": "*fp32"}, {"MODE": mode, "UNSIGNED": False, "MONOID": ""}, num_warps=1)
    output = result.stdout + result.stderr
    assert result.returncode > 0, output
    assert diagnostic in output, output


@triton.jit
def _i32_nonplain_input_scan_column_broadcast_kernel(
        X, W, Y, N, R: tl.constexpr, C: tl.constexpr,
        SCAN_AXIS: tl.constexpr, REVERSE: tl.constexpr, REDUCTION: tl.constexpr,
        SHARED: tl.constexpr, MASKED: tl.constexpr,
        COMPUTED: tl.constexpr, OTHER: tl.constexpr, SNAPSHOT: tl.constexpr,
        FULL_OUTPUT: tl.constexpr, SUBTRACT: tl.constexpr = False):
    program = tl.program_id(0)
    row = tl.arange(0, R)
    column = tl.arange(0, C)
    offset = program * (R * C) + row[:, None] * C + column[None, :]
    if MASKED:
        x = tl.load(X + offset, offset < N, other=OTHER)
    else:
        x = tl.load(X + offset)
    if SHARED:
        if COMPUTED:
            x = x * 2 + 3
        w = x
    else:
        if MASKED:
            w = tl.load(W + offset, offset < N, other=OTHER)
        else:
            w = tl.load(W + offset)
        if COMPUTED:
            w = w * 3 - 1
    scanned = tl.cumsum(x, SCAN_AXIS, reverse=REVERSE)
    if SNAPSHOT:
        tl.store(X + offset, x * 2 - 7, offset < N)
    if REDUCTION == "product":
        reduced = tl.reduce(w, 0, _i32_column_product_combine)
    elif REDUCTION == "and":
        reduced = tl.reduce(w, 0, _i32_row_and_combine)
    elif REDUCTION == "or":
        reduced = tl.reduce(w, 0, _i32_row_or_combine)
    elif REDUCTION == "xor":
        reduced = tl.reduce(w, 0, _i32_row_xor_combine)
    elif REDUCTION == "umin":
        reduced = tl.min(w.to(tl.uint32), 0)
    elif REDUCTION == "umax":
        reduced = tl.max(w.to(tl.uint32), 0)
    elif REDUCTION == "sum":
        reduced = tl.sum(w, 0)
    elif REDUCTION == "min":
        reduced = tl.min(w, 0)
    else:
        reduced = tl.max(w, 0)
    if SUBTRACT:
        result = scanned - reduced[None, :]
    else:
        result = scanned + reduced[None, :]
    if MASKED and not FULL_OUTPUT:
        tl.store(Y + offset, result, offset < N)
    else:
        tl.store(Y + offset, result)


@mps_only
@pytest.mark.parametrize("reduction", ["sum", "min", "max"])
@pytest.mark.parametrize(
    "rows,columns,num_warps,scan_axis,reverse,shared,masked,computed,other,snapshot,full_output", [
        (8, 64, 1, 0, False, True, True, False, 5, False, False),
        (16, 16, 1, 1, True, True, False, True, 0, False, False),
        (8, 128, 1, 0, False, False, True, True, -3, False, True),
        (8, 256, 4, 0, True, True, True, True, 0, True, True),
        (16, 64, 4, 1, False, False, True, False, 7, False, True),
        (8, 256, 4, 0, False, False, False, True, 0, False, False),
        (32, 32, 1, 1, True, True, True, True, -5, True, False),
        (16, 128, 4, 0, True, False, True, True, 0, False, True),
        (32, 16, 4, 0, False, True, False, True, 0, True, False),
        (4, 64, 1, 1, True, False, True, True, 0, False, True),
    ])
def test_multiband_scan_i32_nonplain_column_broadcast_matches_torch(
        rows, columns, num_warps, scan_axis, reverse, shared,
        masked, computed, other, snapshot, full_output, reduction):
    programs = 2
    size = programs * rows * columns
    valid = size - columns - 3 if masked else size
    index = torch.arange(valid)
    if reduction == "sum":
        w_values = (2 ** 30 + index % 31).to(torch.int32)
    else:
        category = (index // columns + index) % 4
        w_values = torch.where(category == 0, -(2 ** 31) + 7 + index % 13,
                               torch.where(category == 1, 2 ** 31 - 9 - index % 11,
                                           torch.where(category == 2, 2 ** 24 + index % 31,
                                                       -(2 ** 24) - index % 37))).to(torch.int32)
    x_values = w_values.clone() if shared else (index % 17 - 8).to(torch.int32)
    x = torch.full((size,), other, dtype=torch.int32)
    w = torch.full((size,), other, dtype=torch.int32)
    x[:valid], w[:valid] = x_values, w_values
    x = x.reshape(programs, rows, columns)
    w = w.reshape(programs, rows, columns)
    if computed:
        if shared:
            x = x * 2 + 3
            w = x
        else:
            w = w * 3 - 1
    axis = scan_axis + 1
    source = x.flip(axis) if reverse else x
    # Every integer operation, including computed inputs and sum, wraps to i32.
    scanned = source.cumsum(axis).to(torch.int32)
    if reverse:
        scanned = scanned.flip(axis)
    if reduction == "sum":
        reduced = w.sum(1).to(torch.int32)
    elif reduction == "min":
        reduced = w.amin(1)
    else:
        reduced = w.amax(1)
    contribution = reduced[:, None, :].expand_as(x)
    expected = scanned + contribution
    sentinel = -999999
    values = x_values.to("mps")
    output = torch.full((size + 7,), sentinel, dtype=torch.int32, device="mps")
    _i32_nonplain_input_scan_column_broadcast_kernel[(programs,)](
        values, w_values.to("mps"), output, valid,
        rows, columns, scan_axis, reverse, reduction, shared,
        masked, computed, other, snapshot, full_output, num_warps=num_warps)
    torch.mps.synchronize()
    actual = output.cpu()
    written = size if full_output else valid
    torch.testing.assert_close(actual[:written], expected.flatten()[:written], atol=0, rtol=0)
    assert (actual[written:] == sentinel).all()
    torch.testing.assert_close(actual[:written] - scanned.flatten()[:written],
                               contribution.flatten()[:written], atol=0, rtol=0)
    expected_input = x.flatten()[:valid] * 2 - 7 if snapshot else x_values
    torch.testing.assert_close(values.cpu(), expected_input, atol=0, rtol=0)


@triton.jit
def _i32_computed_sources_scan_column_broadcast_kernel(
        X, Sum, Min, Max, SharedMax, N, R: tl.constexpr, C: tl.constexpr, SCAN_AXIS: tl.constexpr, MONOIDS: tl.constexpr = False):
    row = tl.arange(0, R)
    column = tl.arange(0, C)
    offset = row[:, None] * C + column[None, :]
    x = tl.load(X + offset, offset < N, other=0)
    w = x * 2 + 3
    v = x * 3 - 1
    scanned = tl.cumsum(x, SCAN_AXIS)
    if MONOIDS:
        summed = tl.reduce(w, 0, _i32_column_product_combine)
        minimum = tl.reduce(w, 0, _i32_row_xor_combine)
        maximum = tl.reduce(v, 0, _i32_row_and_combine)
        shared_maximum = tl.reduce(x, 0, _i32_row_or_combine)
    else:
        summed = tl.sum(w, 0)
        minimum = tl.min(w, 0)
        maximum = tl.max(v, 0)
        shared_maximum = tl.max(x, 0)
    tl.store(Sum + offset, scanned + summed[None, :])
    tl.store(Min + offset, scanned + minimum[None, :])
    tl.store(Max + offset, scanned + maximum[None, :])
    tl.store(SharedMax + offset, scanned + shared_maximum[None, :])


@mps_only
@pytest.mark.parametrize("rows,columns,scan_axis,num_warps", [(8, 128, 0, 1), (16, 32, 1, 4)])
def test_multiband_scan_i32_computed_sources_column_broadcasts_match_torch(rows, columns, scan_axis, num_warps):
    size = rows * columns
    valid = size - columns - 1
    original = (torch.arange(valid) % 17 - 8).to(torch.int32)
    x = torch.zeros(size, dtype=torch.int32)
    x[:valid] = original
    x = x.reshape(rows, columns)
    w, v = x * 2 + 3, x * 3 - 1
    outputs = [torch.empty((rows, columns), dtype=torch.int32, device="mps") for _ in range(4)]
    compiled = _i32_computed_sources_scan_column_broadcast_kernel[(1,)](
        original.to("mps"), *outputs, valid, rows, columns, scan_axis, num_warps=num_warps)
    raw = compiled.asm["metal"]
    msl = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    assert msl.count("threadgroup_barrier(") == 1, msl
    buffers = sorted(int(size) for size in re.findall(
        r"\bthreadgroup\s+(?:u?int32_t|u?int)\s+\w+\s*\[\s*(\d+)\s*\]\s*;", msl))
    # Three SSA inputs sharing one device load need distinct publications;
    # the two W reductions must reuse exactly one of those buffers.
    full_buffers = [size] * 3
    assert buffers in (full_buffers, [32 * num_warps] + full_buffers), msl
    torch.mps.synchronize()
    scanned = x.cumsum(scan_axis).to(torch.int32)
    reductions = [w.sum(0).to(torch.int32), w.amin(0), v.amax(0), x.amax(0)]
    for actual, reduced in zip(outputs, reductions):
        contribution = reduced[None, :].expand_as(x)
        torch.testing.assert_close(actual.cpu(), scanned + contribution, atol=0, rtol=0)
        torch.testing.assert_close(actual.cpu() - scanned, contribution, atol=0, rtol=0)


def test_multiband_scan_i32_computed_sources_column_broadcast_budget_is_named():
    result = _compile_rankn_nested_negative_in_child(
        "_i32_computed_sources_scan_column_broadcast_kernel",
        {"X": "*i32", "Sum": "*i32", "Min": "*i32", "Max": "*i32",
         "SharedMax": "*i32", "N": "i32"},
        {"R": 64, "C": 64, "SCAN_AXIS": 0, "MONOIDS": False}, num_warps=4)
    output = result.stdout + result.stderr
    assert result.returncode > 0, output
    assert "full-tile exchange scratch needs" in output, output
    assert "exceeding the 32768-byte budget" in output, output


@pytest.mark.parametrize("mode,diagnostic", [
    (0, "distinct scan/reduce inputs require same-layout sources defined before the scan "
     "without intervening writes or input exchanges"),
    (1, "distinct scan/reduce inputs require same-layout sources defined before the scan "
     "without intervening writes or input exchanges"),
    (2, "distinct scan/reduce inputs require same-layout sources defined before the scan "
     "without intervening writes or input exchanges"),
    (3, "distinct scan/reduce inputs require same-layout sources defined before the scan "
     "without intervening writes or input exchanges"),
    (4, "a rank-2 tl.reduce cannot share a function with a multi-band staged scan"),
], ids=["load_after_scan", "compute_after_scan", "write_before_scan", "write_after_scan", "transpose"])
def test_multiband_scan_i32_nonplain_column_broadcast_timing_is_named(mode, diagnostic):
    result = _compile_rankn_nested_negative_in_child(
        "_unsupported_different_input_column_broadcast_kernel",
        {"X": "*i32", "W": "*i32", "Y": "*i32"}, {"MODE": mode, "UNSIGNED": False, "MONOID": ""}, num_warps=1)
    output = result.stdout + result.stderr
    assert result.returncode > 0, output
    assert diagnostic in output, output


@mps_only
def test_multiband_scan_i32_nonplain_column_vector_baseline_matches_torch():
    rows, columns = 8, 64
    valid = rows * columns - columns - 3
    original = (torch.arange(valid) % 17 - 8).to(torch.int32)
    x = torch.zeros(rows * columns, dtype=torch.int32)
    x[:valid] = original
    x = x.reshape(rows, columns)
    output = torch.full((rows * columns,), -999999, dtype=torch.int32, device="mps")
    vector = torch.full((columns,), -999999, dtype=torch.int32, device="mps")
    _shared_input_scan_column_reduce_kernel[(1,)](
        original.to("mps"), output, vector, valid, columns,
        rows, columns, 0, False, "sum", False, True, False, num_warps=1)
    torch.mps.synchronize()
    torch.testing.assert_close(vector.cpu(), x.sum(0).to(torch.int32), atol=0, rtol=0)
    torch.testing.assert_close(output.cpu()[:valid], x.cumsum(0).to(torch.int32).flatten()[:valid],
                               atol=0, rtol=0)
    assert (output.cpu()[valid:] == -999999).all()


@mps_only
@pytest.mark.parametrize("reduction", ["sum", "min", "max", "product"])
@pytest.mark.parametrize(
    "rows,columns,num_warps,scan_axis,reverse,shared,masked,computed,other,snapshot,output_masked", [
        (16, 16, 1, 1, True, True, True, False, 5, False, False),
        (8, 64, 1, 0, False, True, False, True, 0, True, False),
        (16, 32, 4, 1, False, True, True, True, 0, True, True),
        (8, 128, 1, 0, True, False, True, True, -3, False, False),
        (8, 256, 4, 0, True, True, True, True, 7, True, False),
        (8, 64, 4, 1, True, False, True, False, -5, False, True),
        (4, 512, 1, 0, False, False, False, True, 0, False, False),
        (8, 512, 4, 0, True, True, True, True, 0, False, True),
        (8, 512, 4, 0, False, True, True, False, 11, False, False),
        (16, 128, 4, 0, True, False, True, True, 0, False, True),
        # C512 already belongs to the plain-load vector-store contract.
        (8, 512, 4, 0, False, True, False, False, 0, False, False),
        (8, 64, 1, 0, False, True, True, False, 1, False, False),
    ])
def test_multiband_scan_i32_nonplain_column_vector_matches_torch(
        rows, columns, num_warps, scan_axis, reverse, shared, masked,
        computed, other, snapshot, output_masked, reduction):
    programs = 2
    size = programs * rows * columns
    valid = size - columns - 3 if masked else size
    if reduction == "product" and masked and columns >= 128:
        # The last program has only a partial first row: its final columns
        # are completely masked, so every product factor is the fill value.
        valid = size - (rows - 1) * columns - 3
    valid_columns = columns - 3 if masked and output_masked else columns
    index = torch.arange(valid)
    if reduction == "product":
        w_values = _i32_row_monoid_values(valid, columns, "product",
                                         even_product=computed and not shared)
    elif reduction == "sum":
        w_values = (2 ** 30 + index % 31).to(torch.int32)
    else:
        category = (index // columns + index) % 4
        w_values = torch.where(category == 0, -(2 ** 31) + 7 + index % 13,
                               torch.where(category == 1, 2 ** 31 - 9 - index % 11,
                                           torch.where(category == 2, 2 ** 24 + index % 31,
                                                       -(2 ** 24) - index % 37))).to(torch.int32)
    x_values = w_values.clone() if shared else (index % 17 - 8).to(torch.int32)
    x = torch.full((size,), other, dtype=torch.int32)
    w = torch.full((size,), other, dtype=torch.int32)
    x[:valid], w[:valid] = x_values, w_values
    x, w = x.reshape(programs, rows, columns), w.reshape(programs, rows, columns)
    if computed:
        if shared:
            x = x * 2 + 3
            w = x
        else:
            w = w * 3 - 1
    axis = scan_axis + 1
    source = x.flip(axis) if reverse else x
    scanned = source.cumsum(axis).to(torch.int32)
    if reverse:
        scanned = scanned.flip(axis)
    if reduction == "product":
        reduced = _i32_row_monoid_reference(w.transpose(1, 2), "product")
        assert (reduced[0] != 0).all(), "unmasked products must retain nonzero wrapped outputs"
    elif reduction == "sum":
        reduced = w.sum(1).to(torch.int32)
    elif reduction == "min":
        reduced = w.amin(1)
    else:
        reduced = w.amax(1)
    sentinel = -999999
    values = x_values.to("mps")
    output = torch.full((size + 7,), sentinel, dtype=torch.int32, device="mps")
    vector = torch.full((programs * columns + 7,), sentinel, dtype=torch.int32, device="mps")
    if shared:
        _shared_input_scan_column_reduce_kernel[(programs,)](
            values, output, vector, valid, valid_columns, rows, columns,
            scan_axis, reverse, reduction, computed, masked, snapshot,
            OTHER=other, OUTPUT_MASKED=output_masked, num_warps=num_warps)
    else:
        _different_input_scan_column_reduce_kernel[(programs,)](
            values, w_values.to("mps"), output, vector, valid, valid_columns,
            rows, columns, scan_axis, reverse, reduction, masked, computed,
            OTHER=other, OUTPUT_MASKED=output_masked, num_warps=num_warps)
    torch.mps.synchronize()
    actual = output.cpu()
    actual_vector = vector.cpu()
    columns_written = actual_vector[:programs * columns].reshape(programs, columns)
    torch.testing.assert_close(actual[:valid], scanned.flatten()[:valid], atol=0, rtol=0)
    torch.testing.assert_close(columns_written[:, :valid_columns],
                               reduced[:, :valid_columns], atol=0, rtol=0)
    assert (actual[valid:] == sentinel).all()
    assert (columns_written[:, valid_columns:] == sentinel).all()
    assert (actual_vector[programs * columns:] == sentinel).all()
    expected_input = x.flatten()[:valid] * 3 + 17 if snapshot else x_values
    torch.testing.assert_close(values.cpu(), expected_input, atol=0, rtol=0)


@triton.jit
def _i32_computed_sources_scan_column_vectors_kernel(
        X, Scanned, Sum, Min, Max, SharedMax, N,
        R: tl.constexpr, C: tl.constexpr, SCAN_AXIS: tl.constexpr, PRODUCT: tl.constexpr = False,
        BITWISE: tl.constexpr = False):
    row = tl.arange(0, R)
    column = tl.arange(0, C)
    offset = row[:, None] * C + column[None, :]
    x = tl.load(X + offset, offset < N, other=0)
    w = x * 2 + 3
    if PRODUCT:
        v = x * 4 + 1
    else:
        v = x * 3 - 1
    scanned = tl.cumsum(x, SCAN_AXIS)
    if BITWISE:
        summed = tl.reduce(w, 0, _i32_row_and_combine)
        minimum = tl.reduce(w, 0, _i32_row_or_combine)
        maximum = tl.reduce(v, 0, _i32_row_xor_combine)
        if PRODUCT:
            shared_maximum = tl.reduce(x, 0, _i32_column_product_combine)
        else:
            shared_maximum = tl.min(x, 0)
    elif PRODUCT:
        summed = tl.reduce(w, 0, _i32_column_product_combine)
        minimum = tl.min(w, 0)
        maximum = tl.reduce(v, 0, _i32_column_product_combine)
        shared_maximum = tl.reduce(x, 0, _i32_column_product_combine)
    else:
        summed = tl.sum(w, 0)
        minimum = tl.min(w, 0)
        maximum = tl.max(v, 0)
        shared_maximum = tl.max(x, 0)
    tl.store(Scanned + offset, scanned)
    tl.store(Sum + column, summed)
    tl.store(Min + column, minimum)
    tl.store(Max + column, maximum)
    tl.store(SharedMax + column, shared_maximum)


@mps_only
@pytest.mark.parametrize("rows,columns,scan_axis,num_warps,vector_exchanges",
                         [(8, 128, 0, 1, 0), (16, 32, 1, 4, 4)])
def test_multiband_scan_i32_computed_sources_column_vectors_match_torch(
        rows, columns, scan_axis, num_warps, vector_exchanges):
    size = rows * columns
    valid = size - columns - 1
    original = (torch.arange(valid) % 17 - 8).to(torch.int32)
    x = torch.zeros(size, dtype=torch.int32)
    x[:valid] = original
    x = x.reshape(rows, columns)
    w, v = x * 2 + 3, x * 3 - 1
    sentinel = -999999
    scanned = torch.full((size + 7,), sentinel, dtype=torch.int32, device="mps")
    outputs = [torch.full((columns + 7,), sentinel, dtype=torch.int32, device="mps") for _ in range(4)]
    compiled = _i32_computed_sources_scan_column_vectors_kernel[(1,)](
        original.to("mps"), scanned, *outputs, valid, rows, columns, scan_axis, num_warps=num_warps)
    raw = compiled.asm["metal"]
    msl = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    assert msl.count("threadgroup_barrier(") == 1 + vector_exchanges, msl
    buffers = sorted(int(size) for size in re.findall(
        r"\bthreadgroup\s+(?:u?int32_t|u?int)\s+\w+\s*\[\s*(\d+)\s*\]\s*;", msl))
    # Publish each computed SSA once, even when inputs share a device load;
    # the sum/min pair must reuse W. The axis-1 layout also exchanges each
    # result vector into the store layout, with its own buffer and barrier.
    assert buffers == [columns] * vector_exchanges + [size] * 3, msl
    torch.mps.synchronize()
    torch.testing.assert_close(scanned.cpu()[:size], x.cumsum(scan_axis).to(torch.int32).flatten(),
                               atol=0, rtol=0)
    assert (scanned.cpu()[size:] == sentinel).all()
    reductions = [w.sum(0).to(torch.int32), w.amin(0), v.amax(0), x.amax(0)]
    for actual, reduced in zip(outputs, reductions):
        torch.testing.assert_close(actual.cpu()[:columns], reduced, atol=0, rtol=0)
        assert (actual.cpu()[columns:] == sentinel).all()


def test_multiband_scan_i32_computed_sources_column_vectors_budget_is_named():
    result = _compile_rankn_nested_negative_in_child(
        "_i32_computed_sources_scan_column_vectors_kernel",
        {"X": "*i32", "Scanned": "*i32", "Sum": "*i32", "Min": "*i32", "Max": "*i32",
         "SharedMax": "*i32", "N": "i32"},
        {"R": 64, "C": 64, "SCAN_AXIS": 0, "PRODUCT": False, "BITWISE": False}, num_warps=4)
    output = result.stdout + result.stderr
    assert result.returncode > 0, output
    assert "full-tile exchange scratch needs" in output, output
    assert "exceeding the 32768-byte budget" in output, output


@triton.jit
def _unsupported_i32_nonplain_scan_column_vector_kernel(X, W, Y, Vector, N, MODE: tl.constexpr, PRODUCT: tl.constexpr = False, UNSIGNED: tl.constexpr = False, BITWISE: tl.constexpr = ""):
    row = tl.arange(0, 16)
    column = tl.arange(0, 16)
    offset = row[:, None] * 16 + column[None, :]
    if MODE == 8:
        x = tl.load(X + offset) * 2 + 3
    else:
        x = tl.load(X + offset, offset < N, other=5)
    if MODE != 0:
        w = tl.load(W + offset, offset < N, other=-3)
        w = w * 3 - 1
    if MODE == 10:
        other_row = tl.arange(0, 8)
        other_column = tl.arange(0, 32)
        other_offset = other_row[:, None] * 32 + other_column[None, :]
        w = tl.load(W + other_offset, other_offset < N, other=-3) * 3 - 1
    if MODE == 2:
        tl.store(W + offset, x + 17, offset < N)
    if MODE == 4:
        w = tl.trans(w)
    if MODE == 12:
        if BITWISE != "":
            reduced = _column_boundary_reduce(w, False, BITWISE)
        elif UNSIGNED:
            reduced = tl.min(w.to(tl.uint32), 0)
        else:
            reduced = tl.reduce(w, 0, _i32_column_product_combine)
    if MODE == 9:
        scanned = x
    else:
        scanned = tl.cumsum(x, 0)
    if MODE == 11:
        second = tl.cumsum(w, 0)
        tl.store(Y + offset, second, offset < N)
    if MODE == 0:
        w = tl.load(W + offset, offset < N, other=-3) * 3 - 1
    elif MODE == 1:
        w = w * 2 + 3
    elif MODE == 3:
        tl.store(W + offset, x + 17, offset < N)
    elif MODE == 5:
        w = scanned * 2 + 3
    if MODE == 7 or MODE == 8:
        reduced = tl.reduce(x, 0, _i32_column_product_combine)
    elif MODE != 12:
        if UNSIGNED:
            reduced = tl.min(w.to(tl.uint32), 0)
        elif BITWISE == "and":
            reduced = tl.reduce(w, 0, _i32_row_and_combine)
        elif BITWISE == "or":
            reduced = tl.reduce(w, 0, _i32_row_or_combine)
        elif BITWISE == "xor":
            reduced = tl.reduce(w, 0, _i32_row_xor_combine)
        elif PRODUCT:
            reduced = tl.reduce(w, 0, _i32_column_product_combine)
        else:
            reduced = tl.sum(w, 0)
    if MODE == 6:
        tl.store(Y + offset, scanned + reduced[None, :], offset < N)
    elif MODE == 13:
        tl.store(Y + offset, scanned + reduced[:, None], offset < N)
    else:
        tl.store(Y + offset, scanned, offset < N)
    if MODE == 10:
        tl.store(Vector + tl.arange(0, 32), reduced)
    elif MODE != 13:
        tl.store(Vector + column, reduced)


@pytest.mark.parametrize("mode,diagnostic", [
    (0, "distinct scan/reduce inputs require same-layout sources defined before the scan "
     "without intervening writes or input exchanges"),
    (1, "distinct scan/reduce inputs require same-layout sources defined before the scan "
     "without intervening writes or input exchanges"),
    (2, "distinct scan/reduce inputs require same-layout sources defined before the scan "
     "without intervening writes or input exchanges"),
    (3, "distinct scan/reduce inputs require same-layout sources defined before the scan "
     "without intervening writes or input exchanges"),
    (4, "a rank-2 tl.reduce cannot share a function with a multi-band staged scan"),
    (5, "distinct scan/reduce inputs require same-layout sources defined before the scan "
     "without intervening writes or input exchanges"),
    (6, "a scan-input column broadcast requires same-layout f32/i32 sources"),
], ids=["load_after_scan", "compute_after_scan", "write_before_scan", "write_after_scan",
        "transpose", "scan_result", "mixed_store"])
def test_multiband_scan_i32_nonplain_column_vector_outside_envelope_is_named(mode, diagnostic):
    result = _compile_rankn_nested_negative_in_child(
        "_unsupported_i32_nonplain_scan_column_vector_kernel",
        {"X": "*i32", "W": "*i32", "Y": "*i32", "Vector": "*i32", "N": "i32"},
        {"MODE": mode, "PRODUCT": False, "UNSIGNED": False, "BITWISE": ""}, num_warps=1)
    output = result.stdout + result.stderr
    assert result.returncode > 0, output
    assert diagnostic in output, output


@pytest.mark.parametrize("reduction", ["sum", "product", "and", "or", "xor"])
@pytest.mark.parametrize("shared", [True, False], ids=["one_source_16x512", "two_sources_8x512"])
def test_multiband_scan_i32_nonplain_column_vector_total_scratch_budget_is_named(shared, reduction):
    # Publication alone fits 32 KiB; existing scan scratch also counts.
    signature = {"X": "*i32", "Y": "*i32", "Column": "*i32", "N": "i32",
                 "VALID_COLUMNS": "i32"}
    constants = {"R": 16 if shared else 8, "C": 512, "SCAN_AXIS": 0,
                 "REVERSE": False, "REDUCTION": reduction, "COMPUTED": not shared,
                 "MASKED": shared, "OTHER": 11, "OUTPUT_MASKED": False}
    if shared:
        kernel = "_shared_input_scan_column_reduce_kernel"
        constants["SNAPSHOT"] = False
    else:
        kernel = "_different_input_scan_column_reduce_kernel"
        signature["W"] = "*i32"
        constants.update(SCAN_PLAIN=False, SCAN_MASKED=False, SCAN_COMPUTED=False)
    result = _compile_rankn_nested_negative_in_child(
        kernel, signature, constants, num_warps=4 if shared else 1)
    output = result.stdout + result.stderr
    assert result.returncode > 0, output
    assert "exceeding the 32768-byte budget" in output, output


@mps_only
def test_multiband_scan_i32_nonplain_row_reduce_baseline_matches_torch():
    rows, columns = 8, 64
    valid = rows * columns - columns - 3
    original = (torch.arange(valid) % 17 - 8).to(torch.int32)
    x = torch.zeros(rows * columns, dtype=torch.int32)
    x[:valid] = original
    x = x.reshape(rows, columns)
    sentinel = -999999
    output = torch.full((rows * columns + 7,), sentinel, dtype=torch.int32, device="mps")
    vector = torch.full((rows + 7,), sentinel, dtype=torch.int32, device="mps")
    _shared_input_scan_row_reduce_kernel[(1,)](
        original.to("mps"), output, vector, valid, rows, columns,
        1, False, True, False, False, True, num_warps=1)
    torch.mps.synchronize()
    torch.testing.assert_close(vector.cpu()[:rows], x.amax(1), atol=0, rtol=0)
    torch.testing.assert_close(output.cpu()[:valid], x.cumsum(1).to(torch.int32).flatten()[:valid],
                               atol=0, rtol=0)
    assert (vector.cpu()[rows:] == sentinel).all()
    assert (output.cpu()[valid:] == sentinel).all()


@mps_only
@pytest.mark.parametrize("reduction", ["sum", "min", "max"])
@pytest.mark.parametrize(
    "rows,columns,num_warps,scan_axis,reverse,shared,masked,computed,other,broadcast,row_masked,full_output,snapshot", [
        (8, 64, 1, 0, False, True, True, False, 5, False, False, False, False),
        (16, 16, 1, 1, True, True, False, True, 0, False, True, False, True),
        (32, 32, 4, 0, True, True, True, True, 0, False, False, True, True),
        (16, 32, 4, 1, False, True, True, True, -7, True, False, True, True),
        (8, 64, 1, 1, True, True, True, False, 11, True, False, False, False),
        (32, 64, 4, 0, False, True, False, True, 0, True, False, True, False),
        (16, 16, 1, 0, True, False, True, False, -5, False, True, False, False),
        (8, 64, 4, 1, False, False, False, True, 0, False, False, False, False),
        (32, 32, 1, 0, False, False, True, True, 0, False, False, True, False),
        (8, 64, 1, 1, True, False, True, True, 0, True, False, True, False),
        (64, 32, 4, 0, True, False, True, False, 9, True, False, True, False),
        (16, 64, 4, 1, False, False, True, True, -3, True, False, False, False),
        # Surviving row vectors may exceed the hardware thread count.
        (128, 16, 1, 1, True, True, True, True, 0, False, False, True, False),
        (256, 8, 1, 1, False, False, True, True, 7, False, True, False, False),
    ])
def test_multiband_scan_i32_nonplain_row_reduce_matches_torch(
        rows, columns, num_warps, scan_axis, reverse, shared, masked, computed,
        other, broadcast, row_masked, full_output, snapshot, reduction):
    programs = 2
    size = programs * rows * columns
    valid = size - columns - 3 if masked else size
    index = torch.arange(valid)
    if reduction == "sum":
        w_values = (2 ** 30 + index % 31).to(torch.int32)
    else:
        category = index % 4
        w_values = torch.where(category == 0, -(2 ** 31) + 7 + index % 13,
                               torch.where(category == 1, 2 ** 31 - 9 - index % 11,
                                           torch.where(category == 2, 2 ** 24 + index % 31,
                                                       -(2 ** 24) - index % 37))).to(torch.int32)
    x_values = w_values.clone() if shared else (index % 17 - 8).to(torch.int32)
    x = torch.full((size,), other, dtype=torch.int32)
    w = torch.full((size,), other, dtype=torch.int32)
    x[:valid], w[:valid] = x_values, w_values
    x, w = x.reshape(programs, rows, columns), w.reshape(programs, rows, columns)
    if computed:
        x = x * 2 + 3
        w = x if shared else w * 3 - 1
    axis = scan_axis + 1
    source = x.flip(axis) if reverse else x
    scanned = source.cumsum(axis).to(torch.int32)
    if reverse:
        scanned = scanned.flip(axis)
    if reduction == "sum":
        reduced = w.sum(2).to(torch.int32)
    elif reduction == "min":
        reduced = w.amin(2)
    else:
        reduced = w.amax(2)
    contribution = reduced[:, :, None].expand_as(x)
    base = x * 2 - scanned
    expected = base + contribution if broadcast else scanned
    sentinel = -999999
    values = x_values.to("mps")
    output = torch.full((size + 7,), sentinel, dtype=torch.int32, device="mps")
    vector = torch.full((programs * rows + 7,), sentinel, dtype=torch.int32, device="mps")
    if shared:
        _shared_input_scan_row_reduce_kernel[(programs,)](
            values, output, vector, valid, rows, columns, scan_axis,
            reverse, reduction == "max", broadcast, computed, masked,
            REDUCTION=reduction, OTHER=other, ROW_MASKED=row_masked,
            FULL_OUTPUT=full_output, SNAPSHOT=snapshot, num_warps=num_warps)
    else:
        _different_input_scan_row_reduce_kernel[(programs,)](
            values, w_values.to("mps"), output, vector, valid, rows, columns,
            scan_axis, reverse, reduction, masked, computed, broadcast, False,
            OTHER=other, ROW_MASKED=row_masked, FULL_OUTPUT=full_output, num_warps=num_warps)
    torch.mps.synchronize()
    actual = output.cpu()
    actual_vector = vector.cpu()
    written = size if full_output else valid
    torch.testing.assert_close(actual[:written], expected.flatten()[:written], atol=0, rtol=0)
    assert (actual[written:] == sentinel).all()
    if broadcast:
        torch.testing.assert_close(actual[:written] - base.flatten()[:written],
                                   contribution.flatten()[:written], atol=0, rtol=0)
        assert (actual_vector == sentinel).all()
    else:
        rows_written = actual_vector[:programs * rows].reshape(programs, rows)
        valid_rows = rows - 1 if row_masked else rows
        torch.testing.assert_close(rows_written[:, :valid_rows], reduced[:, :valid_rows], atol=0, rtol=0)
        assert (rows_written[:, valid_rows:] == sentinel).all()
        assert (actual_vector[programs * rows:] == sentinel).all()
    expected_input = x.flatten()[:valid] * 3 + 17 if snapshot else x_values
    torch.testing.assert_close(values.cpu(), expected_input, atol=0, rtol=0)


@triton.jit
def _i32_computed_sources_scan_row_reducers_kernel(
        X, Scanned, Sum, Min, Max, N, R: tl.constexpr, C: tl.constexpr,
        SCAN_AXIS: tl.constexpr, BROADCAST: tl.constexpr, MONOIDS: tl.constexpr = False,
        PRODUCT: tl.constexpr = False):
    row = tl.arange(0, R)
    column = tl.arange(0, C)
    offset = row[:, None] * C + column[None, :]
    x = tl.load(X + offset, offset < N, other=0)
    w = x * 2 + 3
    if PRODUCT:
        v = x * 4 + 3
    else:
        v = x * 3 - 1
    scanned = tl.cumsum(x, SCAN_AXIS)
    if PRODUCT:
        summed = tl.reduce(w, 1, _i32_column_product_combine)
        minimum = tl.min(w, 1)
        maximum = tl.reduce(v, 1, _i32_column_product_combine)
    elif MONOIDS:
        summed = tl.reduce(w, 1, _i32_column_product_combine)
        minimum = tl.reduce(w, 1, _i32_row_xor_combine)
        maximum = tl.reduce(v, 1, _i32_row_and_combine)
    else:
        summed = tl.sum(w, 1)
        minimum = tl.min(w, 1)
        maximum = tl.max(v, 1)
    tl.store(Scanned + offset, scanned)
    if BROADCAST:
        tl.store(Sum + offset, scanned + summed[:, None])
        tl.store(Min + offset, scanned + minimum[:, None])
        tl.store(Max + offset, scanned + maximum[:, None])
    else:
        tl.store(Sum + row, summed)
        tl.store(Min + row, minimum)
        tl.store(Max + row, maximum)


@mps_only
@pytest.mark.parametrize("rows,columns,scan_axis,num_warps,broadcast,vector_exchanges", [
    (8, 64, 0, 1, False, 3), (16, 32, 1, 4, True, 0),
])
def test_multiband_scan_i32_computed_sources_row_reducers_match_torch(
        rows, columns, scan_axis, num_warps, broadcast, vector_exchanges):
    size = rows * columns
    valid = size - columns - 1
    original = (torch.arange(valid) % 17 - 8).to(torch.int32)
    x = torch.zeros(size, dtype=torch.int32)
    x[:valid] = original
    x = x.reshape(rows, columns)
    w, v = x * 2 + 3, x * 3 - 1
    sentinel = -999999
    result_size = size if broadcast else rows
    scanned = torch.full((size + 7,), sentinel, dtype=torch.int32, device="mps")
    outputs = [torch.full((result_size + 7,), sentinel, dtype=torch.int32, device="mps") for _ in range(3)]
    compiled = _i32_computed_sources_scan_row_reducers_kernel[(1,)](
        original.to("mps"), scanned, *outputs, valid, rows, columns,
        scan_axis, broadcast, num_warps=num_warps)
    raw = compiled.asm["metal"]
    msl = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    buffers = [int(length) for length in re.findall(
        r"\bthreadgroup\s+(?:u?int32_t|u?int)\s+\w+\s*\[\s*(\d+)\s*\]\s*;", msl)]
    # Three distinct SSA inputs share one load; W's sum/min share one publish.
    # Direct row stores also exchange each vector into its output layout.
    assert sorted(buffers) == [rows] * vector_exchanges + [size] * 3, msl
    assert msl.count("threadgroup_barrier(") == 1 + vector_exchanges, msl
    torch.mps.synchronize()
    expected_scan = x.cumsum(scan_axis).to(torch.int32)
    torch.testing.assert_close(scanned.cpu()[:size], expected_scan.flatten(), atol=0, rtol=0)
    assert (scanned.cpu()[size:] == sentinel).all()
    reductions = [w.sum(1).to(torch.int32), w.amin(1), v.amax(1)]
    for actual, reduced in zip(outputs, reductions):
        expected = expected_scan + reduced[:, None] if broadcast else reduced
        torch.testing.assert_close(actual.cpu()[:result_size], expected.flatten(), atol=0, rtol=0)
        assert (actual.cpu()[result_size:] == sentinel).all()
        if broadcast:
            torch.testing.assert_close(actual.cpu()[:size].reshape(rows, columns) - expected_scan,
                                       reduced[:, None].expand_as(x), atol=0, rtol=0)


def test_multiband_scan_i32_computed_sources_row_reducers_budget_is_named():
    result = _compile_rankn_nested_negative_in_child(
        "_i32_computed_sources_scan_row_reducers_kernel",
        {"X": "*i32", "Scanned": "*i32", "Sum": "*i32", "Min": "*i32", "Max": "*i32", "N": "i32"},
        {"R": 64, "C": 64, "SCAN_AXIS": 0, "BROADCAST": False, "MONOIDS": False, "PRODUCT": False}, num_warps=4)
    output = result.stdout + result.stderr
    assert result.returncode > 0, output
    assert "full-tile exchange scratch needs" in output, output
    assert "exceeding the 32768-byte budget" in output, output


@triton.jit
def _i32_row_xor_combine(lhs, rhs):
    return lhs ^ rhs


@triton.jit
def _unsupported_i32_nonplain_scan_row_reduce_kernel(X, W, Y, Z, Row, N, MODE: tl.constexpr, MONOID: tl.constexpr = "sum"):
    row = tl.arange(0, 16)
    column = tl.arange(0, 16)
    offset = row[:, None] * 16 + column[None, :]
    if MODE == 13 or MODE == 14:
        x = tl.load(X + offset) * 2 + 3
    else:
        x = tl.load(X + offset, offset < N, other=5)
    if MODE != 3:
        w = tl.load(W + offset, offset < N, other=-3) * 3 - 1
    if MODE == 1:
        other_row = tl.arange(0, 8)
        other_column = tl.arange(0, 32)
        other_offset = other_row[:, None] * 32 + other_column[None, :]
        w = tl.load(W + other_offset, other_offset < N, other=-3) * 3 - 1
    elif MODE == 6:
        w = tl.trans(w)
    elif MODE == 4:
        tl.store(W + offset, x + 17, offset < N)
    if MODE == 15:
        reduced = _i32_row_monoid_reduce(w, MONOID)
    if MODE == 0:
        scanned = x
    else:
        scanned = tl.cumsum(x, 1)
    if MODE == 2:
        second = tl.cumsum(w, 1)
        tl.store(Z + offset, second)
    elif MODE == 3:
        w = tl.load(W + offset, offset < N, other=-3) * 3 - 1
    elif MODE == 5:
        tl.store(W + offset, x + 17, offset < N)
    elif MODE == 7:
        w = scanned * 2 + 3
    elif MODE == 12:
        w = w * 2 + 3
    if MODE == 10 or MODE == 13:
        reduced = tl.reduce(x, 1, _i32_column_product_combine)
    elif MODE == 11 or MODE == 14:
        reduced = tl.reduce(x, 1, _i32_row_xor_combine)
    elif MODE != 15:
        reduced = _i32_row_monoid_reduce(w, MONOID)
    tl.store(Y + offset, scanned, offset < N)
    if MODE == 8:
        tl.store(Z + offset, scanned + reduced[:, None], offset < N)
        tl.store(Row + row, reduced)
    elif MODE == 9:
        tl.store(Z + offset, scanned + reduced[None, :], offset < N)
    elif MODE == 1:
        tl.store(Row + tl.arange(0, 8), reduced)
    else:
        tl.store(Row + row, reduced)


@pytest.mark.parametrize("mode,diagnostic", [
    (0, "other element types require a direct unmasked tt.load"),
    (1, "other element types require a direct unmasked tt.load"),
    (2, "other element types require a direct unmasked tt.load"),
    (3, "distinct scan/reduce inputs require same-layout sources defined before the scan "
     "without intervening writes or input exchanges"),
    (4, "distinct scan/reduce inputs require same-layout sources defined before the scan "
     "without intervening writes or input exchanges"),
    (5, "distinct scan/reduce inputs require same-layout sources defined before the scan "
     "without intervening writes or input exchanges"),
    (6, "other element types require a direct unmasked tt.load"),
    (7, "distinct scan/reduce inputs require same-layout sources defined before the scan "
     "without intervening writes or input exchanges"),
    (8, "a shared scan-input row reduction requires direct row stores or "
     "same-axis broadcasts into its original tile"),
    (9, "a shared scan-input row reduction requires direct row stores or "
     "same-axis broadcasts into its original tile"),
    (12, "distinct scan/reduce inputs require same-layout sources defined before the scan "
     "without intervening writes or input exchanges"),
    (15, "other element types require a direct unmasked tt.load"),
], ids=["no_scan", "different_shape", "two_scans", "load_after_scan", "write_before_scan",
        "write_after_scan", "transpose", "scan_result", "mixed_store", "cross_axis",
        "compute_after_scan", "reduce_before_scan"])
def test_multiband_scan_i32_nonplain_row_reduce_outside_envelope_is_named(mode, diagnostic):
    result = _compile_rankn_nested_negative_in_child(
        "_unsupported_i32_nonplain_scan_row_reduce_kernel",
        {"X": "*i32", "W": "*i32", "Y": "*i32", "Z": "*i32", "Row": "*i32", "N": "i32"},
        {"MODE": mode, "MONOID": "sum"}, num_warps=1)
    output = result.stdout + result.stderr
    assert result.returncode > 0, output
    assert diagnostic in output, output


@triton.jit
def _i32_row_and_combine(lhs, rhs):
    return lhs & rhs


@triton.jit
def _i32_row_or_combine(lhs, rhs):
    return lhs | rhs


def _i32_row_monoid_reference(source, reduction):
    identity = 1 if reduction == "product" else -1 if reduction == "and" else 0
    result = torch.full(source.shape[:-1], identity, dtype=torch.int32)
    for column in range(source.shape[-1]):
        value = source[..., column]
        if reduction == "product":
            # Explicit modulo-2**32 at every step avoids int64 product overflow.
            result = ((result.to(torch.int64) * value.to(torch.int64)) & 0xffffffff).to(torch.int32)
        elif reduction == "and":
            result = result & value
        elif reduction == "or":
            result = result | value
        else:
            result = result ^ value
    return result


def _i32_row_monoid_values(size, columns, reduction, even_product=False):
    index = torch.arange(size)
    row, column = index // columns, index % columns
    if reduction == "product":
        values = (index % 13 * 2 + (2 if even_product else 1)) * torch.where(index % 3 == 0, -1, 1)
    elif reduction == "and":
        values = 0xa5000080 | ((row % 127) << 8) | (1 << (column % 7))
    elif reduction == "or":
        values = ((row % 127) << 8) | ((row % 2) << 31) | (1 << (column % 7))
    else:
        values = ((row + 1) * 0x01010101) ^ (column * 0x10204081)
        values = values ^ torch.where(column == columns - 1, row + 0x80000001, 0)
    return values.to(torch.int32)


def _run_i32_row_monoid_case(
        rows, columns, num_warps, scan_axis, reverse, shared, masked, computed,
        other, broadcast, row_masked, full_output, snapshot, reduction):
    programs = 2
    size = programs * rows * columns
    valid = size - columns - 3 if masked else size
    w_values = _i32_row_monoid_values(valid, columns, reduction,
                                     even_product=computed and not shared)
    x_values = w_values.clone() if shared else (torch.arange(valid) % 17 - 8).to(torch.int32)
    x = torch.full((size,), other, dtype=torch.int32)
    w = torch.full((size,), other, dtype=torch.int32)
    x[:valid], w[:valid] = x_values, w_values
    x, w = x.reshape(programs, rows, columns), w.reshape(programs, rows, columns)
    if computed:
        x = x * 2 + 3
        w = x if shared else w * 3 - 1
    axis = scan_axis + 1
    source = x.flip(axis) if reverse else x
    scanned = source.cumsum(axis).to(torch.int32)
    if reverse:
        scanned = scanned.flip(axis)
    reduced = _i32_row_monoid_reference(w, reduction)
    if reduction == "product":
        assert (reduced[:, :-2] != 0).all(), "product coverage must retain nonzero wrapped outputs"
    if reduction == "xor":
        assert (reduced[:, :-2] != 0).any(), "XOR coverage must not cancel every row"
    contribution = reduced[:, :, None].expand_as(x)
    base = x * 2 - scanned
    expected = base + contribution if broadcast else scanned
    sentinel = -999999
    values = x_values.to("mps")
    output = torch.full((size + 7,), sentinel, dtype=torch.int32, device="mps")
    vector = torch.full((programs * rows + 7,), sentinel, dtype=torch.int32, device="mps")
    if shared:
        _shared_input_scan_row_reduce_kernel[(programs,)](
            values, output, vector, valid, rows, columns, scan_axis,
            reverse, False, broadcast, computed, masked,
            REDUCTION=reduction, OTHER=other, ROW_MASKED=row_masked,
            FULL_OUTPUT=full_output, SNAPSHOT=snapshot, num_warps=num_warps)
    else:
        _different_input_scan_row_reduce_kernel[(programs,)](
            values, w_values.to("mps"), output, vector, valid, rows, columns,
            scan_axis, reverse, reduction, masked, computed, broadcast, False,
            OTHER=other, ROW_MASKED=row_masked, FULL_OUTPUT=full_output, num_warps=num_warps)
    torch.mps.synchronize()
    actual, actual_vector = output.cpu(), vector.cpu()
    written = size if full_output else valid
    torch.testing.assert_close(actual[:written], expected.flatten()[:written], atol=0, rtol=0)
    assert (actual[written:] == sentinel).all()
    if broadcast:
        torch.testing.assert_close(actual[:written] - base.flatten()[:written],
                                   contribution.flatten()[:written], atol=0, rtol=0)
        assert (actual_vector == sentinel).all()
    else:
        rows_written = actual_vector[:programs * rows].reshape(programs, rows)
        valid_rows = rows - 1 if row_masked else rows
        torch.testing.assert_close(rows_written[:, :valid_rows], reduced[:, :valid_rows], atol=0, rtol=0)
        assert (rows_written[:, valid_rows:] == sentinel).all()
        assert (actual_vector[programs * rows:] == sentinel).all()
    expected_input = x.flatten()[:valid] * 3 + 17 if snapshot else x_values
    torch.testing.assert_close(values.cpu(), expected_input, atol=0, rtol=0)


@mps_only
@pytest.mark.parametrize("reduction,masked", [
    ("product", False), ("and", False), ("or", False), ("xor", False), ("product", True),
], ids=["plain_product", "plain_and", "plain_or", "plain_xor", "masked_product"])
def test_multiband_scan_i32_row_monoid_baseline_matches_torch(reduction, masked):
    other = 1 if reduction == "product" else -1 if reduction == "and" else 0
    _run_i32_row_monoid_case(8, 64, 1, 1, False, True, masked, False,
                            other, False, False, False, False, reduction)


@mps_only
@pytest.mark.parametrize("reduction", ["product", "and", "or", "xor"])
@pytest.mark.parametrize(
    "rows,columns,num_warps,scan_axis,reverse,shared,masked,computed,fill,broadcast,row_masked,full_output,snapshot", [
        (8, 64, 1, 0, False, True, True, False, "identity", False, False, False, False),
        (16, 16, 1, 1, True, True, False, True, "zero", False, True, False, True),
        (32, 32, 4, 0, True, True, True, True, "nonidentity", True, False, True, True),
        (16, 32, 4, 1, False, True, True, True, "zero", False, False, True, True),
        (8, 64, 1, 1, True, False, True, False, "identity", True, False, True, False),
        (16, 64, 4, 0, False, False, False, True, "zero", False, False, False, False),
        (64, 32, 4, 0, True, False, True, True, "nonidentity", True, False, True, False),
        (32, 32, 1, 1, False, False, True, True, "zero", False, True, False, False),
        (128, 16, 1, 1, True, True, True, True, "identity", False, False, True, False),
        (256, 8, 1, 1, False, False, True, True, "zero", False, False, True, False),
    ])
def test_multiband_scan_i32_nonplain_row_monoid_matches_torch(
        rows, columns, num_warps, scan_axis, reverse, shared, masked, computed,
        fill, broadcast, row_masked, full_output, snapshot, reduction):
    if fill == "identity":
        other = 1 if reduction == "product" else -1 if reduction == "and" else 0
    elif fill == "nonidentity":
        other = {"product": -3, "and": -251658241, "or": -2147479551, "xor": -2147479549}[reduction]
    else:
        other = 0
    _run_i32_row_monoid_case(
        rows, columns, num_warps, scan_axis, reverse, shared, masked, computed,
        other, broadcast, row_masked, full_output, snapshot, reduction)


@mps_only
@pytest.mark.parametrize("rows,columns,scan_axis,num_warps,broadcast,vector_exchanges", [
    (8, 64, 0, 1, False, 3), (16, 32, 1, 4, True, 0),
])
def test_multiband_scan_i32_computed_sources_row_monoids_match_torch(
        rows, columns, scan_axis, num_warps, broadcast, vector_exchanges):
    size = rows * columns
    valid = size - columns - 1
    original = _i32_row_monoid_values(valid, columns, "xor")
    x = torch.zeros(size, dtype=torch.int32)
    x[:valid] = original
    x = x.reshape(rows, columns)
    w, v = x * 2 + 3, x * 3 - 1
    sentinel = -999999
    result_size = size if broadcast else rows
    scanned = torch.full((size + 7,), sentinel, dtype=torch.int32, device="mps")
    outputs = [torch.full((result_size + 7,), sentinel, dtype=torch.int32, device="mps") for _ in range(3)]
    compiled = _i32_computed_sources_scan_row_reducers_kernel[(1,)](
        original.to("mps"), scanned, *outputs, valid, rows, columns,
        scan_axis, broadcast, MONOIDS=True, num_warps=num_warps)
    raw = compiled.asm["metal"]
    msl = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    buffers = [int(length) for length in re.findall(
        r"\bthreadgroup\s+(?:u?int32_t|u?int)\s+\w+\s*\[\s*(\d+)\s*\]\s*;", msl)]
    # Preserve the existing publication/layout-exchange contract for these
    # layouts: three SSA sources, with two reductions sharing W's buffer.
    assert sorted(buffers) == [rows] * vector_exchanges + [size] * 3, msl
    assert msl.count("threadgroup_barrier(") == 1 + vector_exchanges, msl
    torch.mps.synchronize()
    expected_scan = x.cumsum(scan_axis).to(torch.int32)
    torch.testing.assert_close(scanned.cpu()[:size], expected_scan.flatten(), atol=0, rtol=0)
    assert (scanned.cpu()[size:] == sentinel).all()
    reductions = [_i32_row_monoid_reference(w, "product"), _i32_row_monoid_reference(w, "xor"),
                  _i32_row_monoid_reference(v, "and")]
    assert (reductions[0] != 0).all()
    assert (reductions[1] != 0).any()
    for actual, reduced in zip(outputs, reductions):
        expected = expected_scan + reduced[:, None] if broadcast else reduced
        torch.testing.assert_close(actual.cpu()[:result_size], expected.flatten(), atol=0, rtol=0)
        assert (actual.cpu()[result_size:] == sentinel).all()
        if broadcast:
            torch.testing.assert_close(actual.cpu()[:size].reshape(rows, columns) - expected_scan,
                                       reduced[:, None].expand_as(x), atol=0, rtol=0)


def test_multiband_scan_i32_computed_sources_row_monoids_budget_is_named():
    result = _compile_rankn_nested_negative_in_child(
        "_i32_computed_sources_scan_row_reducers_kernel",
        {"X": "*i32", "Scanned": "*i32", "Sum": "*i32", "Min": "*i32", "Max": "*i32", "N": "i32"},
        {"R": 64, "C": 64, "SCAN_AXIS": 0, "BROADCAST": False, "MONOIDS": True, "PRODUCT": False}, num_warps=4)
    output = result.stdout + result.stderr
    assert result.returncode > 0, output
    assert "full-tile exchange scratch needs" in output, output
    assert "exceeding the 32768-byte budget" in output, output


@triton.jit
def _i32_row_monoid_reduce(value, REDUCTION: tl.constexpr):
    if REDUCTION == "umin":
        return tl.min(value.to(tl.uint32), 1)
    elif REDUCTION == "umax":
        return tl.max(value.to(tl.uint32), 1)
    elif REDUCTION == "product":
        return tl.reduce(value, 1, _i32_column_product_combine)
    elif REDUCTION == "and":
        return tl.reduce(value, 1, _i32_row_and_combine)
    elif REDUCTION == "or":
        return tl.reduce(value, 1, _i32_row_or_combine)
    elif REDUCTION == "xor":
        return tl.reduce(value, 1, _i32_row_xor_combine)
    else:
        return tl.sum(value, 1)


@pytest.mark.parametrize("mode,reduction,diagnostic", [
    (0, "product", "other element types require a direct unmasked tt.load"),
    (1, "or", "other element types require a direct unmasked tt.load"),
    (2, "and", "other element types require a direct unmasked tt.load"),
    (3, "xor", "distinct scan/reduce inputs require same-layout sources defined before the scan "
     "without intervening writes or input exchanges"),
    (4, "product", "distinct scan/reduce inputs require same-layout sources defined before the scan "
     "without intervening writes or input exchanges"),
    (5, "or", "distinct scan/reduce inputs require same-layout sources defined before the scan "
     "without intervening writes or input exchanges"),
    (6, "and", "other element types require a direct unmasked tt.load"),
    (7, "xor", "distinct scan/reduce inputs require same-layout sources defined before the scan "
     "without intervening writes or input exchanges"),
    (8, "product", "a shared scan-input row reduction requires direct row stores or "
     "same-axis broadcasts into its original tile"),
    (9, "or", "a shared scan-input row reduction requires direct row stores or "
     "same-axis broadcasts into its original tile"),
    (12, "and", "distinct scan/reduce inputs require same-layout sources defined before the scan "
     "without intervening writes or input exchanges"),
    (15, "xor", "other element types require a direct unmasked tt.load"),
], ids=["no_scan", "different_shape", "two_scans", "load_after_scan", "write_before_scan",
        "write_after_scan", "transpose", "scan_result", "mixed_store", "cross_axis",
        "compute_after_scan", "reduce_before_scan"])
def test_multiband_scan_i32_nonplain_row_monoid_outside_envelope_is_named(mode, reduction, diagnostic):
    result = _compile_rankn_nested_negative_in_child(
        "_unsupported_i32_nonplain_scan_row_reduce_kernel",
        {"X": "*i32", "W": "*i32", "Y": "*i32", "Z": "*i32", "Row": "*i32", "N": "i32"},
        {"MODE": mode, "MONOID": reduction}, num_warps=1)
    output = result.stdout + result.stderr
    assert result.returncode > 0, output
    assert diagnostic in output, output


@pytest.mark.parametrize("dtype,reduction,diagnostic", [
    ("*i64", "product", "rank >= 2 scan is implemented for a blocked tile"),
])
def test_multiband_scan_nonplain_row_monoid_excluded_type_is_named(dtype, reduction, diagnostic):
    result = _compile_rankn_nested_negative_in_child(
        "_shared_input_scan_row_reduce_kernel",
        {"X": dtype, "Y": dtype, "Row": dtype, "N": "i32"},
        {"R": 8, "C": 64, "SCAN_AXIS": 1, "REVERSE": False, "MAX": False,
         "BROADCAST": False, "COMPUTED": True, "MASKED": True, "REDUCTION": reduction,
         "OTHER": 1, "ROW_MASKED": False, "FULL_OUTPUT": False, "SNAPSHOT": False}, num_warps=1)
    output = result.stdout + result.stderr
    assert result.returncode > 0, output
    assert diagnostic in output, output


@mps_only
@pytest.mark.parametrize("masked,other", [(False, 1), (True, 1), (True, 0), (True, -3)],
                         ids=["plain", "masked", "zero_fill", "nonidentity_fill"])
def test_multiband_scan_i32_column_product_baseline_matches_torch(masked, other):
    rows, columns = 8, 64
    size = rows * columns
    valid = size - columns - 3 if masked else size
    if masked and other != 1:
        valid = columns - 3
    original = _i32_row_monoid_values(valid, columns, "product")
    x = torch.full((size,), other, dtype=torch.int32)
    x[:valid] = original
    x = x.reshape(rows, columns)
    sentinel = -999999
    output = torch.full((size + 7,), sentinel, dtype=torch.int32, device="mps")
    vector = torch.full((columns + 7,), sentinel, dtype=torch.int32, device="mps")
    _shared_input_scan_column_reduce_kernel[(1,)](
        original.to("mps"), output, vector, valid, columns, rows, columns,
        0, False, "product", False, masked, False,
        OTHER=other, OUTPUT_MASKED=False, num_warps=1)
    torch.mps.synchronize()
    reduced = _i32_row_monoid_reference(x.T, "product")
    if other != 0:
        assert (reduced != 0).all()
    torch.testing.assert_close(vector.cpu()[:columns], reduced, atol=0, rtol=0)
    torch.testing.assert_close(output.cpu()[:valid], x.cumsum(0).to(torch.int32).flatten()[:valid],
                               atol=0, rtol=0)
    assert (vector.cpu()[columns:] == sentinel).all()
    assert (output.cpu()[valid:] == sentinel).all()


@mps_only
@pytest.mark.parametrize("rows,columns,scan_axis,num_warps,vector_exchanges",
                         [(8, 128, 0, 1, 0), (16, 32, 1, 4, 4)])
def test_multiband_scan_i32_computed_sources_column_products_match_torch(
        rows, columns, scan_axis, num_warps, vector_exchanges):
    size = rows * columns
    valid = size - (rows - 1) * columns - 3
    original = _i32_row_monoid_values(valid, columns, "product")
    x = torch.zeros(size, dtype=torch.int32)
    x[:valid] = original
    x = x.reshape(rows, columns)
    w, v = x * 2 + 3, x * 4 + 1
    sentinel = -999999
    scanned = torch.full((size + 7,), sentinel, dtype=torch.int32, device="mps")
    outputs = [torch.full((columns + 7,), sentinel, dtype=torch.int32, device="mps") for _ in range(4)]
    compiled = _i32_computed_sources_scan_column_vectors_kernel[(1,)](
        original.to("mps"), scanned, *outputs, valid, rows, columns, scan_axis,
        PRODUCT=True, num_warps=num_warps)
    raw = compiled.asm["metal"]
    msl = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    buffers = sorted(int(length) for length in re.findall(
        r"\bthreadgroup\s+(?:u?int32_t|u?int)\s+\w+\s*\[\s*(\d+)\s*\]\s*;", msl))
    # W product/min reuse one publication; X/W/V still own distinct buffers.
    assert buffers == [columns] * vector_exchanges + [size] * 3, msl
    assert msl.count("threadgroup_barrier(") == 1 + vector_exchanges, msl
    torch.mps.synchronize()
    torch.testing.assert_close(scanned.cpu()[:size], x.cumsum(scan_axis).to(torch.int32).flatten(),
                               atol=0, rtol=0)
    assert (scanned.cpu()[size:] == sentinel).all()
    reductions = [_i32_row_monoid_reference(w.T, "product"), w.amin(0),
                  _i32_row_monoid_reference(v.T, "product"), _i32_row_monoid_reference(x.T, "product")]
    assert (reductions[0] != 0).all()
    assert (reductions[2] != 0).all()
    for actual, reduced in zip(outputs, reductions):
        torch.testing.assert_close(actual.cpu()[:columns], reduced, atol=0, rtol=0)
        assert (actual.cpu()[columns:] == sentinel).all()


def test_multiband_scan_i32_computed_sources_column_products_budget_is_named():
    result = _compile_rankn_nested_negative_in_child(
        "_i32_computed_sources_scan_column_vectors_kernel",
        {"X": "*i32", "Scanned": "*i32", "Sum": "*i32", "Min": "*i32", "Max": "*i32",
         "SharedMax": "*i32", "N": "i32"},
        {"R": 64, "C": 64, "SCAN_AXIS": 0, "PRODUCT": True, "BITWISE": False}, num_warps=4)
    output = result.stdout + result.stderr
    assert result.returncode > 0, output
    assert "full-tile exchange scratch needs" in output, output
    assert "exceeding the 32768-byte budget" in output, output


@pytest.mark.parametrize("mode,diagnostic", [
    (0, "distinct scan/reduce inputs require same-layout sources defined before the scan "
     "without intervening writes or input exchanges"),
    (1, "distinct scan/reduce inputs require same-layout sources defined before the scan "
     "without intervening writes or input exchanges"),
    (2, "distinct scan/reduce inputs require same-layout sources defined before the scan "
     "without intervening writes or input exchanges"),
    (3, "distinct scan/reduce inputs require same-layout sources defined before the scan "
     "without intervening writes or input exchanges"),
    (4, "rank-2 axis=0 i32 product requires a direct unmasked tt.load"),
    (5, "distinct scan/reduce inputs require same-layout sources defined before the scan "
     "without intervening writes or input exchanges"),
    (6, "rank-2 axis=0 i32 product requires a direct unmasked tt.load"),
    (9, "rank-2 axis=0 i32 product requires a direct unmasked tt.load"),
    (10, "rank-2 axis=0 i32 product requires a direct unmasked tt.load"),
    (11, "rank-2 axis=0 i32 product requires a direct unmasked tt.load"),
    (12, "rank-2 axis=0 i32 product requires a direct unmasked tt.load"),
    (13, "rank-2 axis=0 i32 product requires a direct unmasked tt.load"),
], ids=["load_after_scan", "compute_after_scan", "write_before_scan", "write_after_scan",
        "transpose", "scan_result", "mixed_store", "no_scan", "different_shape",
        "two_scans", "reduce_before_scan", "cross_axis"])
def test_multiband_scan_i32_nonplain_column_product_outside_envelope_is_named(mode, diagnostic):
    result = _compile_rankn_nested_negative_in_child(
        "_unsupported_i32_nonplain_scan_column_vector_kernel",
        {"X": "*i32", "W": "*i32", "Y": "*i32", "Vector": "*i32", "N": "i32"},
        {"MODE": mode, "PRODUCT": True, "UNSIGNED": False, "BITWISE": ""}, num_warps=1)
    output = result.stdout + result.stderr
    assert result.returncode > 0, output
    assert diagnostic in output, output


@mps_only
@pytest.mark.parametrize("reduction", ["umin", "umax"])
def test_multiband_scan_unsigned_row_extrema_baseline_matches_torch(reduction):
    rows, columns = 8, 64
    size = rows * columns
    valid = size - columns - 3
    bits = torch.tensor([0, 1, 0x7fffffff, 0x80000000, 0xffffffff], dtype=torch.int64)
    original = bits[torch.arange(valid) % 5].to(torch.int32)
    other = -1 if reduction == "umin" else 0
    x = torch.full((size,), other, dtype=torch.int32)
    x[:valid] = original
    x = x.reshape(rows, columns)
    unsigned = x.to(torch.int64) & 0xffffffff
    expected = (unsigned.amin(1) if reduction == "umin" else unsigned.amax(1)).to(torch.int32)
    sentinel = -999999
    output = torch.full((size + 7,), sentinel, dtype=torch.int32, device="mps")
    vector = torch.full((rows + 7,), sentinel, dtype=torch.int32, device="mps")
    _shared_input_scan_row_reduce_kernel[(1,)](
        original.to("mps"), output, vector, valid, rows, columns,
        1, False, False, False, False, True, REDUCTION=reduction,
        OTHER=other, num_warps=1)
    torch.mps.synchronize()
    torch.testing.assert_close(vector.cpu()[:rows], expected, atol=0, rtol=0)
    torch.testing.assert_close(output.cpu()[:valid], x.cumsum(1).to(torch.int32).flatten()[:valid],
                               atol=0, rtol=0)
    assert (output.cpu()[valid:] == sentinel).all()
    assert (vector.cpu()[rows:] == sentinel).all()


def _unsigned_extrema_values(size, columns):
    index = torch.arange(size)
    row, column = index // columns, index % columns
    bits = torch.tensor([0, 1, 0x7fffffff, 0x80000000, 0xffffffff, 0x80000001,
                         0x40000000, 0xc0000001, 0x55555555, 0xaaaaaaaa], dtype=torch.int64)
    return bits[(row * 3 + column + row // 3) % len(bits)].to(torch.int32)


@mps_only
@pytest.mark.parametrize("reduction", ["umin", "umax"])
@pytest.mark.parametrize(
    "rows,columns,num_warps,reduce_axis,scan_axis,reverse,shared,masked,computed,fill,broadcast,output_masked,full_output,snapshot", [
        (8, 64, 1, 1, 0, False, True, False, False, "identity", False, False, False, False),
        (16, 16, 1, 1, 1, True, True, True, False, "identity", False, False, True, False),
        (32, 32, 4, 1, 0, True, True, False, True, "zero", True, False, True, True),
        (16, 32, 4, 1, 1, False, False, True, False, "identity", True, False, True, False),
        (32, 32, 1, 1, 0, False, False, True, True, "nonidentity", False, True, False, False),
        (128, 16, 1, 1, 1, True, True, True, True, "zero", False, False, True, True),
        (64, 32, 4, 1, 0, True, False, False, True, "nonidentity", True, False, False, False),
        (8, 64, 1, 0, 1, True, True, True, False, "identity", False, False, False, False),
        (16, 16, 4, 0, 0, False, True, False, True, "nonidentity", False, False, False, False),
        (8, 128, 1, 0, 0, True, False, True, True, "identity", False, False, False, False),
        (4, 512, 1, 0, 0, False, False, True, False, "nonidentity", False, False, False, False),
        (8, 256, 4, 0, 0, True, True, True, True, "nonidentity", False, True, False, True),
        (8, 512, 4, 0, 0, False, True, False, False, "identity", False, False, False, False),
    ])
def test_multiband_scan_unsigned_extrema_matches_torch(
        rows, columns, num_warps, reduce_axis, scan_axis, reverse, shared, masked, computed,
        fill, broadcast, output_masked, full_output, snapshot, reduction):
    programs = 2
    size = programs * rows * columns
    valid = size - columns - 3 if masked else size
    if masked and reduce_axis == 0:
        # Leave complete columns masked in the second program, including
        # columns written by the unmasked vector-store configurations.
        valid = size - (rows - 1) * columns - 3
    other = (-1 if reduction == "umin" else 0) if fill == "identity" else -2147483647 if fill == "nonidentity" else 0
    w_values = _unsigned_extrema_values(valid, columns)
    x_values = w_values.clone() if shared else (torch.arange(valid) % 17 - 8).to(torch.int32)
    x = torch.full((size,), other, dtype=torch.int32)
    w = torch.full((size,), other, dtype=torch.int32)
    x[:valid], w[:valid] = x_values, w_values
    x, w = x.reshape(programs, rows, columns), w.reshape(programs, rows, columns)
    if computed:
        if shared or reduce_axis == 1:
            x = x * 2 + 3
        w = x if shared else w * 3 - 1
    axis = scan_axis + 1
    source = x.flip(axis) if reverse else x
    scanned = source.cumsum(axis).to(torch.int32)
    if reverse:
        scanned = scanned.flip(axis)
    unsigned = w.to(torch.int64) & 0xffffffff
    reduced = (unsigned.amin(reduce_axis + 1) if reduction == "umin"
               else unsigned.amax(reduce_axis + 1)).to(torch.int32)
    signed = w.amin(reduce_axis + 1) if reduction == "umin" else w.amax(reduce_axis + 1)
    assert (reduced[0] != signed[0]).any(), "inputs must distinguish unsigned and signed comparisons"
    sentinel = -999999
    vector_width = rows if reduce_axis == 1 else columns
    valid_vector = vector_width - (1 if reduce_axis == 1 else 3) if output_masked else vector_width
    values = x_values.to("mps")
    output = torch.full((size + 7,), sentinel, dtype=torch.int32, device="mps")
    vector = torch.full((programs * vector_width + 7,), sentinel, dtype=torch.int32, device="mps")
    if reduce_axis == 1:
        if shared:
            _shared_input_scan_row_reduce_kernel[(programs,)](
                values, output, vector, valid, rows, columns, scan_axis,
                reverse, False, broadcast, computed, masked, REDUCTION=reduction,
                OTHER=other, ROW_MASKED=output_masked, FULL_OUTPUT=full_output,
                SNAPSHOT=snapshot, num_warps=num_warps)
        else:
            _different_input_scan_row_reduce_kernel[(programs,)](
                values, w_values.to("mps"), output, vector, valid, rows, columns,
                scan_axis, reverse, reduction, masked, computed, broadcast, False,
                OTHER=other, ROW_MASKED=output_masked, FULL_OUTPUT=full_output, num_warps=num_warps)
    elif shared:
        _shared_input_scan_column_reduce_kernel[(programs,)](
            values, output, vector, valid, valid_vector, rows, columns, scan_axis,
            reverse, reduction, computed, masked, snapshot, OTHER=other,
            OUTPUT_MASKED=output_masked, num_warps=num_warps)
    else:
        _different_input_scan_column_reduce_kernel[(programs,)](
            values, w_values.to("mps"), output, vector, valid, valid_vector,
            rows, columns, scan_axis, reverse, reduction, masked, computed,
            OTHER=other, OUTPUT_MASKED=output_masked, num_warps=num_warps)
    torch.mps.synchronize()
    actual, actual_vector = output.cpu(), vector.cpu()
    base = x * 2 - scanned
    expected = base + reduced[:, :, None] if broadcast else scanned
    written = size if full_output else valid
    torch.testing.assert_close(actual[:written], expected.flatten()[:written], atol=0, rtol=0)
    assert (actual[written:] == sentinel).all()
    if broadcast:
        contribution = reduced[:, :, None].expand_as(x)
        torch.testing.assert_close(actual[:written] - base.flatten()[:written],
                                   contribution.flatten()[:written], atol=0, rtol=0)
        assert (actual_vector == sentinel).all()
    else:
        vectors = actual_vector[:programs * vector_width].reshape(programs, vector_width)
        torch.testing.assert_close(vectors[:, :valid_vector], reduced[:, :valid_vector], atol=0, rtol=0)
        assert (vectors[:, valid_vector:] == sentinel).all()
        assert (actual_vector[programs * vector_width:] == sentinel).all()
    expected_input = x.flatten()[:valid] * 3 + 17 if snapshot else x_values
    torch.testing.assert_close(values.cpu(), expected_input, atol=0, rtol=0)


@triton.jit
def _mixed_signed_unsigned_scan_reducers_kernel(
        X, Scanned, SignedMin, UnsignedMin, SignedMax, UnsignedMax, N,
        R: tl.constexpr, C: tl.constexpr, SCAN_AXIS: tl.constexpr,
        REDUCE_AXIS: tl.constexpr, BROADCAST: tl.constexpr, COLUMN_BROADCAST: tl.constexpr = False):
    row = tl.arange(0, R)
    column = tl.arange(0, C)
    offset = row[:, None] * C + column[None, :]
    x = tl.load(X + offset, offset < N, other=-1)
    w, v = x * 2 + 3, x * 3 - 1
    scanned = tl.cumsum(x, SCAN_AXIS)
    signed_min = tl.min(w, REDUCE_AXIS)
    unsigned_min = tl.min(w.to(tl.uint32), REDUCE_AXIS)
    signed_max = tl.max(v, REDUCE_AXIS)
    unsigned_max = tl.max(v.to(tl.uint32), REDUCE_AXIS)
    tl.store(Scanned + offset, scanned)
    if COLUMN_BROADCAST:
        tl.store(SignedMin + offset, scanned + signed_min[None, :])
        tl.store(UnsignedMin + offset, scanned + unsigned_min[None, :])
        tl.store(SignedMax + offset, scanned + signed_max[None, :])
        tl.store(UnsignedMax + offset, scanned + unsigned_max[None, :])
    elif BROADCAST:
        tl.store(SignedMin + offset, scanned + signed_min[:, None])
        tl.store(UnsignedMin + offset, scanned + unsigned_min[:, None])
        tl.store(SignedMax + offset, scanned + signed_max[:, None])
        tl.store(UnsignedMax + offset, scanned + unsigned_max[:, None])
    else:
        if REDUCE_AXIS == 0:
            index = column
        else:
            index = row
        tl.store(SignedMin + index, signed_min)
        tl.store(UnsignedMin + index, unsigned_min)
        tl.store(SignedMax + index, signed_max)
        tl.store(UnsignedMax + index, unsigned_max)


@mps_only
@pytest.mark.parametrize("rows,columns,scan_axis,reduce_axis,num_warps,broadcast,vector_exchanges", [
    (8, 128, 0, 0, 1, False, 0),
    (16, 32, 1, 1, 4, True, 0),
    (8, 64, 0, 1, 1, False, 4),
])
def test_multiband_scan_mixed_signed_unsigned_reducers_match_torch(
        rows, columns, scan_axis, reduce_axis, num_warps, broadcast, vector_exchanges):
    size = rows * columns
    valid = size - columns - 1
    original = _unsigned_extrema_values(valid, columns)
    x = torch.full((size,), -1, dtype=torch.int32)
    x[:valid] = original
    x = x.reshape(rows, columns)
    w, v = x * 2 + 3, x * 3 - 1
    wu, vu = w.to(torch.int64) & 0xffffffff, v.to(torch.int64) & 0xffffffff
    expected_reductions = [w.amin(reduce_axis), wu.amin(reduce_axis).to(torch.int32),
                           v.amax(reduce_axis), vu.amax(reduce_axis).to(torch.int32)]
    assert (expected_reductions[0] != expected_reductions[1]).any()
    assert (expected_reductions[2] != expected_reductions[3]).any()
    vector_width = columns if reduce_axis == 0 else rows
    result_size = size if broadcast else vector_width
    sentinel = -999999
    scanned = torch.full((size + 7,), sentinel, dtype=torch.int32, device="mps")
    outputs = [torch.full((result_size + 7,), sentinel, dtype=torch.int32, device="mps") for _ in range(4)]
    compiled = _mixed_signed_unsigned_scan_reducers_kernel[(1,)](
        original.to("mps"), scanned, *outputs, valid, rows, columns,
        scan_axis, reduce_axis, broadcast, num_warps=num_warps)
    raw = compiled.asm["metal"]
    msl = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    buffers = sorted(int(length) for length in re.findall(
        r"\bthreadgroup\s+(?:u?int32_t|u?int)\s+\w+\s*\[\s*(\d+)\s*\]\s*;", msl))
    # Signed/unsigned folds share source publications, never comparison results.
    assert buffers == [vector_width] * vector_exchanges + [size] * 3, msl
    assert msl.count("threadgroup_barrier(") == 1 + vector_exchanges, msl
    torch.mps.synchronize()
    expected_scan = x.cumsum(scan_axis).to(torch.int32)
    torch.testing.assert_close(scanned.cpu()[:size], expected_scan.flatten(), atol=0, rtol=0)
    assert (scanned.cpu()[size:] == sentinel).all()
    for actual, reduced in zip(outputs, expected_reductions):
        expected = expected_scan + reduced[:, None] if broadcast else reduced
        torch.testing.assert_close(actual.cpu()[:result_size], expected.flatten(), atol=0, rtol=0)
        assert (actual.cpu()[result_size:] == sentinel).all()


def test_multiband_scan_unsigned_extrema_combined_budget_is_named():
    result = _compile_rankn_nested_negative_in_child(
        "_mixed_signed_unsigned_scan_reducers_kernel",
        {"X": "*i32", "Scanned": "*i32", "SignedMin": "*i32", "UnsignedMin": "*i32",
         "SignedMax": "*i32", "UnsignedMax": "*i32", "N": "i32"},
        {"R": 64, "C": 64, "SCAN_AXIS": 0, "REDUCE_AXIS": 1, "BROADCAST": False, "COLUMN_BROADCAST": False}, num_warps=4)
    output = result.stdout + result.stderr
    assert result.returncode > 0, output
    assert "full-tile exchange scratch needs" in output, output
    assert "exceeding the 32768-byte budget" in output, output


@pytest.mark.parametrize("mode,diagnostic", [
    (0, "reduce combine requires Session L3c (future)"),
    (1, "reduce combine requires Session L3c (future)"),
    (2, "reduce combine requires Session L3c (future)"),
    (3, "distinct scan/reduce inputs require same-layout sources defined before the scan "
     "without intervening writes or input exchanges"),
    (4, "distinct scan/reduce inputs require same-layout sources defined before the scan "
     "without intervening writes or input exchanges"),
    (5, "distinct scan/reduce inputs require same-layout sources defined before the scan "
     "without intervening writes or input exchanges"),
    (6, "reduce combine requires Session L3c (future)"),
    (7, "distinct scan/reduce inputs require same-layout sources defined before the scan "
     "without intervening writes or input exchanges"),
    (8, "a shared scan-input row reduction requires direct row stores or "
     "same-axis broadcasts into its original tile"),
    (9, "a shared scan-input row reduction requires direct row stores or "
     "same-axis broadcasts into its original tile"),
    (12, "distinct scan/reduce inputs require same-layout sources defined before the scan "
     "without intervening writes or input exchanges"),
    (15, "reduce combine requires Session L3c (future)"),
], ids=["no_scan", "different_shape", "two_scans", "load_after_scan", "write_before_scan",
        "write_after_scan", "transpose", "scan_result", "mixed_store", "cross_axis",
        "compute_after_scan", "reduce_before_scan"])
def test_multiband_scan_unsigned_row_extrema_outside_envelope_is_named(mode, diagnostic):
    result = _compile_rankn_nested_negative_in_child(
        "_unsupported_i32_nonplain_scan_row_reduce_kernel",
        {"X": "*i32", "W": "*i32", "Y": "*i32", "Z": "*i32", "Row": "*i32", "N": "i32"},
        {"MODE": mode, "MONOID": "umax"}, num_warps=1)
    output = result.stdout + result.stderr
    assert result.returncode > 0, output
    assert diagnostic in output, output


@pytest.mark.parametrize("mode,diagnostic", [
    (0, "distinct scan/reduce inputs require same-layout sources defined before the scan "
     "without intervening writes or input exchanges"),
    (3, "distinct scan/reduce inputs require same-layout sources defined before the scan "
     "without intervening writes or input exchanges"),
    (4, "axis=0 reduce supports f32/i32 sum/product/max/min"),
    (6, "axis=0 reduce supports f32/i32 sum/product/max/min"),
    (9, "axis=0 reduce supports f32/i32 sum/product/max/min"),
    (12, "axis=0 reduce supports f32/i32 sum/product/max/min"),
    (13, "axis=0 reduce supports f32/i32 sum/product/max/min"),
], ids=["load_after_scan", "write_after_scan", "transpose", "mixed_store", "no_scan",
        "reduce_before_scan", "cross_axis"])
def test_multiband_scan_unsigned_column_extrema_outside_envelope_is_named(mode, diagnostic):
    result = _compile_rankn_nested_negative_in_child(
        "_unsupported_i32_nonplain_scan_column_vector_kernel",
        {"X": "*i32", "W": "*i32", "Y": "*i32", "Vector": "*i32", "N": "i32"},
        {"MODE": mode, "PRODUCT": False, "UNSIGNED": True, "BITWISE": ""}, num_warps=1)
    output = result.stdout + result.stderr
    assert result.returncode > 0, output
    assert diagnostic in output, output


@triton.jit
def _unsupported_unsigned_extrema_rank_kernel(X, Y, RANK: tl.constexpr):
    if RANK == 1:
        offset = tl.arange(0, 128)
        x = tl.load(X + offset).to(tl.uint32)
        tl.store(Y, tl.min(x, 0))
    else:
        batch = tl.arange(0, 2)
        row = tl.arange(0, 8)
        column = tl.arange(0, 16)
        offset = batch[:, None, None] * 128 + row[None, :, None] * 16 + column[None, None, :]
        x = tl.load(X + offset).to(tl.uint32)
        reduced = tl.max(x, 1)
        tl.store(Y + batch[:, None] * 16 + column[None, :], reduced)


@pytest.mark.parametrize("rank,diagnostic", [
    (1, "reduce combine requires Session L3c (future)"),
    (3, "rank >= 3 reduce is implemented for a top-level blocked tile"),
])
def test_unsigned_extrema_without_scan_rank_is_named(rank, diagnostic):
    result = _compile_rankn_nested_negative_in_child(
        "_unsupported_unsigned_extrema_rank_kernel", {"X": "*i32", "Y": "*i32"},
        {"RANK": rank}, num_warps=1)
    output = result.stdout + result.stderr
    assert result.returncode > 0, output
    assert diagnostic in output, output


@pytest.mark.parametrize("dtype,rows,columns,diagnostic", [
    ("*u32", 4, 4, "reduce combine requires Session L3c (future)"),
    ("*u64", 8, 64, "rank >= 2 scan is implemented for a blocked tile"),
], ids=["single_band", "unsigned_i64"])
def test_multiband_scan_unsigned_extrema_stage_and_width_is_named(dtype, rows, columns, diagnostic):
    result = _compile_rankn_nested_negative_in_child(
        "_shared_input_scan_row_reduce_kernel",
        {"X": dtype, "Y": dtype, "Row": dtype, "N": "i32"},
        {"R": rows, "C": columns, "SCAN_AXIS": 1, "REVERSE": False, "MAX": False,
         "BROADCAST": False, "COMPUTED": False, "MASKED": True, "REDUCTION": "min",
         "OTHER": 0, "ROW_MASKED": False, "FULL_OUTPUT": False, "SNAPSHOT": False}, num_warps=1)
    output = result.stdout + result.stderr
    assert result.returncode > 0, output
    assert diagnostic in output, output


@mps_only
@pytest.mark.parametrize("reduction", ["umin", "umax"])
@pytest.mark.parametrize("rows,columns,scan_axis", [(16, 16, 1), (8, 128, 0)])
def test_multiband_scan_unsigned_column_broadcast_baseline_matches_torch(rows, columns, scan_axis, reduction):
    size = rows * columns
    original = _unsigned_extrema_values(size, columns)
    x = original.reshape(rows, columns)
    unsigned = x.to(torch.int64) & 0xffffffff
    reduced = (unsigned.amin(0) if reduction == "umin" else unsigned.amax(0)).to(torch.int32)
    signed = x.amin(0) if reduction == "umin" else x.amax(0)
    assert (reduced != signed).any()
    sentinel = -999999
    output = torch.full((size + 7,), sentinel, dtype=torch.int32, device="mps")
    values = original.to("mps")
    _i32_nonplain_input_scan_column_broadcast_kernel[(1,)](
        values, values, output, size, rows, columns, scan_axis, False,
        reduction, True, False, False, 0, False, True, num_warps=1)
    torch.mps.synchronize()
    scanned = x.cumsum(scan_axis).to(torch.int32)
    expected = scanned + reduced[None, :]
    torch.testing.assert_close(output.cpu()[:size], expected.flatten(), atol=0, rtol=0)
    torch.testing.assert_close(output.cpu()[:size].reshape(rows, columns) - scanned,
                               reduced[None, :].expand_as(x), atol=0, rtol=0)
    assert (output.cpu()[size:] == sentinel).all()


@mps_only
@pytest.mark.parametrize("reduction", ["umin", "umax"])
@pytest.mark.parametrize(
    "rows,columns,num_warps,scan_axis,reverse,shared,masked,computed,fill,snapshot,full_output", [
        (16, 16, 1, 1, True, True, True, False, "identity", False, True),
        (8, 64, 1, 1, False, True, False, True, "zero", True, True),
        (8, 128, 1, 0, True, True, True, True, "zero", True, True),
        (8, 256, 4, 0, False, True, True, False, "nonidentity", False, True),
        (32, 32, 4, 0, True, True, True, True, "identity", True, False),
        (32, 64, 4, 1, False, True, False, False, "identity", False, True),
        (16, 16, 1, 0, True, False, True, False, "identity", False, True),
        (8, 64, 4, 1, False, False, False, True, "zero", False, True),
        (8, 128, 1, 0, False, False, False, False, "identity", False, True),
        (8, 256, 4, 0, True, False, True, True, "zero", False, True),
        (16, 128, 4, 0, False, False, True, True, "nonidentity", False, False),
        (64, 32, 4, 1, True, False, True, False, "nonidentity", False, True),
    ])
def test_multiband_scan_unsigned_column_broadcast_matches_torch(
        rows, columns, num_warps, scan_axis, reverse, shared, masked, computed,
        fill, snapshot, full_output, reduction):
    programs = 2
    size = programs * rows * columns
    valid = size - (rows - 1) * columns - 3 if masked else size
    other = (-1 if reduction == "umin" else 0) if fill == "identity" else -2147483647 if fill == "nonidentity" else 0
    w_values = _unsigned_extrema_values(valid, columns)
    x_values = w_values.clone() if shared else (torch.arange(valid) % 17 - 8).to(torch.int32)
    x = torch.full((size,), other, dtype=torch.int32)
    w = torch.full((size,), other, dtype=torch.int32)
    x[:valid], w[:valid] = x_values, w_values
    x, w = x.reshape(programs, rows, columns), w.reshape(programs, rows, columns)
    if computed:
        if shared:
            x = x * 2 + 3
            w = x
        else:
            w = w * 3 - 1
    axis = scan_axis + 1
    source = x.flip(axis) if reverse else x
    scanned = source.cumsum(axis).to(torch.int32)
    if reverse:
        scanned = scanned.flip(axis)
    unsigned = w.to(torch.int64) & 0xffffffff
    reduced = (unsigned.amin(1) if reduction == "umin" else unsigned.amax(1)).to(torch.int32)
    signed = w.amin(1) if reduction == "umin" else w.amax(1)
    assert (reduced[0] != signed[0]).any(), "input must distinguish unsigned and signed extrema"
    contribution = reduced[:, None, :].expand_as(x)
    expected = scanned + contribution
    sentinel = -999999
    values = x_values.to("mps")
    output = torch.full((size + 7,), sentinel, dtype=torch.int32, device="mps")
    _i32_nonplain_input_scan_column_broadcast_kernel[(programs,)](
        values, w_values.to("mps"), output, valid, rows, columns, scan_axis, reverse,
        reduction, shared, masked, computed, other, snapshot, full_output, num_warps=num_warps)
    torch.mps.synchronize()
    actual = output.cpu()
    written = size if full_output else valid
    torch.testing.assert_close(actual[:written], expected.flatten()[:written], atol=0, rtol=0)
    torch.testing.assert_close(actual[:written] - scanned.flatten()[:written],
                               contribution.flatten()[:written], atol=0, rtol=0)
    assert (actual[written:] == sentinel).all()
    expected_input = x.flatten()[:valid] * 2 - 7 if snapshot else x_values
    torch.testing.assert_close(values.cpu(), expected_input, atol=0, rtol=0)


@mps_only
@pytest.mark.parametrize("rows,columns,scan_axis,num_warps", [(16, 16, 1, 1), (8, 128, 0, 4)])
def test_multiband_scan_mixed_signed_unsigned_column_broadcasts_match_torch(rows, columns, scan_axis, num_warps):
    size = rows * columns
    valid = columns - 3
    original = _unsigned_extrema_values(valid, columns)
    x = torch.full((size,), -1, dtype=torch.int32)
    x[:valid] = original
    x = x.reshape(rows, columns)
    w, v = x * 2 + 3, x * 3 - 1
    wu, vu = w.to(torch.int64) & 0xffffffff, v.to(torch.int64) & 0xffffffff
    reductions = [w.amin(0), wu.amin(0).to(torch.int32), v.amax(0), vu.amax(0).to(torch.int32)]
    assert (reductions[0] != reductions[1]).any()
    assert (reductions[2] != reductions[3]).any()
    sentinel = -999999
    scanned = torch.full((size + 7,), sentinel, dtype=torch.int32, device="mps")
    outputs = [torch.full((size + 7,), sentinel, dtype=torch.int32, device="mps") for _ in range(4)]
    compiled = _mixed_signed_unsigned_scan_reducers_kernel[(1,)](
        original.to("mps"), scanned, *outputs, valid, rows, columns,
        scan_axis, 0, False, COLUMN_BROADCAST=True, num_warps=num_warps)
    raw = compiled.asm["metal"]
    msl = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    buffers = sorted(int(length) for length in re.findall(
        r"\bthreadgroup\s+(?:u?int32_t|u?int)\s+\w+\s*\[\s*(\d+)\s*\]\s*;", msl))
    # Mixed comparison domains reuse their inputs, without publishing vectors.
    assert buffers == [size] * 3, msl
    assert msl.count("threadgroup_barrier(") == 1, msl
    torch.mps.synchronize()
    expected_scan = x.cumsum(scan_axis).to(torch.int32)
    torch.testing.assert_close(scanned.cpu()[:size], expected_scan.flatten(), atol=0, rtol=0)
    assert (scanned.cpu()[size:] == sentinel).all()
    for actual, reduced in zip(outputs, reductions):
        contribution = reduced[None, :].expand_as(x)
        torch.testing.assert_close(actual.cpu()[:size], (expected_scan + contribution).flatten(), atol=0, rtol=0)
        torch.testing.assert_close(actual.cpu()[:size].reshape(rows, columns) - expected_scan,
                                   contribution, atol=0, rtol=0)
        assert (actual.cpu()[size:] == sentinel).all()


def test_multiband_scan_unsigned_column_broadcast_combined_budget_is_named():
    result = _compile_rankn_nested_negative_in_child(
        "_mixed_signed_unsigned_scan_reducers_kernel",
        {"X": "*i32", "Scanned": "*i32", "SignedMin": "*i32", "UnsignedMin": "*i32",
         "SignedMax": "*i32", "UnsignedMax": "*i32", "N": "i32"},
        {"R": 64, "C": 64, "SCAN_AXIS": 0, "REDUCE_AXIS": 0, "BROADCAST": False,
         "COLUMN_BROADCAST": True}, num_warps=4)
    output = result.stdout + result.stderr
    assert result.returncode > 0, output
    assert "full-tile exchange scratch needs" in output, output
    assert "exceeding the 32768-byte budget" in output, output


@triton.jit
def _column_boundary_reduce(value, UNSIGNED: tl.constexpr, MONOID: tl.constexpr):
    if MONOID == "product":
        return tl.reduce(value, 0, _i32_column_product_combine)
    elif MONOID == "and":
        return tl.reduce(value, 0, _i32_row_and_combine)
    elif MONOID == "or":
        return tl.reduce(value, 0, _i32_row_or_combine)
    elif MONOID == "xor":
        return tl.reduce(value, 0, _i32_row_xor_combine)
    elif UNSIGNED:
        return tl.min(value.to(tl.uint32), 0)
    else:
        return tl.sum(value, 0)


@pytest.mark.parametrize("mode,diagnostic", [
    (0, "distinct scan/reduce inputs require same-layout sources defined before the scan "
     "without intervening writes or input exchanges"),
    (1, "distinct scan/reduce inputs require same-layout sources defined before the scan "
     "without intervening writes or input exchanges"),
    (2, "distinct scan/reduce inputs require same-layout sources defined before the scan "
     "without intervening writes or input exchanges"),
    (3, "distinct scan/reduce inputs require same-layout sources defined before the scan "
     "without intervening writes or input exchanges"),
    (4, "axis=0 reduce supports f32/i32 sum/product/max/min"),
    (5, "distinct scan/reduce inputs require same-layout sources defined before the scan "
     "without intervening writes or input exchanges"),
], ids=["load_after_scan", "compute_after_scan", "write_before_scan", "write_after_scan",
        "transpose", "scan_result"])
def test_multiband_scan_unsigned_column_broadcast_source_is_named(mode, diagnostic):
    result = _compile_rankn_nested_negative_in_child(
        "_unsupported_different_input_column_broadcast_kernel",
        {"X": "*i32", "W": "*i32", "Y": "*i32"}, {"MODE": mode, "UNSIGNED": True, "MONOID": ""}, num_warps=1)
    output = result.stdout + result.stderr
    assert result.returncode > 0, output
    assert diagnostic in output, output


@pytest.mark.parametrize("mode", [3, 4, 5, 6, 7, 8, 9, 10],
                         ids=["mixed_store", "cross_axis", "wider_tile", "pre_expand_arithmetic",
                              "multiple_scans", "nested_scan", "region_capture", "downstream_reduce"])
def test_multiband_scan_unsigned_column_broadcast_consumer_is_named(mode):
    result = _compile_rankn_nested_negative_in_child(
        "_unsupported_shared_input_column_broadcast_kernel",
        {"X": "*i32", "W": "*i32", "Y": "*i32", "Scanned": "*i32",
         "Vector": "*i32", "steps": "i32"},
        {"R": 16, "C": 16, "MODE": mode, "UNSIGNED": True, "MONOID": ""}, num_warps=1)
    output = result.stdout + result.stderr
    assert result.returncode > 0, output
    diagnostic = ("nested scan requires a flat load/compute/store loop body" if mode == 8 else
                  "axis=0 reduce supports f32/i32 sum/product/max/min")
    assert diagnostic in output, output


@pytest.mark.parametrize("mode", [0, 3], ids=["reduce_before_scan", "no_scan"])
def test_multiband_scan_unsigned_column_broadcast_phase_is_named(mode):
    result = _compile_rankn_nested_negative_in_child(
        "_unsupported_wide_scan_column_broadcast_kernel",
        {"X": "*i32", "Y": "*i32", "Vector": "*i32"},
        {"MODE": mode, "UNSIGNED": True, "MONOID": ""}, num_warps=1)
    output = result.stdout + result.stderr
    assert result.returncode > 0, output
    assert "axis=0 reduce supports f32/i32 sum/product/max/min" in output, output


@pytest.mark.parametrize("rows,columns,scan_axis,dtype,diagnostic", [
    (4, 1024, 0, "*i32", "axis=0 reduce supports f32/i32 sum/product/max/min"),
    (128, 16, 1, "*i32", "axis=0 reduce supports f32/i32 sum/product/max/min"),
    (16, 16, 1, "*u64", "rank >= 2 scan is implemented for a blocked tile"),
], ids=["wide_columns", "tall_tile", "unsigned_i64"])
def test_multiband_scan_unsigned_column_broadcast_shape_and_type_is_named(
        rows, columns, scan_axis, dtype, diagnostic):
    # REDUCTION=min preserves the pointer's logical unsigned width; no u32 cast.
    result = _compile_rankn_nested_negative_in_child(
        "_shared_input_scan_column_broadcast_kernel", {"X": dtype if dtype == "*u64" else "*u32",
                                                       "Y": dtype, "N": "i32"},
        {"R": rows, "C": columns, "SCAN_AXIS": scan_axis, "REVERSE": False,
         "REDUCTION": "min", "MASKED": False, "COMPUTED": False, "SNAPSHOT": False}, num_warps=1)
    output = result.stdout + result.stderr
    assert result.returncode > 0, output
    assert diagnostic in output, output



def _i32_column_monoid_values(programs, rows, columns, reduction, even_product=False):
    # Generate each column as an independent reduction sequence. In particular,
    # the row-monoid XOR perturbation now survives the column fold.
    return _i32_row_monoid_values(programs * rows * columns, rows, reduction,
                                  even_product=even_product).reshape(programs, columns, rows).transpose(1, 2).contiguous().flatten()


@mps_only
@pytest.mark.parametrize("reduction", ["product", "and", "or", "xor"])
@pytest.mark.parametrize("rows,columns,scan_axis", [(16, 16, 1), (8, 128, 0)])
def test_multiband_scan_i32_column_monoid_broadcast_baseline_matches_torch(rows, columns, scan_axis, reduction):
    size = rows * columns
    original = _i32_column_monoid_values(1, rows, columns, reduction)
    x = original.reshape(rows, columns)
    reduced = _i32_row_monoid_reference(x.T, reduction)
    assert (reduced != 0).any()
    assert (reduced != -1).any()
    if reduction == "product":
        assert (reduced != 0).all()
    sentinel = -999999
    output = torch.full((size + 7,), sentinel, dtype=torch.int32, device="mps")
    values = original.to("mps")
    _i32_nonplain_input_scan_column_broadcast_kernel[(1,)](
        values, values, output, size, rows, columns, scan_axis, False,
        reduction, True, False, False, 0, False, True, num_warps=1)
    torch.mps.synchronize()
    scanned = x.cumsum(scan_axis).to(torch.int32)
    contribution = reduced[None, :].expand_as(x)
    torch.testing.assert_close(output.cpu()[:size], (scanned + contribution).flatten(), atol=0, rtol=0)
    torch.testing.assert_close(output.cpu()[:size].reshape(rows, columns) - scanned,
                               contribution, atol=0, rtol=0)
    assert (output.cpu()[size:] == sentinel).all()


@mps_only
@pytest.mark.parametrize("reduction", ["product", "and", "or", "xor"])
@pytest.mark.parametrize(
    "rows,columns,num_warps,scan_axis,reverse,shared,masked,computed,fill,snapshot,full_output,subtract", [
        (16, 16, 1, 1, True, True, True, False, "identity", False, True, False),
        (8, 64, 1, 1, False, True, False, True, "zero", True, True, True),
        (8, 128, 1, 0, True, True, True, True, "zero", True, True, False),
        (32, 32, 4, 0, True, True, True, True, "nonidentity", True, False, True),
        (8, 128, 1, 0, False, False, False, False, "identity", False, True, True),
        (8, 256, 4, 0, True, False, True, False, "nonidentity", False, True, False),
        (16, 32, 4, 1, False, False, False, True, "zero", False, True, False),
        (16, 128, 4, 0, False, False, True, True, "zero", False, True, True),
    ])
def test_multiband_scan_i32_column_monoid_broadcast_matches_torch(
        rows, columns, num_warps, scan_axis, reverse, shared, masked, computed,
        fill, snapshot, full_output, subtract, reduction):
    programs = 2
    size = programs * rows * columns
    valid = size - (rows - 1) * columns - 3 if masked else size
    if fill == "identity":
        other = 1 if reduction == "product" else -1 if reduction == "and" else 0
    elif fill == "nonidentity":
        other = {"product": -3, "and": -251658241, "or": -2147479551, "xor": -2147479549}[reduction]
    else:
        other = 0
    w_values = _i32_column_monoid_values(programs, rows, columns, reduction,
                                        even_product=computed and not shared)[:valid]
    x_values = w_values.clone() if shared else (torch.arange(valid) % 17 - 8).to(torch.int32)
    x = torch.full((size,), other, dtype=torch.int32)
    w = torch.full((size,), other, dtype=torch.int32)
    x[:valid], w[:valid] = x_values, w_values
    x, w = x.reshape(programs, rows, columns), w.reshape(programs, rows, columns)
    if computed:
        if shared:
            x = x * 2 + 3
            w = x
        else:
            w = w * 3 - 1
    axis = scan_axis + 1
    source = x.flip(axis) if reverse else x
    scanned = source.cumsum(axis).to(torch.int32)
    if reverse:
        scanned = scanned.flip(axis)
    reduced = _i32_row_monoid_reference(w.transpose(1, 2), reduction)
    assert (reduced[0] != 0).any() and (reduced[0] != -1).any()
    assert reduced[0].unique().numel() > 1, "column contributions must vary"
    if reduction == "product":
        assert (reduced[0] != 0).all()
    contribution = reduced[:, None, :].expand_as(x)
    expected = scanned - contribution if subtract else scanned + contribution
    sentinel = -999999
    values = x_values.to("mps")
    output = torch.full((size + 7,), sentinel, dtype=torch.int32, device="mps")
    _i32_nonplain_input_scan_column_broadcast_kernel[(programs,)](
        values, w_values.to("mps"), output, valid, rows, columns, scan_axis, reverse,
        reduction, shared, masked, computed, other, snapshot, full_output,
        SUBTRACT=subtract, num_warps=num_warps)
    torch.mps.synchronize()
    actual = output.cpu()
    written = size if full_output else valid
    torch.testing.assert_close(actual[:written], expected.flatten()[:written], atol=0, rtol=0)
    observed = scanned.flatten()[:written] - actual[:written] if subtract else actual[:written] - scanned.flatten()[:written]
    torch.testing.assert_close(observed, contribution.flatten()[:written], atol=0, rtol=0)
    assert (actual[written:] == sentinel).all()
    expected_input = x.flatten()[:valid] * 2 - 7 if snapshot else x_values
    torch.testing.assert_close(values.cpu(), expected_input, atol=0, rtol=0)


@mps_only
@pytest.mark.parametrize("rows,columns,scan_axis,num_warps", [(16, 32, 1, 4), (8, 128, 0, 1)])
def test_multiband_scan_i32_computed_sources_column_monoids_match_torch(rows, columns, scan_axis, num_warps):
    size = rows * columns
    valid = size - (rows - 1) * columns - 3
    original = _i32_column_monoid_values(1, rows, columns, "xor")[:valid]
    x = torch.zeros(size, dtype=torch.int32)
    x[:valid] = original
    x = x.reshape(rows, columns)
    w, v = x * 2 + 3, x * 3 - 1
    sentinel = -999999
    outputs = [torch.full((size + 7,), sentinel, dtype=torch.int32, device="mps") for _ in range(4)]
    compiled = _i32_computed_sources_scan_column_broadcast_kernel[(1,)](
        original.to("mps"), *outputs, valid, rows, columns, scan_axis,
        MONOIDS=True, num_warps=num_warps)
    raw = compiled.asm["metal"]
    msl = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    buffers = sorted(int(length) for length in re.findall(
        r"\bthreadgroup\s+(?:u?int32_t|u?int)\s+\w+\s*\[\s*(\d+)\s*\]\s*;", msl))
    assert buffers == [size] * 3, msl
    assert msl.count("threadgroup_barrier(") == 1, msl
    torch.mps.synchronize()
    scanned = x.cumsum(scan_axis).to(torch.int32)
    reductions = [_i32_row_monoid_reference(w.T, "product"), _i32_row_monoid_reference(w.T, "xor"),
                  _i32_row_monoid_reference(v.T, "and"), _i32_row_monoid_reference(x.T, "or")]
    assert (reductions[0] != 0).all()
    assert (reductions[1] != 0).any()
    for actual, reduced in zip(outputs, reductions):
        contribution = reduced[None, :].expand_as(x)
        torch.testing.assert_close(actual.cpu()[:size], (scanned + contribution).flatten(), atol=0, rtol=0)
        torch.testing.assert_close(actual.cpu()[:size].reshape(rows, columns) - scanned, contribution, atol=0, rtol=0)
        assert (actual.cpu()[size:] == sentinel).all()


def test_multiband_scan_i32_column_monoid_broadcast_budget_is_named():
    result = _compile_rankn_nested_negative_in_child(
        "_i32_computed_sources_scan_column_broadcast_kernel",
        {"X": "*i32", "Sum": "*i32", "Min": "*i32", "Max": "*i32", "SharedMax": "*i32", "N": "i32"},
        {"R": 64, "C": 64, "SCAN_AXIS": 0, "MONOIDS": True}, num_warps=4)
    output = result.stdout + result.stderr
    assert result.returncode > 0, output
    assert "full-tile exchange scratch needs" in output, output
    assert "exceeding the 32768-byte budget" in output, output


@pytest.mark.parametrize("mode,reduction,diagnostic", [
    (0, "product", "distinct scan/reduce inputs require same-layout sources defined before the scan "
     "without intervening writes or input exchanges"),
    (1, "and", "distinct scan/reduce inputs require same-layout sources defined before the scan "
     "without intervening writes or input exchanges"),
    (2, "or", "distinct scan/reduce inputs require same-layout sources defined before the scan "
     "without intervening writes or input exchanges"),
    (3, "xor", "distinct scan/reduce inputs require same-layout sources defined before the scan "
     "without intervening writes or input exchanges"),
    (4, "and", "axis=0 reduce supports f32/i32 sum/product/max/min"),
    (5, "product", "distinct scan/reduce inputs require same-layout sources defined before the scan "
     "without intervening writes or input exchanges"),
], ids=["load_after_scan", "compute_after_scan", "write_before_scan", "write_after_scan",
        "transpose", "scan_result"])
def test_multiband_scan_i32_column_monoid_broadcast_source_is_named(mode, reduction, diagnostic):
    result = _compile_rankn_nested_negative_in_child(
        "_unsupported_different_input_column_broadcast_kernel",
        {"X": "*i32", "W": "*i32", "Y": "*i32"},
        {"MODE": mode, "UNSIGNED": False, "MONOID": reduction}, num_warps=1)
    output = result.stdout + result.stderr
    assert result.returncode > 0, output
    assert diagnostic in output, output


@pytest.mark.parametrize("mode", [3, 4, 5, 6, 7, 8, 9, 10],
                         ids=["mixed_store", "cross_axis", "wider_tile", "pre_expand_arithmetic",
                              "multiple_scans", "nested_scan", "region_capture", "downstream_reduce"])
def test_multiband_scan_i32_column_monoid_broadcast_consumer_is_named(mode):
    result = _compile_rankn_nested_negative_in_child(
        "_unsupported_shared_input_column_broadcast_kernel",
        {"X": "*i32", "W": "*i32", "Y": "*i32", "Scanned": "*i32",
         "Vector": "*i32", "steps": "i32"},
        {"R": 16, "C": 16, "MODE": mode, "UNSIGNED": False, "MONOID": "xor"}, num_warps=1)
    output = result.stdout + result.stderr
    assert result.returncode > 0, output
    diagnostic = ("nested scan requires a flat load/compute/store loop body" if mode == 8 else
                  "axis=0 reduce supports f32/i32 sum/product/max/min")
    assert diagnostic in output, output


@pytest.mark.parametrize("mode", [0, 3], ids=["reduce_before_scan", "no_scan"])
def test_multiband_scan_i32_column_monoid_broadcast_phase_is_named(mode):
    result = _compile_rankn_nested_negative_in_child(
        "_unsupported_wide_scan_column_broadcast_kernel",
        {"X": "*i32", "Y": "*i32", "Vector": "*i32"},
        {"MODE": mode, "UNSIGNED": False, "MONOID": "and"}, num_warps=1)
    output = result.stdout + result.stderr
    assert result.returncode > 0, output
    assert "axis=0 reduce supports f32/i32 sum/product/max/min" in output, output


@pytest.mark.parametrize("rows,columns,scan_axis,dtype,diagnostic", [
    (4, 1024, 0, "*i32", "axis=0 reduce supports f32/i32 sum/product/max/min"),
    (128, 16, 1, "*i32", "axis=0 reduce supports f32/i32 sum/product/max/min"),
    (8, 128, 1, "*i32", "rank >= 2 scan is implemented for a blocked tile"),
    (16, 16, 1, "*i64", "rank >= 2 scan is implemented for a blocked tile"),
], ids=["wide_columns", "tall_tile", "long_scan_axis", "i64"])
def test_multiband_scan_i32_column_monoid_broadcast_shape_and_type_is_named(
        rows, columns, scan_axis, dtype, diagnostic):
    result = _compile_rankn_nested_negative_in_child(
        "_i32_nonplain_input_scan_column_broadcast_kernel",
        {"X": dtype, "W": dtype, "Y": dtype, "N": "i32"},
        {"R": rows, "C": columns, "SCAN_AXIS": scan_axis, "REVERSE": False,
         "REDUCTION": "and", "SHARED": True, "MASKED": True, "COMPUTED": True,
         "OTHER": 0, "SNAPSHOT": False, "FULL_OUTPUT": True, "SUBTRACT": False}, num_warps=1)
    output = result.stdout + result.stderr
    assert result.returncode > 0, output
    assert diagnostic in output, output


@mps_only
@pytest.mark.parametrize("reduction", ["and", "or", "xor"])
@pytest.mark.parametrize("rows,columns,scan_axis", [(16, 16, 1), (8, 128, 0)])
def test_multiband_scan_i32_bitwise_column_vector_baseline_matches_torch(rows, columns, scan_axis, reduction):
    size = rows * columns
    original = _i32_column_monoid_values(1, rows, columns, reduction)
    x = original.reshape(rows, columns)
    reduced = _i32_row_monoid_reference(x.T, reduction)
    assert (reduced != 0).any() and (reduced != -1).any()
    assert reduced.unique().numel() > 1
    sentinel = -999999
    output = torch.full((size + 7,), sentinel, dtype=torch.int32, device="mps")
    vector = torch.full((columns + 7,), sentinel, dtype=torch.int32, device="mps")
    _shared_input_scan_column_reduce_kernel[(1,)](
        original.to("mps"), output, vector, size, columns, rows, columns,
        scan_axis, False, reduction, False, False, False,
        OTHER=0, OUTPUT_MASKED=False, num_warps=1)
    torch.mps.synchronize()
    torch.testing.assert_close(output.cpu()[:size], x.cumsum(scan_axis).to(torch.int32).flatten(), atol=0, rtol=0)
    torch.testing.assert_close(vector.cpu()[:columns], reduced, atol=0, rtol=0)
    assert (output.cpu()[size:] == sentinel).all()
    assert (vector.cpu()[columns:] == sentinel).all()


@mps_only
@pytest.mark.parametrize("reduction", ["and", "or", "xor"])
@pytest.mark.parametrize(
    "rows,columns,num_warps,scan_axis,reverse,shared,masked,computed,fill,snapshot,output_masked", [
        (16, 16, 1, 1, True, True, False, False, "identity", False, False),
        (8, 64, 1, 1, False, True, True, False, "identity", False, False),
        (8, 128, 1, 0, True, True, True, True, "zero", True, False),
        (8, 256, 4, 0, False, True, False, True, "zero", True, False),
        (8, 512, 4, 0, True, True, True, False, "nonidentity", False, False),
        (16, 16, 4, 0, False, False, True, False, "identity", False, True),
        (16, 32, 4, 1, True, False, False, True, "zero", False, False),
        (4, 512, 1, 0, False, False, True, True, "zero", False, False),
        (16, 128, 4, 0, True, False, True, False, "nonidentity", False, False),
        (8, 64, 1, 1, False, False, True, True, "nonidentity", False, True),
        (2, 128, 1, 0, True, True, True, True, "identity", False, False),
        (64, 32, 4, 0, False, False, True, True, "zero", False, False),
        (8, 512, 4, 0, True, True, True, True, "nonidentity", False, True),
        (64, 2, 1, 0, False, True, True, False, "identity", False, False),
    ])
def test_multiband_scan_i32_bitwise_column_vector_matches_torch(
        rows, columns, num_warps, scan_axis, reverse, shared, masked, computed,
        fill, snapshot, output_masked, reduction):
    programs = 2
    size = programs * rows * columns
    valid = size - (rows - 1) * columns - min(3, columns - 1) if masked else size
    valid_columns = columns - min(3, columns - 1) if masked and output_masked else columns
    if fill == "identity":
        other = -1 if reduction == "and" else 0
    elif fill == "nonidentity":
        other = {"and": -251658241, "or": -2147479551, "xor": -2147479549}[reduction]
    else:
        other = 0
    w_values = _i32_column_monoid_values(programs, rows, columns, reduction)[:valid]
    x_values = w_values.clone() if shared else (torch.arange(valid) % 17 - 8).to(torch.int32)
    x = torch.full((size,), other, dtype=torch.int32)
    w = torch.full((size,), other, dtype=torch.int32)
    x[:valid], w[:valid] = x_values, w_values
    x, w = x.reshape(programs, rows, columns), w.reshape(programs, rows, columns)
    if computed:
        if shared:
            x = x * 2 + 3
            w = x
        else:
            w = w * 3 - 1
    axis = scan_axis + 1
    source = x.flip(axis) if reverse else x
    scanned = source.cumsum(axis).to(torch.int32)
    if reverse:
        scanned = scanned.flip(axis)
    reduced = _i32_row_monoid_reference(w.transpose(1, 2), reduction)
    assert (reduced[0] != 0).any() and (reduced[0] != -1).any()
    assert reduced[0].unique().numel() > 1, "column contributions must vary"
    sentinel = -999999
    values = x_values.to("mps")
    output = torch.full((size + 7,), sentinel, dtype=torch.int32, device="mps")
    vector = torch.full((programs * columns + 7,), sentinel, dtype=torch.int32, device="mps")
    if shared:
        _shared_input_scan_column_reduce_kernel[(programs,)](
            values, output, vector, valid, valid_columns, rows, columns, scan_axis,
            reverse, reduction, computed, masked, snapshot, OTHER=other,
            OUTPUT_MASKED=output_masked, num_warps=num_warps)
    else:
        _different_input_scan_column_reduce_kernel[(programs,)](
            values, w_values.to("mps"), output, vector, valid, valid_columns,
            rows, columns, scan_axis, reverse, reduction, masked, computed,
            OTHER=other, OUTPUT_MASKED=output_masked, num_warps=num_warps)
    torch.mps.synchronize()
    actual, actual_vector = output.cpu(), vector.cpu()
    vectors = actual_vector[:programs * columns].reshape(programs, columns)
    torch.testing.assert_close(actual[:valid], scanned.flatten()[:valid], atol=0, rtol=0)
    torch.testing.assert_close(vectors[:, :valid_columns], reduced[:, :valid_columns], atol=0, rtol=0)
    assert (actual[valid:] == sentinel).all()
    assert (vectors[:, valid_columns:] == sentinel).all()
    assert (actual_vector[programs * columns:] == sentinel).all()
    expected_input = x.flatten()[:valid] * 3 + 17 if snapshot else x_values
    torch.testing.assert_close(values.cpu(), expected_input, atol=0, rtol=0)


@mps_only
@pytest.mark.parametrize("rows,columns,scan_axis,num_warps,vector_exchanges,product",
                         [(8, 128, 0, 1, 0, True), (16, 32, 1, 4, 4, False)])
def test_multiband_scan_i32_computed_sources_column_bitwise_vectors_match_torch(
        rows, columns, scan_axis, num_warps, vector_exchanges, product):
    size = rows * columns
    valid = size - columns - 1
    original = _i32_column_monoid_values(1, rows, columns, "xor")[:valid]
    x = torch.zeros(size, dtype=torch.int32)
    x[:valid] = original
    x = x.reshape(rows, columns)
    w = x * 2 + 3
    v = x * 4 + 1 if product else x * 3 - 1
    sentinel = -999999
    scanned = torch.full((size + 7,), sentinel, dtype=torch.int32, device="mps")
    outputs = [torch.full((columns + 7,), sentinel, dtype=torch.int32, device="mps") for _ in range(4)]
    compiled = _i32_computed_sources_scan_column_vectors_kernel[(1,)](
        original.to("mps"), scanned, *outputs, valid, rows, columns, scan_axis,
        PRODUCT=product, BITWISE=True, num_warps=num_warps)
    raw = compiled.asm["metal"]
    msl = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    buffers = sorted(int(length) for length in re.findall(
        r"\bthreadgroup\s+(?:u?int32_t|u?int)\s+\w+\s*\[\s*(\d+)\s*\]\s*;", msl))
    # AND and OR reuse W, while X/W/V each require one publication. Result
    # layout exchanges remain separate from the shared input publication.
    assert buffers == [columns] * vector_exchanges + [size] * 3, msl
    assert msl.count("threadgroup_barrier(") == 1 + vector_exchanges, msl
    torch.mps.synchronize()
    torch.testing.assert_close(scanned.cpu()[:size], x.cumsum(scan_axis).to(torch.int32).flatten(),
                               atol=0, rtol=0)
    assert (scanned.cpu()[size:] == sentinel).all()
    reductions = [_i32_row_monoid_reference(w.T, "and"), _i32_row_monoid_reference(w.T, "or"),
                  _i32_row_monoid_reference(v.T, "xor"),
                  _i32_row_monoid_reference(x.T, "product") if product else x.amin(0)]
    for actual, reduced in zip(outputs, reductions):
        torch.testing.assert_close(actual.cpu()[:columns], reduced, atol=0, rtol=0)
        assert (actual.cpu()[columns:] == sentinel).all()


def test_multiband_scan_i32_computed_sources_column_bitwise_vectors_budget_is_named():
    result = _compile_rankn_nested_negative_in_child(
        "_i32_computed_sources_scan_column_vectors_kernel",
        {"X": "*i32", "Scanned": "*i32", "Sum": "*i32", "Min": "*i32", "Max": "*i32",
         "SharedMax": "*i32", "N": "i32"},
        {"R": 64, "C": 64, "SCAN_AXIS": 0, "PRODUCT": True, "BITWISE": True}, num_warps=4)
    output = result.stdout + result.stderr
    assert result.returncode > 0, output
    assert "full-tile exchange scratch needs" in output, output
    assert "exceeding the 32768-byte budget" in output, output


@pytest.mark.parametrize("mode,reduction,diagnostic", [
    (0, "and", "distinct scan/reduce inputs require same-layout sources defined before the scan "
     "without intervening writes or input exchanges"),
    (1, "or", "distinct scan/reduce inputs require same-layout sources defined before the scan "
     "without intervening writes or input exchanges"),
    (2, "xor", "distinct scan/reduce inputs require same-layout sources defined before the scan "
     "without intervening writes or input exchanges"),
    (3, "and", "distinct scan/reduce inputs require same-layout sources defined before the scan "
     "without intervening writes or input exchanges"),
    (4, "or", "axis=0 reduce supports f32/i32 sum/product/max/min"),
    (5, "xor", "distinct scan/reduce inputs require same-layout sources defined before the scan "
     "without intervening writes or input exchanges"),
    (6, "and", "axis=0 reduce supports f32/i32 sum/product/max/min"),
    (9, "or", "axis=0 reduce supports f32/i32 sum/product/max/min"),
    (10, "xor", "axis=0 reduce supports f32/i32 sum/product/max/min"),
    (11, "and", "axis=0 reduce supports f32/i32 sum/product/max/min"),
    (12, "or", "axis=0 reduce supports f32/i32 sum/product/max/min"),
    (13, "xor", "axis=0 reduce supports f32/i32 sum/product/max/min"),
], ids=["load_after_scan", "compute_after_scan", "write_before_scan", "write_after_scan",
        "transpose", "scan_result", "mixed_store", "no_scan", "different_shape",
        "two_scans", "reduce_before_scan", "cross_axis"])
def test_multiband_scan_i32_bitwise_column_vector_outside_envelope_is_named(mode, reduction, diagnostic):
    result = _compile_rankn_nested_negative_in_child(
        "_unsupported_i32_nonplain_scan_column_vector_kernel",
        {"X": "*i32", "W": "*i32", "Y": "*i32", "Vector": "*i32", "N": "i32"},
        {"MODE": mode, "PRODUCT": False, "UNSIGNED": False, "BITWISE": reduction}, num_warps=1)
    output = result.stdout + result.stderr
    assert result.returncode > 0, output
    assert diagnostic in output, output


@triton.jit
def _f32_input_scan_column_product_broadcast_kernel(
        X, W, Y, Scanned, N, R: tl.constexpr, C: tl.constexpr,
        SCAN_AXIS: tl.constexpr, REVERSE: tl.constexpr, SHARED: tl.constexpr,
        MASKED: tl.constexpr, COMPUTED: tl.constexpr, OTHER: tl.constexpr,
        SNAPSHOT: tl.constexpr, FULL_OUTPUT: tl.constexpr):
    offset = tl.program_id(0) * R * C + tl.arange(0, R)[:, None] * C + tl.arange(0, C)[None, :]
    if MASKED:
        x = tl.load(X + offset, offset < N, other=OTHER)
    else:
        x = tl.load(X + offset)
    if SHARED:
        if COMPUTED:
            x = x * 2 + 3
        w = x
    else:
        if MASKED:
            w = tl.load(W + offset, offset < N, other=OTHER)
        else:
            w = tl.load(W + offset)
        if COMPUTED:
            w = w * 3 - 1
    scanned = tl.cumsum(x, SCAN_AXIS, reverse=REVERSE)
    if SNAPSHOT:
        tl.store(X + offset, x * 2 - 7, offset < N)
    reduced = tl.reduce(w, 0, _i32_column_product_combine)
    product = tl.broadcast_to(reduced[None, :], (R, C))
    tl.store(Scanned + offset, scanned, offset < N)
    if MASKED and not FULL_OUTPUT:
        tl.store(Y + offset, product, offset < N)
    else:
        tl.store(Y + offset, product)


def _f32_column_product_reference(source):
    # Match the staged column fold: round every multiplication to float32
    # in ascending row order, rather than torch's parallel reduction tree.
    result = torch.ones_like(source[..., 0, :])
    for row in range(source.shape[-2]):
        result = result * source[..., row, :]
    return result


def _f32_column_product_values(programs, rows, columns):
    index = torch.arange(programs * rows * columns).reshape(programs, rows, columns)
    factor = 1 + ((index % columns) % 13 + 1).float() / 1024
    sign = torch.where((index // columns) % 7 == 0, -1., 1.)
    return factor * sign


@mps_only
@pytest.mark.parametrize("shared", [True, False], ids=["shared", "distinct"])
@pytest.mark.parametrize("rows,columns,scan_axis", [(16, 16, 1), (8, 128, 0)])
def test_multiband_scan_f32_column_product_broadcast_baseline_matches_torch(rows, columns, scan_axis, shared):
    size = rows * columns
    w = _f32_column_product_values(1, rows, columns).reshape(rows, columns)
    x = w.clone() if shared else (torch.arange(size).reshape(rows, columns) % 7 - 3).float()
    sentinel = -999999.
    output = torch.full((size + 7,), sentinel, dtype=torch.float32, device="mps")
    scanned = torch.full_like(output, sentinel)
    _f32_input_scan_column_product_broadcast_kernel[(1,)](
        x.to("mps"), w.to("mps"), output, scanned, size, rows, columns,
        scan_axis, False, shared, False, False, 1., False, True,
        num_warps=1)
    torch.mps.synchronize()
    expected = _f32_column_product_reference(w)[None, :].expand_as(w)
    torch.testing.assert_close(output.cpu()[:size].reshape(rows, columns), expected, atol=0, rtol=1e-6)
    torch.testing.assert_close(scanned.cpu()[:size].reshape(rows, columns), x.cumsum(scan_axis), atol=1e-5, rtol=1e-6)
    assert (output.cpu()[size:] == sentinel).all()
    assert (scanned.cpu()[size:] == sentinel).all()


@mps_only
@pytest.mark.parametrize("shared", [True, False], ids=["shared", "distinct"])
@pytest.mark.parametrize("rows,columns,scan_axis,num_warps,reverse,masked,computed,fill,full_output,snapshot", [
    (16, 16, 1, 1, False, False, False, 1., True, False),
    (8, 64, 1, 1, True, True, False, 1., True, True),
    (16, 32, 0, 4, True, True, True, 1.125, True, False),
    (8, 128, 0, 1, False, True, False, 0., True, False),
    (4, 256, 0, 4, True, True, True, 0.75, True, True),
    (8, 256, 0, 1, False, False, True, 1., True, False),
    (64, 2, 0, 1, True, True, False, -1., True, False),
    (2, 64, 1, 1, False, True, True, 1., False, False),
    (32, 32, 1, 4, True, False, False, 1., True, False),
    (64, 32, 0, 4, False, True, False, 1.125, True, False),
    (8, 128, 0, 4, True, True, True, 0., False, False),
    (16, 64, 1, 4, False, False, True, 1., True, True),
    (4, 4, 0, 1, False, True, True, 1.125, True, False),
])
def test_multiband_scan_f32_column_product_broadcast_matches_torch(
        rows, columns, scan_axis, num_warps, reverse, masked, computed,
        fill, full_output, snapshot, shared):
    programs = 2
    size = programs * rows * columns
    valid = size - (rows - 1) * columns - min(3, columns - 1) if masked else size
    target = _f32_column_product_values(programs, rows, columns).flatten()
    # Affine sources stay near +/-1 after computation. Reference the actual
    # rounded input values, including the nonidentity fill of fully masked columns.
    w_values = (target - 3) / 2 if computed and shared else (target + 1) / 3 if computed else target
    if computed and not shared:
        # Dyadic raw values make both affine operations exactly representable,
        # avoiding differences between separate operations and contraction.
        w_values = (w_values * 1024).round() / 1024
    w_values = w_values[:valid].contiguous()
    x_values = w_values.clone() if shared else ((torch.arange(valid) % 9 - 4).float() / 8)
    other = (fill - 3) / 2 if computed and shared else (fill + 1) / 3 if computed else fill
    if computed and not shared:
        other = round(other * 1024) / 1024
    x, w = [torch.full((size,), other, dtype=torch.float32) for _ in range(2)]
    x[:valid], w[:valid] = x_values, w_values
    if computed:
        if shared:
            x = x * 2 + 3
            w = x
        else:
            w = w * 3 - 1
    x, w = x.reshape(programs, rows, columns), w.reshape(programs, rows, columns)
    axis = scan_axis + 1
    scanned = (x.flip(axis) if reverse else x).cumsum(axis)
    if reverse:
        scanned = scanned.flip(axis)
    reduced = _f32_column_product_reference(w)
    assert reduced[0].unique().numel() > 1 and (reduced[0].abs() > 0.5).all()
    sentinel = -999999.
    values = x_values.to("mps")
    output = torch.full((size + 7,), sentinel, dtype=torch.float32, device="mps")
    scan_output = torch.full_like(output, sentinel)
    snapshot = shared and snapshot
    _f32_input_scan_column_product_broadcast_kernel[(programs,)](
        values, w_values.to("mps"), output, scan_output, valid, rows, columns,
        scan_axis, reverse, shared, masked, computed, other, snapshot, full_output,
        num_warps=num_warps)
    torch.mps.synchronize()
    written = size if full_output else valid
    expected = reduced[:, None, :].expand_as(w).flatten()
    torch.testing.assert_close(output.cpu()[:written], expected[:written], atol=1e-6, rtol=2e-5)
    torch.testing.assert_close(scan_output.cpu()[:valid], scanned.flatten()[:valid], atol=2e-5, rtol=2e-6)
    assert (output.cpu()[written:] == sentinel).all()
    assert (scan_output.cpu()[valid:] == sentinel).all()
    expected_input = x.flatten()[:valid] * 2 - 7 if snapshot else x_values
    torch.testing.assert_close(values.cpu(), expected_input, atol=1e-6, rtol=1e-6)


@mps_only
@pytest.mark.parametrize("shared", [True, False], ids=["shared", "distinct"])
@pytest.mark.parametrize("rows,columns,scan_axis,num_warps", [(16, 16, 1, 1), (8, 128, 0, 4)])
def test_multiband_scan_f32_column_product_broadcast_special_values_match_torch(
        rows, columns, scan_axis, num_warps, shared):
    w = torch.ones((rows, columns), dtype=torch.float32)
    for column in range(columns):
        kind = column % 8
        if kind == 0:
            w[0, column] = 0.
        elif kind == 1:
            w[0, column] = -0.
        elif kind == 2:
            w[0, column] = float("nan")
        elif kind == 3:
            w[0, column] = float("inf")
        elif kind == 4:
            w[0, column] = -float("inf")
        elif kind == 5:
            w[:2, column] = 1e20
            w[2:, column] = 1e-20
        elif kind == 6:
            w[:2, column] = 2. ** -126
        else:
            w[0, column] = 0.
            w[1, column] = float("inf")
    # All inputs are normal, zero, or nonfinite. Underflow is far below the
    # subnormal range, so this does not require preserving subnormal values.
    x = w.clone() if shared else torch.ones_like(w)
    size = rows * columns
    sentinel = -999999.
    output = torch.full((size + 7,), sentinel, dtype=torch.float32, device="mps")
    scanned = torch.full_like(output, sentinel)
    _f32_input_scan_column_product_broadcast_kernel[(1,)](
        x.to("mps"), w.to("mps"), output, scanned, size, rows, columns,
        scan_axis, False, shared, False, False, 1., False, True, num_warps=num_warps)
    torch.mps.synchronize()
    expected = _f32_column_product_reference(w)[None, :].expand_as(w)
    actual = output.cpu()[:size].reshape(rows, columns)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0, equal_nan=True)
    zero = expected == 0
    assert torch.equal(actual.signbit()[zero], expected.signbit()[zero])
    torch.testing.assert_close(scanned.cpu()[:size].reshape(rows, columns), x.cumsum(scan_axis),
                               atol=1e-5, rtol=1e-6, equal_nan=True)
    assert (output.cpu()[size:] == sentinel).all()
    assert (scanned.cpu()[size:] == sentinel).all()


@mps_only
@pytest.mark.parametrize("rows,columns,scan_axis,num_warps", [(16, 16, 1, 1), (8, 128, 0, 4)])
def test_multiband_scan_f32_column_products_reuse_publications(rows, columns, scan_axis, num_warps):
    size = rows * columns
    w = _f32_column_product_values(1, rows, columns).reshape(rows, columns)
    x, v = w * 0.9921875, w * 1.0078125
    sentinel = -999999.
    outputs = [torch.full((size + 7,), sentinel, dtype=torch.float32, device="mps") for _ in range(4)]
    compiled = _different_inputs_scan_multiple_column_broadcasts_kernel[(1,)](
        x.to("mps"), w.to("mps"), v.to("mps"), *outputs,
        rows, columns, scan_axis, PRODUCT=True, num_warps=num_warps)
    raw = compiled.asm["metal"]
    msl = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    buffers = sorted(int(length) for length in re.findall(
        r"\bthreadgroup\s+float\s+\w+\s*\[\s*(\d+)\s*\]\s*;", msl))
    # W's product/max pair shares one publication, alongside distinct X/V.
    assert buffers == [size] * 3, msl
    assert msl.count("threadgroup_barrier(") == 1, msl
    torch.mps.synchronize()
    scanned = x.cumsum(scan_axis)
    reductions = [_f32_column_product_reference(w), w.amax(0),
                  _f32_column_product_reference(v), _f32_column_product_reference(x)]
    for output, reduced in zip(outputs, reductions):
        actual = output.cpu()[:size].reshape(rows, columns)
        contribution = reduced[None, :].expand_as(x)
        torch.testing.assert_close(actual, scanned + contribution, atol=2e-5, rtol=2e-6)
        torch.testing.assert_close(actual - scanned, contribution, atol=2e-5, rtol=2e-6)
        assert (output.cpu()[size:] == sentinel).all()


def test_multiband_scan_f32_column_products_publication_budget_is_named():
    result = _compile_rankn_nested_negative_in_child(
        "_different_inputs_scan_multiple_column_broadcasts_kernel",
        {"X": "*fp32", "W": "*fp32", "V": "*fp32", "Sum": "*fp32",
         "Max": "*fp32", "Min": "*fp32", "SharedMax": "*fp32"},
        {"R": 64, "C": 64, "SCAN_AXIS": 0, "PRODUCT": True}, num_warps=4)
    output = result.stdout + result.stderr
    assert result.returncode > 0, output
    assert "full-tile exchange scratch needs" in output, output
    assert "exceeding the 32768-byte budget" in output, output


@pytest.mark.parametrize("mode,diagnostic", [
    (0, "distinct scan/reduce inputs require same-layout sources defined before the scan "
     "without intervening writes or input exchanges"),
    (1, "distinct scan/reduce inputs require same-layout sources defined before the scan "
     "without intervening writes or input exchanges"),
    (2, "distinct scan/reduce inputs require same-layout sources defined before the scan "
     "without intervening writes or input exchanges"),
    (3, "distinct scan/reduce inputs require same-layout sources defined before the scan "
     "without intervening writes or input exchanges"),
    (4, "rank-2 axis=0 f32 product requires a direct unmasked tt.load"),
    (5, "reduce cannot replay its computed source cone across conversion"),
], ids=["load_after_scan", "compute_after_scan", "write_before_scan", "write_after_scan",
        "transpose", "scan_result"])
def test_multiband_scan_f32_column_product_broadcast_source_is_named(mode, diagnostic):
    result = _compile_rankn_nested_negative_in_child(
        "_unsupported_different_input_column_broadcast_kernel",
        {"X": "*fp32", "W": "*fp32", "Y": "*fp32"},
        {"MODE": mode, "UNSIGNED": False, "MONOID": "product"}, num_warps=1)
    output = result.stdout + result.stderr
    assert result.returncode > 0, output
    assert diagnostic in output, output


@pytest.mark.parametrize("mode,diagnostic", [
    (3, "a scan-input column broadcast requires same-layout f32/i32 sources and "
     "direct same-layout consumers in the original tile"),
    (4, "a rank-2 tl.reduce cannot share a function"),
    (5, "a scan-input column broadcast requires same-layout f32/i32 sources and "
     "direct same-layout consumers in the original tile"),
    (7, "a scan-input column broadcast requires same-layout f32/i32 sources and "
     "direct same-layout consumers in the original tile"),
    (8, "nested scan requires a flat load/compute/store loop body"),
    (9, "a scan-input column broadcast requires same-layout f32/i32 sources and "
     "direct same-layout consumers in the original tile"),
], ids=["mixed_store", "cross_axis", "wider_tile", "multiple_scans", "nested_scan", "region_capture"])
def test_multiband_scan_f32_column_product_broadcast_consumer_is_named(mode, diagnostic):
    result = _compile_rankn_nested_negative_in_child(
        "_unsupported_shared_input_column_broadcast_kernel",
        {"X": "*fp32", "W": "*fp32", "Y": "*fp32", "Scanned": "*fp32",
         "Vector": "*fp32", "steps": "i32"},
        {"R": 16, "C": 16, "MODE": mode, "UNSIGNED": False, "MONOID": "product"}, num_warps=1)
    output = result.stdout + result.stderr
    assert result.returncode > 0, output
    assert diagnostic in output, output


@pytest.mark.parametrize("rows,columns,dtype,diagnostic", [
    (4, 1024, "*fp32", "rank-2 axis=0 f32 product requires a direct unmasked tt.load"),
    (128, 16, "*fp32", "rank-2 axis=0 f32 product requires a direct unmasked tt.load"),
    (16, 16, "*fp16", "axis=0 reduce supports f32/i32 sum/product/max/min"),
    (16, 16, "*bf16", "axis=0 reduce supports f32/i32 sum/product/max/min"),
], ids=["wide_columns", "long_reduce_axis", "f16", "bf16"])
def test_multiband_scan_f32_column_product_broadcast_shape_and_type_is_named(rows, columns, dtype, diagnostic):
    result = _compile_rankn_nested_negative_in_child(
        "_f32_input_scan_column_product_broadcast_kernel",
        {"X": dtype, "W": dtype, "Y": dtype, "Scanned": dtype, "N": "i32"},
        {"R": rows, "C": columns, "SCAN_AXIS": 0 if rows <= 64 else 1,
         "REVERSE": False, "SHARED": True, "MASKED": True, "COMPUTED": True,
         "OTHER": 1., "SNAPSHOT": False, "FULL_OUTPUT": True}, num_warps=1)
    output = result.stdout + result.stderr
    assert result.returncode > 0, output
    assert diagnostic in output, output


def _check_f32_column_product_vector(rows, columns, scan_axis, num_warps, shared,
                                     masked, computed, reverse=False, fill=1.,
                                     output_masked=False, snapshot=False):
    programs = 2
    size = programs * rows * columns
    valid = size - (rows - 1) * columns - min(3, columns - 1) if masked else size
    valid_columns = columns - min(3, columns - 1) if masked and output_masked else columns
    target = _f32_column_product_values(programs, rows, columns).flatten()
    raw = (target - 3) / 2 if computed and shared else (target + 1) / 3 if computed else target
    other = (fill - 3) / 2 if computed and shared else (fill + 1) / 3 if computed else fill
    if computed and not shared:
        raw = (raw * 1024).round() / 1024
        other = round(other * 1024) / 1024
    w_values = raw[:valid].contiguous()
    x_values = w_values.clone() if shared else (torch.arange(valid) % 9 - 4).float() / 8
    x, w = [torch.full((size,), other, dtype=torch.float32) for _ in range(2)]
    x[:valid], w[:valid] = x_values, w_values
    if computed:
        if shared:
            x = x * 2 + 3
            w = x
        else:
            w = w * 3 - 1
    x, w = x.reshape(programs, rows, columns), w.reshape(programs, rows, columns)
    axis = scan_axis + 1
    scanned = (x.flip(axis) if reverse else x).cumsum(axis)
    if reverse:
        scanned = scanned.flip(axis)
    reduced = _f32_column_product_reference(w)
    assert reduced[0].unique().numel() > 1 and (reduced[0].abs() > 0.5).all()
    sentinel = -999999.
    values = x_values.to("mps")
    output = torch.full((size + 7,), sentinel, dtype=torch.float32, device="mps")
    vector = torch.full((programs * columns + 7,), sentinel, dtype=torch.float32, device="mps")
    if shared:
        _shared_input_scan_column_reduce_kernel[(programs,)](
            values, output, vector, valid, valid_columns, rows, columns,
            scan_axis, reverse, "product", computed, masked, snapshot,
            OTHER=other, OUTPUT_MASKED=output_masked, num_warps=num_warps)
    else:
        _different_input_scan_column_reduce_kernel[(programs,)](
            values, w_values.to("mps"), output, vector, valid, valid_columns,
            rows, columns, scan_axis, reverse, "product", masked, computed,
            OTHER=other, OUTPUT_MASKED=output_masked, num_warps=num_warps)
    torch.mps.synchronize()
    actual = vector.cpu()[:programs * columns].reshape(programs, columns)
    torch.testing.assert_close(actual[:, :valid_columns], reduced[:, :valid_columns], atol=1e-6, rtol=2e-5)
    torch.testing.assert_close(output.cpu()[:valid], scanned.flatten()[:valid], atol=2e-5, rtol=2e-6)
    assert (actual[:, valid_columns:] == sentinel).all()
    assert (vector.cpu()[programs * columns:] == sentinel).all()
    assert (output.cpu()[valid:] == sentinel).all()
    expected_input = x.flatten()[:valid] * 3 + 17 if snapshot else x_values
    torch.testing.assert_close(values.cpu(), expected_input, atol=2e-6, rtol=1e-6)


@mps_only
@pytest.mark.parametrize("shared", [True, False], ids=["shared", "distinct"])
@pytest.mark.parametrize("rows,columns,scan_axis,masked,computed", [
    (16, 16, 1, True, False),
    (8, 128, 0, False, True),
    (4, 512, 0, True, True),
])
def test_multiband_scan_f32_nonplain_column_product_vector_baseline_matches_torch(
        rows, columns, scan_axis, masked, computed, shared):
    _check_f32_column_product_vector(rows, columns, scan_axis, 1, shared, masked, computed)


@mps_only
@pytest.mark.parametrize(
    "rows,columns,scan_axis,num_warps,shared,masked,computed,reverse,fill,output_masked,snapshot", [
        (16, 16, 1, 1, True, True, False, True, 1., False, False),
        (16, 16, 1, 4, False, True, False, False, 1., True, False),
        (8, 64, 1, 1, True, False, True, False, 1., False, True),
        (8, 64, 1, 4, False, False, True, True, 1., False, False),
        (8, 128, 0, 1, True, True, True, True, 1.125, False, True),
        (8, 128, 0, 4, False, True, True, False, 1.125, False, False),
        (8, 256, 0, 4, True, True, False, False, 0., False, False),
        (8, 256, 0, 1, False, True, False, True, 0., False, False),
        (8, 512, 0, 4, True, True, True, True, 0.75, False, False),
        (4, 512, 0, 1, False, True, True, False, 0.75, False, False),
        (8, 512, 0, 1, True, True, False, False, -1., True, False),
        (4, 512, 0, 4, False, True, True, True, 1., True, False),
        (64, 2, 0, 1, True, True, False, True, -1., False, False),
        (64, 2, 0, 1, False, True, True, False, 1.125, False, False),
        (2, 128, 0, 1, True, True, True, False, 1., False, False),
        (2, 128, 0, 4, False, True, False, True, 1., True, False),
        (64, 32, 0, 4, True, False, True, True, 1., False, True),
        (64, 32, 0, 1, False, True, True, False, 0., False, False),
        (32, 32, 1, 4, True, True, True, False, 0.75, True, False),
        (32, 32, 1, 1, False, True, False, True, -1., False, False),
        (16, 64, 1, 4, True, True, False, False, 1.125, True, False),
        (16, 64, 1, 1, False, False, True, True, 1., False, False),
        (4, 256, 0, 1, True, False, False, True, 1., False, False),
        (4, 256, 0, 4, False, True, True, False, -1., False, False),
    ])
def test_multiband_scan_f32_nonplain_column_product_vector_matches_torch(
        rows, columns, scan_axis, num_warps, shared, masked, computed, reverse, fill, output_masked, snapshot):
    _check_f32_column_product_vector(rows, columns, scan_axis, num_warps, shared,
                                     masked, computed, reverse, fill, output_masked, snapshot)


@mps_only
@pytest.mark.parametrize("shared", [True, False], ids=["shared", "distinct"])
@pytest.mark.parametrize("rows,columns,scan_axis,num_warps", [(16, 16, 1, 4), (8, 128, 0, 1)])
def test_multiband_scan_f32_nonplain_column_product_vector_special_values_match_torch(
        rows, columns, scan_axis, num_warps, shared):
    size = rows * columns
    valid = size - columns
    w = torch.ones((rows, columns), dtype=torch.float32)
    for column in range(columns):
        kind = column % 8
        if kind == 0:
            w[0, column] = 0.
        elif kind == 1:
            w[0, column] = -0.
        elif kind == 2:
            w[0, column] = float("nan")
        elif kind == 3:
            w[0, column] = float("inf")
        elif kind == 4:
            w[0, column] = -float("inf")
        elif kind == 5:
            w[:2, column] = 1e20
            w[2:-1, column] = 1e-20
        elif kind == 6:
            w[:2, column] = 2. ** -126
        else:
            w[0, column], w[1, column] = 0., float("inf")
    x = w.clone() if shared else torch.ones_like(w)
    sentinel = -999999.
    output = torch.full((size + 7,), sentinel, dtype=torch.float32, device="mps")
    vector = torch.full((columns + 7,), sentinel, dtype=torch.float32, device="mps")
    if shared:
        _shared_input_scan_column_reduce_kernel[(1,)](
            x.flatten()[:valid].to("mps"), output, vector, valid, columns,
            rows, columns, scan_axis, False, "product", False, True, False,
            OTHER=1., OUTPUT_MASKED=False, num_warps=num_warps)
    else:
        _different_input_scan_column_reduce_kernel[(1,)](
            x.flatten()[:valid].to("mps"), w.flatten()[:valid].to("mps"), output,
            vector, valid, columns, rows, columns, scan_axis, False, "product", True, False,
            OTHER=1., OUTPUT_MASKED=False, num_warps=num_warps)
    torch.mps.synchronize()
    expected = _f32_column_product_reference(w)
    actual = vector.cpu()[:columns]
    torch.testing.assert_close(actual, expected, atol=0, rtol=0, equal_nan=True)
    zero = expected == 0
    assert torch.equal(actual.signbit()[zero], expected.signbit()[zero])
    torch.testing.assert_close(output.cpu()[:valid], x.cumsum(scan_axis).flatten()[:valid],
                               atol=1e-5, rtol=1e-6, equal_nan=True)
    assert (output.cpu()[valid:] == sentinel).all()
    assert (vector.cpu()[columns:] == sentinel).all()


@mps_only
@pytest.mark.parametrize("rows,columns,scan_axis,num_warps,vector_exchanges",
                         [(8, 128, 0, 1, 0), (16, 32, 1, 4, 4)])
def test_multiband_scan_f32_column_product_vectors_reuse_publications(
        rows, columns, scan_axis, num_warps, vector_exchanges):
    size = rows * columns
    valid = size - columns - 1
    original = (-1 + (torch.arange(valid) % columns).float() / 1024)
    x = torch.zeros(size, dtype=torch.float32)
    x[:valid] = original
    x = x.reshape(rows, columns)
    w, v = x * 2 + 3, x * 4 + 1
    sentinel = -999999.
    scanned = torch.full((size + 7,), sentinel, dtype=torch.float32, device="mps")
    outputs = [torch.full((columns + 7,), sentinel, dtype=torch.float32, device="mps") for _ in range(4)]
    compiled = _i32_computed_sources_scan_column_vectors_kernel[(1,)](
        original.to("mps"), scanned, *outputs, valid, rows, columns, scan_axis,
        PRODUCT=True, BITWISE=False, num_warps=num_warps)
    raw = compiled.asm["metal"]
    msl = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    buffers = sorted(int(length) for length in re.findall(
        r"\bthreadgroup\s+float\s+\w+\s*\[\s*(\d+)\s*\]\s*;", msl))
    assert buffers == [columns] * vector_exchanges + [size] * 3, msl
    assert msl.count("threadgroup_barrier(") == 1 + vector_exchanges, msl
    torch.mps.synchronize()
    torch.testing.assert_close(scanned.cpu()[:size], x.cumsum(scan_axis).flatten(), atol=1e-5, rtol=1e-6)
    assert (scanned.cpu()[size:] == sentinel).all()
    reductions = [_f32_column_product_reference(w), w.amin(0),
                  _f32_column_product_reference(v), _f32_column_product_reference(x)]
    for actual, reduced in zip(outputs, reductions):
        torch.testing.assert_close(actual.cpu()[:columns], reduced, atol=1e-6, rtol=2e-5)
        assert (actual.cpu()[columns:] == sentinel).all()


def test_multiband_scan_f32_column_product_vectors_publication_budget_is_named():
    result = _compile_rankn_nested_negative_in_child(
        "_i32_computed_sources_scan_column_vectors_kernel",
        {"X": "*fp32", "Scanned": "*fp32", "Sum": "*fp32", "Min": "*fp32", "Max": "*fp32",
         "SharedMax": "*fp32", "N": "i32"},
        {"R": 64, "C": 64, "SCAN_AXIS": 0, "PRODUCT": True, "BITWISE": False}, num_warps=4)
    output = result.stdout + result.stderr
    assert result.returncode > 0, output
    assert "full-tile exchange scratch needs" in output, output
    assert "exceeding the 32768-byte budget" in output, output


@pytest.mark.parametrize("shared", [True, False], ids=["one_source_16x512", "two_sources_8x512"])
def test_multiband_scan_f32_nonplain_column_product_vector_total_scratch_budget_is_named(shared):
    signature = {"X": "*fp32", "Y": "*fp32", "Column": "*fp32", "N": "i32", "VALID_COLUMNS": "i32"}
    constants = {"R": 16 if shared else 8, "C": 512, "SCAN_AXIS": 0,
                 "REVERSE": False, "REDUCTION": "product", "COMPUTED": not shared,
                 "MASKED": shared, "OTHER": 1., "OUTPUT_MASKED": False}
    if shared:
        kernel = "_shared_input_scan_column_reduce_kernel"
        constants["SNAPSHOT"] = False
    else:
        kernel = "_different_input_scan_column_reduce_kernel"
        signature["W"] = "*fp32"
        constants.update(SCAN_PLAIN=False, SCAN_MASKED=False, SCAN_COMPUTED=False)
    result = _compile_rankn_nested_negative_in_child(
        kernel, signature, constants, num_warps=4 if shared else 1)
    output = result.stdout + result.stderr
    assert result.returncode > 0, output
    assert "exceeding the 32768-byte budget" in output, output


@pytest.mark.parametrize("mode,diagnostic", [
    (0, "distinct scan/reduce inputs require same-layout sources defined before the scan "
     "without intervening writes or input exchanges"),
    (1, "distinct scan/reduce inputs require same-layout sources defined before the scan "
     "without intervening writes or input exchanges"),
    (2, "distinct scan/reduce inputs require same-layout sources defined before the scan "
     "without intervening writes or input exchanges"),
    (3, "distinct scan/reduce inputs require same-layout sources defined before the scan "
     "without intervening writes or input exchanges"),
    (4, "rank-2 axis=0 f32 product requires a direct unmasked tt.load"),
    (5, "reduce cannot replay its computed source cone across conversion"),
    (6, "rank-2 axis=0 f32 product requires a direct unmasked tt.load"),
    (9, "rank-2 axis=0 f32 product requires a direct unmasked tt.load"),
    (10, "rank-2 axis=0 f32 product requires a direct unmasked tt.load"),
    (11, "rank-2 axis=0 f32 product requires a direct unmasked tt.load"),
    (12, "rank-2 axis=0 f32 product requires a direct unmasked tt.load"),
    (13, "rank-2 axis=0 f32 product requires a direct unmasked tt.load"),
], ids=["load_after_scan", "compute_after_scan", "write_before_scan", "write_after_scan",
        "transpose", "scan_result", "mixed_store", "no_scan", "different_shape",
        "two_scans", "reduce_before_scan", "cross_axis"])
def test_multiband_scan_f32_nonplain_column_product_vector_outside_envelope_is_named(mode, diagnostic):
    result = _compile_rankn_nested_negative_in_child(
        "_unsupported_i32_nonplain_scan_column_vector_kernel",
        {"X": "*fp32", "W": "*fp32", "Y": "*fp32", "Vector": "*fp32", "N": "i32"},
        {"MODE": mode, "PRODUCT": True, "UNSIGNED": False, "BITWISE": ""}, num_warps=1)
    output = result.stdout + result.stderr
    assert result.returncode > 0, output
    assert diagnostic in output, output


@pytest.mark.parametrize("dtype", ["*fp16", "*bf16"])
def test_multiband_scan_f32_nonplain_column_product_vector_other_width_is_named(dtype):
    result = _compile_rankn_nested_negative_in_child(
        "_shared_input_scan_column_reduce_kernel",
        {"X": dtype, "Y": dtype, "Column": dtype, "N": "i32", "VALID_COLUMNS": "i32"},
        {"R": 16, "C": 16, "SCAN_AXIS": 1, "REVERSE": False,
         "REDUCTION": "product", "COMPUTED": True, "MASKED": True, "SNAPSHOT": False,
         "OTHER": 1., "OUTPUT_MASKED": False}, num_warps=1)
    output = result.stdout + result.stderr
    assert result.returncode > 0, output
    assert "axis=0 reduce supports f32/i32 sum/product/max/min" in output, output


def _check_f32_row_product(rows, columns, scan_axis, num_warps, shared, broadcast,
                           masked, computed, reverse=False, fill=1., row_masked=False,
                           full_output=False, snapshot=False):
    programs = 2
    size = programs * rows * columns
    valid = size - columns - 3 if masked else size
    target = _f32_column_product_values(programs, columns, rows).transpose(1, 2).contiguous().flatten()
    raw = (target - 3) / 2 if computed and shared else (target + 1) / 3 if computed else target
    other = (fill - 3) / 2 if computed and shared else (fill + 1) / 3 if computed else fill
    if computed and not shared:
        raw = (raw * 1024).round() / 1024
        other = round(other * 1024) / 1024
    w_values = raw[:valid].contiguous()
    x_values = w_values.clone() if shared else (torch.arange(valid) % 9 - 4).float() / 8
    x, w = [torch.full((size,), other, dtype=torch.float32) for _ in range(2)]
    x[:valid], w[:valid] = x_values, w_values
    if computed:
        x = x * 2 + 3
        w = x if shared else w * 3 - 1
    x, w = x.reshape(programs, rows, columns), w.reshape(programs, rows, columns)
    axis = scan_axis + 1
    scanned = (x.flip(axis) if reverse else x).cumsum(axis)
    if reverse:
        scanned = scanned.flip(axis)
    reduced = _f32_column_product_reference(w.transpose(1, 2))
    assert reduced[0].unique().numel() > 1 and (reduced[0].abs() > 0.5).all()
    contribution = reduced[:, :, None].expand_as(x)
    base = x * 2 - scanned
    expected = base + contribution if broadcast else scanned
    sentinel = -999999.
    values = x_values.to("mps")
    output = torch.full((size + 7,), sentinel, dtype=torch.float32, device="mps")
    vector = torch.full((programs * rows + 7,), sentinel, dtype=torch.float32, device="mps")
    if shared:
        _shared_input_scan_row_reduce_kernel[(programs,)](
            values, output, vector, valid, rows, columns, scan_axis, reverse,
            False, broadcast, computed, masked, REDUCTION="product", OTHER=other,
            ROW_MASKED=row_masked, FULL_OUTPUT=full_output, SNAPSHOT=snapshot, num_warps=num_warps)
    else:
        _different_input_scan_row_reduce_kernel[(programs,)](
            values, w_values.to("mps"), output, vector, valid, rows, columns,
            scan_axis, reverse, "product", masked, computed, broadcast, False,
            OTHER=other, ROW_MASKED=row_masked, FULL_OUTPUT=full_output, num_warps=num_warps)
    torch.mps.synchronize()
    written = size if full_output else valid
    actual = output.cpu()
    torch.testing.assert_close(actual[:written], expected.flatten()[:written], atol=3e-5, rtol=2e-5)
    assert (actual[written:] == sentinel).all()
    if broadcast:
        torch.testing.assert_close(actual[:written] - base.flatten()[:written],
                                   contribution.flatten()[:written], atol=3e-5, rtol=2e-5)
        assert (vector.cpu() == sentinel).all()
    else:
        actual_rows = vector.cpu()[:programs * rows].reshape(programs, rows)
        valid_rows = rows - 1 if row_masked else rows
        torch.testing.assert_close(actual_rows[:, :valid_rows], reduced[:, :valid_rows], atol=1e-6, rtol=2e-5)
        assert (actual_rows[:, valid_rows:] == sentinel).all()
        assert (vector.cpu()[programs * rows:] == sentinel).all()
    expected_input = x.flatten()[:valid] * 3 + 17 if snapshot else x_values
    torch.testing.assert_close(values.cpu(), expected_input, atol=2e-6, rtol=1e-6)


@mps_only
@pytest.mark.parametrize("rows,columns,scan_axis,shared,broadcast,masked,computed", [
    (16, 16, 1, True, False, True, False),
    (16, 16, 1, False, True, True, False),
    (8, 64, 0, True, True, False, True),
    (8, 64, 0, False, False, False, True),
    (128, 16, 1, True, False, True, True),
    (128, 16, 1, False, True, True, True),
])
def test_multiband_scan_f32_nonplain_row_product_baseline_matches_torch(
        rows, columns, scan_axis, shared, broadcast, masked, computed):
    _check_f32_row_product(rows, columns, scan_axis, 1, shared, broadcast, masked, computed)


@mps_only
@pytest.mark.parametrize(
    "rows,columns,scan_axis,num_warps,shared,broadcast,masked,computed,reverse,fill,row_masked,full_output,snapshot", [
        (16, 16, 1, 1, True, False, True, False, True, 1., False, True, False),
        (16, 16, 0, 4, False, True, True, False, False, 1., False, True, False),
        (8, 64, 0, 1, True, True, False, True, False, 1., False, True, True),
        (8, 64, 1, 4, False, False, False, True, True, 1., False, False, False),
        (32, 32, 0, 1, True, False, True, True, True, 1.125, True, True, True),
        (32, 32, 1, 4, False, True, True, True, False, 1.125, False, True, False),
        (16, 64, 1, 4, True, True, True, False, False, 0., False, True, False),
        (16, 64, 0, 1, False, False, True, False, True, 0., False, True, False),
        (128, 16, 1, 4, True, False, True, True, True, 0.75, False, True, False),
        (128, 16, 1, 1, False, True, True, True, False, 0.75, False, True, False),
        (256, 8, 1, 1, True, True, True, False, False, -1., False, True, False),
        (256, 8, 1, 4, False, False, True, True, True, 1., True, True, False),
        (64, 2, 0, 1, True, False, True, False, True, -1., False, True, False),
        (64, 2, 0, 1, False, True, True, True, False, 1.125, False, True, False),
        (2, 64, 0, 1, True, True, True, True, False, 1., False, False, False),
        (2, 64, 1, 1, False, False, True, False, True, 1., True, False, False),
        (64, 32, 0, 4, True, False, False, True, True, 1., False, False, True),
        (64, 32, 1, 1, False, True, True, True, False, 0., False, True, False),
        (32, 32, 1, 4, True, True, True, True, False, 0.75, False, False, False),
        (32, 32, 0, 1, False, False, True, False, True, -1., False, False, False),
        (16, 64, 0, 4, True, False, True, False, False, 1.125, True, False, False),
        (16, 64, 1, 1, False, True, False, True, True, 1., False, True, False),
        (128, 8, 1, 1, True, True, False, True, True, 1., False, False, True),
        (128, 8, 1, 4, False, False, True, True, False, -1., False, True, False),
    ])
def test_multiband_scan_f32_nonplain_row_product_matches_torch(
        rows, columns, scan_axis, num_warps, shared, broadcast, masked, computed,
        reverse, fill, row_masked, full_output, snapshot):
    _check_f32_row_product(rows, columns, scan_axis, num_warps, shared, broadcast,
                           masked, computed, reverse, fill, row_masked, full_output, snapshot)


@triton.jit
def _f32_special_row_product_kernel(X, W, Y, Scanned, N, R: tl.constexpr, C: tl.constexpr,
                                    SCAN_AXIS: tl.constexpr, SHARED: tl.constexpr, BROADCAST: tl.constexpr):
    row = tl.arange(0, R)
    offset = row[:, None] * C + tl.arange(0, C)[None, :]
    x = tl.load(X + offset, offset < N, other=1.)
    if SHARED:
        w = x
    else:
        w = tl.load(W + offset, offset < N, other=1.)
    scanned = tl.cumsum(x, SCAN_AXIS)
    reduced = tl.reduce(w, 1, _i32_column_product_combine)
    tl.store(Scanned + offset, scanned)
    if BROADCAST:
        tl.store(Y + offset, tl.broadcast_to(reduced[:, None], (R, C)))
    else:
        tl.store(Y + row, reduced)


@mps_only
@pytest.mark.parametrize("shared", [True, False], ids=["shared", "distinct"])
@pytest.mark.parametrize("broadcast", [True, False], ids=["broadcast", "vector"])
@pytest.mark.parametrize("rows,columns,scan_axis,num_warps", [(16, 16, 1, 4), (8, 64, 0, 1)])
def test_multiband_scan_f32_nonplain_row_product_special_values_match_torch(
        rows, columns, scan_axis, num_warps, broadcast, shared):
    size = rows * columns
    valid = size - columns
    w = torch.ones((rows, columns), dtype=torch.float32)
    for row in range(rows - 1):
        kind = row % 8
        if kind == 0:
            w[row, 0] = 0.
        elif kind == 1:
            w[row, 0] = -0.
        elif kind == 2:
            w[row, 0] = float("nan")
        elif kind == 3:
            w[row, 0] = float("inf")
        elif kind == 4:
            w[row, 0] = -float("inf")
        elif kind == 5:
            w[row, :2] = 1e20
            w[row, 2:] = 1e-20
        elif kind == 6:
            w[row, :2] = 2. ** -126
        else:
            w[row, 0], w[row, 1] = 0., float("inf")
    x = w.clone() if shared else torch.ones_like(w)
    sentinel = -999999.
    output_size = size if broadcast else rows
    output = torch.full((output_size + 7,), sentinel, dtype=torch.float32, device="mps")
    scanned = torch.full((size + 7,), sentinel, dtype=torch.float32, device="mps")
    _f32_special_row_product_kernel[(1,)](
        x.flatten()[:valid].to("mps"), w.flatten()[:valid].to("mps"), output, scanned,
        valid, rows, columns, scan_axis, shared, broadcast, num_warps=num_warps)
    torch.mps.synchronize()
    reduced = _f32_column_product_reference(w.T)
    expected = reduced[:, None].expand_as(w).flatten() if broadcast else reduced
    actual = output.cpu()[:output_size]
    torch.testing.assert_close(actual, expected, atol=0, rtol=0, equal_nan=True)
    zero = expected == 0
    assert torch.equal(actual.signbit()[zero], expected.signbit()[zero])
    torch.testing.assert_close(scanned.cpu()[:size], x.cumsum(scan_axis).flatten(),
                               atol=1e-5, rtol=1e-6, equal_nan=True)
    assert (output.cpu()[output_size:] == sentinel).all()
    assert (scanned.cpu()[size:] == sentinel).all()


@mps_only
@pytest.mark.parametrize("rows,columns,scan_axis,num_warps,broadcast,vector_exchanges", [
    (8, 64, 0, 1, False, 3), (16, 32, 1, 4, True, 0),
])
def test_multiband_scan_f32_nonplain_row_products_reuse_publications(
        rows, columns, scan_axis, num_warps, broadcast, vector_exchanges):
    size = rows * columns
    valid = size - columns - 1
    original = -1 + (torch.arange(valid) // columns).float() / 1024
    x = torch.zeros(size, dtype=torch.float32)
    x[:valid] = original
    x = x.reshape(rows, columns)
    w, v = x * 2 + 3, x * 4 + 3
    sentinel = -999999.
    result_size = size if broadcast else rows
    scanned = torch.full((size + 7,), sentinel, dtype=torch.float32, device="mps")
    outputs = [torch.full((result_size + 7,), sentinel, dtype=torch.float32, device="mps") for _ in range(3)]
    compiled = _i32_computed_sources_scan_row_reducers_kernel[(1,)](
        original.to("mps"), scanned, *outputs, valid, rows, columns,
        scan_axis, broadcast, PRODUCT=True, num_warps=num_warps)
    raw = compiled.asm["metal"]
    msl = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    buffers = sorted(int(length) for length in re.findall(
        r"\bthreadgroup\s+float\s+\w+\s*\[\s*(\d+)\s*\]\s*;", msl))
    # Distinct computed X/W/V publications share the load; W product/min
    # reuse one buffer. Direct row stores exchange only the result vectors.
    assert buffers == [rows] * vector_exchanges + [size] * 3, msl
    assert msl.count("threadgroup_barrier(") == 1 + vector_exchanges, msl
    torch.mps.synchronize()
    expected_scan = x.cumsum(scan_axis)
    torch.testing.assert_close(scanned.cpu()[:size], expected_scan.flatten(), atol=1e-5, rtol=1e-6)
    assert (scanned.cpu()[size:] == sentinel).all()
    reductions = [_f32_column_product_reference(w.T), w.amin(1), _f32_column_product_reference(v.T)]
    for actual, reduced in zip(outputs, reductions):
        expected = expected_scan + reduced[:, None] if broadcast else reduced
        torch.testing.assert_close(actual.cpu()[:result_size], expected.flatten(), atol=2e-5, rtol=2e-5)
        assert (actual.cpu()[result_size:] == sentinel).all()
        if broadcast:
            torch.testing.assert_close(actual.cpu()[:size].reshape(rows, columns) - expected_scan,
                                       reduced[:, None].expand_as(x), atol=2e-5, rtol=2e-5)


def test_multiband_scan_f32_nonplain_row_products_publication_budget_is_named():
    result = _compile_rankn_nested_negative_in_child(
        "_i32_computed_sources_scan_row_reducers_kernel",
        {"X": "*fp32", "Scanned": "*fp32", "Sum": "*fp32", "Min": "*fp32", "Max": "*fp32", "N": "i32"},
        {"R": 64, "C": 64, "SCAN_AXIS": 0, "BROADCAST": False,
         "MONOIDS": False, "PRODUCT": True}, num_warps=4)
    output = result.stdout + result.stderr
    assert result.returncode > 0, output
    assert "full-tile exchange scratch needs" in output, output
    assert "exceeding the 32768-byte budget" in output, output


def test_multiband_scan_f32_nonplain_row_product_distinct_inputs_budget_is_named():
    result = _compile_rankn_nested_negative_in_child(
        "_different_input_scan_row_reduce_kernel",
        {"X": "*fp32", "W": "*fp32", "Y": "*fp32", "Row": "*fp32", "N": "i32"},
        {"R": 128, "C": 64, "SCAN_AXIS": 1, "REVERSE": False,
         "REDUCTION": "product", "MASKED": True, "COMPUTED": True,
         "BROADCAST": False, "FROM_SCAN_INPUT": False, "OTHER": 1.,
         "ROW_MASKED": False, "FULL_OUTPUT": True}, num_warps=4)
    output = result.stdout + result.stderr
    assert result.returncode > 0, output
    assert "exceeding the 32768-byte budget" in output, output


@pytest.mark.parametrize("mode,diagnostic", [
    (0, "rank-2 axis=1 f32 product requires a direct unmasked tt.load"),
    (1, "rank-2 axis=1 f32 product requires a direct unmasked tt.load"),
    (2, "rank-2 axis=1 f32 product requires a direct unmasked tt.load"),
    (3, "distinct scan/reduce inputs require same-layout sources defined before the scan "
     "without intervening writes or input exchanges"),
    (4, "distinct scan/reduce inputs require same-layout sources defined before the scan "
     "without intervening writes or input exchanges"),
    (5, "distinct scan/reduce inputs require same-layout sources defined before the scan "
     "without intervening writes or input exchanges"),
    (6, "rank-2 axis=1 f32 product requires a direct unmasked tt.load"),
    (7, "reduce cannot replay its computed source cone across conversion"),
    (8, "a shared scan-input row reduction requires direct row stores or "
     "same-axis broadcasts into its original tile"),
    (9, "a shared scan-input row reduction requires direct row stores or "
     "same-axis broadcasts into its original tile"),
    (12, "distinct scan/reduce inputs require same-layout sources defined before the scan "
     "without intervening writes or input exchanges"),
    (15, "rank-2 axis=1 f32 product requires a direct unmasked tt.load"),
], ids=["no_scan", "different_shape", "two_scans", "load_after_scan", "write_before_scan",
        "write_after_scan", "transpose", "scan_result", "mixed_store", "cross_axis",
        "compute_after_scan", "reduce_before_scan"])
def test_multiband_scan_f32_nonplain_row_product_outside_envelope_is_named(mode, diagnostic):
    result = _compile_rankn_nested_negative_in_child(
        "_unsupported_i32_nonplain_scan_row_reduce_kernel",
        {"X": "*fp32", "W": "*fp32", "Y": "*fp32", "Z": "*fp32", "Row": "*fp32", "N": "i32"},
        {"MODE": mode, "MONOID": "product"}, num_warps=1)
    output = result.stdout + result.stderr
    assert result.returncode > 0, output
    assert diagnostic in output, output


@pytest.mark.parametrize("columns,dtype,diagnostic", [
    (128, "*fp32", "a rank-2 tl.reduce cannot share a function with a multi-band staged scan"),
    (16, "*fp16", "reduce combine requires Session L3c (future) — got arith.mulf"),
    (16, "*bf16", "reduce combine requires Session L3c (future) — got arith.mulf"),
], ids=["long_reduce_axis", "f16", "bf16"])
def test_multiband_scan_f32_nonplain_row_product_shape_and_type_is_named(columns, dtype, diagnostic):
    result = _compile_rankn_nested_negative_in_child(
        "_shared_input_scan_row_reduce_kernel",
        {"X": dtype, "Y": dtype, "Row": dtype, "N": "i32"},
        {"R": 16, "C": columns, "SCAN_AXIS": 0, "REVERSE": False,
         "REDUCTION": "product", "MASKED": True, "COMPUTED": True,
         "OTHER": 1., "SNAPSHOT": False, "MAX": False, "BROADCAST": False,
         "ROW_MASKED": False, "FULL_OUTPUT": True}, num_warps=1)
    output = result.stdout + result.stderr
    assert result.returncode > 0, output
    assert diagnostic in output, output


def _check_c512_column_broadcast(rows, shared, dtype, reduction, num_warps=1,
                                 masked=False, computed=False, reverse=False,
                                 full_output=True, snapshot=False):
    columns, programs = 512, 2
    size = programs * rows * columns
    valid = size - (rows - 1) * columns - 3 if masked else size
    index = torch.arange(size)
    if dtype == torch.float32 and reduction == "product":
        target = _f32_column_product_values(programs, rows, columns).flatten()
        raw = (target - 3) / 2 if computed and shared else (target + 1) / 3 if computed else target
        other = (1.125 - 3) / 2 if computed and shared else (1.125 + 1) / 3 if computed else 1.125
        if computed and not shared:
            raw = (raw * 1024).round() / 1024
            other = round(other * 1024) / 1024
    elif dtype == torch.float32:
        raw = 1 + (index % columns).float() / 8 + (index // columns % 7).float() / 4
        other = -0.75
    elif reduction in ("and", "or", "xor", "product"):
        raw = _i32_column_monoid_values(programs, rows, columns, reduction,
                                        even_product=computed and not shared and reduction == "product")
        other = -3
    elif reduction in ("umin", "umax"):
        raw = _unsigned_extrema_values(size, columns)
        other = -2147483647
    elif reduction == "sum":
        raw = (2 ** 30 + index % 31).to(torch.int32)
        other = 7
    else:
        raw = torch.where((index // columns + index) % 2 == 0,
                          -(2 ** 31) + index % 31, 2 ** 31 - 1 - index % 29).to(torch.int32)
        other = -17
    w_values = raw[:valid].contiguous()
    x_values = w_values.clone() if shared else (torch.arange(valid) % 17 - 8).to(dtype)
    x, w = [torch.full((size,), other, dtype=dtype) for _ in range(2)]
    x[:valid], w[:valid] = x_values, w_values
    if computed:
        if shared:
            x = x * 2 + 3
            w = x
        else:
            w = w * 3 - 1
    x, w = x.reshape(programs, rows, columns), w.reshape(programs, rows, columns)
    scanned = (x.flip(1) if reverse else x).cumsum(1).to(dtype)
    if reverse:
        scanned = scanned.flip(1)
    if reduction == "sum":
        reduced = w.sum(1).to(dtype)
    elif reduction in ("min", "max"):
        reduced = w.amin(1) if reduction == "min" else w.amax(1)
    elif reduction in ("umin", "umax"):
        unsigned = w.to(torch.int64) & 0xffffffff
        reduced = (unsigned.amin(1) if reduction == "umin" else unsigned.amax(1)).to(torch.int32)
    elif dtype == torch.float32:
        reduced = _f32_column_product_reference(w)
    else:
        reduced = _i32_row_monoid_reference(w.transpose(1, 2), reduction)
    assert reduced[0].unique().numel() > 1
    assert (reduced[0, 256:] != reduced[0, :256]).any(), "high columns must not alias the first 256"
    contribution = reduced[:, None, :].expand_as(x)
    expected = scanned - contribution
    sentinel = -999999
    values = x_values.to("mps")
    output = torch.full((size + 7,), sentinel, dtype=dtype, device="mps")
    compiled = _i32_nonplain_input_scan_column_broadcast_kernel[(programs,)](
        values, w_values.to("mps"), output, valid, rows, columns, 0, reverse,
        reduction, shared, masked, computed, other, snapshot, full_output,
        SUBTRACT=True, num_warps=num_warps)
    torch.mps.synchronize()
    written = size if full_output else valid
    atol, rtol = (2e-5, 2e-5) if dtype == torch.float32 else (0, 0)
    actual = output.cpu()
    torch.testing.assert_close(actual[:written], expected.flatten()[:written], atol=atol, rtol=rtol)
    torch.testing.assert_close(scanned.flatten()[:written] - actual[:written],
                               contribution.flatten()[:written], atol=atol, rtol=rtol)
    assert (actual[written:] == sentinel).all()
    expected_input = x.flatten()[:valid] * 2 - 7 if snapshot else x_values
    torch.testing.assert_close(values.cpu(), expected_input, atol=atol, rtol=rtol)

    return compiled

@mps_only
@pytest.mark.parametrize("rows,shared,dtype,reduction", [
    (2, True, torch.float32, "sum"),
    (4, False, torch.float32, "product"),
    (8, True, torch.int32, "max"),
    (4, False, torch.int32, "xor"),
    (8, True, torch.int32, "umin"),
    (2, False, torch.int32, "product"),
])
def test_multiband_scan_c512_column_broadcast_baseline_matches_torch(rows, shared, dtype, reduction):
    _check_c512_column_broadcast(rows, shared, dtype, reduction)


@mps_only
@pytest.mark.parametrize("dtype,reduction", [
    (torch.float32, "sum"), (torch.float32, "min"), (torch.float32, "max"), (torch.float32, "product"),
    (torch.int32, "sum"), (torch.int32, "min"), (torch.int32, "max"), (torch.int32, "product"),
    (torch.int32, "and"), (torch.int32, "or"), (torch.int32, "xor"),
    (torch.int32, "umin"), (torch.int32, "umax"),
])
@pytest.mark.parametrize("shared", [True, False], ids=["shared_masked", "distinct_computed"])
def test_multiband_scan_c512_column_broadcast_matches_torch(dtype, reduction, shared):
    _check_c512_column_broadcast(
        8 if shared else 4, shared, dtype, reduction, num_warps=1 if shared else 4,
        masked=True, computed=not shared or reduction in ("product", "xor"),
        reverse=True, full_output=shared or reduction not in ("sum", "max", "or"), snapshot=shared)


@mps_only
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
def test_multiband_scan_c512_column_broadcasts_reuse_publications(dtype):
    rows, columns = 4, 512
    size = rows * columns
    index = torch.arange(size).reshape(rows, columns)
    x = (index % 17 - 8).to(dtype)
    w = ((index * 3 + 5) % 23 - 11).to(dtype)
    v = ((index * 7 + 2) % 31 - 19).to(dtype)
    sentinel = -999999
    outputs = [torch.full((size + 7,), sentinel, dtype=dtype, device="mps") for _ in range(4)]
    compiled = _different_inputs_scan_multiple_column_broadcasts_kernel[(1,)](
        x.to("mps"), w.to("mps"), v.to("mps"), *outputs,
        rows, columns, 0, num_warps=4)
    raw = compiled.asm["metal"]
    msl = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    buffers = sorted(int(length) for length in re.findall(
        r"\bthreadgroup\s+(?:float|u?int32_t|u?int)\s+\w+\s*\[\s*(\d+)\s*\]\s*;", msl))
    assert buffers == [size] * 3, msl
    assert msl.count("threadgroup_barrier(") == 1, msl
    torch.mps.synchronize()
    scanned = x.cumsum(0).to(dtype)
    reductions = [w.sum(0).to(dtype), w.amax(0), v.amin(0), x.amax(0)]
    for output, reduced in zip(outputs, reductions):
        contribution = reduced[None, :].expand_as(x)
        actual = output.cpu()[:size].reshape(rows, columns)
        torch.testing.assert_close(actual, scanned + contribution, atol=0, rtol=0)
        torch.testing.assert_close(actual - scanned, contribution, atol=0, rtol=0)
        assert (reduced[256:] != reduced[:256]).any()
        assert (output.cpu()[size:] == sentinel).all()


@pytest.mark.parametrize("sources", [1, 2, 3])
def test_multiband_scan_c512_column_broadcast_total_budget_is_named(sources):
    if sources == 3:
        kernel = "_different_inputs_scan_multiple_column_broadcasts_kernel"
        signature = {"X": "*fp32", "W": "*fp32", "V": "*fp32", "Sum": "*fp32",
                     "Max": "*fp32", "Min": "*fp32", "SharedMax": "*fp32"}
        constants = {"R": 8, "C": 512, "SCAN_AXIS": 0, "PRODUCT": False}
    else:
        kernel = "_i32_nonplain_input_scan_column_broadcast_kernel"
        signature = {"X": "*i32", "W": "*i32", "Y": "*i32", "N": "i32"}
        constants = {"R": 32 if sources == 1 else 16, "C": 512, "SCAN_AXIS": 0,
                     "REVERSE": False, "REDUCTION": "sum", "SHARED": sources == 1,
                     "MASKED": True, "COMPUTED": True, "OTHER": 1,
                     "SNAPSHOT": False, "FULL_OUTPUT": True, "SUBTRACT": False}
    result = _compile_rankn_nested_negative_in_child(kernel, signature, constants, num_warps=4)
    output = result.stdout + result.stderr
    assert result.returncode > 0, output
    if sources == 1:
        assert "rank >= 2 scan is implemented for a blocked tile" in output, output
        assert "threadgroup budget" in output, output
    else:
        assert "full-tile exchange scratch needs" in output, output
        assert "exceeding the 32768-byte budget" in output, output


@mps_only
@pytest.mark.parametrize("shared,dtype,reduction", [
    (True, torch.float32, "sum"),
    (True, torch.int32, "product"),
    (False, torch.float32, "product"),
    (False, torch.int32, "sum"),
])
def test_multiband_scan_c512_column_broadcast_exact_budget_matches_torch(shared, dtype, reduction):
    rows, columns = (16 if shared else 8), 512
    compiled = _check_c512_column_broadcast(
        rows, shared, dtype, reduction, num_warps=4, masked=True, computed=True,
        reverse=True, full_output=True, snapshot=False)
    raw = compiled.asm["metal"]
    msl = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    buffers = sorted(int(length) for length in re.findall(
        r"\bthreadgroup\s+(?:float|u?int32_t|u?int)\s+\w+\s*\[\s*(\d+)\s*\]\s*;", msl))
    # Broadcast consumers need no result-vector exchange: complete source
    # publications may occupy the full 32 KiB without additional scratch.
    assert buffers == [rows * columns] * (1 if shared else 2), msl
    assert sum(buffers) * 4 == 32768, msl
    assert msl.count("threadgroup_barrier(") == 1, msl


def test_multiband_scan_c512_column_broadcast_snapshot_scratch_budget_is_named():
    result = _compile_rankn_nested_negative_in_child(
        "_i32_nonplain_input_scan_column_broadcast_kernel",
        {"X": "*fp32", "W": "*fp32", "Y": "*fp32", "N": "i32"},
        {"R": 16, "C": 512, "SCAN_AXIS": 0, "REVERSE": True,
         "REDUCTION": "sum", "SHARED": True, "MASKED": True, "COMPUTED": True,
         "OTHER": -0.75, "SNAPSHOT": True, "FULL_OUTPUT": True, "SUBTRACT": True}, num_warps=4)
    output = result.stdout + result.stderr
    assert result.returncode > 0, output
    assert "multidimensional tuple scan scratch needs 33280 bytes" in output, output
    assert "exceeding the 32768-byte budget" in output, output


@triton.jit
def _mixed_row_minimum_combine(a, b):
    return tl.minimum(a, b, propagate_nan=tl.PropagateNan.ALL)


@triton.jit
def _mixed_row_maximum_combine(a, b):
    return tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)


def _f32_row_extremum_reference(source, reduction):
    # Sequential CPU oracle; ALL orders -0 below +0. NONE leaves mixed-sign
    # zero ties unspecified, so tests only require their value and fold parity.
    is_min = reduction in ("min", "minnum")
    propagate_nan = reduction in ("min", "max")
    result = source[..., 0].clone()
    for column in range(1, source.shape[-1]):
        rhs = source[..., column]
        lhs_nan, rhs_nan = torch.isnan(result), torch.isnan(rhs)
        combined = torch.where(rhs < result if is_min else rhs > result, rhs, result)
        zeros = (result == 0) & (rhs == 0)
        negative = (torch.signbit(result) | torch.signbit(rhs) if is_min else
                    torch.signbit(result) & torch.signbit(rhs))
        combined = torch.where(zeros, torch.where(negative, -torch.zeros_like(result),
                                                  torch.zeros_like(result)), combined)
        if propagate_nan:
            result = torch.where(lhs_nan | rhs_nan, float("nan"), combined)
        else:
            result = torch.where(lhs_nan, rhs, torch.where(rhs_nan, result, combined))
    return result


@triton.jit
def _shared_input_scan_mixed_row_sum_kernel(
        X, Y, Scanned, Row, Row2, Y2, N, R: tl.constexpr, C: tl.constexpr,
        SCAN_AXIS: tl.constexpr, REVERSE: tl.constexpr, MASKED: tl.constexpr,
        COMPUTED: tl.constexpr, OTHER: tl.constexpr, ROW_MASKED: tl.constexpr,
        FULL_OUTPUT: tl.constexpr, SNAPSHOT: tl.constexpr, MULTIPLE: tl.constexpr,
        TILE_FIRST: tl.constexpr = False, SNAPSHOT_AFTER: tl.constexpr = False,
        PRODUCT: tl.constexpr = False, REDUCTION: tl.constexpr = "sum",
        PURE_BROADCAST: tl.constexpr = False):
    program = tl.program_id(0)
    row = tl.arange(0, R)
    offset = program * R * C + row[:, None] * C + tl.arange(0, C)[None, :]
    if MASKED:
        x = tl.load(X + offset, offset < N, other=OTHER)
    else:
        x = tl.load(X + offset)
    if COMPUTED:
        x = x * 2 + 3
    scanned = tl.cumsum(x, SCAN_AXIS, reverse=REVERSE)
    if SNAPSHOT:
        tl.store(X + offset, x * 3 + 17, offset < N)
    if PRODUCT:
        reduced = tl.reduce(x, 1, _i32_column_product_combine)
    elif REDUCTION == "smin":
        reduced = tl.min(x, 1)
    elif REDUCTION == "smax":
        reduced = tl.max(x, 1)
    elif REDUCTION == "umin":
        reduced = tl.min(x.to(tl.uint32), 1).to(tl.int32)
    elif REDUCTION == "umax":
        reduced = tl.max(x.to(tl.uint32), 1).to(tl.int32)
    elif REDUCTION == "and":
        reduced = tl.reduce(x, 1, _i32_row_and_combine)
    elif REDUCTION == "or":
        reduced = tl.reduce(x, 1, _i32_row_or_combine)
    elif REDUCTION == "xor":
        reduced = tl.reduce(x, 1, _i32_row_xor_combine)
    elif REDUCTION == "min":
        reduced = tl.reduce(x, 1, _mixed_row_minimum_combine)
    elif REDUCTION == "max":
        reduced = tl.reduce(x, 1, _mixed_row_maximum_combine)
    elif REDUCTION == "minnum":
        reduced = tl.min(x, 1)
    elif REDUCTION == "maxnum":
        reduced = tl.max(x, 1)
    else:
        reduced = tl.sum(x, 1)
    if not TILE_FIRST:
        if ROW_MASKED:
            tl.store(Row + program * R + row, reduced, row < R - 1)
        else:
            tl.store(Row + program * R + row, reduced)
    # Store non-sum reductions independently so combining with the scan cannot
    # hide a wrong fold, including a zero, NaN or fully masked row.
    if PRODUCT or REDUCTION != "sum" or PURE_BROADCAST:
        result = tl.broadcast_to(reduced[:, None], (R, C))
    else:
        result = scanned + reduced[:, None]
    if FULL_OUTPUT:
        tl.store(Y + offset, result)
        tl.store(Scanned + offset, scanned)
    else:
        tl.store(Y + offset, result, offset < N)
        tl.store(Scanned + offset, scanned, offset < N)
    if TILE_FIRST:
        if ROW_MASKED:
            tl.store(Row + program * R + row, reduced, row < R - 1)
        else:
            tl.store(Row + program * R + row, reduced)
    if MULTIPLE:
        tl.store(Row2 + program * R + row, reduced, row > 0)
        second = scanned - reduced[:, None] * 2
        if PURE_BROADCAST:
            # Integer mixed consumers must preserve signed comparison semantics
            # after reading a source publication stored as ui32.
            second = tl.where(reduced[:, None] < 0, second, -second)
        if FULL_OUTPUT:
            tl.store(Y2 + offset, second)
        else:
            tl.store(Y2 + offset, second, offset < N)

    if SNAPSHOT_AFTER:
        tl.store(X + offset, x * 3 + 17, offset < N)


def _check_shared_input_mixed_row_sum(rows, columns, scan_axis, num_warps, masked, computed,
                                     reverse=False, other=0.75, row_masked=False,
                                     full_output=False, snapshot=False, multiple=False,
                                     tile_first=False, snapshot_after=False, product=False, reduction="sum",
                                     distinct_product=False, distinct_extremum=False):
    programs = 2
    size = programs * rows * columns
    valid = size - columns - 3 if masked else size
    index = torch.arange(valid)
    if product:
        # Alternating powers of two keep every sequential prefix bounded.
        # The first factor makes the final products vary across rows.
        exponent = torch.where(index % columns % 2 == 0, 1, -1)
        exponent = torch.where(index % columns == 0, index // columns % 3 - 1, exponent)
        target = 2. ** exponent.float()
        target *= torch.where((index % columns == 0) & (index // columns % 2 == 0), -1., 1.)
        if distinct_product:
            # One dyadic factor distinguishes every row while the alternating
            # powers of two keep all intermediate products finite and exact.
            first = 1 + (index // columns).float() / 128
            first *= torch.where(index // columns % 2 == 0, -1., 1.)
            target = torch.where(index % columns == 0, first, target.abs())
        original = (target - 3) / 2 if computed else target
    else:
        original = 1 + (index // columns % rows).float() / 4 + (index % columns % 7).float() / 8
    if distinct_extremum:
        # Late extrema distinguish every program and row, including columns
        # beyond the first subgroup. All values and scan sums are dyadic.
        original = 4 + (index // columns).float() / 128 + (index % columns).float() / 256
        original = torch.where(index % columns == columns - 1,
                               1 + (index // columns).float() / 128, original)
        original = torch.where(index % columns == columns - 2,
                               16 + (index // columns).float() / 128, original)
    x = torch.full((size,), other, dtype=torch.float32)
    x[:valid] = original
    x = x.reshape(programs, rows, columns)
    if computed:
        x = x * 2 + 3
    axis = scan_axis + 1
    scanned = (x.flip(axis) if reverse else x).cumsum(axis)
    if reverse:
        scanned = scanned.flip(axis)
    if product:
        reduced = _f32_column_product_reference(x.transpose(1, 2))
    elif reduction != "sum":
        reduced = _f32_row_extremum_reference(x, reduction)
    else:
        reduced = x.sum(2)
    assert reduced[0].unique().numel() > 1 and (reduced[0] != 0).all()
    if (distinct_product or distinct_extremum) and not masked:
        assert reduced.unique().numel() == programs * rows
    contribution = reduced[:, :, None].expand_as(x)
    sentinel = -999999.
    values = original.to("mps")
    output, scan_output, output2 = [torch.full((size + 7,), sentinel, dtype=torch.float32, device="mps")
                                    for _ in range(3)]
    row, row2 = [torch.full((programs * rows + 7,), sentinel, dtype=torch.float32, device="mps")
                 for _ in range(2)]
    compiled = _shared_input_scan_mixed_row_sum_kernel[(programs,)](
        values, output, scan_output, row, row2, output2, valid, rows, columns,
        scan_axis, reverse, masked, computed, other, row_masked, full_output, snapshot,
        multiple, TILE_FIRST=tile_first, SNAPSHOT_AFTER=snapshot_after, PRODUCT=product,
        REDUCTION=reduction, num_warps=num_warps)
    torch.mps.synchronize()
    written = size if full_output else valid
    pure_broadcast = product or reduction != "sum"
    expected = contribution if pure_broadcast else scanned + contribution
    actual = output.cpu()
    torch.testing.assert_close(actual[:written], expected.flatten()[:written], atol=0, rtol=0)
    actual_contribution = actual[:written] if pure_broadcast else actual[:written] - scanned.flatten()[:written]
    torch.testing.assert_close(actual_contribution, contribution.flatten()[:written], atol=0, rtol=0)
    torch.testing.assert_close(scan_output.cpu()[:written], scanned.flatten()[:written], atol=0, rtol=0)
    valid_rows = rows - 1 if row_masked else rows
    actual_rows = row.cpu()[:programs * rows].reshape(programs, rows)
    torch.testing.assert_close(actual_rows[:, :valid_rows], reduced[:, :valid_rows], atol=0, rtol=0)
    assert (actual_rows[:, valid_rows:] == sentinel).all()
    assert (row.cpu()[programs * rows:] == sentinel).all()
    assert (actual[written:] == sentinel).all()
    assert (scan_output.cpu()[written:] == sentinel).all()
    if multiple:
        actual_rows2 = row2.cpu()[:programs * rows].reshape(programs, rows)
        torch.testing.assert_close(actual_rows2[:, 1:], reduced[:, 1:], atol=0, rtol=0)
        assert (actual_rows2[:, 0] == sentinel).all()
        assert (row2.cpu()[programs * rows:] == sentinel).all()
        expected2 = scanned - contribution * 2
        torch.testing.assert_close(output2.cpu()[:written], expected2.flatten()[:written], atol=0, rtol=0)
        assert (output2.cpu()[written:] == sentinel).all()
    else:
        assert (row2.cpu() == sentinel).all()
        assert (output2.cpu() == sentinel).all()
    expected_input = x.flatten()[:valid] * 3 + 17 if snapshot or snapshot_after else original
    torch.testing.assert_close(values.cpu(), expected_input, atol=0, rtol=0)
    return compiled


@mps_only
@pytest.mark.parametrize("rows,columns,scan_axis,num_warps,masked,computed", [
    (16, 16, 0, 1, False, False),
    (16, 16, 1, 1, True, False),
    (8, 64, 0, 4, True, True),
    (8, 64, 1, 4, False, True),
    (32, 32, 0, 4, True, False),
    (32, 32, 1, 1, True, True),
])
def test_multiband_scan_shared_input_mixed_row_sum_baseline_matches_torch(
        rows, columns, scan_axis, num_warps, masked, computed):
    _check_shared_input_mixed_row_sum(rows, columns, scan_axis, num_warps, masked, computed)


@mps_only
@pytest.mark.parametrize("multiple", [False, True], ids=["one_each", "multiple_stores_and_broadcasts"])
@pytest.mark.parametrize(
    "rows,columns,scan_axis,num_warps,masked,computed,reverse,other,row_masked,full_output,snapshot", [
        (2, 64, 1, 1, True, False, True, 1.25, True, True, False),
        (64, 2, 0, 1, True, True, False, -0.75, False, False, True),
        (16, 16, 0, 4, False, True, True, 0.75, False, True, True),
        (8, 64, 1, 4, True, True, True, 0.75, True, False, False),
        (32, 32, 1, 1, True, False, False, -1., False, True, False),
        (64, 64, 0, 4, True, True, True, 1.25, True, True, True),
        (64, 32, 1, 4, False, False, False, 0., False, False, False),
        (4, 64, 0, 4, True, False, True, 0.5, True, True, False),
        (16, 64, 1, 1, True, True, False, -0.75, False, True, True),
        (32, 16, 0, 1, True, False, True, 1., False, False, False),
    ])
def test_multiband_scan_shared_input_mixed_row_sum_matches_torch(
        rows, columns, scan_axis, num_warps, masked, computed, reverse, other,
        row_masked, full_output, snapshot, multiple):
    _check_shared_input_mixed_row_sum(
        rows, columns, scan_axis, num_warps, masked, computed, reverse, other,
        row_masked, full_output, snapshot, multiple)


@mps_only
@pytest.mark.parametrize("multiple", [False, True])
def test_multiband_scan_shared_input_mixed_row_sum_reuses_publication(multiple):
    rows, columns = 8, 64
    compiled = _check_shared_input_mixed_row_sum(
        rows, columns, 0, 1, True, True, full_output=True, multiple=multiple)
    raw = compiled.asm["metal"]
    msl = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    buffers = sorted(int(length) for length in re.findall(
        r"\bthreadgroup\s+float\s+\w+\s*\[\s*(\d+)\s*\]\s*;", msl))
    # Splitting the two result layouts must retain one full input publication;
    # multiple stores reuse the one vector conversion, without republishing X.
    # The second masked row store also uses one 32-thread temporary; it is
    # store scratch, not another input publication or reduction exchange.
    expected_buffers = [rows, 32, rows * columns] if multiple else [rows, rows * columns]
    assert buffers == expected_buffers, msl
    assert msl.count("threadgroup_barrier(") == 2, msl


@triton.jit
def _shared_input_mixed_row_sum_budget_kernel(
        X, Y, Row, WRow, VRow, SOURCES: tl.constexpr, PRODUCT: tl.constexpr = False,
        REDUCTION: tl.constexpr = "sum"):
    row = tl.arange(0, 64)
    offset = row[:, None] * 64 + tl.arange(0, 64)[None, :]
    x = tl.load(X + offset)
    w = x * 2 + 3
    if SOURCES == 3:
        v = x * 3 - 1
    scanned = tl.cumsum(x, 0)
    if PRODUCT:
        reduced = tl.reduce(x, 1, _i32_column_product_combine)
    elif REDUCTION == "smin":
        reduced = tl.min(x, 1)
    elif REDUCTION == "smax":
        reduced = tl.max(x, 1)
    elif REDUCTION == "umin":
        reduced = tl.min(x.to(tl.uint32), 1).to(tl.int32)
    elif REDUCTION == "umax":
        reduced = tl.max(x.to(tl.uint32), 1).to(tl.int32)
    elif REDUCTION == "and":
        reduced = tl.reduce(x, 1, _i32_row_and_combine)
    elif REDUCTION == "or":
        reduced = tl.reduce(x, 1, _i32_row_or_combine)
    elif REDUCTION == "xor":
        reduced = tl.reduce(x, 1, _i32_row_xor_combine)
    elif REDUCTION == "min":
        reduced = tl.reduce(x, 1, _mixed_row_minimum_combine)
    elif REDUCTION == "maxnum":
        reduced = tl.max(x, 1)
    else:
        reduced = tl.sum(x, 1)
    other_w = tl.sum(w, 1)
    if SOURCES == 3:
        other_v = tl.sum(v, 1)
    tl.store(Row + row, reduced)
    tl.store(Y + offset, scanned + reduced[:, None])
    tl.store(WRow + row, other_w)
    if SOURCES == 3:
        tl.store(VRow + row, other_v)


@pytest.mark.parametrize("sources", [2, 3])
def test_multiband_scan_shared_input_mixed_row_sum_total_budget_is_named(sources):
    result = _compile_rankn_nested_negative_in_child(
        "_shared_input_mixed_row_sum_budget_kernel",
        {"X": "*fp32", "Y": "*fp32", "Row": "*fp32", "WRow": "*fp32", "VRow": "*fp32"},
        {"SOURCES": sources, "PRODUCT": False, "REDUCTION": "sum"}, num_warps=4)
    output = result.stdout + result.stderr
    assert result.returncode > 0, output
    assert "full-tile exchange scratch needs" in output, output
    assert "exceeding the 32768-byte budget" in output, output


@triton.jit
def _unsupported_shared_input_mixed_row_sum_kernel(
        X, W, Y, Row, Scanned, N, R: tl.constexpr, C: tl.constexpr,
        SCAN_AXIS: tl.constexpr, MODE: tl.constexpr, REDUCTION: tl.constexpr):
    row = tl.arange(0, R)
    column = tl.arange(0, C)
    offset = row[:, None] * C + column[None, :]
    x = tl.load(X + offset, offset < N, other=0)
    if MODE == 0:
        w = tl.load(W + offset, offset < N, other=0)
    elif MODE == 6:
        w = tl.trans(x)
    else:
        w = x
    scanned = tl.cumsum(x, SCAN_AXIS)
    if MODE == 1:
        w = scanned
    if MODE == 2:
        scanned += tl.cumsum(x * 2 + 1, SCAN_AXIS)
    if REDUCTION == "product":
        reduced = tl.reduce(w, 1, _i32_column_product_combine)
    elif REDUCTION == "smin":
        reduced = tl.min(w, 1)
    elif REDUCTION == "smax":
        reduced = tl.max(w, 1)
    elif REDUCTION == "umin":
        reduced = tl.min(w.to(tl.uint32), 1).to(tl.int32)
    elif REDUCTION == "umax":
        reduced = tl.max(w.to(tl.uint32), 1).to(tl.int32)
    elif REDUCTION == "and":
        reduced = tl.reduce(w, 1, _i32_row_and_combine)
    elif REDUCTION == "or":
        reduced = tl.reduce(w, 1, _i32_row_or_combine)
    elif REDUCTION == "min":
        reduced = tl.reduce(w, 1, _mixed_row_minimum_combine)
    elif REDUCTION == "minnum":
        reduced = tl.min(w, 1)
    elif REDUCTION == "maximum":
        reduced = tl.reduce(w, 1, _mixed_row_maximum_combine)
    elif REDUCTION == "xor":
        reduced = tl.reduce(w, 1, _i32_row_xor_combine)
    elif REDUCTION == "max":
        reduced = tl.max(w, 1)
    else:
        reduced = tl.sum(w, 1)
    if MODE == 7:
        if N > 0:
            tl.store(Row + row, reduced)
    else:
        tl.store(Row + row, reduced)
    if MODE == 3:
        tl.store(Y + offset, scanned + reduced[None, :])
    elif MODE == 4:
        wider_column = tl.arange(0, 2 * C)
        wider_offset = row[:, None] * (2 * C) + wider_column[None, :]
        tl.store(Y + wider_offset, reduced[:, None] + wider_column[None, :])
    elif MODE == 5:
        tile = scanned + reduced[:, None]
        if N > 0:
            tl.store(Y + offset, tile)
    elif MODE == 8:
        tile = scanned + reduced[:, None]
        tl.store(Y + offset, tl.trans(tile))
    else:
        tl.store(Y + offset, scanned + reduced[:, None])
    tl.store(Scanned + offset, scanned)


@pytest.mark.parametrize("mode,diagnostic", [
    (0, "a shared scan-input row reduction requires direct row stores or "
     "same-axis broadcasts into its original tile"),
    (1, "reduce cannot replay its computed source cone across conversion"),
    (2, "a rank-2 tl.reduce cannot share a function with a multi-band staged scan"),
    (3, "a shared scan-input row reduction requires direct row stores or "
     "same-axis broadcasts into its original tile"),
    (4, "a shared scan-input row reduction requires direct row stores or "
     "same-axis broadcasts into its original tile"),
    (5, "a shared scan-input row reduction requires direct row stores or "
     "same-axis broadcasts into its original tile"),
    (6, "a rank-2 tl.reduce cannot share a function with a multi-band staged scan"),
    (7, "a shared scan-input row reduction requires direct row stores or "
     "same-axis broadcasts into its original tile"),
    (8, "a shared scan-input row reduction requires direct row stores or "
     "same-axis broadcasts into its original tile"),
], ids=["distinct_source", "scan_result", "multiple_scans", "cross_axis", "wider_tile", "region", "transpose",
        "vector_store_region", "late_transpose"])
def test_multiband_scan_shared_input_mixed_row_sum_source_and_consumer_is_named(mode, diagnostic):
    result = _compile_rankn_nested_negative_in_child(
        "_unsupported_shared_input_mixed_row_sum_kernel",
        {"X": "*fp32", "W": "*fp32", "Y": "*fp32", "Row": "*fp32", "Scanned": "*fp32", "N": "i32"},
        {"R": 16, "C": 16, "SCAN_AXIS": 1, "MODE": mode, "REDUCTION": "sum"}, num_warps=1)
    output = result.stdout + result.stderr
    assert result.returncode > 0, output
    assert diagnostic in output, output


@pytest.mark.parametrize("rows,columns,scan_axis,dtype,reduction,diagnostic", [
    (128, 16, 1, "*fp32", "sum", "a shared scan-input row reduction requires direct row stores or "
     "same-axis broadcasts into its original tile"),
    (16, 128, 0, "*fp32", "sum", "a shared scan-input row reduction requires direct row stores or "
     "same-axis broadcasts into its original tile"),
    (16, 16, 1, "*fp16", "sum", "rank-2 axis=1 reduce over a computed cone is implemented for f32 only; "
     "other element types require a direct unmasked tt.load"),
], ids=["wide_rows", "wide_columns", "f16"])
def test_multiband_scan_shared_input_mixed_row_sum_shape_type_and_combine_is_named(
        rows, columns, scan_axis, dtype, reduction, diagnostic):
    result = _compile_rankn_nested_negative_in_child(
        "_unsupported_shared_input_mixed_row_sum_kernel",
        {"X": dtype, "W": dtype, "Y": dtype, "Row": dtype, "Scanned": dtype, "N": "i32"},
        {"R": rows, "C": columns, "SCAN_AXIS": scan_axis, "MODE": -1, "REDUCTION": reduction}, num_warps=1)
    output = result.stdout + result.stderr
    assert result.returncode > 0, output
    assert diagnostic in output, output


@mps_only
@pytest.mark.parametrize("rows,columns,scan_axis,num_warps,masked,computed", [
    (2, 64, 0, 4, False, False),
    (2, 64, 0, 4, True, False),
    (2, 64, 0, 4, True, True),
    (4, 32, 1, 4, False, True),
    (8, 16, 0, 4, True, False),
    (4, 8, 1, 1, True, True),
])
def test_single_band_scan_shared_input_mixed_row_sum_baseline_matches_torch(
        rows, columns, scan_axis, num_warps, masked, computed):
    assert rows * columns == 32 * num_warps
    _check_shared_input_mixed_row_sum(rows, columns, scan_axis, num_warps, masked, computed,
                                     full_output=True)


@mps_only
@pytest.mark.parametrize("tile_first", [False, True], ids=["row_first", "tile_first"])
@pytest.mark.parametrize(
    "rows,columns,scan_axis,num_warps,masked,computed,reverse,other,row_masked,full_output,multiple,snapshot_after", [
        (2, 16, 0, 1, False, False, True, 0., False, False, False, False),
        (4, 8, 1, 1, True, False, False, 0.75, True, True, True, True),
        (8, 4, 0, 1, True, True, True, -0.75, False, False, True, False),
        (16, 2, 1, 1, False, True, False, 1., False, True, False, True),
        (2, 64, 0, 4, True, False, True, 1.25, True, True, True, False),
        (4, 32, 1, 4, True, True, False, 0., False, False, False, True),
        (8, 16, 0, 4, False, False, True, 0., False, True, True, False),
        (16, 8, 1, 4, True, True, True, 0.75, True, True, True, True),
        (32, 4, 0, 4, True, False, False, -1., False, False, False, True),
        (64, 2, 1, 4, False, True, True, 0., False, True, True, False),
    ])
def test_single_band_scan_shared_input_mixed_row_sum_matches_torch(
        rows, columns, scan_axis, num_warps, masked, computed, reverse, other,
        row_masked, full_output, multiple, snapshot_after, tile_first):
    assert rows * columns == 32 * num_warps
    _check_shared_input_mixed_row_sum(
        rows, columns, scan_axis, num_warps, masked, computed, reverse, other,
        row_masked, full_output, multiple=multiple, tile_first=tile_first,
        snapshot_after=snapshot_after)


@mps_only
@pytest.mark.parametrize("rows,columns,scan_axis,num_warps", [(2, 64, 0, 4), (4, 8, 1, 1)])
def test_single_band_scan_shared_input_mixed_row_sum_separates_result_layouts(
        rows, columns, scan_axis, num_warps):
    compiled = _check_shared_input_mixed_row_sum(rows, columns, scan_axis, num_warps, False, False)
    raw = compiled.asm["metal"]
    msl = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    buffers = sorted(int(length) for length in re.findall(
        r"\bthreadgroup\s+float\s+\w+\s*\[\s*(\d+)\s*\]\s*;", msl))
    # Each consumer layout gets its own row-result buffer. The two CTA-sized
    # buffers belong to the legacy scan and masked tile stores, not publication.
    assert buffers == sorted([rows, rows, 32 * num_warps, 32 * num_warps]), msl
    # The scan has two barriers; each row fold contributes one more.
    assert msl.count("threadgroup_barrier(") == 4, msl


@pytest.mark.parametrize("masked,computed", [(False, False), (True, True)])
def test_single_band_scan_shared_input_mixed_row_sum_snapshot_before_is_named(masked, computed):
    result = _compile_rankn_nested_negative_in_child(
        "_shared_input_scan_mixed_row_sum_kernel",
        {"X": "*fp32", "Y": "*fp32", "Scanned": "*fp32", "Row": "*fp32",
         "Row2": "*fp32", "Y2": "*fp32", "N": "i32"},
        {"R": 2, "C": 64, "SCAN_AXIS": 0, "REVERSE": False, "MASKED": masked,
         "COMPUTED": computed, "OTHER": 0.75, "ROW_MASKED": False,
         "FULL_OUTPUT": True, "SNAPSHOT": True, "MULTIPLE": False,
         "TILE_FIRST": False, "SNAPSHOT_AFTER": False, "PRODUCT": False,
         "REDUCTION": "sum", "PURE_BROADCAST": False}, num_warps=4)
    output = result.stdout + result.stderr
    assert result.returncode > 0, output
    assert ("single-band mixed row reduction requires replayable sources defined before the scan "
            "without intervening writes or input exchanges") in output, output


@triton.jit
def _unsupported_single_band_mixed_row_sum_source_kernel(X, W, Y, Row, N, MODE: tl.constexpr):
    row = tl.arange(0, 8)
    offset = row[:, None] * 8 + tl.arange(0, 8)[None, :]
    x = tl.load(X + offset, offset < N, other=0.75)
    if MODE == 0:
        tl.store(W + offset, x + 1)
    scanned = tl.cumsum(x, 0)
    if MODE == 1:
        tl.store(W + offset, x + 1)
    reduced = tl.sum(x, 1)
    tl.store(Row + row, reduced)
    tl.store(Y + offset, scanned + reduced[:, None])


@pytest.mark.parametrize("mode", [0, 1], ids=["write_before_scan", "write_after_scan"])
def test_single_band_scan_shared_input_mixed_row_sum_source_is_named(mode):
    result = _compile_rankn_nested_negative_in_child(
        "_unsupported_single_band_mixed_row_sum_source_kernel",
        {"X": "*fp32", "W": "*fp32", "Y": "*fp32", "Row": "*fp32", "N": "i32"},
        {"MODE": mode}, num_warps=2)
    output = result.stdout + result.stderr
    assert result.returncode > 0, output
    assert ("single-band mixed row reduction requires replayable sources defined before the scan "
            "without intervening writes or input exchanges") in output, output


@triton.jit
def _unsupported_single_band_mixed_row_product_kernel(
        X, Y, Row, Scanned, N, R: tl.constexpr, C: tl.constexpr,
        SCAN_AXIS: tl.constexpr, MODE: tl.constexpr, REDUCTION: tl.constexpr = "product"):
    row = tl.arange(0, R)
    column = tl.arange(0, C)
    offset = row[:, None] * C + column[None, :]
    x = tl.load(X + offset)
    scanned = tl.cumsum(x, SCAN_AXIS)
    if REDUCTION == "min":
        reduced = tl.reduce(x, 1, _mixed_row_minimum_combine)
    elif REDUCTION == "max":
        reduced = tl.reduce(x, 1, _mixed_row_maximum_combine)
    else:
        reduced = tl.reduce(x, 1, _i32_column_product_combine)
    if MODE == 7:
        if N > 0:
            tl.store(Row + row, reduced)
    else:
        tl.store(Row + row, reduced)
    if MODE == 3:
        tl.store(Y + offset, scanned + reduced[None, :])
    elif MODE == 5:
        tile = scanned + reduced[:, None]
        if N > 0:
            tl.store(Y + offset, tile)
    elif MODE == 8:
        tile = scanned + reduced[:, None]
        tl.store(Y + offset, tl.trans(tile))
    else:
        tl.store(Y + offset, scanned + reduced[:, None])
    tl.store(Scanned + offset, scanned)


@pytest.mark.parametrize("rows,columns,scan_axis,num_warps,mode", [
    (8, 8, 0, 2, 3),
    (2, 128, 0, 8, -1),
    (128, 2, 1, 8, -1),
    (8, 8, 0, 2, 5),
    (8, 8, 0, 2, 7),
    (8, 8, 0, 2, 8),
], ids=["cross_axis", "wide_columns", "wide_rows", "tile_region", "vector_region", "late_transpose"])
def test_single_band_scan_shared_input_mixed_row_sum_consumer_is_named(
        rows, columns, scan_axis, num_warps, mode):
    result = _compile_rankn_nested_negative_in_child(
        "_unsupported_shared_input_mixed_row_sum_kernel",
        {"X": "*fp32", "W": "*fp32", "Y": "*fp32", "Row": "*fp32", "Scanned": "*fp32", "N": "i32"},
        {"R": rows, "C": columns, "SCAN_AXIS": scan_axis, "MODE": mode, "REDUCTION": "sum"}, num_warps=num_warps)
    output = result.stdout + result.stderr
    assert result.returncode > 0, output
    diagnostic = ("ttg.convert_layout: broader staged-transpose deferred to L1d3" if mode == 3 else
                  "single-band mixed row reduction requires direct row stores and same-phase broadcasts "
                  "into its original 2..64 tile")
    assert diagnostic in output, output


@triton.jit
def _single_band_mixed_row_sum_many_results_kernel(
        X, Y, Row, Additional, EXTRAS: tl.constexpr, PRODUCT: tl.constexpr = False,
        REDUCTION: tl.constexpr = "sum"):
    row = tl.arange(0, 64)
    offset = row[:, None] * 2 + tl.arange(0, 2)[None, :]
    x = tl.load(X + offset)
    scanned = tl.cumsum(x, 0)
    if PRODUCT:
        reduced = tl.reduce(x, 1, _i32_column_product_combine)
    elif REDUCTION == "min":
        reduced = tl.reduce(x, 1, _mixed_row_minimum_combine)
    elif REDUCTION == "max":
        reduced = tl.reduce(x, 1, _mixed_row_maximum_combine)
    else:
        reduced = tl.sum(x, 1)
    tl.store(Row + row, reduced)
    tl.store(Y + offset, scanned + reduced[:, None])
    for index in tl.static_range(0, EXTRAS):
        extra = tl.sum(x + (index + 1) * 0.125, 1)
        tl.store(Additional + index * 64 + row, extra)


@mps_only
def test_single_band_scan_shared_input_mixed_row_sum_counts_all_result_scratch():
    rows, columns, extras = 64, 2, 2
    x = (torch.arange(rows * columns).reshape(rows, columns) % 17).float() / 8
    sentinel = -999999.
    output = torch.full((rows * columns + 7,), sentinel, dtype=torch.float32, device="mps")
    row = torch.full((rows + 7,), sentinel, dtype=torch.float32, device="mps")
    additional = torch.full((extras * rows + 7,), sentinel, dtype=torch.float32, device="mps")
    compiled = _single_band_mixed_row_sum_many_results_kernel[(1,)](
        x.to("mps"), output, row, additional, extras, num_warps=4)
    raw = compiled.asm["metal"]
    msl = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    buffers = sorted(int(length) for length in re.findall(
        r"\bthreadgroup\s+float\s+\w+\s*\[\s*(\d+)\s*\]\s*;", msl))
    assert buffers == [rows] * (2 + extras) + [rows * columns], msl
    # Two scan barriers plus one for each of the two split folds and extra folds.
    assert msl.count("threadgroup_barrier(") == 4 + extras, msl
    torch.mps.synchronize()
    reduced = x.sum(1)
    torch.testing.assert_close(output.cpu()[:rows * columns], (x.cumsum(0) + reduced[:, None]).flatten(),
                               atol=0, rtol=0)
    torch.testing.assert_close(row.cpu()[:rows], reduced, atol=0, rtol=0)
    for index in range(extras):
        torch.testing.assert_close(additional.cpu()[index * rows:(index + 1) * rows],
                                   (x + (index + 1) * 0.125).sum(1), atol=0, rtol=0)
    assert (output.cpu()[rows * columns:] == sentinel).all()
    assert (row.cpu()[rows:] == sentinel).all()
    assert (additional.cpu()[extras * rows:] == sentinel).all()


def test_single_band_scan_shared_input_mixed_row_sum_total_budget_is_named():
    result = _compile_rankn_nested_negative_in_child(
        "_single_band_mixed_row_sum_many_results_kernel",
        {"X": "*fp32", "Y": "*fp32", "Row": "*fp32", "Additional": "*fp32"},
        {"EXTRAS": 125, "PRODUCT": False, "REDUCTION": "sum"}, num_warps=4)
    output = result.stdout + result.stderr
    assert result.returncode > 0, output
    assert "single-band mixed row reduction scratch needs 33024 bytes" in output, output
    assert "exceeding the 32768-byte budget" in output, output


@mps_only
@pytest.mark.parametrize("rows,columns,scan_axis,num_warps,masked,computed", [
    (16, 16, 0, 1, False, False),
    (16, 16, 1, 1, True, False),
    (8, 64, 0, 4, True, True),
    (8, 64, 1, 4, False, True),
    (32, 32, 0, 4, True, False),
    (32, 32, 1, 1, True, True),
])
def test_multiband_scan_shared_input_mixed_row_product_baseline_matches_torch(
        rows, columns, scan_axis, num_warps, masked, computed):
    _check_shared_input_mixed_row_sum(
        rows, columns, scan_axis, num_warps, masked, computed,
        other=-1. if computed else 1., full_output=True, product=True)


@mps_only
@pytest.mark.parametrize("rows,columns,scan_axis,num_warps", [
    (2, 64, 0, 4),
    (4, 32, 1, 4),
    (8, 16, 0, 4),
    (4, 8, 1, 1),
    (8, 4, 0, 1),
    (16, 2, 1, 1),
])
def test_single_band_scan_shared_input_mixed_row_product_baseline_matches_torch(
        rows, columns, scan_axis, num_warps):
    assert rows * columns == 32 * num_warps
    _check_shared_input_mixed_row_sum(
        rows, columns, scan_axis, num_warps, False, False,
        full_output=True, product=True, distinct_product=True)


@mps_only
@pytest.mark.parametrize("tile_first", [False, True], ids=["row_first", "tile_first"])
@pytest.mark.parametrize(
    "rows,columns,scan_axis,num_warps,reverse,row_masked,full_output,multiple,snapshot_after", [
        (2, 16, 0, 1, True, False, False, False, False),
        (4, 8, 1, 1, False, True, True, True, True),
        (8, 4, 0, 1, True, False, False, True, False),
        (16, 2, 1, 1, False, True, True, False, True),
        (2, 64, 0, 4, True, True, True, True, False),
        (4, 32, 1, 4, False, False, False, False, True),
        (8, 16, 0, 4, True, False, True, True, False),
        (16, 8, 1, 4, True, True, True, True, True),
        (32, 4, 0, 4, False, False, False, False, True),
        (64, 2, 1, 4, True, False, True, True, False),
    ])
def test_single_band_scan_shared_input_mixed_row_product_matches_torch(
        rows, columns, scan_axis, num_warps, reverse, row_masked, full_output,
        multiple, snapshot_after, tile_first):
    assert rows * columns == 32 * num_warps
    _check_shared_input_mixed_row_sum(
        rows, columns, scan_axis, num_warps, False, False, reverse=reverse,
        row_masked=row_masked, full_output=full_output, multiple=multiple,
        tile_first=tile_first, snapshot_after=snapshot_after, product=True,
        distinct_product=True)


@mps_only
@pytest.mark.parametrize("rows,columns,scan_axis,num_warps", [(2, 64, 0, 4), (4, 8, 1, 1)])
def test_single_band_scan_shared_input_mixed_row_product_separates_result_layouts(
        rows, columns, scan_axis, num_warps):
    compiled = _check_shared_input_mixed_row_sum(
        rows, columns, scan_axis, num_warps, False, False,
        product=True, distinct_product=True)
    raw = compiled.asm["metal"]
    msl = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    buffers = sorted(int(length) for length in re.findall(
        r"\bthreadgroup\s+float\s+\w+\s*\[\s*(\d+)\s*\]\s*;", msl))
    assert buffers == sorted([rows, rows, 32 * num_warps, 32 * num_warps]), msl
    assert msl.count("threadgroup_barrier(") == 4, msl


@pytest.mark.parametrize("masked,computed", [(True, False), (False, True), (True, True)])
def test_single_band_scan_shared_input_mixed_row_product_nonplain_source_is_named(masked, computed):
    result = _compile_rankn_nested_negative_in_child(
        "_shared_input_scan_mixed_row_sum_kernel",
        {"X": "*fp32", "Y": "*fp32", "Scanned": "*fp32", "Row": "*fp32",
         "Row2": "*fp32", "Y2": "*fp32", "N": "i32"},
        {"R": 2, "C": 64, "SCAN_AXIS": 0, "REVERSE": False, "MASKED": masked,
         "COMPUTED": computed, "OTHER": 0.75, "ROW_MASKED": False,
         "FULL_OUTPUT": True, "SNAPSHOT": False, "MULTIPLE": False,
         "TILE_FIRST": False, "SNAPSHOT_AFTER": False, "PRODUCT": True,
         "REDUCTION": "sum", "PURE_BROADCAST": False}, num_warps=4)
    output = result.stdout + result.stderr
    assert result.returncode > 0, output
    assert "single-band mixed row product requires a direct unmasked tt.load" in output, output


def test_single_band_scan_shared_input_mixed_row_product_snapshot_before_is_named():
    result = _compile_rankn_nested_negative_in_child(
        "_shared_input_scan_mixed_row_sum_kernel",
        {"X": "*fp32", "Y": "*fp32", "Scanned": "*fp32", "Row": "*fp32",
         "Row2": "*fp32", "Y2": "*fp32", "N": "i32"},
        {"R": 2, "C": 64, "SCAN_AXIS": 0, "REVERSE": False, "MASKED": False,
         "COMPUTED": False, "OTHER": 1., "ROW_MASKED": False,
         "FULL_OUTPUT": True, "SNAPSHOT": True, "MULTIPLE": False,
         "TILE_FIRST": False, "SNAPSHOT_AFTER": False, "PRODUCT": True,
         "REDUCTION": "sum", "PURE_BROADCAST": False}, num_warps=4)
    output = result.stdout + result.stderr
    assert result.returncode > 0, output
    assert ("single-band mixed row reduction requires replayable sources defined before the scan "
            "without intervening writes or input exchanges") in output, output


@pytest.mark.parametrize("rows,columns,scan_axis,num_warps,mode", [
    (8, 8, 0, 2, 3),
    (2, 128, 0, 8, -1),
    (128, 2, 1, 8, -1),
    (8, 8, 0, 2, 5),
    (8, 8, 0, 2, 7),
    (8, 8, 0, 2, 8),
], ids=["cross_axis", "wide_columns", "wide_rows", "tile_region", "vector_region", "late_transpose"])
def test_single_band_scan_shared_input_mixed_row_product_consumer_is_named(
        rows, columns, scan_axis, num_warps, mode):
    result = _compile_rankn_nested_negative_in_child(
        "_unsupported_single_band_mixed_row_product_kernel",
        {"X": "*fp32", "Y": "*fp32", "Row": "*fp32", "Scanned": "*fp32", "N": "i32"},
        {"R": rows, "C": columns, "SCAN_AXIS": scan_axis, "MODE": mode,
         "REDUCTION": "product"}, num_warps=num_warps)
    output = result.stdout + result.stderr
    assert result.returncode > 0, output
    diagnostic = ("ttg.convert_layout: broader staged-transpose deferred to L1d3" if mode == 3 else
                  "single-band mixed row reduction requires direct row stores and same-phase broadcasts "
                  "into its original 2..64 tile")
    assert diagnostic in output, output


@mps_only
def test_single_band_scan_shared_input_mixed_row_product_counts_all_result_scratch():
    rows, columns, extras = 64, 2, 2
    index = torch.arange(rows * columns).reshape(rows, columns)
    x = torch.where(index % columns == 0, 1 + index.float() / 256, 0.5)
    sentinel = -999999.
    output = torch.full((rows * columns + 7,), sentinel, dtype=torch.float32, device="mps")
    row = torch.full((rows + 7,), sentinel, dtype=torch.float32, device="mps")
    additional = torch.full((extras * rows + 7,), sentinel, dtype=torch.float32, device="mps")
    compiled = _single_band_mixed_row_sum_many_results_kernel[(1,)](
        x.to("mps"), output, row, additional, extras, PRODUCT=True, num_warps=4)
    raw = compiled.asm["metal"]
    msl = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    buffers = sorted(int(length) for length in re.findall(
        r"\bthreadgroup\s+float\s+\w+\s*\[\s*(\d+)\s*\]\s*;", msl))
    assert buffers == [rows] * (2 + extras) + [rows * columns], msl
    assert msl.count("threadgroup_barrier(") == 4 + extras, msl
    torch.mps.synchronize()
    reduced = _f32_column_product_reference(x.T)
    torch.testing.assert_close(output.cpu()[:rows * columns], (x.cumsum(0) + reduced[:, None]).flatten(),
                               atol=0, rtol=0)
    torch.testing.assert_close(row.cpu()[:rows], reduced, atol=0, rtol=0)
    for index in range(extras):
        torch.testing.assert_close(additional.cpu()[index * rows:(index + 1) * rows],
                                   (x + (index + 1) * 0.125).sum(1), atol=0, rtol=0)
    assert (output.cpu()[rows * columns:] == sentinel).all()
    assert (row.cpu()[rows:] == sentinel).all()
    assert (additional.cpu()[extras * rows:] == sentinel).all()


def test_single_band_scan_shared_input_mixed_row_product_total_budget_is_named():
    result = _compile_rankn_nested_negative_in_child(
        "_single_band_mixed_row_sum_many_results_kernel",
        {"X": "*fp32", "Y": "*fp32", "Row": "*fp32", "Additional": "*fp32"},
        {"EXTRAS": 125, "PRODUCT": True, "REDUCTION": "sum"}, num_warps=4)
    output = result.stdout + result.stderr
    assert result.returncode > 0, output
    assert "single-band mixed row reduction scratch needs 33024 bytes" in output, output
    assert "exceeding the 32768-byte budget" in output, output


@mps_only
@pytest.mark.parametrize("scan_axis,reverse", [(0, False), (1, True)])
def test_single_band_scan_shared_input_mixed_row_product_special_values_match_torch(scan_axis, reverse):
    programs, rows, columns = 2, 16, 8
    x = torch.ones((programs, rows, columns), dtype=torch.float32)
    x[:, 0, 0] = -0.
    x[:, 1, 0] = 0.
    x[:, 2, 0] = float("nan")
    x[:, 3, 0] = float("inf")
    x[:, 4, 0] = -float("inf")
    x[:, 5, 0] = float("inf")
    x[:, 5, 1] = 0.
    x[:, 6, :] = 65536.
    x[:, 7, :] = 1 / 4294967296.
    x[:, 7, 0] *= -1
    reduced = _f32_column_product_reference(x.transpose(1, 2))
    assert torch.isinf(reduced[:, 6]).all()
    assert (reduced[:, 7] == 0).all()
    size = x.numel()
    sentinel = -999999.
    output, scanned_output, output2 = [
        torch.full((size + 7,), sentinel, dtype=torch.float32, device="mps") for _ in range(3)]
    row, row2 = [torch.full((programs * rows + 7,), sentinel, dtype=torch.float32, device="mps")
                 for _ in range(2)]
    values = x.flatten().to("mps")
    _shared_input_scan_mixed_row_sum_kernel[(programs,)](
        values, output, scanned_output, row, row2, output2, size, rows, columns,
        scan_axis, reverse, False, False, 1., False, True, False, False,
        TILE_FIRST=reverse, PRODUCT=True, num_warps=4)
    torch.mps.synchronize()
    axis = scan_axis + 1
    scanned = (x.flip(axis) if reverse else x).cumsum(axis)
    if reverse:
        scanned = scanned.flip(axis)
    broadcast = reduced[:, :, None].expand_as(x)
    actual = output.cpu()[:size].reshape_as(x)
    actual_row = row.cpu()[:programs * rows].reshape_as(reduced)
    torch.testing.assert_close(actual, broadcast, atol=0, rtol=0, equal_nan=True)
    torch.testing.assert_close(actual_row, reduced, atol=0, rtol=0, equal_nan=True)
    torch.testing.assert_close(scanned_output.cpu()[:size].reshape_as(x), scanned,
                               atol=0, rtol=0, equal_nan=True)
    for got, expected in [(actual, broadcast), (actual_row, reduced)]:
        zeros = expected == 0
        assert torch.equal(torch.signbit(got[zeros]), torch.signbit(expected[zeros]))
    for buffer, written in [(output, size), (scanned_output, size), (row, programs * rows),
                            (row2, 0), (output2, 0)]:
        assert (buffer.cpu()[written:] == sentinel).all()


@mps_only
@pytest.mark.parametrize("tile_first", [False, True])
@pytest.mark.parametrize(
    "rows,columns,scan_axis,num_warps,masked,computed,reverse,other,row_masked,full_output,snapshot,multiple", [
        (2, 64, 0, 1, False, False, True, 1., False, False, False, False),
        (4, 32, 1, 1, True, False, False, -1., True, True, False, True),
        (8, 64, 0, 4, True, True, True, -1., False, False, True, True),
        (16, 16, 1, 4, False, True, False, -1., False, True, True, False),
        (32, 32, 0, 4, True, True, True, -2., True, True, False, True),
        (32, 8, 1, 1, True, False, False, 2., False, True, True, False),
        (64, 2, 0, 1, True, True, True, -1.5, False, True, True, True),
        (64, 8, 1, 4, True, False, True, 0., True, False, False, True),
        (8, 32, 0, 4, False, False, False, 1., False, True, True, True),
        (16, 64, 1, 1, True, True, False, -2., False, True, True, False),
    ])
def test_multiband_scan_shared_input_mixed_row_product_matches_torch(
        rows, columns, scan_axis, num_warps, masked, computed, reverse, other,
        row_masked, full_output, snapshot, multiple, tile_first):
    assert rows * columns > 32 * num_warps
    _check_shared_input_mixed_row_sum(
        rows, columns, scan_axis, num_warps, masked, computed, reverse, other,
        row_masked, full_output, snapshot, multiple, tile_first, product=True)


@mps_only
@pytest.mark.parametrize("multiple", [False, True])
def test_multiband_scan_shared_input_mixed_row_product_reuses_publication(multiple):
    rows, columns = 8, 64
    compiled = _check_shared_input_mixed_row_sum(
        rows, columns, 0, 1, True, True, other=-1., full_output=True,
        multiple=multiple, product=True)
    raw = compiled.asm["metal"]
    msl = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    buffers = sorted(int(length) for length in re.findall(
        r"\bthreadgroup\s+float\s+\w+\s*\[\s*(\d+)\s*\]\s*;", msl))
    # One source publication and one vector-result exchange serve both folds.
    # The second masked vector store additionally needs a CTA-sized temporary.
    assert buffers == ([rows, 32, rows * columns] if multiple else [rows, rows * columns]), msl
    assert msl.count("threadgroup_barrier(") == 2, msl


@pytest.mark.parametrize("sources", [2, 3])
def test_multiband_scan_shared_input_mixed_row_product_total_budget_is_named(sources):
    result = _compile_rankn_nested_negative_in_child(
        "_shared_input_mixed_row_sum_budget_kernel",
        {"X": "*fp32", "Y": "*fp32", "Row": "*fp32", "WRow": "*fp32", "VRow": "*fp32"},
        {"SOURCES": sources, "PRODUCT": True, "REDUCTION": "sum"}, num_warps=4)
    output = result.stdout + result.stderr
    assert result.returncode > 0, output
    assert "full-tile exchange scratch needs" in output, output
    assert "exceeding the 32768-byte budget" in output, output


@pytest.mark.parametrize("mode,diagnostic", [
    (0, "a shared scan-input row reduction requires direct row stores or "
     "same-axis broadcasts into its original tile"),
    (1, "reduce cannot replay its computed source cone across conversion"),
    (2, "rank-2 axis=1 f32 product requires a direct unmasked tt.load"),
    (3, "a shared scan-input row reduction requires direct row stores or "
     "same-axis broadcasts into its original tile"),
    (4, "a shared scan-input row reduction requires direct row stores or "
     "same-axis broadcasts into its original tile"),
    (5, "a shared scan-input row reduction requires direct row stores or "
     "same-axis broadcasts into its original tile"),
    (6, "rank-2 axis=1 f32 product requires a direct unmasked tt.load"),
    (7, "a shared scan-input row reduction requires direct row stores or "
     "same-axis broadcasts into its original tile"),
    (8, "a shared scan-input row reduction requires direct row stores or "
     "same-axis broadcasts into its original tile"),
], ids=["distinct_source", "scan_result", "multiple_scans", "cross_axis", "wider_tile", "tile_region", "transpose",
        "vector_region", "late_transpose"])
def test_multiband_scan_shared_input_mixed_row_product_source_and_consumer_is_named(mode, diagnostic):
    result = _compile_rankn_nested_negative_in_child(
        "_unsupported_shared_input_mixed_row_sum_kernel",
        {"X": "*fp32", "W": "*fp32", "Y": "*fp32", "Row": "*fp32", "Scanned": "*fp32", "N": "i32"},
        {"R": 16, "C": 16, "SCAN_AXIS": 1, "MODE": mode, "REDUCTION": "product"}, num_warps=1)
    output = result.stdout + result.stderr
    assert result.returncode > 0, output
    assert diagnostic in output, output


@pytest.mark.parametrize("rows,columns,scan_axis,dtype,diagnostic", [
    (128, 16, 1, "*fp32", "a shared scan-input row reduction requires direct row stores or "
     "same-axis broadcasts into its original tile"),
    (16, 128, 0, "*fp32", "a shared scan-input row reduction requires direct row stores or "
     "same-axis broadcasts into its original tile"),
    (16, 16, 1, "*fp16", "reduce combine requires Session L3c (future)"),
], ids=["wide_rows", "wide_columns", "f16"])
def test_multiband_scan_shared_input_mixed_row_product_shape_and_type_is_named(
        rows, columns, scan_axis, dtype, diagnostic):
    result = _compile_rankn_nested_negative_in_child(
        "_unsupported_shared_input_mixed_row_sum_kernel",
        {"X": dtype, "W": dtype, "Y": dtype, "Row": dtype, "Scanned": dtype, "N": "i32"},
        {"R": rows, "C": columns, "SCAN_AXIS": scan_axis, "MODE": -1, "REDUCTION": "product"}, num_warps=1)
    output = result.stdout + result.stderr
    assert result.returncode > 0, output
    assert diagnostic in output, output


@mps_only
@pytest.mark.parametrize("scan_axis,reverse", [(0, False), (1, True)])
def test_multiband_scan_shared_input_mixed_row_product_special_values_match_torch(scan_axis, reverse):
    programs, rows, columns = 2, 8, 64
    x = torch.ones((programs, rows, columns), dtype=torch.float32)
    x[:, 0, 0] = -0.
    x[:, 1, 0] = 0.
    x[:, 2, 0] = float("nan")
    x[:, 3, 0] = float("inf")
    x[:, 4, 0] = -float("inf")
    x[:, 5, 0] = float("inf")
    x[:, 5, 1] = 0.
    # Normal input factors produce overflow/underflow without relying on
    # preservation of subnormal device values.
    x[:, 6, :] = 8.
    x[:, 7, :] = 0.125
    x[:, 7, 0] = -0.125
    reduced = _f32_column_product_reference(x.transpose(1, 2))
    assert torch.isinf(reduced[:, 6]).all()
    assert (reduced[:, 7] == 0).all()
    size = x.numel()
    sentinel = -999999.
    output, scanned_output, output2 = [
        torch.full((size + 7,), sentinel, dtype=torch.float32, device="mps") for _ in range(3)]
    row, row2 = [torch.full((programs * rows + 7,), sentinel, dtype=torch.float32, device="mps")
                 for _ in range(2)]
    values = x.flatten().to("mps")
    _shared_input_scan_mixed_row_sum_kernel[(programs,)](
        values, output, scanned_output, row, row2, output2, size, rows, columns,
        scan_axis, reverse, False, False, 1., False, True, False, False,
        TILE_FIRST=reverse, SNAPSHOT_AFTER=False, PRODUCT=True, num_warps=4)
    torch.mps.synchronize()
    axis = scan_axis + 1
    scanned = (x.flip(axis) if reverse else x).cumsum(axis)
    if reverse:
        scanned = scanned.flip(axis)
    broadcast = reduced[:, :, None].expand_as(x)
    actual = output.cpu()[:size].reshape_as(x)
    actual_row = row.cpu()[:programs * rows].reshape_as(reduced)
    torch.testing.assert_close(actual, broadcast, atol=0, rtol=0, equal_nan=True)
    torch.testing.assert_close(actual_row, reduced, atol=0, rtol=0, equal_nan=True)
    torch.testing.assert_close(scanned_output.cpu()[:size].reshape_as(x), scanned,
                               atol=0, rtol=0, equal_nan=True)
    for got, expected in [(actual, broadcast), (actual_row, reduced)]:
        zeros = expected == 0
        assert torch.equal(torch.signbit(got[zeros]), torch.signbit(expected[zeros]))
    for buffer, written in [(output, size), (scanned_output, size), (row, programs * rows),
                            (row2, 0), (output2, 0)]:
        assert (buffer.cpu()[written:] == sentinel).all()


@mps_only
@pytest.mark.parametrize("reduction", ["min", "max"])
@pytest.mark.parametrize("rows,columns,scan_axis,num_warps", [
    (2, 64, 0, 4), (4, 8, 1, 1), (16, 8, 1, 4),
])
def test_single_band_scan_shared_input_mixed_row_extremum_baseline_matches_torch(
        reduction, rows, columns, scan_axis, num_warps):
    assert rows * columns == 32 * num_warps
    _check_shared_input_mixed_row_sum(
        rows, columns, scan_axis, num_warps, False, False,
        full_output=True, reduction=reduction, distinct_extremum=True)


@mps_only
@pytest.mark.parametrize("reduction", ["min", "max"])
@pytest.mark.parametrize("tile_first", [False, True], ids=["row_first", "tile_first"])
@pytest.mark.parametrize(
    "rows,columns,scan_axis,num_warps,reverse,row_masked,full_output,multiple,snapshot_after", [
        (2, 16, 0, 1, True, False, False, False, False),
        (4, 8, 1, 1, False, True, True, True, True),
        (8, 4, 0, 1, True, False, False, True, False),
        (16, 2, 1, 1, False, True, True, False, True),
        (2, 64, 0, 4, True, True, True, True, False),
        (4, 32, 1, 4, False, False, False, False, True),
        (8, 16, 0, 4, True, False, True, True, False),
        (16, 8, 1, 4, True, True, True, True, True),
        (32, 4, 0, 4, False, False, False, False, True),
        (64, 2, 1, 4, True, False, True, True, False),
    ])
def test_single_band_scan_shared_input_mixed_row_extremum_matches_torch(
        reduction, rows, columns, scan_axis, num_warps, reverse, row_masked, full_output,
        multiple, snapshot_after, tile_first):
    assert rows * columns == 32 * num_warps
    _check_shared_input_mixed_row_sum(
        rows, columns, scan_axis, num_warps, False, False, reverse=reverse,
        row_masked=row_masked, full_output=full_output, multiple=multiple,
        tile_first=tile_first, snapshot_after=snapshot_after, reduction=reduction,
        distinct_extremum=True)


@mps_only
@pytest.mark.parametrize("reduction", ["min", "max"])
@pytest.mark.parametrize("masked,computed", [(True, False), (False, True), (True, True)])
@pytest.mark.parametrize("rows,columns,scan_axis,num_warps,tile_first", [
    (2, 64, 0, 4, False), (4, 16, 1, 2, True),
])
def test_single_band_scan_shared_input_mixed_row_extremum_replays_sources(
        reduction, masked, computed, rows, columns, scan_axis, num_warps, tile_first):
    assert rows * columns == 32 * num_warps
    _check_shared_input_mixed_row_sum(
        rows, columns, scan_axis, num_warps, masked, computed,
        reverse=tile_first, other=0.75, row_masked=True, full_output=True,
        multiple=True, tile_first=tile_first, snapshot_after=tile_first,
        reduction=reduction, distinct_extremum=True)


@mps_only
@pytest.mark.parametrize("reduction", ["min", "max"])
@pytest.mark.parametrize("rows,columns,scan_axis,num_warps", [(2, 64, 0, 4), (4, 8, 1, 1)])
def test_single_band_scan_shared_input_mixed_row_extremum_separates_result_layouts(
        reduction, rows, columns, scan_axis, num_warps):
    compiled = _check_shared_input_mixed_row_sum(
        rows, columns, scan_axis, num_warps, False, False,
        reduction=reduction, distinct_extremum=True)
    raw = compiled.asm["metal"]
    msl = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    buffers = sorted(int(length) for length in re.findall(
        r"\bthreadgroup\s+float\s+\w+\s*\[\s*(\d+)\s*\]\s*;", msl))
    assert buffers == sorted([rows, rows, 32 * num_warps, 32 * num_warps]), msl
    assert msl.count("threadgroup_barrier(") == 4, msl


@pytest.mark.parametrize("reduction", ["min", "max"])
@pytest.mark.parametrize("masked,computed", [(False, False), (True, False), (False, True), (True, True)])
def test_single_band_scan_shared_input_mixed_row_extremum_snapshot_before_is_named(reduction, masked, computed):
    result = _compile_rankn_nested_negative_in_child(
        "_shared_input_scan_mixed_row_sum_kernel",
        {"X": "*fp32", "Y": "*fp32", "Scanned": "*fp32", "Row": "*fp32",
         "Row2": "*fp32", "Y2": "*fp32", "N": "i32"},
        {"R": 2, "C": 64, "SCAN_AXIS": 0, "REVERSE": False, "MASKED": masked,
         "COMPUTED": computed, "OTHER": 0., "ROW_MASKED": False,
         "FULL_OUTPUT": True, "SNAPSHOT": True, "MULTIPLE": False,
         "TILE_FIRST": False, "SNAPSHOT_AFTER": False, "PRODUCT": False,
         "REDUCTION": reduction, "PURE_BROADCAST": False}, num_warps=4)
    output = result.stdout + result.stderr
    assert result.returncode > 0, output
    assert ("single-band mixed row reduction requires replayable sources defined before the scan "
            "without intervening writes or input exchanges") in output, output


@pytest.mark.parametrize("reduction", ["min", "max"])
@pytest.mark.parametrize("rows,columns,scan_axis,num_warps,mode", [
    (8, 8, 0, 2, 3),
    (2, 128, 0, 8, -1),
    (128, 2, 1, 8, -1),
    (8, 8, 0, 2, 5),
    (8, 8, 0, 2, 7),
    (8, 8, 0, 2, 8),
], ids=["cross_axis", "wide_columns", "wide_rows", "tile_region", "vector_region", "late_transpose"])
def test_single_band_scan_shared_input_mixed_row_extremum_consumer_is_named(
        reduction, rows, columns, scan_axis, num_warps, mode):
    result = _compile_rankn_nested_negative_in_child(
        "_unsupported_single_band_mixed_row_product_kernel",
        {"X": "*fp32", "Y": "*fp32", "Row": "*fp32", "Scanned": "*fp32", "N": "i32"},
        {"R": rows, "C": columns, "SCAN_AXIS": scan_axis, "MODE": mode,
         "REDUCTION": reduction}, num_warps=num_warps)
    output = result.stdout + result.stderr
    assert result.returncode > 0, output
    diagnostic = ("ttg.convert_layout: broader staged-transpose deferred to L1d3" if mode == 3 else
                  "single-band mixed row reduction requires direct row stores and same-phase broadcasts "
                  "into its original 2..64 tile")
    assert diagnostic in output, output


@mps_only
@pytest.mark.parametrize("reduction", ["min", "max"])
def test_single_band_scan_shared_input_mixed_row_extremum_counts_all_result_scratch(reduction):
    rows, columns, extras = 64, 2, 2
    x = (torch.arange(rows * columns).reshape(rows, columns) % 17).float() / 8
    sentinel = -999999.
    output = torch.full((rows * columns + 7,), sentinel, dtype=torch.float32, device="mps")
    row = torch.full((rows + 7,), sentinel, dtype=torch.float32, device="mps")
    additional = torch.full((extras * rows + 7,), sentinel, dtype=torch.float32, device="mps")
    compiled = _single_band_mixed_row_sum_many_results_kernel[(1,)](
        x.to("mps"), output, row, additional, extras, REDUCTION=reduction, num_warps=4)
    raw = compiled.asm["metal"]
    msl = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    buffers = sorted(int(length) for length in re.findall(
        r"\bthreadgroup\s+float\s+\w+\s*\[\s*(\d+)\s*\]\s*;", msl))
    assert buffers == [rows] * (2 + extras) + [rows * columns], msl
    assert msl.count("threadgroup_barrier(") == 4 + extras, msl
    torch.mps.synchronize()
    reduced = _f32_row_extremum_reference(x, reduction)
    torch.testing.assert_close(output.cpu()[:rows * columns], (x.cumsum(0) + reduced[:, None]).flatten(),
                               atol=0, rtol=0)
    torch.testing.assert_close(row.cpu()[:rows], reduced, atol=0, rtol=0)
    for index in range(extras):
        torch.testing.assert_close(additional.cpu()[index * rows:(index + 1) * rows],
                                   (x + (index + 1) * 0.125).sum(1), atol=0, rtol=0)
    assert (output.cpu()[rows * columns:] == sentinel).all()
    assert (row.cpu()[rows:] == sentinel).all()
    assert (additional.cpu()[extras * rows:] == sentinel).all()


@pytest.mark.parametrize("reduction", ["min", "max"])
def test_single_band_scan_shared_input_mixed_row_extremum_total_budget_is_named(reduction):
    result = _compile_rankn_nested_negative_in_child(
        "_single_band_mixed_row_sum_many_results_kernel",
        {"X": "*fp32", "Y": "*fp32", "Row": "*fp32", "Additional": "*fp32"},
        {"EXTRAS": 125, "PRODUCT": False, "REDUCTION": reduction}, num_warps=4)
    output = result.stdout + result.stderr
    assert result.returncode > 0, output
    assert "single-band mixed row reduction scratch needs 33024 bytes" in output, output
    assert "exceeding the 32768-byte budget" in output, output


@mps_only
@pytest.mark.parametrize("reduction", ["min", "max", "minnum", "maxnum"])
@pytest.mark.parametrize("rows,columns,scan_axis,num_warps,masked,computed", [
    (16, 16, 1, 1, True, False),
    (8, 64, 0, 4, True, True),
])
def test_multiband_scan_shared_input_mixed_row_extremum_baseline_matches_torch(
        reduction, rows, columns, scan_axis, num_warps, masked, computed):
    _check_shared_input_mixed_row_sum(
        rows, columns, scan_axis, num_warps, masked, computed,
        full_output=True, reduction=reduction)


@mps_only
@pytest.mark.parametrize("reduction", ["min", "max", "minnum", "maxnum"])
@pytest.mark.parametrize(
    "rows,columns,scan_axis,num_warps,masked,computed,reverse,other,row_masked,full_output,snapshot,multiple,tile_first", [
        (2, 64, 1, 1, False, False, True, 0., False, False, False, False, False),
        (4, 64, 0, 4, True, True, False, -2., True, True, True, True, True),
        (16, 16, 1, 1, True, False, True, -1., False, False, False, True, True),
        (32, 32, 0, 4, False, True, False, 0., False, True, True, False, False),
        (64, 2, 0, 1, True, True, True, -1.5, True, True, True, True, False),
        (64, 8, 1, 4, True, False, False, 64., False, True, False, True, True),
    ])
def test_multiband_scan_shared_input_mixed_row_extremum_matches_torch(
        reduction, rows, columns, scan_axis, num_warps, masked, computed, reverse, other,
        row_masked, full_output, snapshot, multiple, tile_first):
    assert rows * columns > 32 * num_warps
    _check_shared_input_mixed_row_sum(
        rows, columns, scan_axis, num_warps, masked, computed, reverse, other,
        row_masked, full_output, snapshot, multiple, tile_first, reduction=reduction)


@mps_only
@pytest.mark.parametrize("reduction,multiple", [("min", False), ("max", True),
                                              ("minnum", True), ("maxnum", False)])
def test_multiband_scan_shared_input_mixed_row_extremum_reuses_publication(reduction, multiple):
    rows, columns = 8, 64
    compiled = _check_shared_input_mixed_row_sum(
        rows, columns, 0, 1, True, True, full_output=True, multiple=multiple, reduction=reduction)
    raw = compiled.asm["metal"]
    msl = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    buffers = sorted(int(length) for length in re.findall(
        r"\bthreadgroup\s+float\s+\w+\s*\[\s*(\d+)\s*\]\s*;", msl))
    # Both extrema folds share one complete source publication and the one
    # vector exchange; multiple masked stores add only their CTA temporary.
    assert buffers == ([rows, 32, rows * columns] if multiple else [rows, rows * columns]), msl
    assert msl.count("threadgroup_barrier(") == 2, msl


@pytest.mark.parametrize("sources,reduction", [(2, "min"), (3, "maxnum")])
def test_multiband_scan_shared_input_mixed_row_extremum_total_budget_is_named(sources, reduction):
    result = _compile_rankn_nested_negative_in_child(
        "_shared_input_mixed_row_sum_budget_kernel",
        {"X": "*fp32", "Y": "*fp32", "Row": "*fp32", "WRow": "*fp32", "VRow": "*fp32"},
        {"SOURCES": sources, "PRODUCT": False, "REDUCTION": reduction}, num_warps=4)
    output = result.stdout + result.stderr
    assert result.returncode > 0, output
    assert "full-tile exchange scratch needs" in output, output
    assert "exceeding the 32768-byte budget" in output, output


@pytest.mark.parametrize("mode,reduction,diagnostic", [
    (0, "min", "a shared scan-input row reduction requires direct row stores or "
     "same-axis broadcasts into its original tile"),
    (1, "maximum", "reduce cannot replay its computed source cone across conversion"),
    (2, "minnum", "a rank-2 tl.reduce cannot share a function with a multi-band staged scan"),
    (3, "max", "a shared scan-input row reduction requires direct row stores or "
     "same-axis broadcasts into its original tile"),
    (4, "min", "a shared scan-input row reduction requires direct row stores or "
     "same-axis broadcasts into its original tile"),
    (5, "maximum", "a shared scan-input row reduction requires direct row stores or "
     "same-axis broadcasts into its original tile"),
    (6, "minnum", "a rank-2 tl.reduce cannot share a function with a multi-band staged scan"),
    (7, "max", "a shared scan-input row reduction requires direct row stores or "
     "same-axis broadcasts into its original tile"),
    (8, "min", "a shared scan-input row reduction requires direct row stores or "
     "same-axis broadcasts into its original tile"),
], ids=["distinct_source", "scan_result", "multiple_scans", "cross_axis", "wider_tile", "tile_region", "transpose",
        "vector_region", "late_transpose"])
def test_multiband_scan_shared_input_mixed_row_extremum_source_and_consumer_is_named(mode, reduction, diagnostic):
    result = _compile_rankn_nested_negative_in_child(
        "_unsupported_shared_input_mixed_row_sum_kernel",
        {"X": "*fp32", "W": "*fp32", "Y": "*fp32", "Row": "*fp32", "Scanned": "*fp32", "N": "i32"},
        {"R": 16, "C": 16, "SCAN_AXIS": 1, "MODE": mode, "REDUCTION": reduction}, num_warps=1)
    output = result.stdout + result.stderr
    assert result.returncode > 0, output
    assert diagnostic in output, output


@pytest.mark.parametrize("rows,columns,scan_axis,dtype,reduction,diagnostic", [
    (128, 16, 1, "*fp32", "min", "a shared scan-input row reduction requires direct row stores or "
     "same-axis broadcasts into its original tile"),
    (16, 128, 0, "*fp32", "maximum", "a shared scan-input row reduction requires direct row stores or "
     "same-axis broadcasts into its original tile"),
    (16, 16, 1, "*fp16", "max", "a rank-2 tl.reduce cannot share a function with a multi-band staged scan"),
], ids=["wide_rows", "wide_columns", "f16"])
def test_multiband_scan_shared_input_mixed_row_extremum_shape_and_type_is_named(
        rows, columns, scan_axis, dtype, reduction, diagnostic):
    result = _compile_rankn_nested_negative_in_child(
        "_unsupported_shared_input_mixed_row_sum_kernel",
        {"X": dtype, "W": dtype, "Y": dtype, "Row": dtype, "Scanned": dtype, "N": "i32"},
        {"R": rows, "C": columns, "SCAN_AXIS": scan_axis, "MODE": -1, "REDUCTION": reduction}, num_warps=1)
    output = result.stdout + result.stderr
    assert result.returncode > 0, output
    assert diagnostic in output, output


def _mixed_row_extremum_special_values(programs, rows, columns):
    x = (torch.arange(programs * rows * columns).reshape(programs, rows, columns) % 17).float() / 8 - 1
    for row, column in [(0, 0), (1, columns // 2), (2, columns - 1)]:
        x[:, row, column] = float("nan")
    x[:, 3, :] = float("nan")
    x[:, 4, :] = 0.
    x[:, 5, :] = -0.
    x[:, 6, ::2], x[:, 6, 1::2] = 0., -0.
    x[:, 7, ::2], x[:, 7, 1::2] = -0., 0.
    x[:, 8, 0] = float("inf")
    x[:, 9, -1] = -float("inf")
    x[:, 10, :] = float("inf")
    x[:, 11, :] = -float("inf")
    x[:, 12, 0], x[:, 12, -1] = float("inf"), -float("inf")
    return x


def _check_mixed_row_extremum_special_values(reduction, columns, scan_axis, num_warps, reverse):
    programs, rows = 2, 16
    x = _mixed_row_extremum_special_values(programs, rows, columns)
    reduced = _f32_row_extremum_reference(x, reduction)
    size = x.numel()
    sentinel = -999999.
    output, scanned_output, output2 = [
        torch.full((size + 7,), sentinel, dtype=torch.float32, device="mps") for _ in range(3)]
    row, row2 = [torch.full((programs * rows + 7,), sentinel, dtype=torch.float32, device="mps")
                 for _ in range(2)]
    _shared_input_scan_mixed_row_sum_kernel[(programs,)](
        x.flatten().to("mps"), output, scanned_output, row, row2, output2,
        size, rows, columns, scan_axis, reverse, False, False, 0., False, True, False, False,
        TILE_FIRST=reverse, SNAPSHOT_AFTER=False, PRODUCT=False, REDUCTION=reduction, num_warps=num_warps)
    torch.mps.synchronize()
    axis = scan_axis + 1
    scanned = (x.flip(axis) if reverse else x).cumsum(axis)
    if reverse:
        scanned = scanned.flip(axis)
    broadcast = reduced[:, :, None].expand_as(x)
    actual = output.cpu()[:size].reshape_as(x)
    actual_row = row.cpu()[:programs * rows].reshape_as(reduced)
    torch.testing.assert_close(actual, broadcast, atol=0, rtol=0, equal_nan=True)
    torch.testing.assert_close(actual_row, reduced, atol=0, rtol=0, equal_nan=True)
    torch.testing.assert_close(scanned_output.cpu()[:size].reshape_as(x), scanned,
                               atol=0, rtol=0, equal_nan=True)
    # Both folds must agree even where numeric min/max permits either zero sign.
    actual_broadcast_row = actual[:, :, 0]
    torch.testing.assert_close(actual_broadcast_row, actual_row, atol=0, rtol=0, equal_nan=True)
    zeros = actual_row == 0
    assert torch.equal(actual_broadcast_row.signbit()[zeros], actual_row.signbit()[zeros])
    check_zero_rows = slice(4, 8) if reduction in ("min", "max") else slice(4, 6)
    for got, expected in [(actual[:, check_zero_rows], broadcast[:, check_zero_rows]),
                          (actual_row[:, check_zero_rows], reduced[:, check_zero_rows])]:
        assert torch.equal(got.signbit(), expected.signbit())
    for buffer, written in [(output, size), (scanned_output, size), (row, programs * rows),
                            (row2, 0), (output2, 0)]:
        assert (buffer.cpu()[written:] == sentinel).all()



@mps_only
@pytest.mark.parametrize("reduction", ["min", "max"])
@pytest.mark.parametrize("columns,scan_axis,num_warps,reverse", [(2, 1, 1, False), (8, 0, 4, True)])
def test_single_band_scan_shared_input_mixed_row_extremum_special_values_match_torch(
        reduction, columns, scan_axis, num_warps, reverse):
    assert 16 * columns == 32 * num_warps
    _check_mixed_row_extremum_special_values(reduction, columns, scan_axis, num_warps, reverse)


@mps_only
@pytest.mark.parametrize("reduction", ["min", "max", "minnum", "maxnum"])
@pytest.mark.parametrize("columns,scan_axis,num_warps,reverse", [(16, 1, 1, False), (32, 0, 4, True)])
def test_multiband_scan_shared_input_mixed_row_extremum_special_values_match_torch(
        reduction, columns, scan_axis, num_warps, reverse):
    _check_mixed_row_extremum_special_values(reduction, columns, scan_axis, num_warps, reverse)

def _mixed_i32_row_product_values(size, columns, computed):
    index = torch.arange(size, dtype=torch.int64)
    row, column = index // columns, index % columns
    desired = (row * 2 + 1) * torch.where(row % 2 == 0, -1, 1)
    # Odd factors have inverses modulo 2**32. Each pair wraps to one, while
    # the first factor gives a distinct nonzero signed product to every row.
    factor = torch.where(row % 2 == 0, 3, 5)
    inverse = torch.where(row % 2 == 0, -1431655765, -858993459)
    target = torch.where(column % 2 == 0, factor, inverse)
    target = torch.where(column == 0, desired, torch.where(column == 1, 1, target))
    raw = (target - 3) // 2 if computed else target
    return raw.to(torch.int32)


def _mixed_i32_row_bitwise_values(size, columns, computed, reduction):
    index = torch.arange(size, dtype=torch.int64)
    row, column = index // columns, index % columns
    # Every source is odd, so x * 2 + 3 can reproduce the same rich bit
    # patterns. Distinct signed row results catch ui32 publication mistakes.
    sign = (row % 2) << 31
    row_bits = ((row + 1) << 8) | sign
    if reduction == "and":
        target = row_bits | 1 | (1 << (column % 7 + 1))
    elif reduction == "or":
        # Split row bits between even and odd columns, so neither one input
        # nor a saturated common mask can stand in for the complete OR fold.
        distributed = torch.where(column % 2 == 0, row_bits & 0x55555555,
                                   row_bits & 0xaaaaaaaa)
        target = distributed | 1 | (1 << (column % 7 + 1))
    else:
        # Repeated pairs cancel, except for one row-specific even residual.
        target = 0x13579bdf ^ ((column // 2) * 0x01020408)
        target ^= torch.where(column == 0, row_bits | 2, 0)
    raw = (target - 3) // 2 if computed else target
    return raw.to(torch.int32)


def _mixed_i32_row_extrema_values(size, columns, computed, reduction):
    index = torch.arange(size, dtype=torch.int64)
    row, column = index // columns, index % columns
    # Put a distinct winner in the final column. The opposite-sign background
    # distinguishes signed and unsigned ordering, including two-column tiles.
    if reduction == "smin":
        winner, background = -2147483647 + row * 4, 1073741825 + row * 4
    elif reduction == "smax":
        winner, background = 2147483647 - row * 4, -1073741823 + row * 4
    elif reduction == "umin":
        winner, background = 5 + row * 4, -1073741823 + row * 4
    else:
        winner, background = -1 - row * 4, 1073741825 + row * 4
    target = torch.where(column == columns - 1, winner, background)
    # Odd targets have exact preimages through the wrapping computed source.
    raw = (target - 3) // 2 if computed else target
    return raw.to(torch.int32)


def _mixed_i32_row_extrema_reference(x, reduction):
    ordered = x.to(torch.int64) & 0xffffffff if reduction.startswith("u") else x
    result = ordered.amin(2) if reduction.endswith("min") else ordered.amax(2)
    return result.to(torch.int32)


def _check_shared_input_mixed_i32_row_sum(
        rows, columns, scan_axis, num_warps, masked, computed, reverse=False,
        other=1073741827, row_masked=False, full_output=True, snapshot=False,
        multiple=False, tile_first=False, product=False, reduction="sum"):
    programs = 2
    size = programs * rows * columns
    valid = size - columns - 3 if masked else size
    index = torch.arange(valid, dtype=torch.int64)
    seeds = torch.tensor([-2147483645, 2147483642, 1073741831, -1073741813, -17, 29, 1, -3],
                         dtype=torch.int64)
    row_bias = torch.where(index // columns % 2 == 0, -536870912, 536870912)
    row_bias = torch.where(index % columns == 0, row_bias, 0)
    extrema = reduction in ("smin", "smax", "umin", "umax")
    original = (_mixed_i32_row_extrema_values(valid, columns, computed, reduction) if extrema else
                _mixed_i32_row_bitwise_values(valid, columns, computed, reduction)
                if reduction != "sum" else
                _mixed_i32_row_product_values(valid, columns, computed) if product else
                (seeds[index % 8] + (index // columns % rows) * 97 +
                 (index % columns) * 13 + (index // (rows * columns)) * 31 + row_bias).to(torch.int32))
    x = torch.full((size,), other, dtype=torch.int32)
    x[:valid] = original
    x = x.reshape(programs, rows, columns)
    if computed:
        x = (x.to(torch.int64) * 2 + 3).to(torch.int32)
    # Every source, scan and output-arithmetic stage wraps independently.
    axis = scan_axis + 1
    scanned = (x.flip(axis) if reverse else x).to(torch.int64).cumsum(axis).to(torch.int32)
    if reverse:
        scanned = scanned.flip(axis)
    reduced = (_mixed_i32_row_extrema_reference(x, reduction) if extrema else
               _i32_row_monoid_reference(x, reduction) if reduction != "sum" else
               _i32_row_monoid_reference(x, "product") if product else
               x.to(torch.int64).sum(2).to(torch.int32))
    if product or reduction != "sum":
        assert (reduced[0] != 0).all()
    assert reduced[0].unique().numel() == rows
    if extrema:
        opposite = "u" + reduction[1:] if reduction.startswith("s") else "s" + reduction[1:]
        assert (reduced[0] != _mixed_i32_row_extrema_reference(x, opposite)[0]).all()
    else:
        assert (reduced[0] < 0).any() and (reduced[0] > 0).any()
    contribution = reduced[:, :, None].expand_as(x)
    doubled = (contribution.to(torch.int64) * 2).to(torch.int32)
    difference = (scanned.to(torch.int64) - doubled.to(torch.int64)).to(torch.int32)
    second = torch.where(contribution < 0, difference.to(torch.int64),
                         -difference.to(torch.int64)).to(torch.int32)
    expected_input = ((x.flatten()[:valid].to(torch.int64) * 3 + 17).to(torch.int32)
                      if snapshot else original)
    sentinel = -123456789
    values = original.to("mps")
    output, scan_output, output2 = [
        torch.full((size + 7,), sentinel, dtype=torch.int32, device="mps") for _ in range(3)]
    row, row2 = [torch.full((programs * rows + 7,), sentinel, dtype=torch.int32, device="mps")
                 for _ in range(2)]
    compiled = _shared_input_scan_mixed_row_sum_kernel[(programs,)](
        values, output, scan_output, row, row2, output2, valid, rows, columns,
        scan_axis, reverse, masked, computed, other, row_masked, full_output, snapshot, multiple,
        TILE_FIRST=tile_first, SNAPSHOT_AFTER=False, PRODUCT=product, REDUCTION=reduction,
        PURE_BROADCAST=True, num_warps=num_warps)
    torch.mps.synchronize()
    written = size if full_output else valid
    for actual, expected in [(output, contribution), (scan_output, scanned)]:
        torch.testing.assert_close(actual.cpu()[:written], expected.flatten()[:written], atol=0, rtol=0)
        assert (actual.cpu()[written:] == sentinel).all()
    valid_rows = rows - 1 if row_masked else rows
    actual_row = row.cpu()[:programs * rows].reshape(programs, rows)
    torch.testing.assert_close(actual_row[:, :valid_rows], reduced[:, :valid_rows], atol=0, rtol=0)
    assert (actual_row[:, valid_rows:] == sentinel).all()
    assert (row.cpu()[programs * rows:] == sentinel).all()
    if multiple:
        actual_row2 = row2.cpu()[:programs * rows].reshape(programs, rows)
        torch.testing.assert_close(actual_row2[:, 1:], reduced[:, 1:], atol=0, rtol=0)
        assert (actual_row2[:, 0] == sentinel).all()
        assert (row2.cpu()[programs * rows:] == sentinel).all()
        torch.testing.assert_close(output2.cpu()[:written], second.flatten()[:written], atol=0, rtol=0)
        assert (output2.cpu()[written:] == sentinel).all()
    else:
        assert (row2.cpu() == sentinel).all()
        assert (output2.cpu() == sentinel).all()
    torch.testing.assert_close(values.cpu(), expected_input, atol=0, rtol=0)
    return compiled


@mps_only
@pytest.mark.parametrize("rows,columns,scan_axis,num_warps,masked,computed", [
    (16, 16, 0, 1, False, False),
    (16, 16, 1, 1, True, False),
    (8, 64, 0, 4, True, True),
    (8, 64, 1, 4, False, True),
    (32, 32, 0, 4, True, False),
    (32, 32, 1, 1, True, True),
])
def test_multiband_scan_shared_input_mixed_i32_row_sum_baseline_matches_torch(
        rows, columns, scan_axis, num_warps, masked, computed):
    _check_shared_input_mixed_i32_row_sum(rows, columns, scan_axis, num_warps, masked, computed)


@mps_only
@pytest.mark.parametrize("tile_first", [False, True])
@pytest.mark.parametrize(
    "rows,columns,scan_axis,num_warps,masked,computed,reverse,other,row_masked,full_output,snapshot,multiple", [
        (2, 64, 0, 1, False, False, True, 3, False, False, False, True),
        (4, 64, 1, 4, True, True, False, -1073741819, True, True, True, True),
        (16, 16, 1, 1, True, False, True, 1073741827, False, False, False, True),
        (32, 32, 0, 4, False, True, False, -3, False, True, True, False),
        (64, 2, 0, 1, True, True, True, 2147483647, True, True, True, True),
        (64, 8, 1, 4, True, False, False, -2147483647, False, True, False, True),
        (8, 32, 0, 4, True, True, True, 0, False, False, True, False),
        (16, 64, 1, 1, True, False, False, 17, False, True, True, True),
    ])
def test_multiband_scan_shared_input_mixed_i32_row_sum_matches_torch(
        rows, columns, scan_axis, num_warps, masked, computed, reverse, other,
        row_masked, full_output, snapshot, multiple, tile_first):
    assert rows * columns > 32 * num_warps
    _check_shared_input_mixed_i32_row_sum(
        rows, columns, scan_axis, num_warps, masked, computed, reverse, other,
        row_masked, full_output, snapshot, multiple, tile_first)


@mps_only
@pytest.mark.parametrize("multiple", [False, True])
def test_multiband_scan_shared_input_mixed_i32_row_sum_reuses_publication(multiple):
    rows, columns = 8, 64
    compiled = _check_shared_input_mixed_i32_row_sum(
        rows, columns, 0, 1, True, True, full_output=True, multiple=multiple)
    raw = compiled.asm["metal"]
    msl = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    buffers = sorted(int(length) for length in re.findall(
        r"\bthreadgroup\s+(?:u?int32_t|u?int)\s+\w+\s*\[\s*(\d+)\s*\]\s*;", msl))
    # Two sum folds reuse one ui32 source and one vector exchange; the second
    # masked row store may allocate its separate CTA-sized store temporary.
    assert buffers == ([rows, 32, rows * columns] if multiple else [rows, rows * columns]), msl
    assert msl.count("threadgroup_barrier(") == 2, msl


@pytest.mark.parametrize("sources", [2, 3])
def test_multiband_scan_shared_input_mixed_i32_row_sum_total_budget_is_named(sources):
    result = _compile_rankn_nested_negative_in_child(
        "_shared_input_mixed_row_sum_budget_kernel",
        {"X": "*i32", "Y": "*i32", "Row": "*i32", "WRow": "*i32", "VRow": "*i32"},
        {"SOURCES": sources, "PRODUCT": False, "REDUCTION": "sum"}, num_warps=4)
    output = result.stdout + result.stderr
    assert result.returncode > 0, output
    assert "full-tile exchange scratch needs" in output, output
    assert "exceeding the 32768-byte budget" in output, output


@pytest.mark.parametrize("mode,diagnostic", [
    (0, "a shared scan-input row reduction requires direct row stores or "
     "same-axis broadcasts into its original tile"),
    (1, "distinct scan/reduce inputs require same-layout sources defined before the scan "
     "without intervening writes or input exchanges"),
    (2, "rank-2 axis=1 reduce over a computed cone is implemented for f32 only; "
     "other element types require a direct unmasked tt.load"),
    (3, "a shared scan-input row reduction requires direct row stores or "
     "same-axis broadcasts into its original tile"),
    (5, "a shared scan-input row reduction requires direct row stores or "
     "same-axis broadcasts into its original tile"),
    (6, "rank-2 axis=1 reduce over a computed cone is implemented for f32 only; "
     "other element types require a direct unmasked tt.load"),
    (7, "a shared scan-input row reduction requires direct row stores or "
     "same-axis broadcasts into its original tile"),
], ids=["distinct_source", "scan_result", "multiple_scans", "cross_axis", "tile_region", "transpose", "vector_region"])
def test_multiband_scan_shared_input_mixed_i32_row_sum_source_and_consumer_is_named(mode, diagnostic):
    result = _compile_rankn_nested_negative_in_child(
        "_unsupported_shared_input_mixed_row_sum_kernel",
        {"X": "*i32", "W": "*i32", "Y": "*i32", "Row": "*i32", "Scanned": "*i32", "N": "i32"},
        {"R": 16, "C": 16, "SCAN_AXIS": 1, "MODE": mode, "REDUCTION": "sum"}, num_warps=1)
    output = result.stdout + result.stderr
    assert result.returncode > 0, output
    assert diagnostic in output, output


@pytest.mark.parametrize("rows,columns,scan_axis,dtype,reduction,diagnostic", [
    (128, 16, 1, "*i32", "sum", "a shared scan-input row reduction requires direct row stores or "
     "same-axis broadcasts into its original tile"),
    (16, 128, 0, "*i32", "sum", "a shared scan-input row reduction requires direct row stores or "
     "same-axis broadcasts into its original tile"),
    (16, 16, 1, "*i64", "sum", "rank >= 2 scan is implemented for a blocked tile"),
    (16, 16, 1, "*fp16", "sum", "rank-2 axis=1 reduce over a computed cone is implemented for f32 only; "
     "other element types require a direct unmasked tt.load"),
], ids=["wide_rows", "wide_columns", "i64", "f16"])
def test_multiband_scan_shared_input_mixed_i32_row_sum_shape_type_and_combine_is_named(
        rows, columns, scan_axis, dtype, reduction, diagnostic):
    result = _compile_rankn_nested_negative_in_child(
        "_unsupported_shared_input_mixed_row_sum_kernel",
        {"X": dtype, "W": dtype, "Y": dtype, "Row": dtype, "Scanned": dtype, "N": "i32"},
        {"R": rows, "C": columns, "SCAN_AXIS": scan_axis, "MODE": -1, "REDUCTION": reduction}, num_warps=1)
    output = result.stdout + result.stderr
    assert result.returncode > 0, output
    assert diagnostic in output, output


@mps_only
@pytest.mark.parametrize("rows,columns,scan_axis,num_warps,masked,computed", [
    (16, 16, 0, 1, False, False),
    (16, 16, 1, 1, True, False),
    (8, 64, 0, 4, True, True),
    (8, 64, 1, 4, False, True),
    (32, 32, 0, 4, True, False),
    (32, 32, 1, 1, True, True),
])
def test_multiband_scan_shared_input_mixed_i32_row_product_baseline_matches_torch(
        rows, columns, scan_axis, num_warps, masked, computed):
    _check_shared_input_mixed_i32_row_sum(
        rows, columns, scan_axis, num_warps, masked, computed, other=3, product=True)


@mps_only
@pytest.mark.parametrize("tile_first", [False, True])
@pytest.mark.parametrize(
    "rows,columns,scan_axis,num_warps,masked,computed,reverse,other,row_masked,full_output,snapshot,multiple", [
        (2, 64, 0, 1, False, False, True, 1, False, False, False, True),
        (4, 64, 1, 4, True, True, False, -1, True, True, True, True),
        (16, 16, 1, 1, True, False, True, -3, False, False, False, True),
        (32, 32, 0, 4, False, True, False, 0, False, True, True, False),
        (64, 2, 0, 1, True, True, True, 2, True, True, True, True),
        (64, 8, 1, 4, True, False, False, 2147483647, False, True, False, True),
        (8, 32, 0, 4, True, True, True, 0, False, False, True, False),
        (16, 64, 1, 1, True, False, False, 1, False, True, True, True),
    ])
def test_multiband_scan_shared_input_mixed_i32_row_product_matches_torch(
        rows, columns, scan_axis, num_warps, masked, computed, reverse, other,
        row_masked, full_output, snapshot, multiple, tile_first):
    assert rows * columns > 32 * num_warps
    _check_shared_input_mixed_i32_row_sum(
        rows, columns, scan_axis, num_warps, masked, computed, reverse, other,
        row_masked, full_output, snapshot, multiple, tile_first, product=True)


@mps_only
@pytest.mark.parametrize("multiple", [False, True])
def test_multiband_scan_shared_input_mixed_i32_row_product_reuses_publication(multiple):
    rows, columns = 8, 64
    compiled = _check_shared_input_mixed_i32_row_sum(
        rows, columns, 0, 1, True, True, other=-1, full_output=True, multiple=multiple, product=True)
    raw = compiled.asm["metal"]
    msl = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    buffers = sorted(int(length) for length in re.findall(
        r"\bthreadgroup\s+(?:u?int32_t|u?int)\s+\w+\s*\[\s*(\d+)\s*\]\s*;", msl))
    # Both modular products consume one publication and share the vector
    # exchange. The second masked row store adds only store scratch.
    assert buffers == ([rows, 32, rows * columns] if multiple else [rows, rows * columns]), msl
    assert msl.count("threadgroup_barrier(") == 2, msl


@pytest.mark.parametrize("sources", [2, 3])
def test_multiband_scan_shared_input_mixed_i32_row_product_total_budget_is_named(sources):
    result = _compile_rankn_nested_negative_in_child(
        "_shared_input_mixed_row_sum_budget_kernel",
        {"X": "*i32", "Y": "*i32", "Row": "*i32", "WRow": "*i32", "VRow": "*i32"},
        {"SOURCES": sources, "PRODUCT": True, "REDUCTION": "sum"}, num_warps=4)
    output = result.stdout + result.stderr
    assert result.returncode > 0, output
    assert "full-tile exchange scratch needs" in output, output
    assert "exceeding the 32768-byte budget" in output, output


@pytest.mark.parametrize("mode,diagnostic", [
    (0, "a shared scan-input row reduction requires direct row stores or "
     "same-axis broadcasts into its original tile"),
    (1, "distinct scan/reduce inputs require same-layout sources defined before the scan "
     "without intervening writes or input exchanges"),
    (2, "rank-2 axis=1 reduce over a computed cone is implemented for f32 only; "
     "other element types require a direct unmasked tt.load"),
    (3, "a shared scan-input row reduction requires direct row stores or "
     "same-axis broadcasts into its original tile"),
    (5, "a shared scan-input row reduction requires direct row stores or "
     "same-axis broadcasts into its original tile"),
    (6, "rank-2 axis=1 reduce over a computed cone is implemented for f32 only; "
     "other element types require a direct unmasked tt.load"),
    (7, "a shared scan-input row reduction requires direct row stores or "
     "same-axis broadcasts into its original tile"),
], ids=["distinct_source", "scan_result", "multiple_scans", "cross_axis", "tile_region", "transpose", "vector_region"])
def test_multiband_scan_shared_input_mixed_i32_row_product_source_and_consumer_is_named(mode, diagnostic):
    result = _compile_rankn_nested_negative_in_child(
        "_unsupported_shared_input_mixed_row_sum_kernel",
        {"X": "*i32", "W": "*i32", "Y": "*i32", "Row": "*i32", "Scanned": "*i32", "N": "i32"},
        {"R": 16, "C": 16, "SCAN_AXIS": 1, "MODE": mode, "REDUCTION": "product"}, num_warps=1)
    output = result.stdout + result.stderr
    assert result.returncode > 0, output
    assert diagnostic in output, output



@pytest.mark.parametrize("rows,columns,scan_axis,dtype,diagnostic", [
    (128, 16, 1, "*i32", "a shared scan-input row reduction requires direct row stores or "
     "same-axis broadcasts into its original tile"),
    (16, 128, 0, "*i32", "a shared scan-input row reduction requires direct row stores or "
     "same-axis broadcasts into its original tile"),
    (16, 16, 1, "*i64", "rank >= 2 scan is implemented for a blocked tile"),
    (16, 16, 1, "*fp16", "reduce combine requires Session L3c (future)"),
], ids=["wide_rows", "wide_columns", "i64", "f16"])
def test_multiband_scan_shared_input_mixed_i32_row_product_shape_and_type_is_named(
        rows, columns, scan_axis, dtype, diagnostic):
    result = _compile_rankn_nested_negative_in_child(
        "_unsupported_shared_input_mixed_row_sum_kernel",
        {"X": dtype, "W": dtype, "Y": dtype, "Row": dtype, "Scanned": dtype, "N": "i32"},
        {"R": rows, "C": columns, "SCAN_AXIS": scan_axis, "MODE": -1, "REDUCTION": "product"}, num_warps=1)
    output = result.stdout + result.stderr
    assert result.returncode > 0, output
    assert diagnostic in output, output


@mps_only
@pytest.mark.parametrize("scan_axis,reverse", [(0, False), (1, True)])
def test_multiband_scan_shared_input_mixed_i32_row_product_special_values_match_torch(scan_axis, reverse):
    programs, rows, columns = 2, 8, 16
    size = programs * rows * columns
    valid = size - columns - 3
    x = torch.ones((programs, rows, columns), dtype=torch.int32)
    x[:, 0, :] = 2
    x[:, 1, columns // 2] = 0
    x[:, 2, 0], x[:, 2, 1] = -2147483648, -1
    x[:, 3, 0] = -1
    x[:, 4, 0], x[:, 4, 1] = 2147483647, 3
    x[:, 5, :] = 65536
    x[:, 6, 0], x[:, 6, 1] = 2, 3
    x[:, 7, 0], x[:, 7, 1] = -3, 5
    original = x.flatten()[:valid].clone()
    x.flatten()[valid:] = 3
    reduced = _i32_row_monoid_reference(x, "product")
    assert (reduced == 0).any() and (reduced != 0).any()
    assert reduced[0, 2] == -2147483648
    # The entirely masked final row retains the nonidentity fill's product.
    assert reduced[-1, -1] == ((3 ** columns + 2 ** 31) % 2 ** 32 - 2 ** 31)
    axis = scan_axis + 1
    scanned = (x.flip(axis) if reverse else x).to(torch.int64).cumsum(axis).to(torch.int32)
    if reverse:
        scanned = scanned.flip(axis)
    sentinel = -123456789
    output, scan_output, output2 = [
        torch.full((size + 7,), sentinel, dtype=torch.int32, device="mps") for _ in range(3)]
    row, row2 = [torch.full((programs * rows + 7,), sentinel, dtype=torch.int32, device="mps")
                 for _ in range(2)]
    _shared_input_scan_mixed_row_sum_kernel[(programs,)](
        original.to("mps"), output, scan_output, row, row2, output2, valid, rows, columns,
        scan_axis, reverse, True, False, 3, False, True, False, False,
        TILE_FIRST=reverse, SNAPSHOT_AFTER=False, PRODUCT=True, REDUCTION="sum",
        PURE_BROADCAST=True, num_warps=1)
    torch.mps.synchronize()
    expected = reduced[:, :, None].expand_as(x)
    torch.testing.assert_close(output.cpu()[:size].reshape_as(x), expected, atol=0, rtol=0)
    torch.testing.assert_close(row.cpu()[:programs * rows].reshape_as(reduced), reduced, atol=0, rtol=0)
    torch.testing.assert_close(scan_output.cpu()[:size].reshape_as(x), scanned, atol=0, rtol=0)
    for buffer, written in [(output, size), (scan_output, size), (row, programs * rows),
                            (row2, 0), (output2, 0)]:
        assert (buffer.cpu()[written:] == sentinel).all()


@mps_only
@pytest.mark.parametrize("reduction", ["and", "or", "xor"])
@pytest.mark.parametrize("rows,columns,scan_axis,num_warps,masked,computed", [
    (16, 16, 0, 1, False, False),
    (16, 16, 1, 1, True, False),
    (8, 64, 0, 4, True, True),
    (8, 64, 1, 4, False, True),
    (32, 32, 0, 4, True, False),
    (32, 32, 1, 1, True, True),
])
def test_multiband_scan_shared_input_mixed_i32_row_bitwise_baseline_matches_torch(
        rows, columns, scan_axis, num_warps, masked, computed, reduction):
    _check_shared_input_mixed_i32_row_sum(
        rows, columns, scan_axis, num_warps, masked, computed, other=-1073741819, reduction=reduction)


@mps_only
@pytest.mark.parametrize("reduction", ["and", "or", "xor"])
@pytest.mark.parametrize("tile_first", [False, True])
@pytest.mark.parametrize(
    "rows,columns,scan_axis,num_warps,masked,computed,reverse,other,row_masked,full_output,snapshot,multiple", [
        (2, 64, 0, 1, False, False, True, 1, False, False, False, True),
        (4, 64, 1, 4, True, True, False, -1, True, True, True, True),
        (16, 16, 1, 1, True, False, True, -3, False, False, False, True),
        (32, 32, 0, 4, False, True, False, 0, False, True, True, False),
        (64, 2, 0, 1, True, True, True, 2, True, True, True, True),
        (64, 8, 1, 4, True, False, False, 2147483647, False, True, False, True),
        (8, 32, 0, 4, True, True, True, 0, False, False, True, False),
        (16, 64, 1, 1, True, False, False, 1, False, True, True, True),
    ])
def test_multiband_scan_shared_input_mixed_i32_row_bitwise_matches_torch(
        rows, columns, scan_axis, num_warps, masked, computed, reverse, other,
        row_masked, full_output, snapshot, multiple, tile_first, reduction):
    assert rows * columns > 32 * num_warps
    _check_shared_input_mixed_i32_row_sum(
        rows, columns, scan_axis, num_warps, masked, computed, reverse, other,
        row_masked, full_output, snapshot, multiple, tile_first, reduction=reduction)


@mps_only
@pytest.mark.parametrize("reduction", ["and", "or", "xor"])
@pytest.mark.parametrize("multiple", [False, True])
def test_multiband_scan_shared_input_mixed_i32_row_bitwise_reuses_publication(multiple, reduction):
    rows, columns = 8, 64
    compiled = _check_shared_input_mixed_i32_row_sum(
        rows, columns, 0, 1, True, True, other=-1, full_output=True, multiple=multiple, reduction=reduction)
    raw = compiled.asm["metal"]
    msl = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    buffers = sorted(int(length) for length in re.findall(
        r"\bthreadgroup\s+(?:u?int32_t|u?int)\s+\w+\s*\[\s*(\d+)\s*\]\s*;", msl))
    # Both bitwise folds consume one publication and share the vector
    # exchange. The second masked row store adds only store scratch.
    assert buffers == ([rows, 32, rows * columns] if multiple else [rows, rows * columns]), msl
    assert msl.count("threadgroup_barrier(") == 2, msl

@pytest.mark.parametrize("reduction", ["and", "or", "xor"])
@pytest.mark.parametrize("sources", [2, 3])
def test_multiband_scan_shared_input_mixed_i32_row_bitwise_total_budget_is_named(sources, reduction):
    result = _compile_rankn_nested_negative_in_child(
        "_shared_input_mixed_row_sum_budget_kernel",
        {"X": "*i32", "Y": "*i32", "Row": "*i32", "WRow": "*i32", "VRow": "*i32"},
        {"SOURCES": sources, "PRODUCT": False, "REDUCTION": reduction}, num_warps=4)
    output = result.stdout + result.stderr
    assert result.returncode > 0, output
    assert "full-tile exchange scratch needs" in output, output
    assert "exceeding the 32768-byte budget" in output, output

@pytest.mark.parametrize("reduction", ["and", "or", "xor"])
@pytest.mark.parametrize("mode,diagnostic", [
    (0, "a shared scan-input row reduction requires direct row stores or "
     "same-axis broadcasts into its original tile"),
    (1, "distinct scan/reduce inputs require same-layout sources defined before the scan "
     "without intervening writes or input exchanges"),
    (2, "rank-2 axis=1 reduce over a computed cone is implemented for f32 only; "
     "other element types require a direct unmasked tt.load"),
    (3, "a shared scan-input row reduction requires direct row stores or "
     "same-axis broadcasts into its original tile"),
    (5, "a shared scan-input row reduction requires direct row stores or "
     "same-axis broadcasts into its original tile"),
    (6, "rank-2 axis=1 reduce over a computed cone is implemented for f32 only; "
     "other element types require a direct unmasked tt.load"),
    (7, "a shared scan-input row reduction requires direct row stores or "
     "same-axis broadcasts into its original tile"),
], ids=["distinct_source", "scan_result", "multiple_scans", "cross_axis", "tile_region", "transpose", "vector_region"])
def test_multiband_scan_shared_input_mixed_i32_row_bitwise_source_and_consumer_is_named(mode, diagnostic, reduction):
    result = _compile_rankn_nested_negative_in_child(
        "_unsupported_shared_input_mixed_row_sum_kernel",
        {"X": "*i32", "W": "*i32", "Y": "*i32", "Row": "*i32", "Scanned": "*i32", "N": "i32"},
        {"R": 16, "C": 16, "SCAN_AXIS": 1, "MODE": mode, "REDUCTION": reduction}, num_warps=1)
    output = result.stdout + result.stderr
    assert result.returncode > 0, output
    assert diagnostic in output, output


@pytest.mark.parametrize("reduction", ["and", "or", "xor"])
@pytest.mark.parametrize("rows,columns,scan_axis,dtype,diagnostic", [
    (128, 16, 1, "*i32", "a shared scan-input row reduction requires direct row stores or "
     "same-axis broadcasts into its original tile"),
    (16, 128, 0, "*i32", "a shared scan-input row reduction requires direct row stores or "
     "same-axis broadcasts into its original tile"),
    (16, 16, 1, "*i64", "rank >= 2 scan is implemented for a blocked tile"),
], ids=["wide_rows", "wide_columns", "i64"])
def test_multiband_scan_shared_input_mixed_i32_row_bitwise_shape_and_type_is_named(
        rows, columns, scan_axis, dtype, diagnostic, reduction):
    result = _compile_rankn_nested_negative_in_child(
        "_unsupported_shared_input_mixed_row_sum_kernel",
        {"X": dtype, "W": dtype, "Y": dtype, "Row": dtype, "Scanned": dtype, "N": "i32"},
        {"R": rows, "C": columns, "SCAN_AXIS": scan_axis, "MODE": -1, "REDUCTION": reduction}, num_warps=1)
    output = result.stdout + result.stderr
    assert result.returncode > 0, output
    assert diagnostic in output, output


@mps_only
@pytest.mark.parametrize("reduction", ["and", "or", "xor"])
@pytest.mark.parametrize("scan_axis,reverse", [(0, False), (1, True)])
def test_multiband_scan_shared_input_mixed_i32_row_bitwise_special_values_match_torch(
        scan_axis, reverse, reduction):
    programs, rows, columns = 2, 8, 16
    size = programs * rows * columns
    valid = size - columns - 3
    x = torch.zeros((programs, rows, columns), dtype=torch.int32)
    x[:, 1, :] = -1
    x[:, 2, 0] = -2147483648
    x[:, 3, 0] = 2147483647
    x[:, 4, ::2], x[:, 4, 1::2] = 0x55555555, -1431655766
    x[:, 5, 0] = -1
    x[:, 6, :] = -2147483648
    x[:, 7, :] = 0x12345679
    original = x.flatten()[:valid].clone()
    other = -1431655765
    x.flatten()[valid:] = other
    reduced = _i32_row_monoid_reference(x, reduction)
    assert (reduced == 0).any() and (reduced < 0).any() and (reduced > 0).any()
    # AND/OR retain the nonidentity fill; an even XOR fold cancels it.
    assert reduced[-1, -1] == (0 if reduction == "xor" else other)
    axis = scan_axis + 1
    scanned = (x.flip(axis) if reverse else x).to(torch.int64).cumsum(axis).to(torch.int32)
    if reverse:
        scanned = scanned.flip(axis)
    sentinel = -123456789
    output, scan_output, output2 = [
        torch.full((size + 7,), sentinel, dtype=torch.int32, device="mps") for _ in range(3)]
    row, row2 = [torch.full((programs * rows + 7,), sentinel, dtype=torch.int32, device="mps")
                 for _ in range(2)]
    _shared_input_scan_mixed_row_sum_kernel[(programs,)](
        original.to("mps"), output, scan_output, row, row2, output2, valid, rows, columns,
        scan_axis, reverse, True, False, other, False, True, False, False,
        TILE_FIRST=reverse, SNAPSHOT_AFTER=False, PRODUCT=False, REDUCTION=reduction,
        PURE_BROADCAST=True, num_warps=1)
    torch.mps.synchronize()
    expected = reduced[:, :, None].expand_as(x)
    torch.testing.assert_close(output.cpu()[:size].reshape_as(x), expected, atol=0, rtol=0)
    torch.testing.assert_close(row.cpu()[:programs * rows].reshape_as(reduced), reduced, atol=0, rtol=0)
    torch.testing.assert_close(scan_output.cpu()[:size].reshape_as(x), scanned, atol=0, rtol=0)
    for buffer, written in [(output, size), (scan_output, size), (row, programs * rows),
                            (row2, 0), (output2, 0)]:
        assert (buffer.cpu()[written:] == sentinel).all()


@mps_only
@pytest.mark.parametrize("reduction", ["smin", "smax", "umin", "umax"])
@pytest.mark.parametrize("rows,columns,scan_axis,num_warps,masked,computed", [
    (16, 16, 0, 1, False, False),
    (16, 16, 1, 1, True, False),
    (8, 64, 0, 4, True, True),
    (8, 64, 1, 4, False, True),
    (32, 32, 0, 4, True, False),
    (32, 32, 1, 1, True, True),
])
def test_multiband_scan_shared_input_mixed_i32_row_extrema_baseline_matches_torch(
        rows, columns, scan_axis, num_warps, masked, computed, reduction):
    _check_shared_input_mixed_i32_row_sum(
        rows, columns, scan_axis, num_warps, masked, computed, other=-1073741819, reduction=reduction)


@mps_only
@pytest.mark.parametrize("reduction", ["smin", "smax", "umin", "umax"])
@pytest.mark.parametrize("tile_first", [False, True])
@pytest.mark.parametrize(
    "rows,columns,scan_axis,num_warps,masked,computed,reverse,other,row_masked,full_output,snapshot,multiple", [
        (2, 64, 0, 1, False, False, True, 1, False, False, False, True),
        (4, 64, 1, 4, True, True, False, -1, True, True, True, True),
        (16, 16, 1, 1, True, False, True, -3, False, False, False, True),
        (32, 32, 0, 4, False, True, False, 0, False, True, True, False),
        (64, 2, 0, 1, True, True, True, 2, True, True, True, True),
        (64, 8, 1, 4, True, False, False, 2147483647, False, True, False, True),
        (8, 32, 0, 4, True, True, True, 0, False, False, True, False),
        (16, 64, 1, 1, True, False, False, 1, False, True, True, True),
    ])
def test_multiband_scan_shared_input_mixed_i32_row_extrema_matches_torch(
        rows, columns, scan_axis, num_warps, masked, computed, reverse, other,
        row_masked, full_output, snapshot, multiple, tile_first, reduction):
    assert rows * columns > 32 * num_warps
    _check_shared_input_mixed_i32_row_sum(
        rows, columns, scan_axis, num_warps, masked, computed, reverse, other,
        row_masked, full_output, snapshot, multiple, tile_first, reduction=reduction)


@mps_only
@pytest.mark.parametrize("reduction", ["smin", "smax", "umin", "umax"])
@pytest.mark.parametrize("multiple", [False, True])
def test_multiband_scan_shared_input_mixed_i32_row_extrema_reuses_publication(multiple, reduction):
    rows, columns = 8, 64
    compiled = _check_shared_input_mixed_i32_row_sum(
        rows, columns, 0, 1, True, True, other=-1, full_output=True, multiple=multiple, reduction=reduction)
    raw = compiled.asm["metal"]
    msl = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    buffers = sorted(int(length) for length in re.findall(
        r"\bthreadgroup\s+(?:u?int32_t|u?int)\s+\w+\s*\[\s*(\d+)\s*\]\s*;", msl))
    # Signed extrema may use si32 vector storage. Both folds still share one
    # source publication and vector exchange; extra masked stores add scratch.
    assert buffers == ([rows, 32, rows * columns] if multiple else [rows, rows * columns]), msl
    assert msl.count("threadgroup_barrier(") == 2, msl


@pytest.mark.parametrize("reduction", ["smin", "smax", "umin", "umax"])
@pytest.mark.parametrize("sources", [2, 3])
def test_multiband_scan_shared_input_mixed_i32_row_extrema_total_budget_is_named(sources, reduction):
    result = _compile_rankn_nested_negative_in_child(
        "_shared_input_mixed_row_sum_budget_kernel",
        {"X": "*i32", "Y": "*i32", "Row": "*i32", "WRow": "*i32", "VRow": "*i32"},
        {"SOURCES": sources, "PRODUCT": False, "REDUCTION": reduction}, num_warps=4)
    output = result.stdout + result.stderr
    assert result.returncode > 0, output
    assert "full-tile exchange scratch needs" in output, output
    assert "exceeding the 32768-byte budget" in output, output


@pytest.mark.parametrize("reduction", ["smin", "smax", "umin", "umax"])
@pytest.mark.parametrize("mode,diagnostic", [
    (0, "a shared scan-input row reduction requires direct row stores or "
     "same-axis broadcasts into its original tile"),
    (1, "distinct scan/reduce inputs require same-layout sources defined before the scan "
     "without intervening writes or input exchanges"),
    (2, "rank-2 axis=1 reduce over a computed cone is implemented for f32 only; "
     "other element types require a direct unmasked tt.load"),
    (3, "a shared scan-input row reduction requires direct row stores or "
     "same-axis broadcasts into its original tile"),
    (5, "a shared scan-input row reduction requires direct row stores or "
     "same-axis broadcasts into its original tile"),
    (6, "rank-2 axis=1 reduce over a computed cone is implemented for f32 only; "
     "other element types require a direct unmasked tt.load"),
    (7, "a shared scan-input row reduction requires direct row stores or "
     "same-axis broadcasts into its original tile"),
], ids=["distinct_source", "scan_result", "multiple_scans", "cross_axis", "tile_region", "transpose", "vector_region"])
def test_multiband_scan_shared_input_mixed_i32_row_extrema_source_and_consumer_is_named(mode, diagnostic, reduction):
    result = _compile_rankn_nested_negative_in_child(
        "_unsupported_shared_input_mixed_row_sum_kernel",
        {"X": "*i32", "W": "*i32", "Y": "*i32", "Row": "*i32", "Scanned": "*i32", "N": "i32"},
        {"R": 16, "C": 16, "SCAN_AXIS": 1, "MODE": mode, "REDUCTION": reduction}, num_warps=1)
    output = result.stdout + result.stderr
    assert result.returncode > 0, output
    if reduction.startswith("u") and mode in (2, 6):
        diagnostic = "reduce combine requires Session L3c (future)"
    assert diagnostic in output, output


@pytest.mark.parametrize("rows,columns,scan_axis,dtype,diagnostic,reduction", [
    (*case, reduction)
    for reduction in ("smin", "smax", "umin", "umax")
    for case in (
        (128, 16, 1, "*i32", "a shared scan-input row reduction requires direct row stores or "
         "same-axis broadcasts into its original tile"),
        (16, 128, 0, "*i32", "a shared scan-input row reduction requires direct row stores or "
         "same-axis broadcasts into its original tile"),
        (16, 16, 1, "*i64", "rank >= 2 scan is implemented for a blocked tile"),
    )
] + [
    (2, 16, 1, "*i32", "reduce combine requires Session L3c (future)", reduction)
    for reduction in ("umin", "umax")
])
def test_multiband_scan_shared_input_mixed_i32_row_extrema_shape_and_type_is_named(
        rows, columns, scan_axis, dtype, diagnostic, reduction):
    result = _compile_rankn_nested_negative_in_child(
        "_unsupported_shared_input_mixed_row_sum_kernel",
        {"X": dtype, "W": dtype, "Y": dtype, "Row": dtype, "Scanned": dtype, "N": "i32"},
        {"R": rows, "C": columns, "SCAN_AXIS": scan_axis, "MODE": -1, "REDUCTION": reduction}, num_warps=1)
    output = result.stdout + result.stderr
    assert result.returncode > 0, output
    assert diagnostic in output, output


@mps_only
@pytest.mark.parametrize("reduction", ["smin", "smax", "umin", "umax"])
@pytest.mark.parametrize("scan_axis,reverse,computed", [(0, False, False), (1, True, True)])
def test_multiband_scan_shared_input_mixed_i32_row_extrema_special_values_match_torch(
        scan_axis, reverse, computed, reduction):
    programs, rows, columns = 2, 8, 16
    size = programs * rows * columns
    valid = size - columns - 3
    source = torch.zeros((programs, rows, columns), dtype=torch.int32)
    source[:, 1, :] = -1
    source[:, 2, -1] = -2147483648
    source[:, 3, :] = -1
    source[:, 3, -1] = 2147483647
    source[:, 4, ::2], source[:, 4, 1::2] = -2147483648, 2147483647
    source[:, 5, :], source[:, 5, -1] = -2147483648, 0
    source[:, 6, :], source[:, 6, -1] = 2147483647, -1
    source[:, 7, :] = 0x12345679
    if computed:
        # Exercise both sides of the signed wrap boundary, and retain whole
        # negative rows so both signed comparison branches remain observable.
        source[:, 5, :], source[:, 5, -1] = -1073741825, 0
        source[:, 6, :], source[:, 6, -1] = 1073741823, -1
        source[:, 7, :] = 1073741823
    original = source.flatten()[:valid].clone()
    other = -1431655765
    source.flatten()[valid:] = other
    x = (source.to(torch.int64) * 2 + 3).to(torch.int32) if computed else source
    reduced = _mixed_i32_row_extrema_reference(x, reduction)
    opposite = "u" + reduction[1:] if reduction.startswith("s") else "s" + reduction[1:]
    assert (reduced != _mixed_i32_row_extrema_reference(x, opposite)).any()
    assert (reduced < 0).any() and (reduced > 0).any()
    filled = ((other * 2 + 3 + 2**31) % 2**32 - 2**31) if computed else other
    assert reduced[-1, -1] == filled
    axis = scan_axis + 1
    scanned = (x.flip(axis) if reverse else x).to(torch.int64).cumsum(axis).to(torch.int32)
    if reverse:
        scanned = scanned.flip(axis)
    contribution = reduced[:, :, None].expand_as(x)
    difference = (scanned.to(torch.int64) - contribution.to(torch.int64) * 2).to(torch.int32)
    second = torch.where(contribution < 0, difference.to(torch.int64), -difference.to(torch.int64)).to(torch.int32)
    sentinel = -123456789
    output, scan_output, output2 = [
        torch.full((size + 7,), sentinel, dtype=torch.int32, device="mps") for _ in range(3)]
    row, row2 = [torch.full((programs * rows + 7,), sentinel, dtype=torch.int32, device="mps")
                 for _ in range(2)]
    _shared_input_scan_mixed_row_sum_kernel[(programs,)](
        original.to("mps"), output, scan_output, row, row2, output2, valid, rows, columns,
        scan_axis, reverse, True, computed, other, False, True, False, True,
        TILE_FIRST=reverse, SNAPSHOT_AFTER=False, PRODUCT=False, REDUCTION=reduction,
        PURE_BROADCAST=True, num_warps=1)
    torch.mps.synchronize()
    for actual, expected in [(output, contribution), (scan_output, scanned), (output2, second)]:
        torch.testing.assert_close(actual.cpu()[:size].reshape_as(x), expected, atol=0, rtol=0)
        assert (actual.cpu()[size:] == sentinel).all()
    torch.testing.assert_close(row.cpu()[:programs * rows].reshape_as(reduced), reduced, atol=0, rtol=0)
    actual_row2 = row2.cpu()[:programs * rows].reshape_as(reduced)
    torch.testing.assert_close(actual_row2[:, 1:], reduced[:, 1:], atol=0, rtol=0)
    assert (actual_row2[:, 0] == sentinel).all()
    for buffer in (row, row2):
        assert (buffer.cpu()[programs * rows:] == sentinel).all()
