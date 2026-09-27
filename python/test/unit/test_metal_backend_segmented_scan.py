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


def test_scan_result_broadcast_beyond_one_pass_is_named(capfd):
    with pytest.raises(Exception):
        _scan_bcast_kernel[(1,)](torch.zeros(64, device="mps"),
                                 torch.zeros(4, 64, device="mps"), 4, 64, False,
                                 num_warps=1)
    assert "a rank-1 tl.scan result broadcast into a rank-2 tile" in \
        capfd.readouterr().err


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


# A rank-2 reduce beside a multi-band staged scan died inside the conversion
# or, stored on its own along axis 0, returned a wrong answer; a rank-3 one was
# wrong whenever its layout was not thread-major. Both are refused by name.
@pytest.mark.parametrize("A,B,C,w,rank3,msg", [
    (1, 16, 16, 1, False, "a rank-2 tl.reduce cannot share a function"),
    (1, 8, 64, 4, False, "a rank-2 tl.reduce cannot share a function"),
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
