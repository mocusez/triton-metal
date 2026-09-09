"""Tuple associative scans on the Metal backend.

These tests pin the unmodified LeetTriton medium segmented-prefix-sum kernels:
each pass must compile to Metal, and the original three-pass ``solve`` entry
must match a CPU segmented-exclusive reference on MPS.

They also pin the two-state affine composition used by the medium linear
recurrence.  That algebra has a dedicated narrow lowering.

A general tuple scan no longer has to be one of those two shapes: an inclusive
Hillis-Steele prefix over one threadgroup buffer per carried value replays the
scan's own combine region, so any associative combine over rank-1 blocked
f32/i32 sources holding at most one element per thread is lowered.  What is
still refused -- a narrower slot type, a rank-2 tile, more than one element per
thread -- is refused before conversion, by the same predicate the pattern
dispatches on.
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
def test_tuple_scan_above_one_element_per_thread_is_named(capfd):
    """Outside the envelope the refusal has to come from the pre-conversion
    validator: a pattern that declines inside `applyFullConversion` does not
    raise, it takes the process down."""
    x = torch.randn(256, device="mps")
    prefix = torch.zeros(256, device="mps")
    count = torch.zeros(256, device="mps")
    with pytest.raises(Exception) as error:
        _tuple_sum_and_count_kernel[(1,)](x, prefix, count, BLOCK=256,
                                          num_warps=1)
    assert "convert-tritongpu-to-metal failed" in str(error.value)
    reason = capfd.readouterr()
    assert "at most one element per thread" in reason.err + reason.out

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
