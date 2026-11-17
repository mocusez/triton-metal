"""Per-element masked `tl.atomic_add` and histogram output on Metal.

`tt.atomic_rmw fadd` in tensor form is the lock-free accumulation the
layer-norm backward uses in place of the tutorial's global spin lock (Apple
GPUs give no cross-threadgroup forward-progress guarantee, so a `while
atomic_cas(Lock,0,1): pass` spin lock can deadlock). Many programs atomically
add their partial row into a shared buffer; the atomic serializes the colliding
read-modify-writes.

`AtomicRmwLowering`'s tensor branch models it on the masked device store: the
typeconverter scalarizes value/index/mask to the per-thread cone, the E>1 tile
loop replicates the op in place, and the device atomic is guarded by the mask
cone so masked-off lanes never touch a potentially-OOB (and, zero-copy, live)
address. Covers E==1 / E>1, a real `cols<N` mask, and the exact grouped
`_dw[lock_id*N + cols]` accumulation shape of `_layer_norm_bwd_dx_fused`.

The integer cases cover the `tl.histogram` workload: one output lane scans the
logical input tile for its bin, then a masked i32 tensor atomic accumulates the
per-program counts into the final histogram.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")

import triton  # noqa: E402
import triton.language as tl  # noqa: E402

libmetal = pytest.importorskip(
    "triton._C.libtriton.metal",
    reason="Metal backend pybind module not built into libtriton",
)
if not torch.backends.mps.is_available():
    pytest.skip(
        "Metal backend requires an MPS-enabled PyTorch (Apple Silicon)",
        allow_module_level=True,
    )


LEET_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "metal_leet"
SUBARRAY_SUM_PATH = LEET_FIXTURES / "medium-subarray_sum.py"
BFS_PATH = LEET_FIXTURES / "hard-bfs_shortest_path.py"


@triton.jit
def _scatter_add_rows(In, Out, N, BLOCK: tl.constexpr):
    # grid=(G,): every program atomically adds its row into the shared Out[cols].
    g = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    v = tl.load(In + g * N + cols, mask=mask, other=0.0)
    tl.atomic_add(Out + cols, v, mask=mask)


# E==1 (BLOCK==tpb at 32), E>1 (BLOCK>tpb at 256/1024), non-pow2 N (masked).
@pytest.mark.parametrize("G, N", [(8, 32), (5, 200), (16, 256), (3, 1000), (7, 1024)])
def test_atomic_add_scatter(G, N):
    torch.manual_seed(G * 100 + N)
    inp = torch.randn(G, N, device="mps")
    out = torch.zeros(N, device="mps")
    BLOCK = triton.next_power_of_2(N)
    _scatter_add_rows[(G,)](inp, out, N, BLOCK=BLOCK)
    torch.mps.synchronize()
    ref = inp.cpu().sum(0)
    # Float-add reorder across programs is non-bit-deterministic; close in fp32.
    torch.testing.assert_close(out.cpu(), ref, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize(
    ("N", "S", "E"),
    [(32, 0, 0), (1024, 0, 1023), (2500, 113, 2317)],
)
def test_original_subarray_sum_solve_matches_reference(N, S, E):
    spec = importlib.util.spec_from_file_location("subarray_sum", SUBARRAY_SUM_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    values_cpu = (torch.arange(N, dtype=torch.float32) % 29) * 0.125 - 1.5
    values = values_cpu.to("mps")
    output = torch.zeros(1, dtype=torch.float32, device="mps")
    module.solve(values, output, N, S, E)
    torch.mps.synchronize()
    torch.testing.assert_close(
        output.cpu()[0], values_cpu[S : E + 1].sum(), atol=2e-3, rtol=2e-5
    )


@triton.jit
def _grouped_accumulate(In, DW, N, GROUP_SIZE_M, BLOCK: tl.constexpr):
    # Exact shape of the layer-norm backward dw accumulation:
    # DW is [GROUP_SIZE_M, N]; row `r` accumulates into bucket `r % GROUP`.
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    lock_id = row % GROUP_SIZE_M
    v = tl.load(In + row * N + cols, mask=mask, other=0.0)
    tl.atomic_add(DW + lock_id * N + cols, v, mask=mask)


@pytest.mark.parametrize("M, N, GROUP", [(8, 128, 4), (16, 256, 4), (10, 200, 3), (12, 1024, 8)])
def test_atomic_add_grouped(M, N, GROUP):
    torch.manual_seed(M * 1000 + N + GROUP)
    inp = torch.randn(M, N, device="mps")
    dw = torch.zeros(GROUP, N, device="mps")
    BLOCK = triton.next_power_of_2(N)
    _grouped_accumulate[(M,)](inp, dw, N, GROUP, BLOCK=BLOCK)
    torch.mps.synchronize()
    # reference: rows folded by (row % GROUP)
    ref = torch.zeros(GROUP, N)
    ic = inp.cpu()
    for r in range(M):
        ref[r % GROUP] += ic[r]
    torch.testing.assert_close(dw.cpu(), ref, atol=1e-4, rtol=1e-4)


@triton.jit
def _atomic_add_ones(Out, N, BLOCK: tl.constexpr):
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    tl.atomic_add(Out + cols, tl.full([BLOCK], 1.0, tl.float32), mask=mask)


@pytest.mark.parametrize("G, N", [(64, 128), (100, 300)])
def test_atomic_add_ones_exact(G, N):
    # Adding 1.0 G times is exact regardless of order -> bit-exact contention check.
    out = torch.zeros(N, device="mps")
    BLOCK = triton.next_power_of_2(N)
    _atomic_add_ones[(G,)](out, N, BLOCK=BLOCK)
    torch.mps.synchronize()
    ref = torch.full((N,), float(G))
    torch.testing.assert_close(out.cpu(), ref, atol=0, rtol=0)


@triton.jit
def _histogram_atomic(In, Out, N, NUM_BINS, PADDED_BINS: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < N
    values = tl.load(In + offsets, mask=valid)
    hist = tl.histogram(values, PADDED_BINS)
    # A masked load without an explicit `other` contributes zero for its
    # inactive lanes. Match the leet kernel by removing those padding zeros.
    hist -= (BLOCK - tl.sum(valid.to(tl.int32))) * (tl.arange(0, PADDED_BINS) == 0)
    bins = tl.arange(0, PADDED_BINS)
    tl.atomic_add(Out + bins, hist, mask=(bins < NUM_BINS) & (hist != 0))


@pytest.mark.parametrize("N, NUM_BINS", [(1024, 16), (1000, 16), (2500, 37)])
def test_histogram_atomic_matches_bincount(N, NUM_BINS):
    torch.manual_seed(N + NUM_BINS)
    inp = torch.randint(0, NUM_BINS, (N,), dtype=torch.int32, device="mps")
    out = torch.zeros(NUM_BINS, dtype=torch.int32, device="mps")
    _histogram_atomic[(triton.cdiv(N, 1024),)](
        inp,
        out,
        N,
        NUM_BINS,
        PADDED_BINS=triton.next_power_of_2(NUM_BINS),
        BLOCK=1024,
    )
    torch.mps.synchronize()
    ref = torch.bincount(inp.cpu().to(torch.int64), minlength=NUM_BINS)
    torch.testing.assert_close(out.cpu().to(torch.int64), ref, atol=0, rtol=0)


@triton.jit
def _masked_histogram_store(In, Out, N, BINS: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    valid = offsets < N
    values = tl.load(In + offsets, mask=valid)
    hist = tl.histogram(values, BINS, mask=valid)
    tl.store(Out + tl.arange(0, BINS), hist)


@pytest.mark.parametrize("N", [100, 1000])
def test_masked_histogram_matches_bincount(N):
    BINS = 16
    torch.manual_seed(N)
    inp = torch.randint(0, BINS, (N,), dtype=torch.int32, device="mps")
    out = torch.zeros(BINS, dtype=torch.int32, device="mps")
    _masked_histogram_store[(1,)](inp, out, N, BINS=BINS, BLOCK=1024)
    torch.mps.synchronize()
    ref = torch.bincount(inp.cpu().to(torch.int64), minlength=BINS)
    torch.testing.assert_close(out.cpu().to(torch.int64), ref, atol=0, rtol=0)


@triton.jit
def _atomic_max_scalar_f32(In, Out):
    value = tl.load(In + tl.program_id(0))
    tl.atomic_max(Out, value)


@pytest.mark.parametrize(
    "values",
    [
        [1.0, 7.0, 3.0, 29.0, 11.0],
        [-91.0, -7.0, -31.0, -2.0, -18.0],
    ],
)
def test_atomic_max_scalar_f32_contended(values):
    inp = torch.tensor(values, dtype=torch.float32, device="mps")
    out = torch.full((1,), -2147483648.0, dtype=torch.float32, device="mps")
    _atomic_max_scalar_f32[(len(values),)](inp, out)
    torch.mps.synchronize()
    assert out.cpu().item() == max(values)


@triton.jit
def _atomic_min_scalar_i32(In, Out):
    value = tl.load(In + tl.program_id(0))
    tl.atomic_min(Out, value)


@pytest.mark.parametrize(
    "values",
    [
        [17, 5, 93, 11, 42],
        [12, -7, 31, -44, 0, 19],
    ],
)
def test_atomic_min_scalar_i32_contended(values):
    inp = torch.tensor(values, dtype=torch.int32, device="mps")
    out = torch.full((1,), 2147483647, dtype=torch.int32, device="mps")
    _atomic_min_scalar_i32[(len(values),)](inp, out)
    torch.mps.synchronize()
    assert out.cpu().item() == min(values)


@triton.jit
def _atomic_minmax_rows(In, Out, N, BLOCK: tl.constexpr, DO_MAX: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    values = tl.load(In + row * N + cols, mask=mask)
    if DO_MAX:
        tl.atomic_max(Out + cols, values, mask=mask)
    else:
        tl.atomic_min(Out + cols, values, mask=mask)


@pytest.mark.parametrize("do_max", [False, True])
@pytest.mark.parametrize("N", [16, 200])
def test_atomic_minmax_tensor_i32(do_max, N):
    G = 5
    torch.manual_seed(9000 + N + do_max)
    inp = torch.randint(-1000, 1000, (G, N), dtype=torch.int32, device="mps")
    initial = -2147483648 if do_max else 2147483647
    out = torch.full((N,), initial, dtype=torch.int32, device="mps")
    _atomic_minmax_rows[(G,)](
        inp, out, N, BLOCK=triton.next_power_of_2(N), DO_MAX=do_max
    )
    torch.mps.synchronize()
    ref = inp.cpu().amax(0) if do_max else inp.cpu().amin(0)
    torch.testing.assert_close(out.cpu(), ref, atol=0, rtol=0)


@pytest.mark.parametrize("do_max", [False, True])
def test_atomic_minmax_tensor_u32_uses_unsigned_order(do_max):
    values = torch.tensor(
        [
            [0, 2**31 - 1, 2**31, 2**32 - 1],
            [17, 2**31, 2**32 - 1, 3],
            [2**31 + 9, 1, 5, 2**31 + 1],
        ],
        dtype=torch.uint32,
        device="mps",
    )
    initial = 0 if do_max else 2**32 - 1
    out = torch.full((4,), initial, dtype=torch.uint32, device="mps")
    _atomic_minmax_rows[(values.shape[0],)](
        values, out, 4, BLOCK=4, DO_MAX=do_max
    )
    torch.mps.synchronize()
    rows = values.cpu().tolist()
    expected = [
        (max if do_max else min)(row[col] for row in rows)
        for col in range(values.shape[1])
    ]
    assert out.cpu().tolist() == expected


@pytest.mark.parametrize("do_max", [False, True])
def test_atomic_minmax_tensor_f32_ordered_bits(do_max):
    values = torch.tensor(
        [
            [-91.5, -0.25, 7.0, 2.0],
            [-7.0, 3.5, -31.0, 29.0],
            [-18.0, -2.0, 11.0, -5.0],
        ],
        dtype=torch.float32,
        device="mps",
    )
    initial = float("-inf") if do_max else float("inf")
    out = torch.full((4,), initial, dtype=torch.float32, device="mps")
    _atomic_minmax_rows[(values.shape[0],)](
        values, out, 4, BLOCK=4, DO_MAX=do_max
    )
    torch.mps.synchronize()
    ref = values.cpu().amax(0) if do_max else values.cpu().amin(0)
    torch.testing.assert_close(out.cpu(), ref, atol=0, rtol=0)


@triton.jit
def _atomic_minmax_rank2(
    In,
    Out,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    DO_MAX: tl.constexpr,
):
    batch = tl.program_id(0)
    rows = tl.arange(0, BLOCK_M)[:, None]
    cols = tl.arange(0, BLOCK_N)[None, :]
    offsets = rows * N + cols
    mask = (rows < M) & (cols < N)
    values = tl.load(In + batch * M * N + offsets, mask=mask)
    if DO_MAX:
        tl.atomic_max(Out + offsets, values, mask=mask)
    else:
        tl.atomic_min(Out + offsets, values, mask=mask)


@pytest.mark.parametrize("do_max", [False, True])
def test_atomic_minmax_tensor_rank2_masked_sub_tpb(do_max):
    G, M, N = 5, 3, 5
    torch.manual_seed(12000 + do_max)
    values = torch.randint(
        -1000, 1000, (G, M, N), dtype=torch.int32, device="mps"
    )
    initial = -2147483648 if do_max else 2147483647
    out = torch.full((M, N), initial, dtype=torch.int32, device="mps")
    _atomic_minmax_rank2[(G,)](
        values, out, M, N, BLOCK_M=4, BLOCK_N=8, DO_MAX=do_max
    )
    torch.mps.synchronize()
    ref = values.cpu().amax(0) if do_max else values.cpu().amin(0)
    torch.testing.assert_close(out.cpu(), ref, atol=0, rtol=0)


@triton.jit
def _atomic_min_return_old(In, Out, Old, N, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < N
    values = tl.load(In + offsets, mask=mask)
    old = tl.atomic_min(Out + offsets, values, mask=mask)
    tl.store(Old + offsets, old, mask=mask)


def test_atomic_min_tensor_returns_old_value_once():
    N, BLOCK = 17, 32
    values = torch.arange(N, dtype=torch.int32, device="mps") - 50
    out = torch.full((N,), 1234, dtype=torch.int32, device="mps")
    old = torch.empty_like(out)
    compiled = _atomic_min_return_old.warmup(
        values, out, old, N, BLOCK=BLOCK, grid=(1,)
    )
    msl = compiled.asm["metal"]
    if isinstance(msl, bytes):
        msl = msl.decode()
    assert msl.count("atomic_fetch_min_explicit") == 1
    assert "int32_t v" in msl

    _atomic_min_return_old[(1,)](values, out, old, N, BLOCK=BLOCK)
    torch.mps.synchronize()
    torch.testing.assert_close(old.cpu(), torch.full((N,), 1234, dtype=torch.int32))
    torch.testing.assert_close(out.cpu(), values.cpu())


@pytest.mark.parametrize("do_max", [False, True])
def test_atomic_minmax_tensor_u64_void_modify(do_max):
    values = torch.tensor(
        [
            [0, 2**63 - 1, 2**63, 2**64 - 1],
            [9, 2**63, 2**64 - 1, 7],
            [2**63 + 11, 1, 5, 2**63 + 3],
        ],
        dtype=torch.uint64,
        device="mps",
    )
    initial = 0 if do_max else 2**64 - 1
    # torch.mps supports uint64 tensors, but torch.full(uint64, device="mps")
    # currently rejects the scalar fill path. Constructing from explicit data
    # exercises the same storage without depending on that unrelated PyTorch
    # limitation.
    out = torch.tensor([initial] * 4, dtype=torch.uint64, device="mps")
    compiled = _atomic_minmax_rows.warmup(
        values, out, 4, BLOCK=4, DO_MAX=do_max, grid=(values.shape[0],)
    )
    msl = compiled.asm["metal"]
    if isinstance(msl, bytes):
        msl = msl.decode()
    expected_function = "atomic_max_explicit" if do_max else "atomic_min_explicit"
    assert f"{expected_function}((device atomic_ulong*)" in msl
    assert "atomic_fetch_" not in msl

    _atomic_minmax_rows[(values.shape[0],)](
        values, out, 4, BLOCK=4, DO_MAX=do_max
    )
    torch.mps.synchronize()
    rows = values.cpu().tolist()
    expected = [
        (max if do_max else min)(row[col] for row in rows)
        for col in range(values.shape[1])
    ]
    assert out.cpu().tolist() == expected


@triton.jit
def _atomic_minmax_scalar_u64(Out, value, DO_MAX: tl.constexpr):
    if DO_MAX:
        tl.atomic_max(Out, value)
    else:
        tl.atomic_min(Out, value)


@pytest.mark.parametrize("do_max", [False, True])
def test_atomic_minmax_scalar_u64_void_modify(do_max):
    host_values = [0, 2**63 - 1, 2**63, 2**64 - 1, 17]
    initial = 0 if do_max else 2**64 - 1
    out = torch.tensor([initial], dtype=torch.uint64, device="mps")
    for value in host_values:
        _atomic_minmax_scalar_u64[(1,)](out, value, DO_MAX=do_max)
    torch.mps.synchronize()
    assert out.cpu().item() == (max if do_max else min)(host_values)


# --- 64-bit add/and/or/xor ----------------------------------------------------
#
# MSL's atomic_ulong has only void min/max. A 64-bit add/and/or/xor whose old
# value is unused runs as two 32-bit atomics on the low and high words; add
# carries from the low word's fetched value into the high word. The value after
# the kernel is exact (additions commute; the halves of a bitwise op are
# independent). A 64-bit snapshot is not available, so the old value cannot be
# returned, and mixing kinds in one kernel is refused because a different
# 64-bit atomic could act on a half-updated value.

_SPLIT64_OPS = ["add", "and", "or", "xor"]


@triton.jit
def _atomic_split64(In, Out, N, CELLS: tl.constexpr, BLOCK: tl.constexpr, OP: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    value = tl.load(In + offsets, mask=mask, other=0)
    if OP == "add":
        tl.atomic_add(Out + offsets % CELLS, value, mask=mask)
    elif OP == "and":
        tl.atomic_and(Out + offsets % CELLS, value, mask=mask)
    elif OP == "or":
        tl.atomic_or(Out + offsets % CELLS, value, mask=mask)
    else:
        tl.atomic_xor(Out + offsets % CELLS, value, mask=mask)


def _split64_reference(op, initial, values, cells):
    result = list(initial)
    for i, value in enumerate(values):
        cell = i % cells
        if op == "add":
            result[cell] = (result[cell] + value) % 2**64
        elif op == "and":
            result[cell] &= value
        elif op == "or":
            result[cell] |= value
        else:
            result[cell] ^= value
    return result


def _random_u64(count, gen):
    hi = torch.randint(0, 2**32, (count,), generator=gen, dtype=torch.int64)
    lo = torch.randint(0, 2**32, (count,), generator=gen, dtype=torch.int64)
    return [(h << 32) | l for h, l in zip(hi.tolist(), lo.tolist())]


# BLOCK=1024 at one warp is the tile loop; CELLS=1 puts every lane of every
# program on one cell, so low-word carries race the high-word adds.
@pytest.mark.parametrize("BLOCK, num_warps, N, CELLS", [(64, 1, 1024, 64), (256, 4, 1000, 7), (1024, 1, 4096, 1)])
@pytest.mark.parametrize("dtype", [torch.int64, torch.uint64])
@pytest.mark.parametrize("op", _SPLIT64_OPS)
def test_atomic_64bit_split_ops_are_exact(op, dtype, BLOCK, num_warps, N, CELLS):
    gen = torch.Generator().manual_seed(N + CELLS)
    values = _random_u64(N, gen)
    if op == "add":
        initial = [2**32 - 1] * CELLS  # the first add of any odd low word carries
    elif op == "and":
        initial = [2**64 - 1] * CELLS
        values = [v | w for v, w in zip(values, _random_u64(N, gen))]
    else:
        initial = _random_u64(CELLS, gen)

    def to_dtype(seq):
        if dtype == torch.uint64:
            return torch.tensor(seq, dtype=torch.uint64)
        return torch.tensor([v - 2**64 if v >= 2**63 else v for v in seq], dtype=torch.int64)

    out = to_dtype(initial).to("mps")
    _atomic_split64[(triton.cdiv(N, BLOCK),)](
        to_dtype(values).to("mps"), out, N, CELLS=CELLS, BLOCK=BLOCK, OP=op, num_warps=num_warps
    )
    torch.mps.synchronize()
    got = [v % 2**64 for v in out.cpu().tolist()]
    assert got == _split64_reference(op, initial, values, CELLS)


@triton.jit
def _atomic_split64_scalar(In, Out, OP: tl.constexpr):
    value = tl.load(In + tl.program_id(0))
    if OP == "add":
        tl.atomic_add(Out, value)
    else:
        tl.atomic_xor(Out, value)


@pytest.mark.parametrize("op", ["add", "xor"])
def test_atomic_64bit_split_scalar_contention(op):
    gen = torch.Generator().manual_seed(1)
    values = _random_u64(64, gen)
    out = torch.tensor([2**32 - 1], dtype=torch.uint64, device="mps")
    _atomic_split64_scalar[(64,)](torch.tensor(values, dtype=torch.uint64).to("mps"), out, OP=op)
    torch.mps.synchronize()
    assert out.cpu().item() % 2**64 == _split64_reference(op, [2**32 - 1], values, 1)[0]


@triton.jit
def _atomic_split64_old_value(Out, Old, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    tl.store(Old + offsets, tl.atomic_add(Out + offsets, tl.full((BLOCK,), 1, tl.int64)))


def test_atomic_64bit_split_old_value_is_refused_by_name(capfd):
    out = torch.zeros(32, dtype=torch.int64, device="mps")
    old = torch.zeros(32, dtype=torch.int64, device="mps")
    with pytest.raises(Exception):
        _atomic_split64_old_value[(1,)](out, old, BLOCK=32)
    assert "64-bit integer atomic add cannot return its old value" in capfd.readouterr().err
    # The process is still usable after the refusal.
    _atomic_split64[(1,)](torch.ones(32, dtype=torch.int64, device="mps"), out, 32, CELLS=32, BLOCK=32, OP="add")
    torch.mps.synchronize()
    assert torch.equal(out.cpu(), torch.ones(32, dtype=torch.int64))


@triton.jit
def _atomic_split64_mixed(Out, BLOCK: tl.constexpr, OTHER: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    tl.atomic_add(Out + offsets, tl.full((BLOCK,), 1, tl.int64))
    if OTHER == "xor":
        tl.atomic_xor(Out + offsets, tl.full((BLOCK,), 1, tl.int64))
    else:
        tl.atomic_max(Out + offsets, tl.full((BLOCK,), 1, tl.uint64))


@pytest.mark.parametrize("other, message", [
    ("xor", "64-bit integer atomic add runs as two 32-bit atomics and cannot share a kernel with a 64-bit atomic xor"),
    ("umax", "64-bit integer atomic add runs as two 32-bit atomics and cannot share a kernel with a 64-bit atomic umax"),
])
def test_atomic_64bit_split_mixed_kinds_are_refused_by_name(capfd, other, message):
    dtype = torch.int64 if other == "xor" else torch.uint64
    out = torch.tensor([0] * 32, dtype=dtype, device="mps")
    with pytest.raises(Exception):
        _atomic_split64_mixed[(1,)](out, BLOCK=32, OTHER=other)
    assert message in capfd.readouterr().err


def test_atomic_ulong_compile_error_has_capability_context(monkeypatch):
    from triton.backends.metal import driver as metal_driver

    compiler_error = RuntimeError("MSL compiler rejected atomic_ulong")

    def reject_shader(_source):
        raise compiler_error

    monkeypatch.setattr(metal_driver, "_use_mps_runtime", lambda: True)
    monkeypatch.setattr(torch.mps, "compile_shader", reject_shader)
    with pytest.raises(RuntimeError, match="Apple8-or-newer") as error:
        metal_driver.MetalUtils().load_binary(
            "atomic_u64", "kernel void atomic_u64(device atomic_ulong*)", 0, 0
        )
    assert error.value.__cause__ is compiler_error


def test_atomic_ulong_capability_surface_matches_backend_gate():
    header = "#include <metal_stdlib>\nusing namespace metal;\n"

    def source(name, statement):
        return (
            header
            + f"kernel void {name}(device ulong* out [[buffer(0)]]) {{ "
            + statement
            + " }"
        )

    # Apple8+ exposes the void u64 min/max surface used by the backend.
    torch.mps.compile_shader(
        source(
            "u64_min_control",
            "atomic_min_explicit((device atomic_ulong*)&out[0], "
            "(ulong)1, memory_order_relaxed);",
        )
    )

    unsupported = {
        "u64_fetch_add": (
            "atomic_fetch_add_explicit((device atomic_ulong*)&out[0], "
            "(ulong)1, memory_order_relaxed);"
        ),
        "u64_fetch_and": (
            "atomic_fetch_and_explicit((device atomic_ulong*)&out[0], "
            "(ulong)1, memory_order_relaxed);"
        ),
        "u64_fetch_or": (
            "atomic_fetch_or_explicit((device atomic_ulong*)&out[0], "
            "(ulong)1, memory_order_relaxed);"
        ),
        "u64_fetch_xor": (
            "atomic_fetch_xor_explicit((device atomic_ulong*)&out[0], "
            "(ulong)1, memory_order_relaxed);"
        ),
        "u64_exchange": (
            "atomic_exchange_explicit((device atomic_ulong*)&out[0], "
            "(ulong)1, memory_order_relaxed);"
        ),
        "u64_cas": (
            "ulong expected = 0; "
            "atomic_compare_exchange_weak_explicit("
            "(device atomic_ulong*)&out[0], &expected, (ulong)1, "
            "memory_order_relaxed, memory_order_relaxed);"
        ),
    }
    for name, statement in unsupported.items():
        with pytest.raises((RuntimeError, SyntaxError), match="no matching function"):
            torch.mps.compile_shader(source(name, statement))


@triton.jit
def _max_subarray_sum_kernel(
    input, output, N, windows_size, length, BLOCK_SIZE: tl.constexpr
):
    # Verbatim kernel body from python/test/unit/fixtures/metal_leet/medium-max_subarray_sum.py.
    offs = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < length
    result = tl.zeros([BLOCK_SIZE], tl.float32)
    for i in range(windows_size):
        offs_i = offs + i
        mask_i = offs_i < N
        val = tl.load(input + offs_i, mask_i)
        result += val
    ret = tl.where(mask, result, float("-inf"))
    ret = tl.max(ret)
    tl.atomic_max(output, ret)


@pytest.mark.parametrize(
    "N, window_size",
    [(1, 1), (31, 5), (32, 7), (33, 7), (1000, 37), (1025, 64)],
)
def test_max_subarray_sum_matches_torch(N, window_size):
    torch.manual_seed(N * 100 + window_size)
    inp = torch.randn(N, dtype=torch.float32, device="mps")
    # Pin the all-negative case as well as mixed random inputs.
    if N == 33:
        inp = -inp.abs() - 1.0
    out = torch.empty(1, dtype=torch.float32, device="mps")
    out.fill_(-2147483648)
    length = N - window_size + 1
    BLOCK_SIZE = 32
    _max_subarray_sum_kernel[(triton.cdiv(N, BLOCK_SIZE),)](
        inp, out, N, window_size, length, BLOCK_SIZE
    )
    torch.mps.synchronize()
    expected = inp.cpu().unfold(0, window_size, 1).sum(1).max()
    torch.testing.assert_close(out.cpu()[0], expected, atol=1e-5, rtol=1e-5)


# --- atomics through an address that folded to a uniform pointer -------------
#
# `tl.atomic_add(out + offs * 0, v)` collapses the whole tile onto one cell, so
# Triton folds the `tt.addptr` away and hands the backend a bare `tt.splat`.
# The tensor branch used to require an addptr, and an op that matches no pattern
# is a process kill here rather than an error: this shape crashed the caller on
# 7 of 8 runs (SIGSEGV or abort, varying run to run).
#
# It is lowered rather than rejected because, unlike the store of the same
# shape, it is not a race: every lane read-modify-writes the one cell
# atomically, which is what the shape means. The sums below are exact, so a
# dropped or duplicated lane shows up as a hard mismatch and not as noise.


@triton.jit
def _atomic_add_fold_to_one_cell(In, Out, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    tl.atomic_add(Out + offsets * 0, tl.load(In + offsets))


@pytest.mark.parametrize("num_warps", [1, 2, 4])
@pytest.mark.parametrize("BLOCK", [1, 8, 32, 128, 1024])
def test_atomic_add_uniform_address_accumulates_every_lane(BLOCK, num_warps):
    inp = torch.ones(BLOCK, dtype=torch.float32, device="mps")
    out = torch.zeros(1, dtype=torch.float32, device="mps")
    _atomic_add_fold_to_one_cell[(1,)](inp, out, BLOCK, num_warps=num_warps)
    torch.mps.synchronize()
    assert out.cpu()[0].item() == float(BLOCK)


@triton.jit
def _atomic_add_fold_to_one_cell_i32(In, Out, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    tl.atomic_add(Out + offsets * 0, tl.load(In + offsets))


@pytest.mark.parametrize("BLOCK", [8, 64])
def test_atomic_add_uniform_address_i32(BLOCK):
    inp = torch.arange(BLOCK, dtype=torch.int32, device="mps")
    out = torch.zeros(1, dtype=torch.int32, device="mps")
    _atomic_add_fold_to_one_cell_i32[(1,)](inp, out, BLOCK)
    torch.mps.synchronize()
    assert out.cpu()[0].item() == BLOCK * (BLOCK - 1) // 2


@triton.jit
def _atomic_add_uniform_masked(In, Out, KEEP, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    tl.atomic_add(
        Out + offsets * 0, tl.load(In + offsets), mask=(offsets == KEEP)
    )


@pytest.mark.parametrize("keep", [0, 5, 31])
def test_atomic_add_uniform_address_masked_to_one_lane(keep):
    BLOCK = 32
    inp = torch.arange(BLOCK, dtype=torch.float32, device="mps")
    out = torch.zeros(1, dtype=torch.float32, device="mps")
    _atomic_add_uniform_masked[(1,)](inp, out, keep, BLOCK)
    torch.mps.synchronize()
    assert out.cpu()[0].item() == float(keep)


@triton.jit
def _atomic_max_uniform(In, Out, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    tl.atomic_max(Out + offsets * 0, tl.load(In + offsets))


def test_atomic_max_uniform_address_reduces_the_tile():
    BLOCK = 64
    torch.manual_seed(7)
    inp = torch.randint(-500, 500, (BLOCK,), dtype=torch.int32, device="mps")
    out = torch.full((1,), -2147483648, dtype=torch.int32, device="mps")
    _atomic_max_uniform[(1,)](inp, out, BLOCK)
    torch.mps.synchronize()
    assert out.cpu()[0].item() == inp.cpu().max().item()


@triton.jit
def _atomic_add_uniform_returns_old(In, Out, Old, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    old = tl.atomic_add(Out + offsets * 0, tl.load(In + offsets))
    tl.store(Old + offsets, old)


def test_atomic_add_uniform_address_old_values_are_a_permutation():
    # Every lane adds 1 to the same cell, so the fetched old values are the
    # partial sums in whatever order the hardware serialized them: as a
    # multiset, exactly 0..BLOCK-1. Order is not part of the contract.
    BLOCK = 32
    inp = torch.ones(BLOCK, dtype=torch.float32, device="mps")
    out = torch.zeros(1, dtype=torch.float32, device="mps")
    old = torch.empty(BLOCK, dtype=torch.float32, device="mps")
    _atomic_add_uniform_returns_old[(1,)](inp, out, old, BLOCK)
    torch.mps.synchronize()
    assert out.cpu()[0].item() == float(BLOCK)
    assert sorted(old.cpu().tolist()) == [float(i) for i in range(BLOCK)]


@triton.jit
def _atomic_add_uniform_i32_masked_returns_old(Out, Old, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    active = (offsets % 3) != 1
    old = tl.atomic_add(
        Out + tl.zeros([BLOCK], tl.int32),
        tl.full([BLOCK], 1, tl.int32),
        mask=active,
    )
    tl.store(Old + offsets, old, mask=active)


def test_atomic_add_uniform_i32_masked_old_values_are_tickets():
    BLOCK = 64
    active_count = sum(1 for offset in range(BLOCK) if offset % 3 != 1)
    out = torch.zeros(1, dtype=torch.int32, device="mps")
    old = torch.full((BLOCK,), -7, dtype=torch.int32, device="mps")

    _atomic_add_uniform_i32_masked_returns_old[(1,)](out, old, BLOCK)
    torch.mps.synchronize()

    old_cpu = old.cpu()
    active = [offset for offset in range(BLOCK) if offset % 3 != 1]
    inactive = [offset for offset in range(BLOCK) if offset % 3 == 1]
    assert out.cpu()[0].item() == active_count
    assert sorted(old_cpu[active].tolist()) == list(range(active_count))
    assert old_cpu[inactive].tolist() == [-7] * len(inactive)


def _cpu_shortest_path(grid, start, end):
    from collections import deque

    rows, cols = len(grid), len(grid[0])
    if start == end:
        return 0
    seen = {start}
    q = deque([(start[0], start[1], 0)])
    while q:
        row, col, depth = q.popleft()
        for nrow, ncol in (
            (row - 1, col),
            (row + 1, col),
            (row, col - 1),
            (row, col + 1),
        ):
            if not (0 <= nrow < rows and 0 <= ncol < cols):
                continue
            if grid[nrow][ncol] != 0 or (nrow, ncol) in seen:
                continue
            if (nrow, ncol) == end:
                return depth + 1
            seen.add((nrow, ncol))
            q.append((nrow, ncol, depth + 1))
    return -1


def _bfs_case(name):
    if name == "same":
        return [[0]], (0, 0), (0, 0), False
    if name == "open":
        return [[0, 0, 0], [1, 1, 0], [0, 0, 0]], (0, 0), (2, 0), False
    if name == "detour":
        return [[0, 0, 1, 0], [1, 0, 1, 0], [0, 0, 0, 0]], (0, 0), (0, 3), False
    if name == "unreachable":
        return [[0, 1, 0], [1, 1, 1], [0, 1, 0]], (0, 0), (2, 2), False
    if name == "noncontiguous":
        return [[0, 0, 0, 0], [1, 1, 1, 0], [0, 0, 0, 0]], (0, 0), (2, 0), True
    if name == "wide_frontier":
        rows, cols = 75, 75
        return [[0] * cols for _ in range(rows)], (37, 37), (74, 74), False
    if name == "deep_corridor":
        return [[0] * 1025], (0, 0), (0, 1024), False
    if name == "large_open":
        rows, cols = 257, 257
        return [[0] * cols for _ in range(rows)], (0, 0), (256, 256), False
    raise AssertionError(name)


def _run_bfs_child(case_name):
    source = f"""
import importlib.util
import json

import torch

path = {str(BFS_PATH)!r}
case_name = {case_name!r}

spec = importlib.util.spec_from_file_location("leet_bfs_shortest_path", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

def make_case(name):
    if name == "same":
        return [[0]], (0, 0), (0, 0), False
    if name == "open":
        return [[0, 0, 0], [1, 1, 0], [0, 0, 0]], (0, 0), (2, 0), False
    if name == "detour":
        return [[0, 0, 1, 0], [1, 0, 1, 0], [0, 0, 0, 0]], (0, 0), (0, 3), False
    if name == "unreachable":
        return [[0, 1, 0], [1, 1, 1], [0, 1, 0]], (0, 0), (2, 2), False
    if name == "noncontiguous":
        return [[0, 0, 0, 0], [1, 1, 1, 0], [0, 0, 0, 0]], (0, 0), (2, 0), True
    if name == "wide_frontier":
        rows, cols = 75, 75
        return [[0] * cols for _ in range(rows)], (37, 37), (74, 74), False
    if name == "deep_corridor":
        return [[0] * 1025], (0, 0), (0, 1024), False
    if name == "large_open":
        rows, cols = 257, 257
        return [[0] * cols for _ in range(rows)], (0, 0), (256, 256), False
    raise AssertionError(name)

grid_host, start, end, noncontiguous = make_case(case_name)
rows = len(grid_host)
cols = len(grid_host[0])
grid_cpu = torch.tensor(grid_host, dtype=torch.int32)
if noncontiguous:
    base = torch.ones((rows, cols * 2), dtype=torch.int32)
    base[:, ::2] = grid_cpu
    grid = base.to("mps")[:, ::2]
else:
    grid = grid_cpu.to("mps")
result = torch.full((1,), -999, dtype=torch.int32, device="mps")

repeat = 20 if case_name == "wide_frontier" else 1
values = []
for _ in range(repeat):
    result.fill_(-999)
    ret = module.solve(grid, result, rows, cols, start[0], start[1], end[0], end[1])
    torch.mps.synchronize()
    values.append([int(ret), int(result.cpu()[0].item())])

print(json.dumps(values))
"""
    env = dict(os.environ)
    env.pop("TRITON_INTERPRET", None)
    return subprocess.run(
        [sys.executable, "-c", source],
        capture_output=True,
        text=True,
        env=env,
    )


@pytest.mark.parametrize(
    "case_name",
    [
        "same",
        "open",
        "detour",
        "unreachable",
        "noncontiguous",
        "wide_frontier",
        "deep_corridor",
        "large_open",
    ],
)
def test_bfs_shortest_path_original_solve_matches_cpu(case_name):
    import json

    grid, start, end, _ = _bfs_case(case_name)
    expected = _cpu_shortest_path(grid, start, end)
    proc = _run_bfs_child(case_name)
    assert proc.returncode == 0, (
        f"BFS child for {case_name} exited {proc.returncode}\n"
        f"--- stdout ---\n{proc.stdout[-1000:]}\n"
        f"--- stderr ---\n{proc.stderr[-3000:]}"
    )
    observed = json.loads(proc.stdout)
    assert observed
    assert all(pair == [expected, expected] for pair in observed)


# --- bitwise / exchange RMW kinds, and the scalar old-value broadcast --------
#
# `tt.atomic_rmw` accepted add/fadd/min/max/umin/umax only; and/or/xor/xchg were
# rejected outright even though MSL has all four (verified by compiling
# `atomic_fetch_and_explicit` and friends through `torch.mps.compile_shader` —
# 32-bit only, there is no atomic_ulong bitwise overload, while exchange also
# has an atomic_float one).


@triton.jit
def _atomic_bitwise_scalar(In, Out, OP: tl.constexpr):
    v = tl.load(In)
    if OP == 0:
        tl.atomic_and(Out, v)
    elif OP == 1:
        tl.atomic_or(Out, v)
    else:
        tl.atomic_xor(Out, v)


@pytest.mark.parametrize(
    "op, initial, value, expected",
    [
        (0, 0b1011, 0b0110, 0b0010),
        (1, 0b1001, 0b0110, 0b1111),
        (2, 0b1011, 0b0110, 0b1101),
        (0, -1, 0x0F0F0F0F, 0x0F0F0F0F),
        (2, -1, -1, 0),
    ],
)
def test_atomic_bitwise_scalar_i32(op, initial, value, expected):
    inp = torch.full((1,), value, dtype=torch.int32, device="mps")
    out = torch.full((1,), initial, dtype=torch.int32, device="mps")
    _atomic_bitwise_scalar[(1,)](inp, out, op)
    torch.mps.synchronize()
    assert out.cpu()[0].item() == expected


@triton.jit
def _atomic_xor_tensor(In, Out, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    tl.atomic_xor(Out + offsets, tl.load(In + offsets))


def test_atomic_xor_tensor_i32():
    BLOCK = 64
    torch.manual_seed(3)
    inp = torch.randint(0, 1 << 20, (BLOCK,), dtype=torch.int32, device="mps")
    out = torch.randint(0, 1 << 20, (BLOCK,), dtype=torch.int32, device="mps")
    ref = out.cpu() ^ inp.cpu()
    _atomic_xor_tensor[(1,)](inp, out, BLOCK)
    torch.mps.synchronize()
    torch.testing.assert_close(out.cpu(), ref, atol=0, rtol=0)


@triton.jit
def _atomic_xchg_scalar(In, Out):
    tl.atomic_xchg(Out, tl.load(In))


def test_atomic_xchg_scalar_f32_replaces_the_cell():
    inp = torch.full((1,), 4.5, dtype=torch.float32, device="mps")
    out = torch.full((1,), -1.0, dtype=torch.float32, device="mps")
    _atomic_xchg_scalar[(1,)](inp, out)
    torch.mps.synchronize()
    assert out.cpu()[0].item() == 4.5


@triton.jit
def _atomic_add_scalar_returns_old(Out, Seen, BLOCK: tl.constexpr):
    # One ticket per program; every lane of the program must observe the SAME
    # ticket, which is what the threadgroup publish + barrier is for.
    old = tl.atomic_add(Out, 1)
    pid = tl.program_id(0)
    tl.store(Seen + pid * BLOCK + tl.arange(0, BLOCK), old)


def test_atomic_add_scalar_old_value_reaches_every_lane():
    G, BLOCK = 4, 32
    out = torch.zeros(1, dtype=torch.int32, device="mps")
    seen = torch.full((G * BLOCK,), -1, dtype=torch.int32, device="mps")
    _atomic_add_scalar_returns_old[(G,)](out, seen, BLOCK, num_warps=2)
    torch.mps.synchronize()
    assert out.cpu()[0].item() == G
    tickets = seen.cpu().reshape(G, BLOCK)
    # Uniform within a program...
    for row in tickets.tolist():
        assert len(set(row)) == 1, row
    # ...and one distinct ticket per program. Which program drew which is up to
    # the hardware's serialization and is not part of the contract.
    assert sorted(tickets[:, 0].tolist()) == list(range(G))


# --- f32 atomic compare-and-swap -------------------------------------------


@triton.jit
def _atomic_cas_f32_tensor(Target, Expected, Desired, Old, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    old = tl.atomic_cas(
        Target + offsets,
        tl.load(Expected + offsets),
        tl.load(Desired + offsets),
    )
    tl.store(Old + offsets, old)


def test_atomic_cas_f32_uses_bitwise_compare_and_returns_old_value():
    # Include the two IEEE-754 cases that a numeric `==` retry test gets wrong:
    # signed zero compares unequal for CAS, while equal NaN payloads compare
    # equal because compare-exchange operates on the object representation.
    target_bits = torch.tensor(
        [
            0,
            -0x80000000,
            0x7FC00001,
            0x7FC00001,
            0x3FC00000,
            0x40000000,
            0x40800000,
            0x40A00000,
        ],
        dtype=torch.int32,
    )
    expected_bits = torch.tensor(
        [
            0,
            0,
            0x7FC00001,
            0x7FC00002,
            0x3FC00000,
            0x40400000,
            0x40800000,
            0x40C00000,
        ],
        dtype=torch.int32,
    )
    desired_bits = torch.tensor(
        [
            0x41200000,
            0x41300000,
            0x41400000,
            0x41500000,
            0x41600000,
            0x41700000,
            0x41800000,
            0x41900000,
        ],
        dtype=torch.int32,
    )
    target = target_bits.view(torch.float32).to("mps")
    expected = expected_bits.view(torch.float32).to("mps")
    desired = desired_bits.view(torch.float32).to("mps")
    old = torch.empty_like(target)

    _atomic_cas_f32_tensor[(1,)](
        target, expected, desired, old, BLOCK=target_bits.numel()
    )
    torch.mps.synchronize()

    final_bits = torch.where(target_bits == expected_bits, desired_bits, target_bits)
    assert torch.equal(old.cpu().view(torch.int32), target_bits)
    assert torch.equal(target.cpu().view(torch.int32), final_bits)


@triton.jit
def _atomic_cas_f32_scalar_contended(Target, Seen, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    desired = (pid + 1).to(tl.float32)
    old = tl.atomic_cas(Target, 0.0, desired)
    tl.store(Seen + pid * BLOCK + tl.arange(0, BLOCK), old)


def test_atomic_cas_f32_scalar_contention_and_old_value_broadcast():
    programs = 4
    block = 32
    target = torch.zeros(1, dtype=torch.float32, device="mps")
    seen = torch.full(
        (programs * block,), -1.0, dtype=torch.float32, device="mps"
    )

    _atomic_cas_f32_scalar_contended[(programs,)](
        target, seen, BLOCK=block, num_warps=2
    )
    torch.mps.synchronize()

    final = target.cpu().item()
    assert final in {1.0, 2.0, 3.0, 4.0}
    rows = seen.cpu().reshape(programs, block)
    for row in rows:
        assert torch.equal(row, torch.full_like(row, row[0]))
    tickets = rows[:, 0]
    assert torch.count_nonzero(tickets == 0.0).item() == 1
    assert torch.all((tickets == 0.0) | (tickets == final))


# --- 16-bit atomic_cas -------------------------------------------------------
#
# MSL has no 16-bit atomics. A 16-bit compare-and-swap reads the 32-bit word
# that contains the element, returns the element's observed bits when they
# differ from `cmp`, and otherwise swaps in the word with only that element
# replaced. A failed exchange re-reads and retries, so an update of the OTHER
# half of the word is never lost and never mistaken for a mismatch.

_CAS16_DTYPES = [torch.float16, torch.bfloat16, torch.int16]


@triton.jit
def _atomic_cas_16bit_tensor(Target, Expected, Desired, Old, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    old = tl.atomic_cas(Target + offsets, tl.load(Expected + offsets), tl.load(Desired + offsets))
    tl.store(Old + offsets, old)


@pytest.mark.parametrize("dtype", _CAS16_DTYPES)
def test_atomic_cas_16bit_compares_bits_and_returns_old_value(dtype):
    # For the float types this includes the two cases a numeric `==` gets
    # wrong: -0.0 against +0.0 must NOT swap, and two equal NaN payloads must.
    target_bits = torch.tensor([0, -0x8000, 0x7E01, 0x7E01, 0x3C00, 0x4000, 0x4400, 0x4500], dtype=torch.int16)
    expected_bits = torch.tensor([0, 0, 0x7E01, 0x7E02, 0x3C00, 0x4200, 0x4400, 0x4600], dtype=torch.int16)
    desired_bits = torch.arange(0x4900, 0x4908, dtype=torch.int16)
    target = target_bits.view(dtype).to("mps")
    old = torch.empty_like(target)
    _atomic_cas_16bit_tensor[(1,)](
        target, expected_bits.view(dtype).to("mps"), desired_bits.view(dtype).to("mps"), old, BLOCK=8
    )
    torch.mps.synchronize()
    final_bits = torch.where(target_bits == expected_bits, desired_bits, target_bits)
    assert torch.equal(old.cpu().view(torch.int16), target_bits)
    assert torch.equal(target.cpu().view(torch.int16), final_bits)


@triton.jit
def _atomic_cas_16bit_race(Target, Old, Values, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    old = tl.atomic_cas(Target + offsets, tl.load(Values + offsets * 0), tl.load(Values + 1 + pid + offsets * 0))
    tl.store(Old + pid * BLOCK + offsets, old)


@pytest.mark.parametrize("num_warps", [1, 4])
@pytest.mark.parametrize("dtype", _CAS16_DTYPES)
def test_atomic_cas_16bit_adjacent_cells_race(dtype, num_warps):
    # Four programs race on every cell, and neighbouring cells share a word, so
    # most failed exchanges are caused by the OTHER half. Exactly one program
    # per cell must win; the rest must see the winner's value.
    programs, block = 4, 64
    values = torch.arange(0, programs + 1, dtype=torch.float32).to(dtype).to("mps")
    target = torch.zeros(block, dtype=dtype, device="mps")
    old = torch.full((programs, block), -7, dtype=dtype, device="mps")
    _atomic_cas_16bit_race[(programs,)](target, old, values, BLOCK=block, num_warps=num_warps)
    torch.mps.synchronize()
    final = target.cpu().float()
    seen = old.cpu().float()
    for cell in range(block):
        winner = int(final[cell].item())
        assert winner in range(1, programs + 1)
        assert seen[winner - 1, cell].item() == 0.0
        assert torch.count_nonzero(seen[:, cell] == 0.0).item() == 1
        assert torch.count_nonzero(seen[:, cell] == float(winner)).item() == programs - 1


@triton.jit
def _atomic_cas_16bit_scalar(Target, Seen, Values, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    old = tl.atomic_cas(Target, tl.load(Values), tl.load(Values + 1 + pid))
    tl.store(Seen + pid * BLOCK + tl.arange(0, BLOCK), old)


@pytest.mark.parametrize("dtype", _CAS16_DTYPES)
def test_atomic_cas_16bit_scalar_contention_and_old_value_broadcast(dtype):
    programs, block = 4, 32
    values = torch.arange(0, programs + 1, dtype=torch.float32).to(dtype).to("mps")
    target = torch.zeros(1, dtype=dtype, device="mps")
    seen = torch.full((programs * block,), -1, dtype=dtype, device="mps")
    _atomic_cas_16bit_scalar[(programs,)](target, seen, values, BLOCK=block, num_warps=2)
    torch.mps.synchronize()
    final = target.cpu().float().item()
    assert final in {1.0, 2.0, 3.0, 4.0}
    rows = seen.cpu().float().reshape(programs, block)
    for row in rows:
        assert torch.equal(row, torch.full_like(row, row[0]))
    tickets = rows[:, 0]
    assert torch.count_nonzero(tickets == 0.0).item() == 1
    assert torch.all((tickets == 0.0) | (tickets == final))


@pytest.mark.parametrize("dtype", _CAS16_DTYPES)
def test_atomic_cas_16bit_on_a_view_that_starts_mid_word(dtype):
    base = torch.full((20,), 3, dtype=dtype, device="mps")
    view = base[1:17]
    expected = torch.full((16,), 3, dtype=dtype, device="mps")
    expected[1::2] = 5  # every other compare misses
    desired = torch.arange(10, 26, dtype=torch.float32).to(dtype).to("mps")
    old = torch.empty(16, dtype=dtype, device="mps")
    _atomic_cas_16bit_tensor[(1,)](view, expected, desired, old, BLOCK=16)
    torch.mps.synchronize()
    want = torch.full((20,), 3.0)
    want[1:17] = torch.where(torch.arange(16) % 2 == 0, torch.arange(10, 26).float(), torch.tensor(3.0))
    assert torch.equal(base.cpu().float(), want)
    assert torch.equal(old.cpu().float(), torch.full((16,), 3.0))


# --- f16 atomic add ---------------------------------------------------------
#
# MSL has no atomic half, so `tl.atomic_add` on an f16 buffer was rejected. It
# is emulated with a compare-exchange loop over the 32-bit word that CONTAINS
# the element: read the pair, add into this element's lane, swap the pair back,
# retry if another thread got there first. The word is 4-byte aligned because
# the buffer base is and the index is masked down to an even element.
#
# bf16 uses the same word-level loop, but its helper adds on the f32 bit
# pattern and rounds back to nearest-even by hand. The `bfloat2` form of the
# half helper makes Apple's shader compiler service die with
# XPC_ERROR_CONNECTION_INTERRUPTED once the call sits inside another loop, which
# is what a BLOCK > tpb tile does; the bit-level form builds.


@triton.jit
def _atomic_add_f16_rows(In, Out, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    tl.atomic_add(Out + offsets, tl.load(In + offsets))


@pytest.mark.parametrize("num_warps", [1, 4])
@pytest.mark.parametrize("BLOCK", [8, 64, 256])
def test_atomic_add_f16_accumulates_across_programs(BLOCK, num_warps):
    inp = torch.ones(BLOCK, dtype=torch.float16, device="mps")
    out = torch.zeros(BLOCK, dtype=torch.float16, device="mps")
    _atomic_add_f16_rows[(4,)](inp, out, BLOCK, num_warps=num_warps)
    torch.mps.synchronize()
    assert torch.equal(out.cpu(), torch.full((BLOCK,), 4.0, dtype=torch.float16))


@triton.jit
def _atomic_add_f16_one_cell(In, Out, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    tl.atomic_add(Out + offsets * 0, tl.load(In + offsets))


@pytest.mark.parametrize("BLOCK", [8, 64])
def test_atomic_add_f16_contended_single_cell(BLOCK):
    # Every lane hits the same half, so this only comes out right if the
    # compare-exchange actually retries.
    inp = torch.ones(BLOCK, dtype=torch.float16, device="mps")
    out = torch.zeros(1, dtype=torch.float16, device="mps")
    _atomic_add_f16_one_cell[(1,)](inp, out, BLOCK)
    torch.mps.synchronize()
    assert out.cpu()[0].item() == float(BLOCK)


@triton.jit
def _atomic_add_f16_old(In, Out, Old, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    tl.store(Old + offsets, tl.atomic_add(Out + offsets, tl.load(In + offsets)))


def test_atomic_add_f16_returns_the_old_value():
    BLOCK = 64
    inp = torch.ones(BLOCK, dtype=torch.float16, device="mps")
    out = torch.full((BLOCK,), 3.0, dtype=torch.float16, device="mps")
    old = torch.zeros(BLOCK, dtype=torch.float16, device="mps")
    _atomic_add_f16_old[(1,)](inp, out, old, BLOCK)
    torch.mps.synchronize()
    assert torch.equal(old.cpu(), torch.full((BLOCK,), 3.0, dtype=torch.float16))
    assert torch.equal(out.cpu(), torch.full((BLOCK,), 4.0, dtype=torch.float16))


@triton.jit
def _atomic_add_16bit_view_old(In, Out, Old, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    tl.store(Old + offsets, tl.atomic_add(Out + offsets, tl.load(In + offsets)))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("start", [1, 2, 3])
def test_atomic_add_16bit_on_a_view_that_starts_mid_word(dtype, start):
    # `base[1:]` starts two bytes into a 32-bit word. The compare-exchange has to
    # find the word from the element's address: masking the INDEX to an even
    # element named a misaligned word and every add landed one element early,
    # writing base[0] and leaving the view's last element untouched.
    BLOCK = 64
    base = torch.full((BLOCK + 4,), 5.0, dtype=dtype, device="mps")
    view = base[start:start + BLOCK]
    old = torch.zeros(BLOCK, dtype=dtype, device="mps")
    values = torch.arange(1, BLOCK + 1, dtype=torch.float32).to(dtype)
    for _ in range(3):
        _atomic_add_16bit_view_old[(1,)](values.to("mps"), view, old, BLOCK)
    torch.mps.synchronize()
    expected = torch.full((BLOCK + 4,), 5.0)
    expected[start:start + BLOCK] += 3 * torch.arange(1, BLOCK + 1, dtype=torch.float32)
    assert torch.equal(base.cpu().float(), expected.to(dtype).float())
    assert torch.equal(old.cpu().float(), (5.0 + 2 * torch.arange(1, BLOCK + 1, dtype=torch.float32)).to(dtype).float())


@triton.jit
def _atomic_add_bf16_rows(In, Out, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    tl.atomic_add(Out + offsets, tl.load(In + offsets))


# BLOCK=1024 at one warp is 32 elements per thread: the tile loop that used to
# kill Apple's shader compiler service.
@pytest.mark.parametrize("num_warps", [1, 4])
@pytest.mark.parametrize("BLOCK", [8, 64, 256, 1024])
def test_atomic_add_bf16_accumulates_across_programs(BLOCK, num_warps):
    inp = torch.ones(BLOCK, dtype=torch.bfloat16, device="mps")
    out = torch.zeros(BLOCK, dtype=torch.bfloat16, device="mps")
    _atomic_add_bf16_rows[(4,)](inp, out, BLOCK, num_warps=num_warps)
    torch.mps.synchronize()
    assert torch.equal(out.cpu(), torch.full((BLOCK,), 4.0, dtype=torch.bfloat16))


@pytest.mark.parametrize("num_warps", [1, 4])
@pytest.mark.parametrize("BLOCK", [64, 256])
def test_atomic_add_bf16_contended_single_cell_returns_every_old_value(BLOCK, num_warps):
    # Every lane adds 1 to the same bf16 cell, so the old values must be exactly
    # 0..BLOCK-1 in some order (256 is still an exact bf16 integer).
    inp = torch.ones(BLOCK, dtype=torch.bfloat16, device="mps")
    out = torch.zeros(BLOCK, dtype=torch.bfloat16, device="mps")
    old = torch.full((BLOCK,), -1.0, dtype=torch.bfloat16, device="mps")
    _atomic_add_bf16_one_cell_old[(1,)](inp, out, old, BLOCK, num_warps=num_warps)
    torch.mps.synchronize()
    assert out.cpu()[0].item() == float(BLOCK)
    assert torch.equal(old.cpu().float().sort().values, torch.arange(BLOCK, dtype=torch.float32))


@triton.jit
def _atomic_add_bf16_one_cell_old(In, Out, Old, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    tl.store(Old + offsets, tl.atomic_add(Out + offsets * 0, tl.load(In + offsets)))


@triton.jit
def _atomic_add_bf16_cells_old(In, Out, Old, N, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    old = tl.atomic_add(Out + offsets, tl.load(In + offsets, mask=mask), mask=mask)
    tl.store(Old + offsets, old, mask=mask)


def test_atomic_add_bf16_rounds_like_torch():
    # One uncontended add per cell, so the result must be torch's own bf16 sum:
    # f32 add, round to nearest even. Ties (256 + 1, 256 + 3), carries into the
    # exponent, overflow to inf, inf - inf, NaN and signed zero are all covered.
    special = torch.tensor(
        [256.0, 256.0, 255.0, 3.0e38, float("inf"), float("nan"), -0.0, 0.0, -1.5, 1.0e-30],
        dtype=torch.bfloat16,
    )
    addend = torch.tensor(
        [1.0, 3.0, 1.0, 3.0e38, float("-inf"), 1.0, -0.0, -0.0, 1.5, 1.0e-30],
        dtype=torch.bfloat16,
    )
    gen = torch.Generator().manual_seed(0)
    base = torch.cat([special, (torch.randn(1000, generator=gen) * 100).to(torch.bfloat16)])
    vals = torch.cat([addend, (torch.randn(1000, generator=gen) * 100).to(torch.bfloat16)])
    n = base.numel()
    out = base.to("mps")
    old = torch.zeros(n, dtype=torch.bfloat16, device="mps")
    _atomic_add_bf16_cells_old[(triton.cdiv(n, 256),)](vals.to("mps"), out, old, n, BLOCK=256)
    torch.mps.synchronize()
    expected = (base.float() + vals.float()).to(torch.bfloat16)
    got = out.cpu()
    nan = expected.isnan()
    assert torch.equal(got.isnan(), nan)
    assert torch.equal(got[~nan].view(torch.int16), expected[~nan].view(torch.int16))
    assert torch.equal(old.cpu().view(torch.int16), base.view(torch.int16))


@triton.jit
def _atomic_add_rank3(
    Out,
    I,
    J,
    K,
    S0,
    S1,
    BI: tl.constexpr,
    BJ: tl.constexpr,
    BK: tl.constexpr,
):
    i = tl.arange(0, BI)[:, None, None]
    j = tl.arange(0, BJ)[None, :, None]
    k = tl.arange(0, BK)[None, None, :]
    offsets = i * S0 + j * S1 + k
    mask = (i < I) & (j < J) & (k < K)
    tl.atomic_add(Out + offsets, tl.full((BI, BJ, BK), 1, tl.int32), mask=mask)


@triton.jit
def _atomic_add_rank4(Out, S0, S1, S2):
    i = tl.arange(0, 2)[:, None, None, None]
    j = tl.arange(0, 4)[None, :, None, None]
    k = tl.arange(0, 4)[None, None, :, None]
    l = tl.arange(0, 8)[None, None, None, :]
    offsets = i * S0 + j * S1 + k * S2 + l
    tl.atomic_add(Out + offsets, tl.full((2, 4, 4, 8), 1, tl.int32))


@triton.jit
def _atomic_add_rank5(Out, S0, S1, S2, S3):
    i = tl.arange(0, 2)[:, None, None, None, None]
    j = tl.arange(0, 2)[None, :, None, None, None]
    k = tl.arange(0, 4)[None, None, :, None, None]
    l = tl.arange(0, 4)[None, None, None, :, None]
    m = tl.arange(0, 8)[None, None, None, None, :]
    offsets = i * S0 + j * S1 + k * S2 + l * S3 + m
    tl.atomic_add(Out + offsets, tl.full((2, 2, 4, 4, 8), 1, tl.int32))


@pytest.mark.parametrize("num_warps", [1, 4])
@pytest.mark.parametrize(
    "shape,block",
    [((2, 3, 5), (2, 4, 8)), ((4, 8, 16), (4, 8, 16))],
)
def test_atomic_add_tensor_rank3_strided_masked(shape, block, num_warps):
    strides = (shape[1] * shape[2] + 7, shape[2] + 3, 1)
    storage = sum((size - 1) * stride for size, stride in zip(shape, strides)) + 1
    programs = 3
    initial = torch.full((storage,), 77, dtype=torch.int32)
    out = initial.to("mps")

    _atomic_add_rank3[(programs,)](
        out,
        *shape,
        strides[0],
        strides[1],
        BI=block[0],
        BJ=block[1],
        BK=block[2],
        num_warps=num_warps,
    )

    expected = initial.clone()
    expected.as_strided(shape, strides)[:] += programs
    assert torch.equal(out.cpu(), expected)


def test_atomic_add_tensor_rank4_strided():
    shape = (2, 4, 4, 8)
    strides = (173, 43, 11, 1)
    storage = sum((size - 1) * stride for size, stride in zip(shape, strides)) + 1
    programs = 3
    initial = torch.full((storage,), 91, dtype=torch.int32)
    out = initial.to("mps")

    _atomic_add_rank4[(programs,)](out, *strides[:3])

    expected = initial.clone()
    expected.as_strided(shape, strides)[:] += programs
    assert torch.equal(out.cpu(), expected)


def test_atomic_add_tensor_rank5_strided():
    shape = (2, 2, 4, 4, 8)
    strides = (401, 197, 53, 11, 1)
    storage = sum((size - 1) * stride for size, stride in zip(shape, strides)) + 1
    programs = 3
    initial = torch.full((storage,), 53, dtype=torch.int32)
    out = initial.to("mps")

    _atomic_add_rank5[(programs,)](out, *strides[:4])

    expected = initial.clone()
    expected.as_strided(shape, strides)[:] += programs
    assert torch.equal(out.cpu(), expected)


# A scalar store or atomic is one effect per program, but every register band
# of a tile runs the kernel body again. With more elements than threads the
# function's band loop repeated each scalar effect once per band: a counter
# beside a 256-element tile at one warp went up by 8, not 1, at top level and
# inside a user loop alike, and `tl.store(p, tl.load(p) + 1)` did the same.
@triton.jit
def _scalar_effect_beside_tile(X, O, C, T, N: tl.constexpr, FORM: tl.constexpr):
    i = tl.arange(0, N)
    x = tl.load(X + i)
    if FORM == "top":
        tl.atomic_add(C, 1.0)
        tl.store(O + i, x)
    elif FORM == "loop":
        for t in range(T):
            tl.atomic_add(C, 1.0)
            tl.store(O + t * N + i, x)
    elif FORM == "while":
        t = 0
        while t < T:
            tl.atomic_add(C, 2.0)
            tl.store(O + t * N + i, x)
            t += 1
    elif FORM == "xchg":
        for t in range(T):
            tl.atomic_xchg(C, t * 1.0)
            tl.store(O + t * N + i, x)
    elif FORM == "rmw":
        for t in range(T):
            tl.store(C, tl.load(C) + tl.sum(x * (t + 1), 0))
            tl.store(O + t * N + i, x)
    elif FORM == "old":
        old = tl.atomic_add(C, 1.0)
        tl.store(O + i, x + old)


@pytest.mark.parametrize("form", ["top", "loop", "while", "xchg", "rmw", "old"])
@pytest.mark.parametrize("N, num_warps", [(64, 1), (256, 1), (256, 2), (1024, 4), (64, 2)])
def test_scalar_effect_runs_once_per_program_across_bands(form, N, num_warps):
    steps = 3
    torch.manual_seed(N + num_warps)
    x = torch.randn(N)
    counter = torch.full((1,), 5.0, device="mps")
    out = torch.zeros(steps * N, device="mps")
    _scalar_effect_beside_tile[(1,)](x.to("mps"), out, counter, steps, N, form,
                                     num_warps=num_warps)
    torch.mps.synchronize()
    stored = 1 if form in ("top", "old") else steps
    want_out = torch.cat([x] * stored)
    if form == "old":
        want_out = want_out + 5.0
    torch.testing.assert_close(out.cpu()[:stored * N], want_out, atol=1e-4, rtol=1e-4)
    want_counter = {
        "top": 6.0, "old": 6.0, "loop": 5.0 + steps, "while": 5.0 + 2 * steps,
        "xchg": float(steps - 1), "rmw": 5.0 + x.sum().item() * sum(range(1, steps + 1)),
    }[form]
    assert counter.item() == pytest.approx(want_counter, rel=1e-4, abs=1e-3)


@triton.jit
def _scalar_old_value_in_banded_loop(X, O, C, T, N: tl.constexpr):
    i = tl.arange(0, N)
    x = tl.load(X + i)
    for t in range(T):
        old = tl.atomic_add(C, 1.0)
        tl.store(O + t * N + i, x + old)


def test_scalar_atomic_old_value_in_banded_loop_is_named(capfd):
    # Band 0 runs every iteration of the user loop the band loop encloses, so a
    # fetched value published once per band cannot reach the later bands.
    x = torch.randn(256, device="mps")
    out = torch.zeros(3 * 256, device="mps")
    counter = torch.zeros(1, device="mps")
    with pytest.raises(Exception, match="convert-tritongpu-to-metal failed"):
        _scalar_old_value_in_banded_loop[(1,)](x, out, counter, 3, 256, num_warps=1)
    diagnostic = capfd.readouterr()
    assert "a scalar atomic whose old value is used cannot sit in a loop" in (
        diagnostic.err + diagnostic.out)
    # At one element per thread there is no band loop, and it is exact.
    out = torch.zeros(3 * 64, device="mps")
    counter = torch.zeros(1, device="mps")
    _scalar_old_value_in_banded_loop[(1,)](x[:64].contiguous(), out, counter, 3, 64,
                                           num_warps=2)
    torch.mps.synchronize()
    want = torch.cat([x[:64].cpu() + t for t in range(3)])
    torch.testing.assert_close(out.cpu(), want)
    assert counter.item() == 3.0
