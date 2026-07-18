"""End-to-end LSD radix sort on the Metal backend (leet-triton hard-radix_sort).

Covers the constructs this kernel pair was the first to need, each of which was
a separate hard blocker:

  * `tl.cast(<i1 predicate>, tl.int32)`            -> arith.extui
  * `tl.sum` over a `(v >> shift) & 0xF` cone      -> reduce cone with shifts
  * that same cone containing an integer compare   -> signless/ui32 reconciliation
  * `tl.cumsum` on i32                             -> integer tt.scan
  * 16 unrolled cumsums in one kernel              -> pooled scan buffers (32 KB
                                                      threadgroup budget)
  * `tl.store(dst + idx_tensor, v, mask=...)`      -> masked scatter through a
                                                      ttg.convert_layout relabel

The ragged sizes matter: a scatter whose mask is taken from the store's
destination layout instead of being re-derived at the source index masks the
wrong lanes, and that is invisible whenever N is an exact multiple of BLOCK.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import triton
import triton.language as tl

libmetal = pytest.importorskip(
    "triton._C.libtriton.metal",
    reason="Metal backend pybind module not built into libtriton",
)
if not torch.backends.mps.is_available():
    pytest.skip(
        "Metal backend requires an MPS-enabled PyTorch (Apple Silicon)",
        allow_module_level=True,
    )

BLOCK_SIZE = 1024


@triton.jit
def count_kernel(src_ptr, count_ptr, N, shift, num_blocks, BLOCK_SIZE: tl.constexpr):
    src_ptr = src_ptr.to(tl.pointer_type(tl.uint32))

    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    vals = tl.load(src_ptr + offsets, mask=mask, other=0)

    digits = (vals >> shift) & 0xF
    digits = tl.where(mask, digits, 16)

    for b in tl.static_range(16):
        is_b = (digits == b)
        sum_b = tl.sum(tl.cast(is_b, tl.int32))
        tl.store(count_ptr + b * num_blocks + pid, sum_b)


@triton.jit
def scatter_kernel(src_ptr, dst_ptr, offset_ptr, N, shift, num_blocks,
                   BLOCK_SIZE: tl.constexpr):
    src_ptr = src_ptr.to(tl.pointer_type(tl.uint32))
    dst_ptr = dst_ptr.to(tl.pointer_type(tl.uint32))

    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    vals = tl.load(src_ptr + offsets, mask=mask, other=0)
    digits = (vals >> shift) & 0xF
    digits = tl.where(mask, digits, 16)

    loc_offsets = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
    glob_offsets = tl.zeros([BLOCK_SIZE], dtype=tl.int32)

    for b in tl.static_range(16):
        is_b = (digits == b)
        is_b_int = tl.cast(is_b, tl.int32)

        cum_b = tl.cumsum(is_b_int)
        loc_offsets = tl.where(is_b, cum_b - is_b_int, loc_offsets)

        g_off = tl.load(offset_ptr + b * num_blocks + pid)
        glob_offsets = tl.where(is_b, g_off, glob_offsets)

    write_idx = glob_offsets + loc_offsets
    tl.store(dst_ptr + write_idx, vals, mask=mask)


def _radix_sort(inp: torch.Tensor, out: torch.Tensor, N: int) -> None:
    num_blocks = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    buf = torch.empty_like(inp, memory_format=torch.contiguous_format)
    counts = torch.empty((16, num_blocks), dtype=torch.int32, device=inp.device)
    global_offsets = torch.empty_like(counts)

    for pass_idx in range(8):
        shift = pass_idx * 4
        if pass_idx == 0:
            src, dst = inp, buf
        elif pass_idx % 2 == 1:
            src, dst = buf, out
        else:
            src, dst = out, buf

        count_kernel[(num_blocks,)](src, counts, N, shift, num_blocks,
                                    BLOCK_SIZE=BLOCK_SIZE)

        counts_flat = counts.view(-1)
        global_offsets_flat = global_offsets.view(-1)
        global_offsets_flat[0] = 0
        if counts_flat.numel() > 1:
            global_offsets_flat[1:] = torch.cumsum(counts_flat[:-1], dim=0)

        scatter_kernel[(num_blocks,)](src, dst, global_offsets, N, shift,
                                      num_blocks, BLOCK_SIZE=BLOCK_SIZE)


# 1024 == exactly one full tile; 12345 spans 13 programs with a ragged tail.
@pytest.mark.parametrize("N", [1024, 4096, 100, 1500, 5000, 12345])
def test_radix_sort_matches_torch_sort(N):
    torch.manual_seed(0x5A17)
    inp = torch.randint(0, 2**31 - 1, (N,), dtype=torch.int32)
    out = torch.zeros_like(inp)

    _radix_sort(inp, out, N)

    expected, _ = torch.sort(inp.to(torch.int64))
    torch.testing.assert_close(out.to(torch.int64), expected, rtol=0, atol=0)


def test_radix_count_kernel_histogram_exact():
    """The count phase alone: a per-digit `tl.sum` over a shift/mask cone."""
    N = 4096
    num_blocks = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    shift = 4

    torch.manual_seed(0x5A17)
    src = torch.randint(0, 2**31 - 1, (N,), dtype=torch.int32)
    counts = torch.zeros((16, num_blocks), dtype=torch.int32)

    count_kernel[(num_blocks,)](src, counts, N, shift, num_blocks,
                                BLOCK_SIZE=BLOCK_SIZE)

    digits = (src.to(torch.int64) >> shift) & 0xF
    expected = torch.zeros((16, num_blocks), dtype=torch.int64)
    for b in range(num_blocks):
        lo, hi = b * BLOCK_SIZE, min((b + 1) * BLOCK_SIZE, N)
        for d in range(16):
            expected[d, b] = (digits[lo:hi] == d).sum()

    torch.testing.assert_close(counts.to(torch.int64), expected, rtol=0, atol=0)


@pytest.mark.parametrize("ndig", [1, 2, 16])
def test_scan_buffer_pooling_local_rank(ndig):
    """`ndig` unrolled i32 cumsums sharing one pooled threadgroup buffer pair.

    ndig=1 takes the private-allocation path; ndig>1 pools. Regression guard for
    the read-after-overwrite this exposed: the emitter inlines a single-use
    value at its USE site, which floated each scan's placeholder read past the
    next scan's refill, so every arm read the last scan's prefix sums.
    """

    @triton.jit
    def locrank(src, out, shift, NDIG: tl.constexpr, BLOCK: tl.constexpr):
        o = tl.arange(0, BLOCK)
        v = tl.load(src + o)
        digits = (v >> shift) & 0xF
        loc = tl.zeros([BLOCK], dtype=tl.int32)
        for b in tl.static_range(NDIG):
            is_b = digits == b
            is_b_int = tl.where(is_b, 1, 0)
            cum_b = tl.cumsum(is_b_int)
            loc = tl.where(is_b, cum_b - is_b_int, loc)
        tl.store(out + o, loc)

    BLOCK = 1024
    shift = 4
    torch.manual_seed(0x5A17)
    src = torch.randint(0, 2**31 - 1, (BLOCK,), dtype=torch.int32)
    out = torch.zeros(BLOCK, dtype=torch.int32)

    locrank[(1,)](src, out, shift, NDIG=ndig, BLOCK=BLOCK)

    d = (src.to(torch.int64) >> shift) & 0xF
    expected = torch.zeros(BLOCK, dtype=torch.int64)
    for b in range(ndig):
        is_b = (d == b).to(torch.int64)
        expected = torch.where(d == b, torch.cumsum(is_b, 0) - is_b, expected)

    torch.testing.assert_close(out.to(torch.int64), expected, rtol=0, atol=0)
