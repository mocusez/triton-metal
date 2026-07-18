"""L1d2c Phase B AC.T3 sweep: parametrized (BLOCK_N, num_warps) over a
masked-cvt kernel.

Per `.omc/specs/deep-interview-leet-triton-l1d2c-phase-b-fix.md` AC.T3:
sweep (BLOCK_N, num_warps) ∈ {(8,2), (16,8), (32,4)} over a masked-cvt
kernel; each case PASS bit-exact ×10 runs.

RESOLVED 2026-06-03 (XPASS disposition): the in-envelope cases (8x8_nw2,
16x16_nw8; single-threadgroup, sizePerThread=[1,1]) now pass bit-exact and
deterministically (30/30) — the lowering no longer emits the cross-lane
`tg_load_indexed` reload that triggered the Apple Metal lane-aliasing
miscompile, so AC.T3 is met for these shapes. They are plain-pass and
retained as regression guards.

RESOLVED 2026-06-23 (32x32_nw4): the BLOCK>tpb (E=8, sizePerThread>1) case
now passes bit-exact. The rank-2 producer-cone normalization
(`normalizeBlockedDivergentCvts` in TritonGPUToMetal.cpp) rewrites the load
cone from the row-major #blocked to the column-major #blocked1 store layout,
collapsing the transpose `ttg.convert_layout` to an identity (direct
gather/scatter) instead of routing it through the deferred L1d3 threadgroup
staging path. No runtime staging, no cross-lane reload.

The kernel matches the canonical Triton matmul-pre-transpose shape:
load `tile` from input, transpose via implicit ttg.convert_layout, store
to output with the same mask. For (BLOCK_N, num_warps) ∈ {(8,2),
(16,8), (32,4)} the staged-transpose envelope is exercised:

  * (8, 2)  → 64 threads, sizePerThread=[1,1] on both sides — in-envelope.
  * (16, 8) → 256 threads, sizePerThread=[1,1] — in-envelope (matches the
              L1d2b masked transpose anchor).
  * (32, 4) → 128 threads, sizePerThread=[1,1] (32*32=1024 / 128 = E=8;
              elem_per_thread > 1 → tile loop wrapped). This case lands
              OUTSIDE the strict [1,1] envelope for E==1 — the tile loop
              wraps the body but the cvt staging body still fires on each
              iteration. Phase B's rewrite is exercised once per iteration.
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

N_REPEATS = 10


@triton.jit
def _masked_transpose_sweep(
    input_ptr,
    output_ptr,
    rows,
    cols,
    sir,
    sic,
    sor,
    soc,
    BLOCK_N: tl.constexpr,
):
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    x = pid_x * BLOCK_N + tl.arange(0, BLOCK_N)
    y = pid_y * BLOCK_N + tl.arange(0, BLOCK_N)
    xi = x[None, :]
    yi = y[:, None]
    mask = (xi < cols) & (yi < rows)
    tile = tl.load(input_ptr + yi * sir + xi * sic, mask=mask, other=0.0)
    tl.store(output_ptr + xi * sor + yi * soc, tile, mask=mask)


# RESOLVED 2026-06-03 (XPASS disposition): the L1d2c lane-aliasing
# miscompile no longer reproduces for the single-threadgroup
# (sizePerThread=[1,1], E=1) shapes — they now lower to a direct
# address-arithmetic gather/scatter with no cross-lane `tg_load_indexed`
# reload, and pass bit-exact deterministically (30/30). 8x8_nw2 and
# 16x16_nw8 are flipped to plain pass.
# RESOLVED 2026-06-23 (32x32_nw4): the BLOCK>tpb (E=8, spt>1) case is also a
# plain pass now — `normalizeBlockedDivergentCvts` rewrites the load cone to
# the store's #blocked1 layout, collapsing the transpose cvt to a direct
# gather/scatter (rank-2 generalization of the rank-1 divergent-cvt normalize),
# so it no longer hits the L1d3 staged-transpose gate.
@pytest.mark.parametrize(
    "BLOCK_N,num_warps",
    [
        pytest.param(8, 2, id="8x8_nw2"),
        pytest.param(16, 8, id="16x16_nw8"),
        pytest.param(32, 4, id="32x32_nw4"),
    ],
)
def test_masked_store_sweep_bit_exact(BLOCK_N, num_warps):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    if device == "cpu":
        pytest.skip("MPS device required to launch metal kernels")

    for iteration in range(N_REPEATS):
        torch.manual_seed(0xC0FFEE + iteration)
        rows = BLOCK_N
        cols = BLOCK_N
        inp = (
            torch.arange(rows * cols, dtype=torch.float32, device=device)
            .reshape(rows, cols)
            .contiguous()
        )
        out = torch.zeros(cols, rows, dtype=torch.float32, device=device).contiguous()
        grid = (triton.cdiv(cols, BLOCK_N), triton.cdiv(rows, BLOCK_N))
        _masked_transpose_sweep[grid](
            inp,
            out,
            rows,
            cols,
            cols,
            1,
            rows,
            1,
            BLOCK_N,
            num_warps=num_warps,
        )
        expected = inp.t().contiguous()
        assert torch.equal(out, expected), (
            f"masked-cvt sweep BLOCK_N={BLOCK_N} num_warps={num_warps} "
            f"iteration={iteration}/{N_REPEATS} max-abs-err="
            f"{(out - expected).abs().max().item()}\n"
            f"out=\n{out.cpu()}\nexpected=\n{expected.cpu()}"
        )


# --- Sub-tile threads clobbering another program's rows ---------------------
#
# A store mask is a GLOBAL bound (`pid*BLOCK + arange < n_rows`), not a per-tile
# one. When the stored tensor has fewer elements than the threadgroup has
# threads, the threads PAST the tile still satisfy that bound and store a value
# belonging to some other row — overwriting the output of whichever program owns
# that row. With BLOCK=32, tpb=128 and n_rows=64 over two programs, program 0's
# threads 32..63 pass `localTid < 64` and clobber program 1's rows.
#
# `StoreLowering` has always emitted a `localTid < numElements` guard for this;
# `MaskedStoreLowering` did not, relying on the user mask alone. Invisible
# whenever the mask bound coincides with the tile — single-program launches, or
# n_rows == BLOCK — which is why the sweeps above stayed green.


@triton.jit
def _masked_store_subtile_kernel(x_ptr, y_ptr, n_rows, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    v = tl.load(x_ptr + off, mask=off < n_rows, other=0.0)
    tl.store(y_ptr + off, v * 2.0, mask=off < n_rows)


@pytest.mark.parametrize("BLOCK, n_rows, num_warps", [
    (32, 64, 4),    # tpb=128, tile=32: threads 32..63 pass a 64-wide mask
    (32, 128, 4),   # four programs
    (16, 64, 4),    # tile 16 of 128
    (32, 64, 1),    # tpb=32 == tile: guard is a no-op, must stay correct
    (64, 256, 8),   # tpb=256, tile=64
])
def test_masked_store_subtile_no_cross_program_clobber(BLOCK, n_rows, num_warps):
    torch.manual_seed(BLOCK * 1000 + n_rows)
    x = torch.randn(n_rows, dtype=torch.float32, device="mps")
    y = torch.zeros(n_rows, dtype=torch.float32, device="mps")
    grid = (triton.cdiv(n_rows, BLOCK),)
    _masked_store_subtile_kernel[grid](x, y, n_rows, BLOCK, num_warps=num_warps)
    torch.mps.synchronize()
    assert torch.equal(y.cpu(), (x * 2.0).cpu()), (
        f"BLOCK={BLOCK} n_rows={n_rows} num_warps={num_warps}: "
        f"max err {(y - x * 2.0).abs().max().item()}"
    )
