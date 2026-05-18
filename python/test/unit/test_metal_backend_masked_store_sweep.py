"""L1d2c Phase B AC.T3 sweep: parametrized (BLOCK_N, num_warps) over a
masked-cvt kernel.

Per `.omc/specs/deep-interview-leet-triton-l1d2c-phase-b-fix.md` AC.T3:
sweep (BLOCK_N, num_warps) ∈ {(8,2), (16,8), (32,4)} over a masked-cvt
kernel; each case PASS bit-exact ×10 runs.

HONEST DIVERGENCE (per spec §"Reporting expectations" item 6): the
Apple Metal MSL `tg_load_indexed` lane-aliasing miscompile that
motivates L1d2c Phase B is NOT eliminated by MaskedStoreLowering's
scratch-sentinel + value-select rewrite (verified empirically; see the
file-level docstring in `test_metal_backend_transpose.py` and the
canary lit fixture
`test/Dialect/Metal/convert-tritongpu-to-metal/masked_store_unconditional.mlir`).
The sweep cases are therefore marked `xfail(strict=False)` rather than
expected-pass; AC.T3 is unmet and surfaced as honest divergence.

These sweep cases stay in the test suite so that any future fix landing
upstream (Apple compiler update, MSL emitter workaround, or
`ConvertLayoutLowering` cvt-body redesign) will flip them from xfail to
pass automatically.

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
if not hasattr(libmetal, "launch_kernel_with_pipeline"):
    pytest.skip(
        "Metal runtime not compiled (non-Darwin build or Xcode CLT absent)",
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


@pytest.mark.xfail(
    reason=(
        "L1d2c Phase B honest divergence: Apple Metal MSL "
        "`tg_load_indexed` lane-aliasing miscompile persists after "
        "MaskedStoreLowering's scratch-sentinel + value-select rewrite. "
        "See `test_metal_backend_transpose.py` file docstring and the "
        "Phase B fix-locus comment in "
        "`third_party/metal/lib/Conversion/TritonGPUToMetal/TritonGPUToMetal.cpp`."
    ),
    strict=False,
)
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
