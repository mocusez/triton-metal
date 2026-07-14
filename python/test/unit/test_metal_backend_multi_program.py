"""Multi-program GPU correctness: grid > (1, 1, 1).

Acceptance test for `.omc/specs/deep-interview-metal-pid-lowering.md`
(AC.P5). Confirms that `tt.get_program_id x` lowers to a real
`metal.threadgroup_id "x"` that yields the correct per-threadgroup
program id on Apple GPU. With 8 programs each handling 128 elements,
the full 1024-element vector_add must produce bit-exact output.
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


@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)


def test_multi_program_vector_add_grid8():
    BLOCK_SIZE = 128
    N = 1024  # 1024 / 128 = 8 programs, each handling 128 contiguous elements
    grid = (N // BLOCK_SIZE, 1, 1)  # (8, 1, 1)

    torch.manual_seed(0xC0FFEE)
    x = torch.rand(N, dtype=torch.float32)
    y = torch.rand(N, dtype=torch.float32)
    out = torch.zeros(N, dtype=torch.float32)

    add_kernel[grid](x, y, out, N, BLOCK_SIZE=BLOCK_SIZE)

    expected = x + y
    # Full N range: every program contributes its 128-elem block; mask
    # is `offsets < N` and N == grid[0]*BLOCK_SIZE so it's always true.
    torch.testing.assert_close(out, expected, atol=0, rtol=0)


@triton.jit
def _per_row_masked_copy_kernel(X, Y, N, BLOCK: tl.constexpr):
    # Per-row masked elementwise load/store: each program (grid=(M,)) handles one
    # row X[row, :], masking the padded columns. The mask index is the PER-ROW
    # column (`cols < N`, no program offset) — distinct from vector-add's
    # global-flat `pid*BLOCK+arange < n`.
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * N + cols, mask=mask, other=0.0)
    tl.store(Y + row * N + cols, x + 1.0, mask=mask)


@pytest.mark.parametrize("M, N", [(1, 100), (4, 100), (8, 300), (3, 700)])
def test_per_row_masked_copy_multiprogram(M, N):
    # Regression: the masked-load mask used a GLOBAL flat index (id.x < N) even
    # for per-row masks, so every program k>=1 (id.x = k*tpb+local >= N) masked
    # out its whole row and read 0. Now the mask reads the actual per-row cone.
    torch.manual_seed(M * 100 + N)
    x = torch.randn((M, N), dtype=torch.float32, device="mps")
    y = torch.zeros((M, N), dtype=torch.float32, device="mps")
    _per_row_masked_copy_kernel[(M,)](x, y, N, BLOCK=triton.next_power_of_2(N))
    torch.mps.synchronize()
    torch.testing.assert_close(y.cpu(), (x + 1.0).cpu(), atol=0, rtol=0)
