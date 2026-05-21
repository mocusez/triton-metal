"""Parametrized matrix test: sizePerThread x mask coverage for the Metal backend.

Locks the regression matrix that the previous ralph identified as a CI coverage
gap (sizePerThread x mask was tested as independent axes but never crossed). The
8 1D rows exercise the AC4-v6 fix surface from the prior ralph (BLOCK_SIZE
auto-selects sizePerThread via the Triton frontend's num_warps split, so the
1024+mask case is the exact TTGIR shape that triggered the original failure).
The 2 2D rows lock the AC4-v6 follow-up's 2D MSL-translate ptr-cast bug
(currently `@pytest.mark.xfail(strict=True)` until that fix lands).

See `.omc/plans/metal-deferred-followups.md` AC8.
"""

import platform

import pytest
import torch

import triton
import triton.language as tl


pytestmark = pytest.mark.skipif(
    not (platform.system() == "Darwin" and platform.machine() == "arm64"),
    reason="Metal backend tests run only on Apple Silicon.",
)


# --- 1D kernel under test --------------------------------------------------------


@triton.jit
def _add_1d_kernel(
    x_ptr, y_ptr, out_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr, USE_MASK: tl.constexpr, USE_OTHER: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    if USE_MASK:
        mask = offsets < n_elements
        if USE_OTHER:
            x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
            y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
        else:
            x = tl.load(x_ptr + offsets, mask=mask)
            y = tl.load(y_ptr + offsets, mask=mask)
        tl.store(out_ptr + offsets, x + y, mask=mask)
    else:
        x = tl.load(x_ptr + offsets)
        y = tl.load(y_ptr + offsets)
        tl.store(out_ptr + offsets, x + y)


# --- 2D kernel under test --------------------------------------------------------


@triton.jit
def _add_2d_kernel(
    x_ptr, out_ptr, M, N,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, USE_OTHER: tl.constexpr,
):
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    mask_m = offs_m[:, None] < M
    mask_n = offs_n[None, :] < N
    mask = mask_m & mask_n
    offs = offs_m[:, None] * N + offs_n[None, :]
    if USE_OTHER:
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    else:
        x = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x, mask=mask)


# --- 1D parametrizations (8 + 1 = 9 rows) ----------------------------------------


@pytest.mark.parametrize(
    "block_size,num_warps,use_mask,use_other",
    [
        # spt=1 (BLOCK_SIZE=128 with num_warps=4 → 128/(4*32)=1 elem per thread)
        (128,  4, False, False),
        (128,  4, True,  False),
        # spt=2 (BLOCK_SIZE=256, num_warps=4 → 2 elem per thread)
        (256,  4, False, False),
        (256,  4, True,  False),
        # spt=4 (BLOCK_SIZE=1024, num_warps=4 → 8 elem per thread; spt=[4] per layout)
        (1024, 4, False, False),
        (1024, 4, True,  False),
        # spt=4 + other=0.0 — the original-bug TTGIR shape from the prior ralph
        (1024, 4, True,  True),
        # spt=8 (BLOCK_SIZE=2048, num_warps=4 → 16 elem per thread; spt=[8])
        (2048, 4, False, False),
    ],
    ids=[
        "spt1_unmasked", "spt1_masked",
        "spt2_unmasked", "spt2_masked",
        "spt4_unmasked", "spt4_masked",
        "spt4_masked_other0",
        "spt8_unmasked",
    ],
)
def test_vector_add_1d_matrix(block_size, num_warps, use_mask, use_other):
    """1D vector add across `sizePerThread x mask` matrix.

    Uses N = block_size + 13 (non-multiple) so the mask path is exercised when
    `use_mask=True`. With `use_mask=False`, N == block_size exactly.
    """
    n = block_size + (13 if use_mask else 0)
    x = torch.rand(n, device="cpu", dtype=torch.float32)
    y = torch.rand(n, device="cpu", dtype=torch.float32)
    out = torch.empty_like(x)
    grid = (triton.cdiv(n, block_size),)
    _add_1d_kernel[grid](
        x, y, out, n,
        BLOCK_SIZE=block_size, USE_MASK=use_mask, USE_OTHER=use_other,
        num_warps=num_warps,
    )
    expected = x + y
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)


# --- 2D parametrizations (2 rows) ------------------------------------------------
# Note: the AC4-v6 DCE narrowing applied in this same ralph (Step 3, US-002)
# transitively fixed the Python-frontend 2D pathway — the previously-anticipated
# `i32→!tt.ptr<f32>→ui32` round-trip cast emission no longer blocks this surface
# end-to-end. A hand-crafted-IR edge case still emits the cast inside scf.if
# regions (see `.omc/specs/deep-interview-metal-2d-msl-translate-ptr.md`), but
# real-frontend kernels lower cleanly. xfail markers removed (US-001 (1e)).


@pytest.mark.parametrize(
    "M,N,BLOCK_M,BLOCK_N",
    [(8, 128, 8, 128), (16, 256, 16, 256)],
    ids=["2d_8x128", "2d_16x256"],
)
def test_vector_add_2d_matrix(M, N, BLOCK_M, BLOCK_N):
    """2D vector copy across 2D-AND mask path; locks the AC4-v6 follow-up bug."""
    x = torch.rand(M * N, device="cpu", dtype=torch.float32)
    out = torch.empty_like(x)
    _add_2d_kernel[(1,)](
        x, out, M, N,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, USE_OTHER=False,
        num_warps=8,
    )
    torch.testing.assert_close(out, x, rtol=1e-5, atol=1e-5)
