"""A tile loop runs as many trips as the LARGEST tile in the kernel needs.

An op whose own tile holds fewer elements per thread is asked for more
iterations than it has elements, and the per-iteration index keeps counting --
so those trips address past the end of that op's own tensor. The Metal backend
binds tensors ZERO-COPY, a host tensor IS the device buffer, so such a write
lands in whatever live allocation follows it.

The window is exactly `1 < E_op < E_loop`. Below it (E_op <= 1) the index
ignores the iteration and every trip rewrites the same slot; at it, the indices
cover the tensor exactly. In between, nothing stopped them: the guard that
existed covered only the sub-threadgroup case, and its comment asserted that a
tile loop made the rest safe -- true only while the loop is sized by the op's
own tile.

Each test writes through a VIEW into a larger buffer whose tail is a canary, so
the assertion is about memory outside the tensor rather than about the values
inside it, which were correct all along.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import triton  # noqa: E402
import triton.language as tl  # noqa: E402

pytest.importorskip(
    "triton._C.libtriton.metal",
    reason="Metal backend pybind module not built into libtriton",
)
if not torch.backends.mps.is_available():
    pytest.skip(
        "Metal backend requires an MPS-enabled PyTorch (Apple Silicon)",
        allow_module_level=True,
    )

CANARY = -7.0


def _canary_view(elements, spare=4096):
    big = torch.full((elements + spare,), CANARY, device="mps")
    return big, big[:elements]


def _assert_untouched(big, elements):
    tail = big[elements:].cpu()
    clobbered = int((tail != CANARY).sum())
    assert clobbered == 0, (
        f"{clobbered} elements past the end of the tensor were written; "
        f"first at {int((tail != CANARY).nonzero()[0]) + elements}")


@triton.jit
def _wide_tile_and_narrow_vector(x_ptr, y_ptr, tile_ptr, vec_ptr,
                                 M: tl.constexpr, N: tl.constexpr):
    rows = tl.arange(0, M)[:, None]
    cols = tl.arange(0, N)[None, :]
    tile = tl.load(x_ptr + rows * N + cols)
    tl.store(tile_ptr + rows * N + cols, tile * 2.0)
    vec = tl.load(y_ptr + tl.arange(0, N))
    tl.store(vec_ptr + tl.arange(0, N), vec + 1.0)


@pytest.mark.parametrize("M,N,num_warps", [(8, 128, 1), (4, 64, 1),
                                           (8, 256, 4), (16, 128, 2)])
def test_narrow_vector_store_stays_inside_its_tensor(M, N, num_warps):
    """The [M, N] tile sizes the loop; the [N] vector has M times fewer
    elements per thread and used to keep writing for the remaining trips."""
    torch.manual_seed(M * 31 + N)
    x = torch.randn(M, N, device="mps")
    y = torch.randn(N, device="mps")
    out_tile = torch.zeros(M, N, device="mps")
    big, out_vec = _canary_view(N)
    _wide_tile_and_narrow_vector[(1,)](x, y, out_tile, out_vec, M=M, N=N,
                                       num_warps=num_warps)
    torch.mps.synchronize()
    torch.testing.assert_close(out_vec.cpu(), y.cpu() + 1.0, atol=1e-6,
                               rtol=1e-6)
    torch.testing.assert_close(out_tile.cpu(), x.cpu() * 2.0, atol=1e-6,
                               rtol=1e-6)
    _assert_untouched(big, N)


@triton.jit
def _wide_tile_and_uniform_masked_vector(x_ptr, y_ptr, tile_ptr, vec_ptr, live,
                                         M: tl.constexpr, N: tl.constexpr):
    rows = tl.arange(0, M)[:, None]
    cols = tl.arange(0, N)[None, :]
    tile = tl.load(x_ptr + rows * N + cols)
    tl.store(tile_ptr + rows * N + cols, tile * 2.0)
    offs = tl.arange(0, N)
    vec = tl.load(y_ptr + offs)
    tl.store(vec_ptr + offs, vec + 1.0, mask=live > 0)


def test_uniform_masked_narrow_vector_store_stays_inside_its_tensor():
    """A user mask does not subsume the guard in general. The common
    `arange < bound` shape happens to: past the tile the index is already over
    the bound, which is why the masked form did NOT reproduce this. A mask that
    does not read the element index -- here a uniform flag -- leaves nothing to
    stop the extra trips."""
    torch.manual_seed(5)
    M, N = 8, 128
    x = torch.randn(M, N, device="mps")
    y = torch.randn(N, device="mps")
    out_tile = torch.zeros(M, N, device="mps")
    big, out_vec = _canary_view(N)
    _wide_tile_and_uniform_masked_vector[(1,)](x, y, out_tile, out_vec, 1, M=M,
                                               N=N, num_warps=1)
    torch.mps.synchronize()
    torch.testing.assert_close(out_vec.cpu(), y.cpu() + 1.0, atol=1e-6,
                               rtol=1e-6)
    _assert_untouched(big, N)


@triton.jit
def _wide_tile_and_narrow_atomic(x_ptr, tile_ptr, vec_ptr, M: tl.constexpr,
                                 N: tl.constexpr):
    rows = tl.arange(0, M)[:, None]
    cols = tl.arange(0, N)[None, :]
    tile = tl.load(x_ptr + rows * N + cols)
    tl.store(tile_ptr + rows * N + cols, tile * 2.0)
    tl.atomic_add(vec_ptr + tl.arange(0, N), tl.full((N,), 1.0, tl.float32))


def test_narrow_atomic_stays_inside_its_tensor():
    """The atomics carry the same per-thread index and the same window."""
    torch.manual_seed(6)
    M, N = 8, 128
    x = torch.randn(M, N, device="mps")
    out_tile = torch.zeros(M, N, device="mps")
    big, out_vec = _canary_view(N)
    out_vec.zero_()
    _wide_tile_and_narrow_atomic[(1,)](x, out_tile, out_vec, M=M, N=N,
                                       num_warps=1)
    torch.mps.synchronize()
    torch.testing.assert_close(out_vec.cpu(), torch.ones(N), atol=0, rtol=0)
    _assert_untouched(big, N)


@triton.jit
def _two_tiles_kernel(A, B, C, D, R1: tl.constexpr, C1: tl.constexpr,
                      R2: tl.constexpr, C2: tl.constexpr,
                      SHARE_ROWS: tl.constexpr):
    r1 = tl.arange(0, R1)
    c1 = tl.arange(0, C1)
    a = tl.load(A + r1[:, None] * C1 + c1[None, :])
    tl.store(B + r1[:, None] * C1 + c1[None, :], a * 2.0)
    if SHARE_ROWS:
        r2 = r1
    else:
        r2 = tl.arange(0, R2)
    c2 = tl.arange(0, C2)
    c = tl.load(C + r2[:, None] * C2 + c2[None, :])
    tl.store(D + r2[:, None] * C2 + c2[None, :], c + 1.0)


# Two INDEPENDENT rank-2 tiles of different shapes in one kernel. Every rank-2
# range decomposes against a tile, and a range that knew its own one was only
# allowed to use it when that tile held as many elements as the kernel's
# largest -- so a `[4, 2]` store beside a `[4, 4]` one wrote `tid / 4, tid % 4`,
# and a `[R, 1]` column was never recognised as a tile at all. 📏
# `s_two_rank2_tiles.py`: 156 of 270 such kernels silently wrong, now 0. The
# cells cover a unit column, a unit row, the smaller tile above one pass
# (E = 2), and the LARGER tile going wrong beside a smaller one. Both outputs go
# through canary views: a smaller tile runs under the larger one's tile loop.
@pytest.mark.parametrize("first,second,num_warps,share", [
    ((4, 1), (4, 4), 1, False),
    ((4, 1), (4, 4), 1, True),
    ((16, 1), (16, 16), 2, True),
    ((4, 2), (4, 4), 1, False),
    ((1, 64), (16, 16), 1, False),
    ((64, 1), (64, 32), 1, True),
    ((32, 2), (8, 128), 1, False),
    ((16, 16), (64, 32), 1, False),
    ((64, 32), (16, 16), 4, False),
    ((64, 8), (16, 16), 2, False),
])
def test_two_independent_rank2_tiles_each_use_their_own_shape(
        first, second, num_warps, share):
    (r1, c1), (r2, c2) = first, second
    torch.manual_seed(r1 * 131 + c1 * 7 + r2 + c2 + num_warps)
    a = torch.randn(r1, c1, device="mps")
    c = torch.randn(r2, c2, device="mps")
    big_b, b = _canary_view(r1 * c1)
    big_d, d = _canary_view(r2 * c2)
    _two_tiles_kernel[(1,)](a, b, c, d, r1, c1, r2, c2, share,
                            num_warps=num_warps)
    torch.mps.synchronize()
    assert torch.equal(b.cpu().reshape(r1, c1), a.cpu() * 2)
    assert torch.equal(d.cpu().reshape(r2, c2), c.cpu() + 1)
    _assert_untouched(big_b, r1 * c1)
    _assert_untouched(big_d, r2 * c2)
