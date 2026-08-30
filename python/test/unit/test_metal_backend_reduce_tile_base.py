"""Rank-2 axis=1 reduce: tile base address, and uniform (splat) tensor loads.

Two independent Metal-backend fixes are covered here.

1. **Tile base address.** The rank-2 axis=1 reduce never re-materialised the
   load's address; it fabricated `rowBase = (r + tgid*tpb)*N` and added only the
   SCALAR tt.addptr offsets. `accumulateScalarAddPtrOffsets` stopped walking at
   `tt.broadcast`, so for a 2D tile — built as
   `addptr(broadcast(addptr(splat(addptr(arg, pid*S)), rows*N)), cols)` — the
   real `pid*S` term was never found and the fabricated `tgid*tpb*N` stood in
   for it. That is correct only when `S == tpb*N`, i.e. exactly one contiguous
   tile per program. Any other base (`pid*K*M*N` below) was read from the wrong
   offset and silently produced wrong numbers — no error, no diagnostic.

   The walker now traverses `tt.broadcast`/`tt.expand_dims`, and when a real
   scalar term is found it REPLACES the fabricated one rather than adding to it
   (adding would double-count the per-program offset). When no scalar term
   exists — the program offset folded into the row tensor, as in
   `offs_m = pid*BLOCK_M + arange(...)` — the fabricated term is still used, so
   that shape is unchanged.

2. **Uniform (splat) pointer accesses.** `tt.load` on a bare `tt.splat` of a
   scalar pointer — no `tt.addptr`, every lane reading the SAME address — hit
   `"tt.load expects a tt.addptr feeding ptr"` and failed to legalize. Triton
   emits exactly this whenever the per-element offset folds away, which a
   1-element tile always does (`tl.arange(0, 1) * stride` is 0). It is not
   specific to size 1: the `uniform_load` cases below cover wider tiles too.

The store lowering intentionally covers only rank-1 singleton tensors, where
lane 0 has unambiguous ownership. Wider splat stores would write multiple
values to one address and remain unsupported. A FAILED legalization poisons
the MLIR context, so the NEXT compile in the same process aborts (SIGABRT)
instead of raising; compile-failure xfails are therefore unsafe in this shared
pytest process.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

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


@triton.jit
def _reduce_strided_base_kernel(x_ptr, out_ptr, K, BM: tl.constexpr,
                                BN: tl.constexpr):
    # Per-program stride is K tiles, not one — the case the fabricated
    # `tgid*tpb*N` got wrong.
    pid = tl.program_id(0)
    om, on = tl.arange(0, BM), tl.arange(0, BN)
    x = tl.load(x_ptr + pid * K * BM * BN + om[:, None] * BN + on[None, :])
    tl.store(out_ptr + pid * BM + om, tl.sum(x, axis=1))


@pytest.mark.parametrize("K", [1, 2, 4])
@pytest.mark.parametrize("BM, BN, num_warps", [(32, 16, 1), (64, 16, 2)])
def test_rank2_reduce_strided_tile_base(K, BM, BN, num_warps):
    G = 8
    torch.manual_seed(K * 100 + BM)
    x = torch.randn(G, K, BM, BN, dtype=torch.float32, device="mps")
    out = torch.zeros(G, BM, dtype=torch.float32, device="mps")
    _reduce_strided_base_kernel[(G,)](x, out, K, BM=BM, BN=BN,
                                      num_warps=num_warps)
    torch.mps.synchronize()
    # Each program reduces the FIRST tile of its K-tile stride.
    ref = x.cpu().double()[:, 0, :, :].sum(-1)
    err = (out.cpu().double() - ref).abs().max() / ref.abs().max()
    assert err <= 2e-5, f"rel_err={err:.3e}"


@triton.jit
def _reduce_in_loop_no_carry_kernel(x_ptr, out_ptr, T, BM: tl.constexpr,
                                    BN: tl.constexpr):
    # Reduce inside a loop with a trip-varying address, and NO iter_args. The
    # fill must be emitted per trip; hoisting it would reduce trip 0 every time.
    pid = tl.program_id(0)
    om, on = tl.arange(0, BM), tl.arange(0, BN)
    tile = BM * BN
    for t in range(T):
        x = tl.load(x_ptr + pid * T * tile + t * tile
                    + om[:, None] * BN + on[None, :])
        tl.store(out_ptr + pid * T * BM + t * BM + om, tl.sum(x, axis=1))


@pytest.mark.parametrize("BM, BN, num_warps", [(32, 16, 1), (64, 16, 2)])
def test_rank2_reduce_in_loop_no_iter_args(BM, BN, num_warps):
    G, T = 8, 8
    torch.manual_seed(BM)
    x = torch.randn(G, T, BM, BN, dtype=torch.float32, device="mps")
    out = torch.zeros(G, T * BM, dtype=torch.float32, device="mps")
    _reduce_in_loop_no_carry_kernel[(G,)](x, out, T, BM=BM, BN=BN,
                                          num_warps=num_warps)
    torch.mps.synchronize()
    ref = x.cpu().double().sum(-1)
    got = out.cpu().double().reshape(G, T, BM)
    err = (got - ref).abs().max() / ref.abs().max()
    assert err <= 2e-5, f"rel_err={err:.3e}"


@triton.jit
def _uniform_load_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
    # `tl.arange(0, BLOCK) * 0` folds the per-element offset away, so Triton
    # emits tt.load on a bare tt.splat: every lane reads x_ptr[0].
    idx = tl.arange(0, BLOCK) * 0
    v = tl.load(x_ptr + idx)
    tl.store(out_ptr + tl.arange(0, BLOCK), v)


# BLOCK=1 is covered by the reduce test below, which stores through a scalar
# pointer. Wider blocks exercise the uniform load without introducing an
# intentionally unsupported wider uniform store.
@pytest.mark.parametrize("BLOCK", [2, 8, 32, 64])
def test_uniform_splat_load(BLOCK):
    torch.manual_seed(BLOCK)
    x = torch.randn(16, dtype=torch.float32, device="mps")
    out = torch.zeros(BLOCK, dtype=torch.float32, device="mps")
    _uniform_load_kernel[(1,)](x, out, BLOCK=BLOCK)
    torch.mps.synchronize()
    # Every lane read the same element, so the output is x[0] broadcast.
    assert torch.equal(out.cpu(), x.cpu()[0].repeat(BLOCK))


@triton.jit
def _uniform_load_reduce_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
    idx = tl.arange(0, BLOCK) * 0
    v = tl.load(x_ptr + idx)
    tl.store(out_ptr, tl.sum(v, axis=0))


@pytest.mark.parametrize("BLOCK", [1, 4, 32])
def test_uniform_splat_load_feeding_reduce(BLOCK):
    torch.manual_seed(BLOCK)
    x = torch.randn(16, dtype=torch.float32, device="mps")
    out = torch.zeros(1, dtype=torch.float32, device="mps")
    _uniform_load_reduce_kernel[(1,)](x, out, BLOCK=BLOCK)
    torch.mps.synchronize()
    ref = x.cpu().double()[0] * BLOCK
    assert (out.cpu().double()[0] - ref).abs() <= 2e-5 * max(abs(ref.item()), 1.0)


@triton.jit
def _singleton_masked_store_kernel(out_ptr, limit, BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.full((BLOCK,), 7.0, tl.float32)
    tl.store(out_ptr + offset, value, mask=offset < limit)


def test_singleton_masked_splat_store_respects_false_mask():
    out = torch.full((2,), -1.0, dtype=torch.float32, device="mps")
    _singleton_masked_store_kernel[(2,)](out, 1, BLOCK=1)
    torch.mps.synchronize()
    torch.testing.assert_close(out.cpu(), torch.tensor([7.0, -1.0]))


def _load_sparse_matvec_module():
    path = (Path(__file__).resolve().parent / "fixtures" / "metal_leet" /
            "medium-sparse_matrix-vector_multiplication.py")
    assert path.is_file(), f"required Metal Leet fixture not present: {path}"
    spec = importlib.util.spec_from_file_location("leet_sparse_matvec", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "M,N,density", [(1, 1, 0.0), (3, 5, 0.3), (5, 1031, 0.05)]
)
def test_leet_sparse_matvec_coo(M, N, density):
    """Sparse COO solve paths match PyTorch without a dense GPU A."""
    module = _load_sparse_matvec_module()
    torch.manual_seed(0x5A00 + M * 17 + N)
    a_cpu = torch.randn((M, N), dtype=torch.float32)
    a_cpu *= torch.rand((M, N)) < density
    a_sparse = a_cpu.to_sparse_coo().coalesce()
    nnz = a_sparse._nnz()
    assert a_sparse.layout == torch.sparse_coo
    assert a_sparse.values().numel() == nnz
    x_cpu = torch.randn((N,), dtype=torch.float32)
    x = x_cpu.to("mps").contiguous()
    y = torch.empty((M,), dtype=torch.float32, device="mps")
    y_prepacked = torch.empty_like(y)

    module.solve(a_sparse, x, y, M, N, nnz)
    row_indices, col_indices, values = module.prepare_coo(
        a_sparse, "mps", M, N, nnz
    )
    module.solve_coo(
        row_indices,
        col_indices,
        values,
        x,
        y_prepacked,
        M,
        N,
        nnz,
    )
    torch.mps.synchronize()

    expected = a_cpu @ x_cpu
    torch.testing.assert_close(y.cpu(), expected, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(
        y_prepacked.cpu(), expected, atol=1e-4, rtol=1e-4
    )


def test_leet_sparse_matvec_coalesces_duplicate_coordinates():
    module = _load_sparse_matvec_module()
    indices = torch.tensor([[0, 0, 1], [1, 1, 0]], dtype=torch.int64)
    values = torch.tensor([1.25, 2.75, -3.0], dtype=torch.float32)
    a_sparse = torch.sparse_coo_tensor(indices, values, (2, 3))
    assert not a_sparse.is_coalesced()
    x_cpu = torch.tensor([2.0, -1.0, 4.0], dtype=torch.float32)
    x = x_cpu.to("mps")
    y = torch.empty((2,), dtype=torch.float32, device="mps")
    y_prepacked = torch.empty_like(y)

    module.solve(a_sparse, x, y, 2, 3, a_sparse._nnz())
    rows, cols, packed_values = module.prepare_coo(
        a_sparse, "mps", 2, 3, a_sparse._nnz()
    )
    assert packed_values.numel() == 2
    module.solve_coo(
        rows, cols, packed_values, x, y_prepacked, 2, 3,
        packed_values.numel()
    )
    torch.mps.synchronize()

    expected = a_sparse.to_dense() @ x_cpu
    torch.testing.assert_close(y.cpu(), expected)
    torch.testing.assert_close(y_prepacked.cpu(), expected)


def test_leet_sparse_matvec_dense_input_compatibility():
    module = _load_sparse_matvec_module()
    torch.manual_seed(0x5A5A)
    M, N = 3, 5
    a_cpu = torch.randn((M, N), dtype=torch.float32)
    x_cpu = torch.randn((N,), dtype=torch.float32)
    y = torch.empty((M,), dtype=torch.float32, device="mps")

    module.solve(
        a_cpu.to("mps"),
        x_cpu.to("mps"),
        y,
        M,
        N,
        int(torch.count_nonzero(a_cpu)),
    )
    torch.mps.synchronize()

    torch.testing.assert_close(y.cpu(), a_cpu @ x_cpu, atol=1e-4, rtol=1e-4)

    a_noncontiguous = torch.randn(
        (M, N * 2), dtype=torch.float32, device="mps"
    )[:, ::2]
    assert not a_noncontiguous.is_contiguous()
    with pytest.raises(ValueError, match="contiguous"):
        module.solve(a_noncontiguous, x_cpu.to("mps"), y, M, N, M * N)


# --- 3. Slice-encoded rank-1 loads -----------------------------------------
#
# A 1D vector that gets expand_dims'd/broadcast into a 2D tile carries
# `#ttg.slice<{dim, parent = #blocked2D}>`, not a blocked encoding, so its
# `tt.load` failed to legalize outright ("missing ttg.blocked layout").
#
# The subtle half is the LIVE path. Under slice<dim=1,parent=B> thread t holds
# row t/N; under the rank-1 #blocked1 the store wants, it holds row t. When
# both consumers exist Triton usually just duplicates the load — but if a
# `tt.reduce` result (inherently slice-encoded) is combined with the value, the
# two paths are forced to share an encoding and only a `ttg.convert_layout`
# separates them. That cvt was classified as a scalar identity, so every lane
# ended up reading element t/N, i.e. element 0 for any N >= tpb.


@triton.jit
def _slice_load_two_consumers_kernel(skip_ptr, a_ptr, t_ptr, y_ptr, d_model,
                                     BD: tl.constexpr, BN: tl.constexpr):
    d_off, n_off = tl.arange(0, BD), tl.arange(0, BN)
    d_mask = d_off < d_model
    skip = tl.load(skip_ptr + d_off, mask=d_mask, other=0.0)
    idx2d = d_off[:, None] * BN + n_off[None, :]
    a = tl.load(a_ptr + idx2d)
    # Consumer A: into the 2D tile (this is what forces the slice encoding).
    tl.store(t_ptr + idx2d, tl.expand_dims(skip, 1) * a)
    # Consumer B: live rank-1 store.
    tl.store(y_ptr + d_off, skip * 2.0, mask=d_mask)


@pytest.mark.parametrize("BD, BN, num_warps", [(32, 64, 4), (16, 32, 2),
                                               (64, 16, 4)])
def test_slice_encoded_load_two_consumers(BD, BN, num_warps):
    torch.manual_seed(BD * BN)
    skip = torch.randn(BD, dtype=torch.float32, device="mps")
    a = torch.randn(BD * BN, dtype=torch.float32, device="mps")
    t = torch.zeros(BD * BN, dtype=torch.float32, device="mps")
    y = torch.zeros(BD, dtype=torch.float32, device="mps")
    _slice_load_two_consumers_kernel[(1,)](skip, a, t, y, BD, BD=BD, BN=BN,
                                           num_warps=num_warps)
    torch.mps.synchronize()
    torch.testing.assert_close(t.cpu(), (skip.cpu()[:, None] * a.cpu().reshape(BD, BN)).reshape(-1))
    torch.testing.assert_close(y.cpu(), skip.cpu() * 2.0)


@triton.jit
def _slice_load_plus_reduce_kernel(a_ptr, skip_ptr, y_ptr, d_model,
                                   BD: tl.constexpr, BN: tl.constexpr):
    d_off, n_off = tl.arange(0, BD), tl.arange(0, BN)
    d_mask = d_off < d_model
    a = tl.load(a_ptr + d_off[:, None] * BN + n_off[None, :])
    s = tl.sum(a, axis=1)
    skip = tl.load(skip_ptr + d_off, mask=d_mask, other=0.0)
    # `s` is slice-encoded, so this addf drags `skip` into the slice encoding
    # too and the store needs a real (data-moving) convert_layout.
    tl.store(y_ptr + d_off, s + skip, mask=d_mask)


@pytest.mark.parametrize("BD, BN, num_warps", [(32, 64, 4), (16, 32, 2),
                                               (64, 16, 4), (32, 8, 1)])
def test_slice_encoded_load_added_to_reduce_result(BD, BN, num_warps):
    torch.manual_seed(BD * BN + 1)
    a = torch.randn(BD * BN, dtype=torch.float32, device="mps")
    skip = torch.randn(BD, dtype=torch.float32, device="mps")
    y = torch.zeros(BD, dtype=torch.float32, device="mps")
    _slice_load_plus_reduce_kernel[(1,)](a, skip, y, BD, BD=BD, BN=BN,
                                         num_warps=num_warps)
    torch.mps.synchronize()
    ref = a.cpu().double().reshape(BD, BN).sum(1) + skip.cpu().double()
    err = (y.cpu().double() - ref).abs().max() / ref.abs().max()
    assert err <= 2e-5, f"rel err {err}"
    # Guard the specific failure mode: every lane collapsing to element 0.
    assert not torch.allclose(y.cpu() - a.cpu().reshape(BD, BN).sum(1),
                              skip.cpu()[0].expand(BD), atol=1e-4)
