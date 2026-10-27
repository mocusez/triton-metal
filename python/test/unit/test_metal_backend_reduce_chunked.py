"""Rank-2 axis=1 reduce — `(M, N)` sweep including M*N >> tpb shapes.

Exercises `ReduceLowering`'s rank-2 axis=1 body across a sweep of `(M, N)`
shapes. The kernel sums a 2D `(M, N)` tile along axis=1 and writes the
resulting `(M,)` vector.

The L3a-tileloop-2 redesign makes the body self-contained: it walks back to
the producing `tt.load` and reduces each row directly from device memory into
a per-row threadgroup buffer (`rowBuf[M]`), hoisted above the FuncOpLowering
tile loop so it runs once. This removed the old `M*N == tpb` assumption (and
the old chunked body's threadgroup over-allocation), so every shape below —
including the large `M*N >> tpb` ones that were previously xfail — now produces
correct results.
"""

from __future__ import annotations

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
def reduce_sum_axis1_kernel(
    x_ptr,
    out_ptr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    addr = offs_m[:, None] * BLOCK_N + offs_n[None, :]
    x = tl.load(x_ptr + addr)
    s = tl.sum(x, axis=1)
    tl.store(out_ptr + offs_m, s)


@pytest.mark.parametrize(
    "M, N",
    [
        # In-budget shape: M*N == tpb (the single-(row,col)-per-thread case).
        (8, 16),     # 128 B, M*N == 128 == tpb (4 warps × 32).
        # M*N >> tpb shapes. Previously xfail (the old chunked body assumed
        # M*N == tpb and produced wrong results / over-allocated threadgroup
        # memory). The L3a-tileloop-2 redesign reduces each row directly from
        # device memory into a per-row threadgroup buffer, so these now pass.
        (1024, 8),
        (1024, 16),
        (1024, 64),    # matches conv1d's reduce shape.
        (2048, 64),
        (512, 128),
    ],
)
def test_reduce_chunked_axis1_f32(M, N):
    torch.manual_seed(0xC0FFEE)
    x = torch.randn((M, N), dtype=torch.float32).contiguous()
    out = torch.zeros((M,), dtype=torch.float32).contiguous()
    reduce_sum_axis1_kernel[(1, 1, 1)](
        x, out, BLOCK_M=M, BLOCK_N=N
    )
    expected = torch.sum(x.cpu(), dim=1).to(torch.float32)
    torch.testing.assert_close(out, expected, atol=1e-4, rtol=1e-4)


@triton.jit
def reduce_sum_axis1_strided_kernel(
    x_ptr,
    out_ptr,
    row_stride,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    OP: tl.constexpr,
):
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    x = tl.load(x_ptr + offs_m[:, None] * row_stride + offs_n[None, :])
    if OP == "max":
        r = tl.max(x, axis=1)
    elif OP == "min":
        r = tl.min(x, axis=1)
    else:
        r = tl.sum(x, axis=1)
    tl.store(out_ptr + offs_m, r)


@triton.jit
def reduce_sum_axis1_program_kernel(
    x_ptr,
    out_ptr,
    row_stride,
    prog_stride,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BASE_MODE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_n = tl.arange(0, BLOCK_N)
    if BASE_MODE == "scalar":
        rows = tl.arange(0, BLOCK_M)[:, None]
        x = tl.load(x_ptr + pid * prog_stride + rows * row_stride
                    + offs_n[None, :])
    else:
        rows = (pid * BLOCK_M + tl.arange(0, BLOCK_M))[:, None]
        x = tl.load(x_ptr + rows * row_stride + offs_n[None, :])
    s = tl.sum(x, axis=1)
    tl.store(out_ptr + pid * BLOCK_M + tl.arange(0, BLOCK_M), s)


@pytest.mark.parametrize(
    "M, N, row_stride, num_warps, op",
    [
        # The row stride is where the real address lives. Reading it from the
        # tile's own width instead reduces rows packed at stride N: row 0 is
        # right and every later row comes from the wrong place.
        (8, 128, 256, 1, "sum"),
        (8, 128, 129, 1, "sum"),
        (8, 128, 3, 1, "sum"),
        (8, 128, 128, 1, "sum"),      # stride == N: the shape that always held
        (4, 32, 64, 1, "max"),
        (16, 64, 200, 2, "min"),
        (8, 128, 256, 4, "sum"),
        (64, 16, 40, 4, "sum"),
    ],
    ids=[
        "stride_2n", "stride_n_plus_1", "stride_below_n", "stride_eq_n",
        "max_stride_2n", "min_stride_odd", "warps4_stride_2n",
        "tall_stride_below_n",
    ],
)
def test_reduce_axis1_row_stride_matches_torch(M, N, row_stride, num_warps, op):
    torch.manual_seed(0xC0FFEE)
    extent = M * max(row_stride, N) + N
    flat = torch.randn((extent,), dtype=torch.float32).contiguous()
    rows = torch.arange(M)[:, None]
    cols = torch.arange(N)[None, :]
    tile = flat[rows * row_stride + cols]
    if op == "max":
        expected = tile.max(dim=1).values
    elif op == "min":
        expected = tile.min(dim=1).values
    else:
        expected = tile.sum(dim=1)
    out = torch.zeros((M,), dtype=torch.float32).contiguous()
    reduce_sum_axis1_strided_kernel[(1, 1, 1)](
        flat, out, row_stride, BLOCK_M=M, BLOCK_N=N, OP=op,
        num_warps=num_warps)
    torch.testing.assert_close(out, expected, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize(
    "base_mode, programs, M, N, row_stride",
    [
        ("scalar", 3, 8, 64, 128),
        ("folded", 3, 8, 64, 128),
        ("scalar", 1, 16, 32, 100),
        ("folded", 2, 16, 32, 100),
    ],
    ids=["scalar_base_3progs", "folded_base_3progs", "scalar_base_1prog",
         "folded_base_2progs"],
)
def test_reduce_axis1_row_stride_multi_program(base_mode, programs, M, N,
                                               row_stride):
    """The program base and the row stride must both come from the chain.

    `scalar` carries the program term as a scalar addptr; `folded` hides it in
    the row tensor. Both are replayed from the load's own pointer chain now, so
    neither needs the fabricated per-program term the fast path used.
    """
    torch.manual_seed(0x5EED)
    prog_stride = M * row_stride
    extent = programs * prog_stride + M * row_stride + N
    flat = torch.randn((extent,), dtype=torch.float32).contiguous()
    cols = torch.arange(N)[None, :]
    expected = []
    for pid in range(programs):
        if base_mode == "scalar":
            index = pid * prog_stride + torch.arange(M)[:, None] * row_stride
        else:
            index = (pid * M + torch.arange(M))[:, None] * row_stride
        expected.append(flat[index + cols].sum(dim=1))
    expected = torch.cat(expected)
    out = torch.zeros((programs * M,), dtype=torch.float32).contiguous()
    reduce_sum_axis1_program_kernel[(programs, 1, 1)](
        flat, out, row_stride, prog_stride, BLOCK_M=M, BLOCK_N=N,
        BASE_MODE=base_mode, num_warps=4)
    torch.testing.assert_close(out, expected, atol=1e-4, rtol=1e-4)


def test_reduce_axis1_row_buffer_budget_is_refused_then_recovers(tmp_path):
    from test_metal_backend_reduce_rank2_computed import _refusal_child

    result, output = _refusal_child(tmp_path, "row_buffer_budget", """
        x = torch.zeros(16384, 2, device="mps")
        out = torch.zeros(16384, device="mps")
        try:
            module.reduce_sum_axis1_kernel[(1,)](
                x, out, BLOCK_M=16384, BLOCK_N=2)
        except RuntimeError:
            pass
        else:
            raise AssertionError("an over-budget row buffer compiled")

        x = torch.arange(
            8192 * 2, dtype=torch.float32, device="mps").reshape(8192, 2)
        out = torch.empty(8192, device="mps")
        module.reduce_sum_axis1_kernel[(1,)](
            x, out, BLOCK_M=8192, BLOCK_N=2)
        torch.mps.synchronize()
        torch.testing.assert_close(out, x.sum(dim=1))
        print("RECOVERED")
    """, module_path=__file__)
    needle = "per-row size exceeds 32 KiB; chunking impossible"
    assert result.returncode == 0, output
    assert result.stderr.count(needle) == 1, output
    assert "RECOVERED" in result.stdout, output
    assert "failed to legalize" not in output, output
