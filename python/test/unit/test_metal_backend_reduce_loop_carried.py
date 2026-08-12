"""Rank-2 axis=1 reduce over a cone that contains a LOOP-CARRIED per-row scalar.

Isolates the staged-leaf reduce (Increment 2.5): the reduce input `acc[:, None]
* x` re-derives, per column, a per-row leaf `acc` that is the enclosing
`scf.for`'s iter_arg — a control-flow value the re-emission cone evaluator cannot
reconstruct. At M <= tpb each fill thread reduces its own row (r == localTid), so
`acc[r]` is exactly the thread's converted per-thread scalar (getRemappedValue);
the reduce fill is emitted inline inside the loop. Compared bit-exact against a
CPU replay of the same recurrence.
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
def loop_carried_reduce_kernel(x_ptr, out_ptr, STEPS,
                               M: tl.constexpr, N: tl.constexpr):
    row = tl.arange(0, M)
    col = tl.arange(0, N)
    acc = tl.zeros([M], dtype=tl.float32)          # loop-carried per-row leaf
    for _ in range(STEPS):
        x = tl.load(x_ptr + row[:, None] * N + col[None, :])
        score = acc[:, None] * x + x                # acc[:,None] is the staged leaf
        m = tl.max(score, axis=1)                   # rank-2 reduce over the cone
        acc = acc + m * 0.01
    tl.store(out_ptr + row, acc)


def _reference(x, STEPS, M, N):
    acc = torch.zeros(M)
    xc = x.cpu().reshape(M, N)
    for _ in range(STEPS):
        m = (acc[:, None] * xc + xc).max(dim=1).values
        acc = acc + m * 0.01
    return acc


@pytest.mark.parametrize("M, N, STEPS", [(128, 64, 5), (8, 16, 3), (128, 64, 1)])
def test_reduce_loop_carried_leaf(M, N, STEPS):
    torch.manual_seed(0xC0FFEE + M * N + STEPS)
    x = torch.randn(M * N, dtype=torch.float32, device="mps").contiguous()
    out = torch.zeros(M, dtype=torch.float32, device="mps").contiguous()
    loop_carried_reduce_kernel[(1,)](x, out, STEPS, M, N, num_warps=4)
    torch.mps.synchronize()
    torch.testing.assert_close(out.cpu(), _reference(x, STEPS, M, N),
                               atol=1e-4, rtol=1e-4)


@triton.jit
def _loop_varying_attention_stats_kernel(q_ptr, k_ptr, out_ptr, STEPS,
                                         M: tl.constexpr, N: tl.constexpr):
    row = tl.arange(0, M)
    col = tl.arange(0, N)
    q = tl.load(q_ptr + col)
    running_max = -float("inf")
    running_sum = 0.0
    for step in range(STEPS):
        k = tl.load(k_ptr + step * M * N + row[:, None] * N + col[None, :])
        scores = tl.sum(q[None, :] * k, axis=1)
        block_max = tl.max(scores, axis=0)
        new_max = tl.maximum(running_max, block_max)
        running_sum = running_sum * tl.exp(running_max - new_max)
        running_sum += tl.sum(tl.exp(scores - new_max), axis=0)
        running_max = new_max
    tl.store(out_ptr, running_max)
    tl.store(out_ptr + 1, running_sum)


@pytest.mark.parametrize("M, N, STEPS", [(32, 16, 2), (32, 32, 3)])
def test_slice_rank1_reduce_in_loop_varying_attention_stats(M, N, STEPS):
    """The reduced tile address varies inside a loop carrying softmax state."""
    torch.manual_seed(0xA77E + M * N + STEPS)
    q = torch.randn(N, dtype=torch.float32, device="mps")
    k = torch.randn(STEPS, M, N, dtype=torch.float32, device="mps")
    out = torch.empty(2, dtype=torch.float32, device="mps")

    _loop_varying_attention_stats_kernel[(1,)](
        q, k, out, STEPS, M=M, N=N, num_warps=4
    )
    torch.mps.synchronize()

    scores = (k.cpu().double() * q.cpu().double()[None, None, :]).sum(dim=2)
    flat_scores = scores.reshape(-1)
    expected_max = flat_scores.max()
    expected_sum = torch.exp(flat_scores - expected_max).sum()
    torch.testing.assert_close(
        out[0].cpu().double(), expected_max, atol=2e-5, rtol=1e-5
    )
    torch.testing.assert_close(
        out[1].cpu().double(), expected_sum, atol=2e-4, rtol=1e-4
    )


@triton.jit
def _int8_kv_attention_kernel(q_ptr, k_ptr, v_ptr, k_scale_ptr, v_scale_ptr,
                              out_ptr, seq_len, head_dim, sm_scale,
                              BLOCK_SEQ: tl.constexpr, HEAD_DIM: tl.constexpr):
    head = tl.program_id(0)
    col = tl.arange(0, HEAD_DIM)
    col_mask = col < head_dim
    q = tl.load(q_ptr + head * head_dim + col, mask=col_mask, other=0.0)

    running_max = -float("inf")
    running_sum = 0.0
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)
    for start in range(0, seq_len, BLOCK_SEQ):
        row = start + tl.arange(0, BLOCK_SEQ)
        row_mask = row < seq_len
        tile_mask = row_mask[:, None] & col_mask[None, :]
        offsets = head * seq_len * head_dim + row[:, None] * head_dim + col[None, :]

        k = tl.load(k_ptr + offsets, mask=tile_mask, other=0).to(tl.float32)
        k_scale = tl.load(k_scale_ptr + head * seq_len + row,
                          mask=row_mask, other=0.0)
        scores = tl.sum(q[None, :] * k * k_scale[:, None], axis=1) * sm_scale
        scores = tl.where(row_mask, scores, -float("inf"))

        block_max = tl.max(scores, axis=0)
        new_max = tl.maximum(running_max, block_max)
        alpha = tl.exp(running_max - new_max)
        beta = tl.exp(scores - new_max)
        running_sum = running_sum * alpha + tl.sum(beta, axis=0)

        v = tl.load(v_ptr + offsets, mask=tile_mask, other=0).to(tl.float32)
        v_scale = tl.load(v_scale_ptr + head * seq_len + row,
                          mask=row_mask, other=0.0)
        acc = acc * alpha + tl.sum(beta[:, None] * v * v_scale[:, None], axis=0)
        running_max = new_max

    tl.store(out_ptr + head * head_dim + col, acc / running_sum, mask=col_mask)


def test_int8_kv_attention_online_softmax_end_to_end():
    """Covers signed int8 casts and both reductions in a ragged 3-block loop."""
    heads, seq_len, head_dim = 2, 65, 24
    torch.manual_seed(0x1A77E)
    q_cpu = torch.randn(heads, head_dim, dtype=torch.float32)
    k_cpu = torch.randint(-32, 33, (heads, seq_len, head_dim), dtype=torch.int8)
    v_cpu = torch.randint(-32, 33, (heads, seq_len, head_dim), dtype=torch.int8)
    k_scale_cpu = torch.rand(heads, seq_len, dtype=torch.float32) * 0.05 + 0.001
    v_scale_cpu = torch.rand(heads, seq_len, dtype=torch.float32) * 0.05 + 0.001

    scores = (
        q_cpu[:, None, :] * k_cpu.float() * k_scale_cpu[:, :, None]
    ).sum(dim=2) * (head_dim ** -0.5)
    expected = (
        torch.softmax(scores, dim=1)[:, :, None]
        * v_cpu.float()
        * v_scale_cpu[:, :, None]
    ).sum(dim=1)

    q, k, v, k_scale, v_scale = [
        tensor.to("mps")
        for tensor in (q_cpu, k_cpu, v_cpu, k_scale_cpu, v_scale_cpu)
    ]
    out = torch.empty(heads, head_dim, dtype=torch.float32, device="mps")
    _int8_kv_attention_kernel[(heads,)](
        q, k, v, k_scale, v_scale, out, seq_len, head_dim,
        head_dim ** -0.5, BLOCK_SEQ=32,
        HEAD_DIM=triton.next_power_of_2(head_dim), num_warps=4,
    )
    torch.mps.synchronize()
    torch.testing.assert_close(out.cpu(), expected, atol=3e-4, rtol=3e-4)


# --- Loop-carried rank-2 TILE ----------------------------------------------
#
# The staged-leaf mechanism above maps a leaf to ONE per-thread scalar, i.e. a
# per-ROW value. A tile that also varies along the row has no representation
# there, and an `scf.for` iter_arg is a BlockArgument the cone evaluator cannot
# re-emit anyway — a scan is not a reduction, so there is no reassociation to
# fall back on either. The [M,N] tile is therefore materialised in threadgroup
# memory by a pre-pass (`preprocessLoopCarriedReduceTiles`), which also seeds it
# with the loop's init and hands ReduceLowering the mapping.
#
# Detection must be pre-conversion: by the time the pattern runs, the SCF
# structural type conversion has rebuilt the loop with per-thread scalar
# iter_args and detached the original body block.
#
# Restricted to M <= tpb, where the reduce's row fill degenerates to one row per
# thread, so thread r owns row r of the tile outright — seed, update and read
# all happen on that thread and the buffer needs no barrier.


@triton.jit
def _loop_carried_tile_kernel(c_ptr, y_ptr, STEPS,
                              M: tl.constexpr, N: tl.constexpr):
    row = tl.arange(0, M)
    col = tl.arange(0, N)
    idx2d = row[:, None] * N + col[None, :]
    h = tl.zeros((M, N), dtype=tl.float32)     # loop-carried [M,N] tile
    for t in range(0, STEPS):
        cc = tl.load(c_ptr + t * M * N + idx2d)
        h = h * 0.5 + cc
        tl.store(y_ptr + t * M + row, tl.sum(cc * h, axis=1))


def _tile_reference(c, STEPS, M, N):
    h = torch.zeros(M, N, dtype=torch.float64)
    out = []
    cc_all = c.cpu().double()
    for t in range(STEPS):
        cc = cc_all[t * M * N:(t + 1) * M * N].reshape(M, N)
        h = h * 0.5 + cc
        out.append((cc * h).sum(1))
    return torch.cat(out)


@pytest.mark.parametrize("M, N, STEPS", [(32, 64, 4), (16, 32, 3), (8, 8, 5),
                                         (128, 16, 2)])
def test_reduce_loop_carried_tile(M, N, STEPS):
    torch.manual_seed(0xB0A + M * N + STEPS)
    c = torch.randn(STEPS * M * N, dtype=torch.float32, device="mps").contiguous()
    y = torch.zeros(STEPS * M, dtype=torch.float32, device="mps").contiguous()
    _loop_carried_tile_kernel[(1,)](c, y, STEPS, M, N, num_warps=4)
    torch.mps.synchronize()
    ref = _tile_reference(c, STEPS, M, N)
    err = (y.cpu().double() - ref).abs().max() / max(ref.abs().max().item(), 1.0)
    assert err <= 2e-5, f"rel err {err}"


# --- Hoisted base pointer in a re-emitted cone ------------------------------
#
# `evalRank1ValueAt` scalarised only the OUTERMOST tt.addptr offset and then
# added the SCALAR offsets from the rest of the chain. A tensor offset on an
# INNER addptr — where the per-element index lives whenever the base pointer is
# hoisted out of the loop — was dropped:
#
#   base = tt.addptr(splat(p + pid*S), offs)   <- index, tensor, DROPPED
#   addr = tt.addptr(base, splat(t*S))         <- outermost, no index
#
# The mask is scalarised down a separate path and kept its index, so the read
# came out silently off-row rather than obviously broken. Every kernel that
# hoists a row/column base out of its scan loop hits this — the SSM selective
# scan does it for all four of u/delta/B/C.


@triton.jit
def _hoisted_base_kernel(v_ptr, c_ptr, y_ptr, STEPS, n_valid,
                         M: tl.constexpr, N: tl.constexpr):
    row = tl.arange(0, M)
    col = tl.arange(0, N)
    v_base = v_ptr + col                       # hoisted OUT of the loop
    idx2d = row[:, None] * N + col[None, :]
    h = tl.zeros((M, N), dtype=tl.float32)
    for t in range(0, STEPS):
        v = tl.load(v_base + t * N, mask=col < n_valid, other=0.0)
        cc = tl.load(c_ptr + t * M * N + idx2d)
        h = h * 0.5 + cc * v[None, :]
        tl.store(y_ptr + t * M + row, tl.sum(cc * h, axis=1))


@pytest.mark.parametrize("M, N, STEPS, n_valid", [(32, 64, 4, 64),
                                                  (32, 64, 3, 16),
                                                  (16, 32, 4, 20)])
def test_reduce_loop_carried_tile_hoisted_base(M, N, STEPS, n_valid):
    torch.manual_seed(0x1DE + M * N + STEPS + n_valid)
    v = torch.randn(STEPS * N, dtype=torch.float32, device="mps").contiguous()
    c = torch.randn(STEPS * M * N, dtype=torch.float32, device="mps").contiguous()
    y = torch.zeros(STEPS * M, dtype=torch.float32, device="mps").contiguous()
    _hoisted_base_kernel[(1,)](v, c, y, STEPS, n_valid, M, N, num_warps=4)
    torch.mps.synchronize()

    h = torch.zeros(M, N, dtype=torch.float64)
    vc, cc_all = v.cpu().double(), c.cpu().double()
    out = []
    for t in range(STEPS):
        vt = vc[t * N:(t + 1) * N].clone()
        vt[n_valid:] = 0.0
        cc = cc_all[t * M * N:(t + 1) * M * N].reshape(M, N)
        h = h * 0.5 + cc * vt[None, :]
        out.append((cc * h).sum(1))
    ref = torch.cat(out)
    err = (y.cpu().double() - ref).abs().max() / max(ref.abs().max().item(), 1.0)
    assert err <= 2e-5, f"rel err {err}"


# --- Cone load with its own row stride --------------------------------------
#
# evalRank2ConeAt's tt.load leaf used to read `device[rowBase + n]`, where
# rowBase is derived ONCE from the reduce's representative load. That assumes
# every load in the cone shares the reduce tile's row stride and per-program
# base. False as soon as a cone load has a different shape: the SSM scan's
# reduce tile is (BLOCK_D, BLOCK_N) while its `A` is (d_model, d_state), so A
# was read with stride BLOCK_N and carrying another buffer's batch offset. The
# address is now re-materialised from the load's own tt.addptr chain at (r, n).


@triton.jit
def _cone_load_own_stride_kernel(a_ptr, c_ptr, y_ptr, STEPS,
                                 M: tl.constexpr, N: tl.constexpr,
                                 STRIDE: tl.constexpr):
    row = tl.arange(0, M)
    col = tl.arange(0, N)
    idx2d = row[:, None] * N + col[None, :]
    # Row stride STRIDE != N: a view into a WIDER buffer.
    a = tl.load(a_ptr + row[:, None] * STRIDE + col[None, :])
    h = tl.zeros((M, N), dtype=tl.float32)
    for t in range(0, STEPS):
        cc = tl.load(c_ptr + t * M * N + idx2d)
        h = h * 0.5 + cc * a
        tl.store(y_ptr + t * M + row, tl.sum(cc * h, axis=1))


@pytest.mark.parametrize("M, N, STEPS, STRIDE", [(32, 64, 4, 96), (16, 32, 3, 40),
                                                 (32, 16, 4, 64)])
def test_reduce_loop_carried_tile_cone_load_own_stride(M, N, STEPS, STRIDE):
    torch.manual_seed(0x5D + M * N + STRIDE)
    a = torch.randn(M * STRIDE, dtype=torch.float32, device="mps").contiguous()
    c = torch.randn(STEPS * M * N, dtype=torch.float32, device="mps").contiguous()
    y = torch.zeros(STEPS * M, dtype=torch.float32, device="mps").contiguous()
    _cone_load_own_stride_kernel[(1,)](a, c, y, STEPS, M, N, STRIDE, num_warps=4)
    torch.mps.synchronize()

    ac = a.cpu().double().reshape(M, STRIDE)[:, :N]
    cc_all = c.cpu().double()
    h = torch.zeros(M, N, dtype=torch.float64)
    out = []
    for t in range(STEPS):
        cc = cc_all[t * M * N:(t + 1) * M * N].reshape(M, N)
        h = h * 0.5 + cc * ac
        out.append((cc * h).sum(1))
    ref = torch.cat(out)
    err = (y.cpu().double() - ref).abs().max() / max(ref.abs().max().item(), 1.0)
    assert err <= 2e-5, f"rel err {err}"


# --- Loop-carried tile across several row-block programs ---------------------
#
# Every loop-carried test above launches a single program, so the staged tile
# only ever had one threadgroup's worth of state. Here the row block comes from
# program_id(1), matching the SSM selective scan's (batch, d_blocks) launch:
# each program owns its own threadgroup buffer and its own slice of the output.


@triton.jit
def _loop_carried_tile_multiprog_kernel(c_ptr, y_ptr, STEPS, n_rows,
                                        M: tl.constexpr, N: tl.constexpr):
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    row = pid_m * M + tl.arange(0, M)
    col = tl.arange(0, N)
    rmask = row < n_rows
    h = tl.zeros((M, N), dtype=tl.float32)
    for t in range(0, STEPS):
        cc = tl.load(c_ptr + pid_b * STEPS * n_rows * N + t * n_rows * N
                     + row[:, None] * N + col[None, :])
        h = h * 0.5 + cc
        tl.store(y_ptr + pid_b * STEPS * n_rows + t * n_rows + row,
                 tl.sum(cc * h, axis=1), mask=rmask)


@pytest.mark.parametrize("batch, M, N, STEPS, n_rows", [
    (1, 32, 64, 4, 64),    # two row-blocks
    (2, 32, 64, 4, 64),    # two row-blocks x two batches
    (1, 32, 64, 3, 128),   # four row-blocks
    (1, 16, 32, 4, 64),
])
def test_reduce_loop_carried_tile_multiprogram(batch, M, N, STEPS, n_rows):
    torch.manual_seed(0x2D + n_rows + M)
    c = torch.randn(batch * STEPS * n_rows * N, dtype=torch.float32,
                    device="mps").contiguous()
    y = torch.zeros(batch * STEPS * n_rows, dtype=torch.float32,
                    device="mps").contiguous()
    _loop_carried_tile_multiprog_kernel[(batch, n_rows // M)](
        c, y, STEPS, n_rows, M, N, num_warps=4)
    torch.mps.synchronize()

    cc = c.cpu().double().reshape(batch, STEPS, n_rows, N)
    h = torch.zeros(batch, n_rows, N, dtype=torch.float64)
    out = []
    for t in range(STEPS):
        h = h * 0.5 + cc[:, t]
        out.append((cc[:, t] * h).sum(-1))
    ref = torch.stack(out, 1).reshape(-1)
    err = (y.cpu().double() - ref).abs().max() / max(ref.abs().max().item(), 1.0)
    assert err <= 2e-5, f"rel err {err}"
