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
