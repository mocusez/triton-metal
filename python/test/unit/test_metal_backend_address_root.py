"""A 2-D tile address with a UNIFORM axis: it must run, and it must be right.

Triton folds an axis whose address contribution is uniform and hands the access
a `tt.broadcast` of a lower-rank pointer tensor —
`tensor<64x1x!tt.ptr<f32>> -> tensor<64x64x!tt.ptr<f32>>` is what
`tl.full((BLOCK,), k, tl.int32)[None, :]` compiles to. Every gate in the backend
used to refuse that root, and because all four decline with
`notifyMatchFailure` INSIDE `applyFullConversion`, the refusal did not raise: the
failed conversion's rollback leaves the module unverifiable and the process
exits on SIGSEGV or SIGABRT, intermittently. The verbatim Floyd-Warshall kernel
from the LeetGPU corpus exited 139 before `N=4` finished.

Two things fixed that, and this file pins both:

  1. `unsupportedTensorAddressReason` is shared between the patterns and
     `validateTensorAddressSupport`, which runs before conversion — so a
     refused address is an ERROR, never a signal.
  2. The broadcast root is accepted, because nothing needs re-deriving for it:
     the offset beneath was emitted against the CONSUMER's layout already.

Each case runs in its own subprocess and checks its own answer. If a regression
brings the crash back, an in-process test would take the whole pytest session
with it; a subprocess turns it into a readable assertion.

The exhaustive grid — dtypes, BLOCK, num_warps, non-multiple-of-16 extents —
lives in `python/test/metal_sweeps/sweeps/s_bcast_ptr.py` (288 cells). This file
is the in-tree net.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

torch = pytest.importorskip("torch")

pytest.importorskip(
    "triton._C.libtriton.metal",
    reason="Metal backend pybind module not built into libtriton",
)
if not torch.backends.mps.is_available():
    pytest.skip(
        "Metal backend requires an MPS-enabled PyTorch (Apple Silicon)",
        allow_module_level=True,
    )

HEAD = textwrap.dedent("""
    import torch, triton
    import triton.language as tl

    BLOCK, KROW, KCOL, N = 64, 3, 5, 64
    ref = torch.arange(BLOCK * BLOCK, dtype=torch.float32).reshape(BLOCK, BLOCK)
    x = ref.to("mps")
    out = torch.zeros(BLOCK, BLOCK, device="mps")


    def check(want):
        got = out.cpu()
        if not torch.equal(got, want):
            bad = (got != want).nonzero()
            i, j = bad[0].tolist()
            print(f"WRONG {len(bad)}/{want.numel()} cells; "
                  f"[{i},{j}] got={got[i, j].item()} want={want[i, j].item()}")
    """)

# Read a COLUMN through a uniform column axis — the Floyd-Warshall shape.
UNIFORM_COLUMN = """
@triton.jit
def k(X, O, sx, KCOL, N, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    j = tl.arange(0, BLOCK)
    ko = tl.full((BLOCK,), KCOL, dtype=tl.int32)
    v = tl.load(X + i[:, None] * sx + ko[None, :]{mask})
    tl.store(O + i[:, None] * sx + j[None, :], v)

k[(1,)](x, out, BLOCK, KCOL, N, BLOCK=BLOCK, num_warps={nw})
torch.mps.synchronize()
check(ref[:, KCOL:KCOL + 1].expand(BLOCK, BLOCK).contiguous())
"""

# The mirror: a uniform ROW axis.
UNIFORM_ROW = """
@triton.jit
def k(X, O, sx, KROW, N, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    j = tl.arange(0, BLOCK)
    ko = tl.full((BLOCK,), KROW, dtype=tl.int32)
    v = tl.load(X + ko[:, None] * sx + j[None, :]{mask})
    tl.store(O + i[:, None] * sx + j[None, :], v)

k[(1,)](x, out, BLOCK, KROW, N, BLOCK=BLOCK, num_warps={nw})
torch.mps.synchronize()
check(ref[KROW:KROW + 1, :].expand(BLOCK, BLOCK).contiguous())
"""

# A genuine 2-D addptr whose column term FOLDS to a uniform value. Same family,
# reached without writing `tl.full` at all — which is why the discriminator is
# "does the axis fold", not "did the author write a splat".
FOLDED_COLUMN = """
@triton.jit
def k(X, O, sx, KCOL, N, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    j = tl.arange(0, BLOCK)
    v = tl.load(X + i[:, None] * sx + (j * 0 + KCOL)[None, :]{mask})
    tl.store(O + i[:, None] * sx + j[None, :], v)

k[(1,)](x, out, BLOCK, KCOL, N, BLOCK=BLOCK, num_warps={nw})
torch.mps.synchronize()
check(ref[:, KCOL:KCOL + 1].expand(BLOCK, BLOCK).contiguous())
"""

# Write side: the destination's row axis is uniform. Masked down to one lane per
# column, so the shape is a column write rather than a race.
STORE_UNIFORM_ROW = """
@triton.jit
def k(X, O, sx, KROW, N, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    j = tl.arange(0, BLOCK)
    ko = tl.full((BLOCK,), KROW, dtype=tl.int32)
    v = tl.load(X + i[:, None] * sx + j[None, :])
    tl.store(O + ko[:, None] * sx + j[None, :], v, mask=(i[:, None] == 0))

k[(1,)](x, out, BLOCK, KROW, N, BLOCK=BLOCK, num_warps={nw})
torch.mps.synchronize()
want = torch.zeros(BLOCK, BLOCK)
want[KROW, :] = ref[0, :]
check(want)
"""

SHAPES = {
    "uniform_column": UNIFORM_COLUMN,
    "uniform_row": UNIFORM_ROW,
    "folded_column": FOLDED_COLUMN,
    "store_uniform_row": STORE_UNIFORM_ROW,
}

MASKS = {
    "nomask": "",
    "mask": ", mask=(i[:, None] < N)",
    "other": ", mask=(i[:, None] < N), other=0.0",
}


def _run(tmp_path, name, source):
    path = tmp_path / f"{name}.py"
    path.write_text(HEAD + textwrap.dedent(source))
    env = dict(os.environ)
    # A per-case cache keeps one case's compile from masking the next one's.
    env["TRITON_CACHE_DIR"] = str(tmp_path / "cache" / name)
    return subprocess.run([sys.executable, str(path)], capture_output=True,
                          text=True, env=env, timeout=300)


@pytest.mark.parametrize("shape", sorted(SHAPES))
@pytest.mark.parametrize("mask", sorted(MASKS))
@pytest.mark.parametrize("num_warps", [1, 4])
def test_uniform_axis_address(tmp_path, shape, mask, num_warps):
    """Compiles, runs, and matches torch — and above all never exits on a signal.

    `num_warps` is swept because it decides how many layouts a kernel has: at one
    warp 32 threads cannot cover a 64-element tile one element each, so Triton
    mints a second encoding. The same address shape was measured NAMED at one
    warp count and KILLER at another before the shared predicate landed.
    """
    name = f"{shape}_{mask}_nw{num_warps}"
    src = SHAPES[shape].format(mask=MASKS[mask], nw=num_warps)
    proc = _run(tmp_path, name, src)

    # Negative return codes are signals (-11 SIGSEGV, -6 SIGABRT).
    assert proc.returncode >= 0, (
        f"{name}: the Metal backend killed the process with signal "
        f"{-proc.returncode} instead of reporting a reason.\n"
        f"stderr tail:\n{proc.stderr[-2000:]}")
    assert proc.returncode == 0, (
        f"{name}: exit {proc.returncode}\n{proc.stderr[-2000:]}")
    assert "WRONG" not in proc.stdout, f"{name}: {proc.stdout.strip()}"
