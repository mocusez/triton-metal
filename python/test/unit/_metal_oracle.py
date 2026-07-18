"""Shared oracle utilities for the Triton Metal backend.

This is a *helper module*, not a pytest test file (underscore prefix → pytest
does not collect it). It is imported by Metal-backend tests and by the
diagnostic probes that the deferred-work specs need before they can be
implemented efficiently.

It provides two independent tools that close the two recurring "unknowns" in
the Metal backend backlog:

  Part A — Blocked-layout element oracle  (`blocked_layout`)
      A pure-Python golden source for the Triton `BlockedEncodingAttr`
      thread→element mapping. Given a blocked encoding, it answers
      "which logical tensor element does (localThreadId, registerIv) own?".
      This is the independent reference for validating the C++ helper
      `emitBlockedFlatPos` that the staged-transpose work (L1d3 session-2)
      must implement — see
      `.omc/specs/deep-interview-l1d3-bailout-stub.md` (findings F2/F3).
      Without it, `emitBlockedFlatPos` would be "verified" only by the same
      hand-simulation that authored it. NO GPU / torch needed.

  Part B — Numerical divergence report  (`diff_report`)
      Turns a bare `assert_close`-style "Tensors are not close" failure into a
      structured report: where the mismatches are, by how much, and — crucially
      for layout/accumulation bugs — what *pattern* they follow (periodic
      stride, correct-prefix-then-garbage, uniform scale factor). This is the
      experiment harness the runtime-bit-exactness carry-forwards need:
        - L3 chunked reduce  (`test_metal_backend_reduce_chunked.py` xfails)
        - L3a-tileloop-2     (`test_metal_backend_reduce_per_thread.py` xfails)
      Backend-agnostic: accepts torch tensors, numpy arrays, or nested lists.

The Part-A formula is validated at import-time-free `__main__` self-test
against the hand-computed golden case in the L1d3 spec (thread 0 of a 16×16
`sizePerThread=[4,1]` transpose layout → flat positions
{0,16,32,48,8,24,40,56}).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Part A — Blocked-layout element oracle
# ---------------------------------------------------------------------------
#
# Canonical Triton `BlockedEncodingAttr` semantics (validated against upstream
# `lib/Dialect/TritonGPU/IR/Dialect.cpp` and the L1d3 F2 hand-computation):
#
#   tileSize[d] = sizePerThread[d] * threadsPerWarp[d] * warpsPerCTA[d]
#   reps[d]     = ceil(shape[d] / tileSize[d])
#
#   For a thread identified by per-dim (lane[d] in [0,T[d]), warp[d] in [0,W[d)))
#   and a per-thread register decomposed into (elem[d] in [0,S[d]),
#   rep[d] in [0,reps[d])):
#
#       coord[d] = rep[d]  * tileSize[d]
#                + warp[d] * (threadsPerWarp[d] * sizePerThread[d])
#                + lane[d] * sizePerThread[d]
#                + elem[d]
#
#   The linear local thread id decomposes over `order` (fastest dim first):
#       lane linearizes over order with radices threadsPerWarp[order[i]];
#       warp linearizes over order with radices warpsPerCTA[order[i]].
#
#   The per-thread register index `iv` linearizes (fastest first):
#       elem[order[0]], elem[order[1]], ..., rep[order[0]], rep[order[1]], ...
#
#   The returned position is the LOGICAL row-major flat index of the tensor
#   element (last dim contiguous) — this is the element *identity*, independent
#   of whatever physical scratch addressing a lowering later chooses.


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def _row_major_strides(shape: Sequence[int]) -> List[int]:
    strides = [1] * len(shape)
    for d in range(len(shape) - 2, -1, -1):
        strides[d] = strides[d + 1] * shape[d + 1]
    return strides


def _delinearize(linear: int, radices_in_order: Sequence[int],
                 order: Sequence[int]) -> List[int]:
    """Decompose a linear index into per-dim components.

    `radices_in_order[i]` is the radix for dimension `order[i]` (order[0] is the
    fastest-varying). Returns a list indexed by the natural dimension number.
    """
    comps = [0] * len(order)
    rem = linear
    for i, dim in enumerate(order):
        r = radices_in_order[i]
        comps[dim] = rem % r
        rem //= r
    return comps


@dataclass(frozen=True)
class BlockedLayout:
    """A Triton blocked encoding, with the canonical element oracle attached."""

    shape: Tuple[int, ...]
    size_per_thread: Tuple[int, ...]
    threads_per_warp: Tuple[int, ...]
    warps_per_cta: Tuple[int, ...]
    order: Tuple[int, ...]

    # derived (filled in __post_init__ via object.__setattr__ for frozen dc)
    tile_size: Tuple[int, ...] = field(default=(), compare=False)
    reps: Tuple[int, ...] = field(default=(), compare=False)

    def __post_init__(self) -> None:
        rank = len(self.shape)
        for name, v in (
            ("size_per_thread", self.size_per_thread),
            ("threads_per_warp", self.threads_per_warp),
            ("warps_per_cta", self.warps_per_cta),
            ("order", self.order),
        ):
            if len(v) != rank:
                raise ValueError(f"{name} length {len(v)} != rank {rank}")
        if sorted(self.order) != list(range(rank)):
            raise ValueError(f"order {self.order} is not a permutation of 0..{rank-1}")
        tile = tuple(
            self.size_per_thread[d] * self.threads_per_warp[d] * self.warps_per_cta[d]
            for d in range(rank)
        )
        reps = tuple(_ceil_div(self.shape[d], tile[d]) for d in range(rank))
        object.__setattr__(self, "tile_size", tile)
        object.__setattr__(self, "reps", reps)

    # -- shape facts -------------------------------------------------------
    @property
    def rank(self) -> int:
        return len(self.shape)

    @property
    def threads_per_cta(self) -> int:
        out = 1
        for t in self.threads_per_warp:
            out *= t
        for w in self.warps_per_cta:
            out *= w
        return out

    @property
    def elems_per_thread(self) -> int:
        out = 1
        for d in range(self.rank):
            out *= self.size_per_thread[d] * self.reps[d]
        return out

    # -- the oracle --------------------------------------------------------
    def per_iter_flat_position(self, local_tid: int, iv: int) -> int:
        """Logical row-major flat position owned by (local_tid, register iv).

        This is exactly the value the C++ helper `emitBlockedFlatPos(blocked,
        localTid, iv)` must compute. Validate the lowering by comparing against
        this for every (local_tid, iv) in range.
        """
        rank = self.rank
        # Decompose the local thread id into per-dim lane + warp components.
        lane = _delinearize(
            local_tid % _prod(self.threads_per_warp),
            [self.threads_per_warp[d] for d in self.order],
            self.order,
        )
        warp = _delinearize(
            local_tid // _prod(self.threads_per_warp),
            [self.warps_per_cta[d] for d in self.order],
            self.order,
        )
        # Decompose the register index into per-dim (elem, rep) components.
        # Fastest-first radices: elem[order[i]] then rep[order[i]].
        elem_radices = [self.size_per_thread[d] for d in self.order]
        rep_radices = [self.reps[d] for d in self.order]
        elem_block = _prod(self.size_per_thread)
        elem = _delinearize(iv % elem_block, elem_radices, self.order)
        rep = _delinearize(iv // elem_block, rep_radices, self.order)

        strides = _row_major_strides(self.shape)
        flat = 0
        for d in range(rank):
            coord = (
                rep[d] * self.tile_size[d]
                + warp[d] * (self.threads_per_warp[d] * self.size_per_thread[d])
                + lane[d] * self.size_per_thread[d]
                + elem[d]
            )
            flat += coord * strides[d]
        return flat

    def thread_positions(self, local_tid: int) -> List[int]:
        """All flat positions owned by a thread, in register (iv) order."""
        return [self.per_iter_flat_position(local_tid, iv)
                for iv in range(self.elems_per_thread)]

    def coverage(self) -> Dict[int, List[Tuple[int, int]]]:
        """Map flat position -> list of (local_tid, iv) that claim it.

        A correct layout covers every in-tile position exactly once. Positions
        with len != 1 reveal aliasing (the bug class L1d2c/L1d2e chased) or
        gaps. Only positions inside the populated tile region are reported.
        """
        out: Dict[int, List[Tuple[int, int]]] = {}
        for tid in range(self.threads_per_cta):
            for iv in range(self.elems_per_thread):
                pos = self.per_iter_flat_position(tid, iv)
                out.setdefault(pos, []).append((tid, iv))
        return out

    def aliasing(self) -> Dict[int, List[Tuple[int, int]]]:
        """Positions claimed by more than one (tid, iv) — should be empty."""
        return {pos: owners for pos, owners in self.coverage().items()
                if len(owners) > 1}


def _prod(xs: Sequence[int]) -> int:
    out = 1
    for x in xs:
        out *= x
    return out


# ---------------------------------------------------------------------------
# Part B — Numerical divergence report
# ---------------------------------------------------------------------------


@dataclass
class DiffReport:
    total: int
    n_mismatch: int
    examples: List[Tuple[int, Tuple[int, ...], float, float, float, float]]
    patterns: List[str]
    shape: Tuple[int, ...]
    atol: float
    rtol: float

    @property
    def fraction(self) -> float:
        return self.n_mismatch / self.total if self.total else 0.0

    @property
    def ok(self) -> bool:
        return self.n_mismatch == 0

    def __str__(self) -> str:  # human-readable summary
        if self.ok:
            return f"diff_report: MATCH ({self.total} elems, shape={self.shape})"
        lines = [
            f"diff_report: {self.n_mismatch}/{self.total} mismatch "
            f"({self.fraction:.1%}), shape={self.shape}, "
            f"atol={self.atol}, rtol={self.rtol}",
        ]
        for flat, idx, got, exp, abserr, relerr in self.examples:
            lines.append(
                f"  [{flat:>8}] idx={idx} got={got:+.6g} exp={exp:+.6g} "
                f"abs={abserr:.3g} rel={relerr:.3g}"
            )
        if self.patterns:
            lines.append("  patterns:")
            lines.extend(f"    - {p}" for p in self.patterns)
        return "\n".join(lines)


def _to_flat_lists(t):
    """Return (flat_values: list[float], shape: tuple[int,...])."""
    # torch tensor
    if hasattr(t, "detach") and hasattr(t, "cpu"):
        t = t.detach().cpu()
        shape = tuple(t.shape)
        return t.reshape(-1).tolist(), shape
    # numpy array
    if hasattr(t, "ravel") and hasattr(t, "shape"):
        return t.ravel().tolist(), tuple(t.shape)
    # nested list / sequence
    import numpy as _np  # local import; numpy is always available in this env
    a = _np.asarray(t)
    return a.ravel().tolist(), tuple(a.shape)


def _unravel(flat: int, shape: Sequence[int]) -> Tuple[int, ...]:
    idx = []
    for s in reversed(shape):
        idx.append(flat % s)
        flat //= s
    return tuple(reversed(idx))


def diff_report(got, expected, *, atol: float = 1e-4, rtol: float = 1e-4,
                max_examples: int = 10) -> DiffReport:
    """Structured element-wise divergence report between two tensors.

    `|got - exp| > atol + rtol*|exp|` defines a mismatch (same rule as
    `torch.testing.assert_close`). Returns a `DiffReport` whose `str()` is a
    diagnostic summary and whose `.patterns` flags common layout/accumulation
    bug signatures. Backend-agnostic (torch / numpy / lists).
    """
    g, gshape = _to_flat_lists(got)
    e, eshape = _to_flat_lists(expected)
    if len(g) != len(e):
        raise ValueError(f"length mismatch: got {len(g)} vs expected {len(e)}")
    shape = gshape if gshape == eshape else (len(g),)

    mismatches: List[int] = []
    examples = []
    for i, (gv, ev) in enumerate(zip(g, e)):
        abserr = abs(gv - ev)
        tol = atol + rtol * abs(ev)
        if abserr > tol:
            mismatches.append(i)
            if len(examples) < max_examples:
                relerr = abserr / abs(ev) if ev != 0 else float("inf")
                examples.append((i, _unravel(i, shape), gv, ev, abserr, relerr))

    patterns = _detect_patterns(g, e, mismatches, shape) if mismatches else []
    return DiffReport(
        total=len(g), n_mismatch=len(mismatches), examples=examples,
        patterns=patterns, shape=shape, atol=atol, rtol=rtol,
    )


def _detect_patterns(g: List[float], e: List[float], mism: List[int],
                     shape: Sequence[int]) -> List[str]:
    """Heuristics that point at the *class* of bug, not just its presence."""
    out: List[str] = []
    n = len(g)

    # 1. Correct prefix then all-wrong tail (off-by-tile / truncated emission).
    first_bad = mism[0]
    if first_bad > 0 and len(mism) == n - first_bad and mism[-1] == n - 1:
        out.append(
            f"correct prefix [0,{first_bad}) then ALL of [{first_bad},{n}) wrong "
            f"— suggests truncated/early-terminated emission or off-by-one tile bound"
        )

    # 2. Periodic mismatch stride (per-thread / per-tile-iv aliasing).
    if len(mism) >= 3:
        deltas = [mism[i + 1] - mism[i] for i in range(len(mism) - 1)]
        d0 = deltas[0]
        if d0 > 1 and all(d == d0 for d in deltas):
            out.append(
                f"mismatches are periodic with stride {d0} — points at a "
                f"per-thread/per-tile-iv indexing bug (compare stride to tpb / "
                f"sizePerThread / tile extent)"
            )

    # 3. Uniform scale factor (missing accumulation of k tiles/iters).
    ratios = []
    for i in mism[:256]:
        if g[i] != 0:
            ratios.append(e[i] / g[i])
    if ratios:
        r0 = ratios[0]
        if abs(r0) > 1e-6 and all(abs(r - r0) < 1e-3 * abs(r0) for r in ratios):
            out.append(
                f"expected ≈ {r0:.4g}× got on mismatches — suggests got summed "
                f"only 1/{r0:.4g} of the elements (missing tile-iv / chunk "
                f"accumulation)"
            )

    # 4. Magnitude summary.
    abserrs = [abs(g[i] - e[i]) for i in mism]
    out.append(
        f"abs-error range [{min(abserrs):.3g}, {max(abserrs):.3g}] over "
        f"{len(mism)} mismatched elems"
    )
    return out


# ---------------------------------------------------------------------------
# Import-free self-test (run: `python python/test/unit/_metal_oracle.py`)
# ---------------------------------------------------------------------------

def _self_test() -> None:
    # Golden case from .omc/specs/deep-interview-l1d3-bailout-stub.md, finding F2:
    # the sizePerThread>1 transpose layout. Thread 0 must own exactly
    # {0, 16, 32, 48, 8, 24, 40, 56} in row-major iv order.
    lay = BlockedLayout(
        shape=(16, 16),
        size_per_thread=(4, 1),
        threads_per_warp=(4, 8),
        warps_per_cta=(1, 1),
        order=(0, 1),
    )
    assert lay.tile_size == (16, 8), lay.tile_size
    assert lay.reps == (1, 2), lay.reps
    assert lay.elems_per_thread == 8, lay.elems_per_thread
    got = lay.thread_positions(0)
    expected = [0, 16, 32, 48, 8, 24, 40, 56]
    assert got == expected, f"L1d3 F2 oracle mismatch:\n got={got}\n exp={expected}"

    # A correct layout tiling its region must have zero aliasing.
    assert lay.aliasing() == {}, f"unexpected aliasing: {lay.aliasing()}"

    # Sanity: a trivial 1D contiguous layout (sPT=4, 32 lanes, 1 warp) over 128
    # elements — thread 0 owns the first contiguous chunk {0,1,2,3}.
    lay1d = BlockedLayout((128,), (4,), (32,), (1,), (0,))
    assert lay1d.thread_positions(0) == [0, 1, 2, 3], lay1d.thread_positions(0)
    assert lay1d.thread_positions(1) == [4, 5, 6, 7], lay1d.thread_positions(1)
    assert lay1d.aliasing() == {}

    # Part B: diff_report basics + a periodic-stride pattern.
    rep_ok = diff_report([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert rep_ok.ok and rep_ok.n_mismatch == 0

    # every other element wrong -> periodic stride 2 pattern.
    g = [1.0, 9.0, 1.0, 9.0, 1.0, 9.0]
    e = [1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
    rep = diff_report(g, e, atol=1e-6, rtol=0.0)
    assert rep.n_mismatch == 3, rep.n_mismatch
    assert any("periodic with stride 2" in p for p in rep.patterns), rep.patterns

    print("PASS: _metal_oracle self-test")
    print()
    print("Demo — Part A, thread 0 of the L1d3 transpose layout:")
    print(f"  flat positions (iv order): {got}")
    print()
    print("Demo — Part B, periodic-stride divergence report:")
    print(rep)


if __name__ == "__main__":
    _self_test()
