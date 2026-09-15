# Differential sweeps for the Metal backend

Two harnesses, two questions.

| harness | question | run it with |
|---|---|---|
| `sweep_lib.py` | does the backend give the RIGHT answer across a grid? | `./drive.sh sweeps/s_foo.py` |
| `probe_lib.py` | where does the backend REFUSE, and does it refuse properly? | `pixi run --frozen python sweeps/probe_foo.py` |

Both write to `$SWEEP_OUT_DIR`, defaulting to `<repo>/build/metal-sweeps`
(`build/` is gitignored). Nothing a sweep produces belongs in the tree.

## Why the driver exists

A decline inside `applyFullConversion` does not raise — it **SIGSEGVs or
SIGABRTs the process**. A plain loop stops at the first unsupported shape and
every case after it goes unrun. That is not hypothetical: converting 40 such
crashes in the `tl.dot` sweep into clean skips is what let it finish, and it
immediately surfaced **14 wrong answers that had been hiding behind them**.

`drive.sh` re-invokes the script with whatever crashed added to `SWEEP_SKIP`,
up to 40 times, then prints unique WRONG answers, unique rejections, and the
crashed tags. It exits non-zero if anything crashed, because a crashed cell is a
P0 gap on its own: the decline is in the wrong place.

A compile-time REJECTION is not a bug, and is counted separately. Only a WRONG
ANSWER is.

## The crash is intermittent, so one run proves nothing

`flaky.sh <script.py> [runs=30]` histograms exit codes: `0` ran, `1` clean
reject, `134` SIGABRT, `139` SIGSEGV. Measured on the 128x32 masked-load gap,
six runs of ONE unchanged shape gave `139 134 139 134 139 139` on one build and
`139 139 139 1 1 1` on the next. Both are "~50% crash"; neither run alone says
so. Before blaming a change for a crash, histogram both builds.

## The verdict vocabulary

`probe_lib.classify` assigns exactly one of:

| verdict | meaning |
|---|---|
| `OK` | compiled and ran |
| `WRONG` | compiled, ran, did not match the oracle — a **false capability claim** |
| `NAMED` | exit 1 carrying a `Metal backend: …` diagnostic — a good rejection |
| `UNNAMED` | exit 1 with no named reason — reaches conversion teardown, **P0 gap** |
| `KILLER` | died on a signal — **P0 gap** |
| `FRONT` | the Triton **frontend** refused it; not a backend gap |

`FRONT` is not a nicety. In the round-11 audit 12 of 164 probes were
`ast_to_ttir` refusing f16 `floor`/`ceil`/`sigmoid` before the backend saw
anything. A harness that only greps for "error" files all twelve as backend
gaps.

## Oracles

`torch` on CPU for the reference. When a kernel's own correctness is in doubt,
`TRITON_INTERPRET=1` runs it through Triton's reference semantics on CPU — that
is what distinguishes "my kernel is wrong" from "the backend is wrong", and it
settled the transposed-B bug in one run.

## Where the bugs are

**Degenerate runtime extents, above all `N == 1`.** Triton specializes an
argument equal to 1 into a constant AND flips the tile order, so that corner
exercises different code than any power-of-two shape.

**A runtime dim NOT divisible by 16.** Divisibility is the other thing Triton
specializes on, and losing it does not degrade gently — it mints a whole second
`#ttg.blocked` with the opposite `order` for the rank-1 column cone.

Both corners work by changing which **layouts** exist, not which shapes. That is
why they keep paying out and why sweeping tile extents alone does not.

⚠️ **Sweep these against more than one `num_warps`.** `BLOCK_N == tpb` collapses
the second layout and the bug vanishes — D=65 is exact at `nw=4` and wrong at
`nw=8`, and the in-tree suite happened to hold only the hiding cells for both
that bug and its `tl.cat` companion.

**A cooperative pass indexed by thread id, when the tile is wider than the
threadgroup.** The Metal emitters publish a tile by having each thread take
`_fa_tid`, `ltid.x`, or a lane id and walk `for (c = tid; c < SIZE; c += tpb)`.
That idiom is correct at any size. The defects come from the places that wrote
`q = tid` under an `if (q < BM)` BOUND instead — a bound is not a loop, and the
rows past the threadgroup width are silently never computed. `metal.fused_attention`
had this in six places at once (`s_attn_rows.py`), because the schedule caps the
launch at `min(num_warps, BM/8) * 32` and nothing raises it back to `BM`.

⚠️ So **sweep the block extent against `num_warps`, not just against the shape**:
the discriminator is a RELATION between the two, and every cell where the source
asked for enough warps hides it. In that family only `BM = 64, num_warps = 1`
was wrong; `BM = 64` at 2, 4 and 8 warps were all exact.

⚠️ **Ordering primitives must be fed a permutation.** `tl.sort`, `tl.topk` and
anything built on them return the correct prefix of ALREADY SORTED input even
when broken, so an `arange`-fed sweep reports OK for cells that are measurably
wrong. This is not a mistake you make once: it hid the topk budget-fallback
defect for an entire round, then hid it again in a hand-written re-check by
someone who had just read this warning. Use `torch.randperm` under a fixed seed,
and before trusting any OK from an ordering primitive, look at the input the
probe actually fed it.

## Writing a sweep

```python
"""<what axis this sweeps, WHY it is the axis, and the MEASURED before/after.>"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
import torch, triton
import triton.language as tl
from sweep_lib import Sweep

@triton.jit
def kernel(...): ...

sw = Sweep("s_myaxis")
torch.manual_seed(0)
for D in DS:
    for nw in (1, 2, 4, 8):
        tag = f"D{D}_nw{nw}_div16{'Y' if D % 16 == 0 else 'n'}"
        def f(D=D, nw=nw):              # bind loop vars as defaults
            o = torch.zeros(..., device="mps")
            kernel[(1,)](..., num_warps=nw)
            torch.mps.synchronize()
            err = (o.cpu() - reference).abs().max().item()
            return None if err < 1e-5 else f"maxerr={err:.2e}"
        sw.run("case_" + tag, f)
sw.finish()
```

**The tag must encode every swept axis.** That is what makes the results file
greppable, and it is the only way to show a claim like "every WRONG carries
`div16n`".

## Writing a probe round

```python
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
import probe_lib as P

P.SCRATCH = P.scratch_dir("myround")
TEMPLATE = """
@triton.jit
def k(X, O, BLOCK: tl.constexpr):
    ...
"""
for ...:
    P.PROBES[tag] = TEMPLATE.format(...)

sys.exit(P.main())
```

`REPEAT=30 pixi run --frozen python sweeps/probe_foo.py <tag>` re-runs one probe
to settle an intermittent verdict; a `KILLER` in any repeat wins.
