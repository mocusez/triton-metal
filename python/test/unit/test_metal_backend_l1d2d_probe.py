"""L1d2d — tg_load_indexed lane-aliasing regression matrix.

Per the implementation notes,
re-diagnose the Apple Metal lane-aliasing miscompile after L1d2c Phase B's
3-variant sweep disproved Phase A's masked-store-shape hypothesis. New
working hypothesis: the trigger is `metal.tg_load_indexed` with a
non-identity index expression reading post-barrier threadgroup memory.

Each cell varies exactly one of three binary axes:

  * B1 (index permutation): identity vs non-identity (transpose)
  * B2 (barrier-before-load): present vs absent
  * B3 (warp-locality): in-warp (16-thread / 1-warp tg) vs cross-warp (64-thread / 2-warp tg)

D0  identity     absent   in-warp     -> PASS
D1  identity     absent   cross-warp  -> PASS
D2  identity     present  in-warp     -> PASS
D3  identity     present  cross-warp  -> PASS
D4  non-id       absent   in-warp     -> translation only (unsynchronized)
D5  non-id       absent   cross-warp  -> translation only (unsynchronized)
D6  non-id       present  in-warp     -> PASS
D7  non-id       present  cross-warp  -> PASS

Cells are hand-crafted .mlir files under
`test/Dialect/Metal/l1d2d_probe/cell_D{0..7}.mlir`. The harness fires each
through `triton-metal-opt | triton-metal-translate --mlir-to-msl`,
compiles defined-memory cells via `torch.mps.compile_shader`, and dispatches
on MPS tensors with deterministic `arange(N)` input. Each GPU cell runs ITERS
times. D4/D5 remain historical translation fixtures: being in one SIMD group
does not supply the missing memory synchronization. D7scratch also remains
translation-only because its scratch read precedes initialization. Passing
dispatches of these historical probes cannot establish numerical correctness.
"""
from __future__ import annotations

import pathlib
import re
import subprocess

import pytest
import torch

pytest.importorskip(
    "triton._C.libtriton.metal",
    reason="Metal backend pybind module not built into libtriton",
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
PROBE_DIR = REPO_ROOT / "test" / "Dialect" / "Metal" / "l1d2d_probe"
BUILD_BIN = REPO_ROOT / "build" / "cmake.macosx-11.0-arm64-cpython-3.12" / "bin"
METAL_OPT = BUILD_BIN / "triton-metal-opt"
METAL_TRANSLATE = BUILD_BIN / "triton-metal-translate"

ITERS = 30

# GPU cell metadata: (cell_id, threadgroup_size, num_threadgroups, description).
# D4/D5 have unsynchronized cross-lane reads; D7scratch reads uninitialized
# scratch. Keep their original IR in the translation-only tests below.
# Most cells dispatch as 1 threadgroup. D7mtg dispatches as 2 threadgroups
# (multi-tg grid matching L1d2c Phase A C6's exact MSL shape).
CELLS = [
    ("D0", 16, 1, "PASS"),
    ("D1", 64, 1, "PASS"),
    ("D2", 16, 1, "PASS"),
    ("D3", 64, 1, "PASS"),
    ("D6", 16, 1, "PASS"),
    ("D7", 64, 1, "PASS"),
    # Extension probes (added after the cube revealed D7 passes):
    # D7mask = D7 + masked-store wrap downstream of trailing barrier
    #          (tests Phase A's `if(mask){devstore}` shape hypothesis).
    # D7mtg  = D7mask + multi-threadgroup grid + local-index = id.x-tgid.x*64
    #          (full L1d2c Phase A C6 MSL shape reproduction).
    ("D7mask", 64, 1, "EXT: tests if(mask){devstore} wrap shape"),
    ("D7mtg",  64, 2, "EXT: D7mask + multi-tg + lid=id.x-tgid.x*64"),
]


def _is_non_identity(cell_id: str) -> bool:
    """Numerically tested cells with a transpose index expression."""
    return cell_id in {"D6", "D7", "D7mask", "D7mtg"}


def _block_n(threadgroup_size: int) -> int:
    """BLOCK_N for the transpose permutation. tg of 16 -> 4x4; tg of 64 -> 8x8."""
    return 4 if threadgroup_size == 16 else 8


def _reference(cell_id: str, threadgroup_size: int, num_threadgroups: int) -> list[float]:
    """Algorithmic reference: out[gid] = in[(local_idx(lid))+tgid*tg_size]
    where lid = gid % tg_size. For identity cells idx==lid, for non-identity
    idx==(lid%N)*N + lid/N (transpose-within-threadgroup)."""
    N = threadgroup_size
    total = N * num_threadgroups
    out = []
    for gid in range(total):
        tgid = gid // N
        lid = gid - tgid * N
        if _is_non_identity(cell_id):
            n = _block_n(N)
            idx_in_tg = (lid % n) * n + lid // n
        else:
            idx_in_tg = lid
        out.append(float(tgid * N + idx_in_tg))
    return out


def _translate_to_msl(cell_id: str) -> str:
    """Run triton-metal-opt | triton-metal-translate --mlir-to-msl."""
    src = PROBE_DIR / f"cell_{cell_id}.mlir"
    assert src.exists(), f"missing probe MLIR: {src}"
    opt_proc = subprocess.run(
        [str(METAL_OPT), str(src)], capture_output=True, check=True, text=True
    )
    translate_proc = subprocess.run(
        [str(METAL_TRANSLATE), "--mlir-to-msl"],
        input=opt_proc.stdout, capture_output=True, check=True, text=True,
    )
    return translate_proc.stdout


def _dispatch(msl: str, kernel_name: str, threadgroup_size: int,
              num_threadgroups: int, in_floats: list[float]) -> list[float]:
    """Compile MSL via torch.mps.compile_shader and dispatch with
    threads=(threadgroup_size*num_threadgroups,1,1),
    group_size=(threadgroup_size,1,1) on MPS tensors; return the output as a
    list of len(in_floats). The cell kernels take two `device float*` buffers
    (in, out), bound positionally exactly as the live MetalLauncher binds them.
    """
    lib = torch.mps.compile_shader(msl)
    kfn = getattr(lib, kernel_name)
    n = len(in_floats)
    in_t = torch.tensor(in_floats, dtype=torch.float32, device="mps")
    out_t = torch.zeros(n, dtype=torch.float32, device="mps")
    kfn(
        in_t, out_t,
        threads=(threadgroup_size * num_threadgroups, 1, 1),
        group_size=(threadgroup_size, 1, 1),
    )
    torch.mps.synchronize()
    return out_t.cpu().tolist()


@pytest.mark.parametrize("diagnostic,synchronized", [("D4", "D6"), ("D5", "D7")])
def test_l1d2d_transpose_barrier_precedes_shared_read(diagnostic, synchronized):
    """Retain the historical contrast without dispatching a racy shader."""
    for cell_id in (diagnostic, synchronized):
        msl = _translate_to_msl(cell_id)
        buffer = re.search(r"threadgroup float (\w+)\[\d+\];", msl)
        assert buffer, msl
        name = buffer.group(1)
        store = re.search(rf"{name}\[[^\n]+\] = [^\n]+;", msl)
        load = re.search(rf"float \w+ = {name}\[[^\n]+\];", msl)
        assert store and load, msl
        assert store.end() < load.start(), msl
        barriers = msl[store.end():load.start()].count(
            "threadgroup_barrier(mem_flags::mem_threadgroup)")
        assert barriers == (1 if cell_id == synchronized else 0), msl
        # A trailing barrier cannot order the preceding cross-lane read.
        assert "threadgroup_barrier(mem_flags::mem_threadgroup)" in msl[load.end():]


def test_l1d2d_uninitialized_scratch_fixture_translates_without_dispatch():
    """Preserve the original reproduction, including its undefined read."""
    msl = _translate_to_msl("D7scratch")
    buffers = re.findall(r"threadgroup float (\w+)\[64\];", msl)
    assert len(buffers) == 2, msl
    scratch = buffers[0]
    load = re.search(rf"float \w+ = {scratch}\[[^\n]+\];", msl)
    store = re.search(rf"{scratch}\[[^\n]+\] = [^\n]+;", msl)
    assert load and store, msl
    assert load.end() < store.start(), msl


@pytest.mark.skipif(not torch.backends.mps.is_available(),
                    reason="Metal backend requires an MPS-enabled PyTorch")
@pytest.mark.parametrize("cell_id,threadgroup_size,num_threadgroups,doc", CELLS)
def test_l1d2d_cell(cell_id, threadgroup_size, num_threadgroups, doc):
    """Run a single cell repeatedly and require bit-exact output."""
    msl = _translate_to_msl(cell_id)
    dump_path = pathlib.Path(f"/tmp/l1d2d_dump/{cell_id}.metal")
    dump_path.parent.mkdir(parents=True, exist_ok=True)
    dump_path.write_text(msl)

    n_total = threadgroup_size * num_threadgroups
    in_floats = [float(i) for i in range(n_total)]
    expected = _reference(cell_id, threadgroup_size, num_threadgroups)
    pass_count = 0
    fail_runs: list[tuple[int, list[float]]] = []
    for it in range(ITERS):
        out = _dispatch(msl, f"cell_{cell_id}", threadgroup_size,
                        num_threadgroups, in_floats)
        if out == expected:
            pass_count += 1
        else:
            fail_runs.append((it, out))
    summary = (
        f"[L1d2d {cell_id}] {pass_count}/{ITERS} runs PASS"
        f" (tg={threadgroup_size}x{num_threadgroups}, doc={doc!r})"
    )
    print(summary)
    if fail_runs:
        print(f"  first failing run #{fail_runs[0][0]}: out={fail_runs[0][1]}")
        print(f"  expected:                       {expected}")
        if len(fail_runs) > 1:
            for it, out in fail_runs[1:]:
                if out != fail_runs[0][1]:
                    print(f"  distinct failing pattern run #{it}: out={out}")
                    break

    assert pass_count == ITERS, (
        f"L1d2d {cell_id} unexpected failure: {pass_count}/{ITERS} runs PASS"
    )
