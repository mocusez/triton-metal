"""L1d2d — 8-cell tg_load_indexed re-diagnosis probe.

Per `.omc/specs/deep-interview-leet-triton-l1d2d-tg-load-rediagnosis.md`,
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
D4  non-id       absent   in-warp     -> ?
D5  non-id       absent   cross-warp  -> ?
D6  non-id       present  in-warp     -> likely FAIL if Apple's race is per-warp-shuffle
D7  non-id       present  cross-warp  -> EXPECTED FAIL (L1d2 anchor)

Cells are hand-crafted .mlir files under
`test/Dialect/Metal/l1d2d_probe/cell_D{0..7}.mlir`. The harness fires each
through `triton-metal-opt | triton-metal-translate --mlir-to-msl`,
compiles to a metallib via `libmetal.compile_msl_to_metallib`, and
dispatches with deterministic `arange(N)` input. Each cell runs ITERS
times (>= 10) to surface nondeterministic races.
"""
from __future__ import annotations

import os
import pathlib
import struct
import subprocess

import pytest

libmetal = pytest.importorskip(
    "triton._C.libtriton.metal",
    reason="Metal backend pybind module not built into libtriton",
)
if not hasattr(libmetal, "launch_kernel_with_pipeline"):
    pytest.skip(
        "Metal runtime not compiled (non-Darwin build or Xcode CLT absent)",
        allow_module_level=True,
    )

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
PROBE_DIR = REPO_ROOT / "test" / "Dialect" / "Metal" / "l1d2d_probe"
BUILD_BIN = REPO_ROOT / "build" / "cmake.macosx-11.0-arm64-cpython-3.12" / "bin"
METAL_OPT = BUILD_BIN / "triton-metal-opt"
METAL_TRANSLATE = BUILD_BIN / "triton-metal-translate"

ITERS = 10

# Cell metadata: (cell_id, threadgroup_size, num_threadgroups, expected_outcome_doc).
# Most cells dispatch as 1 threadgroup. D7mtg dispatches as 2 threadgroups
# (multi-tg grid matching L1d2c Phase A C6's exact MSL shape).
CELLS = [
    ("D0", 16, 1, "PASS"),
    ("D1", 64, 1, "PASS"),
    ("D2", 16, 1, "PASS"),
    ("D3", 64, 1, "PASS"),
    ("D4", 16, 1, "?"),
    ("D5", 64, 1, "?"),
    ("D6", 16, 1, "likely-FAIL-if-in-warp-race"),
    ("D7", 64, 1, "EXPECTED-FAIL-L1d2-anchor"),
    # Extension probes (added after the cube revealed D7 passes):
    # D7mask = D7 + masked-store wrap downstream of trailing barrier
    #          (tests Phase A's `if(mask){devstore}` shape hypothesis).
    # D7mtg  = D7mask + multi-threadgroup grid + local-index = id.x-tgid.x*64
    #          (full L1d2c Phase A C6 MSL shape reproduction).
    ("D7mask", 64, 1, "EXT: tests if(mask){devstore} wrap shape"),
    ("D7mtg",  64, 2, "EXT: D7mask + multi-tg + lid=id.x-tgid.x*64"),
    ("D7scratch", 64, 2, "EXT: D7mtg + Phase-B scratch RMW select-on-value (full current C6 MSL shape)"),
]


def _is_non_identity(cell_id: str) -> bool:
    """Cells D4-D7* use a non-identity (transpose) index expression."""
    return cell_id in {"D4", "D5", "D6", "D7", "D7mask", "D7mtg", "D7scratch"}


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
    """Compile MSL -> metallib, allocate IO buffers, launch with
    grid=(threadgroup_size*num_threadgroups,1,1) and tg=(threadgroup_size,1,1),
    copy out, return as list of len(in_floats)."""
    metallib = libmetal.compile_msl_to_metallib(msl)
    lib_handle, fn_handle, pso_handle, _max_threads = libmetal.load_metallib(
        metallib, kernel_name
    )
    in_bytes = b"".join(struct.pack("<f", x) for x in in_floats)
    nbytes = len(in_bytes)
    n = len(in_floats)
    in_buf = libmetal.alloc_buffer(nbytes)
    out_buf = libmetal.alloc_buffer(nbytes)
    libmetal.copy_h2d(in_buf, in_bytes)
    libmetal.copy_h2d(out_buf, b"\x00" * nbytes)
    try:
        libmetal.launch_kernel_with_pipeline(
            pso_handle, [in_buf, out_buf],
            (threadgroup_size * num_threadgroups, 1, 1),
            (threadgroup_size, 1, 1),
        )
        raw = libmetal.copy_d2h(out_buf, nbytes)
    finally:
        libmetal.free_buffer(in_buf)
        libmetal.free_buffer(out_buf)
        libmetal.free_pipeline(pso_handle)
        libmetal.free_function(fn_handle)
        libmetal.free_library(lib_handle)
    return [struct.unpack_from("<f", raw, 4 * i)[0] for i in range(n)]


@pytest.mark.parametrize("cell_id,threadgroup_size,num_threadgroups,doc", CELLS)
def test_l1d2d_cell(cell_id, threadgroup_size, num_threadgroups, doc, request):
    """Run a single cell 10 times, record pass/fail count."""
    msl = _translate_to_msl(cell_id)
    dump_path = pathlib.Path(f"/tmp/l1d2d_dump/{cell_id}.msl")
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

    # Cells with known lane-aliasing failures: mark xfail until L1d2e ships.
    # Spec predicted D6/D7 would fail; the actual cube run shows D5
    # (cross-warp, non-identity, no-barrier) is the race-prone bare-cube
    # cell. D7mtg is the L1d2 anchor reproduction.
    if cell_id in {"D5", "D6", "D7", "D7mask", "D7mtg", "D7scratch"} and fail_runs:
        pytest.xfail(
            f"L1d2d {cell_id} reproduces lane-aliasing race "
            f"({pass_count}/{ITERS} runs PASS); pending L1d2e fix"
        )
    assert pass_count == ITERS, (
        f"L1d2d {cell_id} unexpected failure: {pass_count}/{ITERS} runs PASS"
    )
