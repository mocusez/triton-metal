#!/usr/bin/env python3
"""Verify a captured TTGIR artifact matches the shape the metal-translate
TTGIR→MSL pipeline expects.

Looks under the given dump directory for any file with TTGIR-shaped name
(`*.ttgir` or `*ttgir*`) and asserts the content contains all of:

  - `tt.func`
  - `tt.get_program_id`
  - `tt.load`
  - `tt.store`
  - `arith.addf`
  - `#triton_gpu.blocked<`

Exits 0 on success, 1 with a list of missing tokens on failure.

This is the env-setup acceptance gate (AC4) from
`.omc/specs/deep-interview-triton-dev-env-ttgir.md`.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Each entry is a tuple of acceptable alternatives — Triton's MLIR printer
# uses `#ttg.blocked<` as the current alias for the triton_gpu blocked
# layout, while older versions printed `#triton_gpu.blocked<`. The shape
# check accepts either.
EXPECTED_TOKENS = (
    ("tt.func",),
    ("tt.get_program_id",),
    ("tt.load",),
    ("tt.store",),
    ("arith.addf",),
    ("#ttg.blocked<", "#triton_gpu.blocked<"),
)


def find_ttgir_files(root: Path) -> list[Path]:
    matches: list[Path] = []
    for dirpath, _, files in os.walk(root):
        for name in files:
            if "ttgir" in name.lower():
                matches.append(Path(dirpath) / name)
    return sorted(matches)


def check_file(path: Path) -> tuple[bool, list[str]]:
    text = path.read_text(errors="replace")
    missing: list[str] = []
    for alternatives in EXPECTED_TOKENS:
        if not any(alt in text for alt in alternatives):
            missing.append(" | ".join(alternatives))
    return (not missing), missing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        type=Path,
        help="Directory to scan for TTGIR artifacts (typically TRITON_KERNEL_DUMP_DIR / TRITON_DUMP_DIR).",
    )
    args = parser.parse_args()

    if not args.root.exists():
        print(f"FAIL: dump dir does not exist: {args.root}", file=sys.stderr)
        return 1

    files = find_ttgir_files(args.root)
    if not files:
        print(
            f"FAIL: no TTGIR-shaped files found under {args.root}. "
            "Was the kernel actually compiled? Is TRITON_KERNEL_DUMP=1 set?",
            file=sys.stderr,
        )
        return 1

    print(f"Scanning {len(files)} candidate file(s) under {args.root}:")
    for p in files:
        print(f"  - {p}")

    any_passed = False
    for p in files:
        ok, missing = check_file(p)
        status = "OK" if ok else "MISS"
        print(f"[{status}] {p}")
        if missing:
            print(f"        missing tokens: {missing}")
        if ok:
            any_passed = True

    if any_passed:
        print("PASS: at least one TTGIR artifact contains the full expected op-name set.")
        return 0
    print(
        f"FAIL: no file contained the full expected op-name set: {list(EXPECTED_TOKENS)}",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
