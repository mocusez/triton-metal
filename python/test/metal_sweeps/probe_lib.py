"""Subprocess probe harness with the decline-verdict taxonomy.

`sweep_lib.py` answers "does the backend give the RIGHT answer". This answers
"where does the backend REFUSE, and does it refuse properly" — the audit that
found every P0 gap so far.

Each probe is a standalone source string run in its own subprocess, because an
in-conversion decline kills the process and would otherwise take the whole run
with it. The verdict vocabulary is the point:

  OK       compiled and ran
  WRONG    compiled, ran, and did not match the oracle — a false capability claim
  NAMED    exit 1 carrying a "Metal backend: ..." diagnostic — a good rejection
  UNNAMED  exit 1 with no named reason — reaches conversion teardown, P0 gap
  KILLER   died on a signal — P0 gap, and the signal varies between runs
  HANG     no result within PROBE_TIMEOUT seconds (default 300); killed — P0 gap
  FRONT    the Triton FRONTEND refused it; not a backend gap

FRONT is not a nicety. In the round-11 audit 12 of 164 probes were `ast_to_ttir`
refusing f16 `floor`/`ceil`/`sigmoid` before the backend saw anything; a harness
that only greps for "error" files all twelve as backend gaps.

A probe reports WRONG by printing a line containing "WRONG" — the numeric
comparison lives in the probe source, next to the oracle it uses.

Usage from a round script:

    import probe_lib as P
    P.SCRATCH = P.scratch_dir("myround")
    P.PROBES["some_tag"] = SOURCE_STRING
    P.main()
"""

import os
import subprocess
import sys
import textwrap

# Repeat each probe N times. An in-conversion decline is intermittent (measured
# at ~50% on one unchanged shape), so a single run cannot tell "I broke this"
# from "it was already broken". REPEAT=30 is the setting that answers that.
REPEAT = int(os.environ.get("REPEAT", "1"))

HEAD = "import torch, triton\nimport triton.language as tl\ntorch.manual_seed(0)\n"

PROBES = {}


def scratch_dir(round_name):
    """Per-round scratch under `$SWEEP_OUT_DIR` (default `<repo>/build/...`)."""
    from sweep_lib import out_dir

    d = os.path.join(out_dir(), f"probes_{round_name}")
    os.makedirs(d, exist_ok=True)
    return d


SCRATCH = None


def run_src(tag, src):
    path = os.path.join(SCRATCH, f"{tag}.py")
    with open(path, "w") as fh:
        fh.write(src)
    env = dict(os.environ)
    # A per-probe cache keeps one probe's compile from masking the next one's.
    env["TRITON_CACHE_DIR"] = os.path.join(SCRATCH, "cache", tag)
    # A probe that never finishes -- a GPU kernel that does not complete, or a
    # compiler loop -- is a verdict of its own, not a reason to abandon every
    # probe after it. `subprocess.run` kills the child when the timeout fires.
    try:
        return subprocess.run([sys.executable, path], capture_output=True,
                              text=True, env=env, timeout=PROBE_TIMEOUT)
    except subprocess.TimeoutExpired as e:
        def text(v):
            return v.decode(errors="replace") if isinstance(v, bytes) else (v or "")
        return subprocess.CompletedProcess(e.cmd, HANG_RC, text(e.stdout),
                                           text(e.stderr))


# Seconds one probe may run; `run_src` reports a probe past it with `HANG_RC`.
PROBE_TIMEOUT = int(os.environ.get("PROBE_TIMEOUT", "300"))
HANG_RC = -1000


def classify(rc, out, err):
    if rc == HANG_RC:
        return "HANG", f"no result after {PROBE_TIMEOUT}s; killed"
    if rc == 0:
        # The probe checked its own answer and said so.
        if "WRONG" in out or "WRONG" in err:
            line = [l for l in (out + "\n" + err).splitlines() if "WRONG" in l]
            return "WRONG", line[0].strip()[:150]
        return "OK", ""
    named = [l for l in err.splitlines() if "Metal backend:" in l]
    if rc not in (0, 1):
        return "KILLER", f"rc={rc}"
    if named:
        return "NAMED", named[0].strip()[:150]
    # A frontend refusal never reaches the backend, so it is not our gap.
    front = [l for l in err.splitlines()
             if ("Unexpected dtype" in l or "not supported" in l
                 or "AttributeError" in l or "TypeError" in l
                 or "ValueError" in l or "unsupported AST" in l
                 or "must be" in l)]
    if front and "convert-tritongpu-to-metal" not in err:
        return "FRONT", front[-1].strip()[:150]
    tail = [l for l in err.strip().splitlines() if l.strip()]
    return "UNNAMED", (tail[-1].strip()[:150] if tail else "")


def main():
    if SCRATCH is None:
        raise SystemExit("probe_lib.SCRATCH not set; call scratch_dir(<round>)")
    names = [a for a in sys.argv[1:] if not a.startswith("--")] or list(PROBES)
    tally = {}
    findings = []
    for name in names:
        codes, verdict, why = [], "OK", ""
        for _ in range(REPEAT):
            r = run_src(name, HEAD + textwrap.dedent(PROBES[name]))
            codes.append(r.returncode)
            v, w = classify(r.returncode, r.stdout, r.stderr)
            # A killer or a hang anywhere in the repeats wins: both are
            # intermittent.
            if v in ("KILLER", "HANG") or (verdict == "OK" and v != "OK"):
                verdict, why = v, w
        tally[verdict] = tally.get(verdict, 0) + 1
        line = f"{verdict:8} {name:40} {sorted(set(codes))}"
        if verdict in ("KILLER", "HANG", "UNNAMED", "WRONG"):
            findings.append(f"{line}\n         | {why}")
        print(line + (f"\n         | {why}" if verdict != "OK" else ""), flush=True)
    print("\n# " + " / ".join(f"{v} {n}" for v, n in sorted(tally.items())),
          flush=True)
    if findings:
        print("\n# P0 GAPS (crash, hang, unnamed decline, or wrong answer):",
              flush=True)
        for f in findings:
            print(f, flush=True)
    # Exit non-zero so a driver or CI step can gate on it.
    return 1 if findings else 0
