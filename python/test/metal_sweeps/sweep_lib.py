"""Crash-tolerant differential sweep harness for the Metal backend.

A case the backend declines INSIDE `applyFullConversion` does not raise — it
takes the process down, and it does so INTERMITTENTLY. A plain loop therefore
stops at the first such case and hides every case after it.

So each case records its tag to a progress file before running. `drive.sh`
re-invokes the script with that tag added to `SWEEP_SKIP` until the script
completes, and reports the crashed tags as a category of their own. The
progress file's age is also how `drive.sh` notices a cell that never ends -- a
GPU hang blocks in `torch.mps.synchronize()` and ran 6.5 hours once -- so the
script writes its own pid for the watchdog to kill. That is what
turned 40 process kills in the `tl.dot` sweep into clean skips and exposed the
14 wrong answers hiding behind them.

A compile-time REJECTION is not a bug and is counted separately from a WRONG
ANSWER. The point of a sweep is answers that differ from the oracle, not
coverage — see `probe_lib.py` when the question IS coverage.

Output goes to `$SWEEP_OUT_DIR`, defaulting to `<repo>/build/metal-sweeps`
(`build/` is gitignored), never next to the sweep script.
"""

import os
import sys
import traceback


def out_dir():
    """Directory for `<name>.progress` / `<name>.results`, created on demand."""
    d = os.environ.get("SWEEP_OUT_DIR")
    if not d:
        here = os.path.dirname(os.path.abspath(__file__))
        repo = os.path.abspath(os.path.join(here, os.pardir, os.pardir, os.pardir))
        d = os.path.join(repo, "build", "metal-sweeps")
    os.makedirs(d, exist_ok=True)
    return d


class Sweep:
    def __init__(self, name):
        self.name = name
        d = out_dir()
        self.progress = os.path.join(d, f"{name}.progress")
        self.results = os.path.join(d, f"{name}.results")
        self.skip = set(x for x in os.environ.get("SWEEP_SKIP", "").split(",") if x)
        self.wrong = self.rejected = self.ok = 0
        # The python process itself, not whatever launcher wrapped it: killing
        # `pixi run` leaves its child blocked on the GPU.
        self.pidfile = os.path.join(d, f"{name}.pid")
        with open(self.pidfile, "w") as f:
            f.write(str(os.getpid()))

    def run(self, tag, fn):
        """Run one cell. `fn()` returns None for OK, or a message for WRONG.

        The tag must encode EVERY swept axis. That is what makes the results
        file greppable, and it is the only way to show that (say) every WRONG
        carries one particular level of one particular axis.
        """
        if tag in self.skip:
            return
        with open(self.progress, "w") as f:
            f.write(tag)
            f.flush()
            os.fsync(f.fileno())
        try:
            msg = fn()
        except Exception as e:
            first = (str(e).strip().splitlines() or [""])[0]
            self.rejected += 1
            self._log(f"REJECT\t{tag}\t{first[:90]}")
            return
        except BaseException:
            self.rejected += 1
            self._log(f"REJECT\t{tag}\t{traceback.format_exc().splitlines()[-1][:90]}")
            return
        if msg:
            self.wrong += 1
            self._log(f"WRONG\t{tag}\t{msg}")
        else:
            self.ok += 1

    def _log(self, line):
        with open(self.results, "a") as f:
            f.write(line + "\n")
        print(line, flush=True)

    def finish(self):
        os.path.exists(self.progress) and os.remove(self.progress)
        os.path.exists(self.pidfile) and os.remove(self.pidfile)
        # Put the cell count in the RESULTS file, not only on stdout. Only a
        # WRONG or a REJECT writes a line, so a sweep that finds nothing leaves
        # an EMPTY results file -- which reads exactly like a sweep that never
        # ran, and `drive.sh` discards stdout. "0 wrong answers" is only
        # evidence when it comes with the number of cells that produced it.
        line = (f"SUMMARY\t{self.name}\tok={self.ok}\tWRONG={self.wrong}"
                f"\trejected={self.rejected}\tskipped={len(self.skip)}")
        with open(self.results, "a") as f:
            f.write(line + "\n")
        print(line, flush=True)


if __name__ == "__main__":
    print(out_dir())
    sys.exit(0)
