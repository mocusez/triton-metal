#!/bin/zsh
# Re-invoke a sweep until it completes, skipping whatever crashed the process
# and whatever never finished.
#
# Needed because an in-conversion decline on this backend does not raise — it
# SIGSEGVs or SIGABRTs the process, so a plain run stops at the first such cell
# and every cell after it goes unmeasured.
#
# A cell can also HANG: a GPU kernel that never completes blocks the process in
# `torch.mps.synchronize()` forever (one ran 6.5 hours before anyone noticed),
# and a compiler loop does the same at compile time. A watchdog kills the
# process once one cell's progress marker is older than `SWEEP_CELL_TIMEOUT`
# seconds, records the cell as HANG, and skips it on the next attempt.
#
# usage: drive.sh <sweep.py>
#        SWEEP_OUT_DIR=<dir> drive.sh <sweep.py>
# env:   SWEEP_CELL_TIMEOUT  seconds one cell may run (default 300)
#        SWEEP_MAX_ATTEMPTS  re-invocations before giving up (default 40)
script=$1
[[ -z "$script" ]] && { print -u2 "usage: drive.sh <sweep.py>"; exit 2 }
name=$(basename $script .py)
zmodload zsh/datetime

here=${0:A:h}
repo=${here:h:h:h}
outdir=${SWEEP_OUT_DIR:-$repo/build/metal-sweeps}
mkdir -p $outdir
export SWEEP_OUT_DIR=$outdir

py=${SWEEP_PY:-"pixi run python"}
timeout=${SWEEP_CELL_TIMEOUT:-300}
max_attempts=${SWEEP_MAX_ATTEMPTS:-40}
progress=$outdir/$name.progress
pidfile=$outdir/$name.pid

rm -f $outdir/$name.results $progress $pidfile
skip=""
crashed=()
hung=()
finished=0
for attempt in {1..$max_attempts}; do
  SWEEP_SKIP="$skip" ${=py} -u $script >/dev/null 2>&1 &
  launcher=$!
  timed_out=""
  while kill -0 $launcher 2>/dev/null; do
    sleep 1
    [ -f $progress ] || continue
    age=$(( EPOCHSECONDS - $(stat -f %m $progress 2>/dev/null || echo $EPOCHSECONDS) ))
    if [ $age -gt $timeout ]; then
      timed_out=$(cat $progress)
      [ -f $pidfile ] && kill -9 $(cat $pidfile) 2>/dev/null
      kill -9 $launcher 2>/dev/null
      break
    fi
  done
  wait $launcher 2>/dev/null
  code=$?
  rm -f $pidfile
  if [ -n "$timed_out" ]; then
    hung+=("$timed_out")
    print "HANG\t$timed_out\tno result after ${timeout}s; killed" >> $outdir/$name.results
    if [ -z "$skip" ]; then skip="$timed_out"; else skip="$skip,$timed_out"; fi
    rm -f $progress
    continue
  fi
  if [ $code -eq 0 ]; then finished=1; break; fi
  if [ -f $progress ]; then
    tag=$(cat $progress)
    crashed+=("$tag")
    if [ -z "$skip" ]; then skip="$tag"; else skip="$skip,$tag"; fi
    rm -f $progress
  else
    echo "  (exit $code with no progress marker; stopping)"; break
  fi
done
echo "--- $name ---"
echo "results: $outdir/$name.results"
sort -u $outdir/$name.results 2>/dev/null | grep -c "^WRONG" | sed 's/^/WRONG answers (unique): /'
sort -u $outdir/$name.results 2>/dev/null | grep "^WRONG" | head -20
sort -u $outdir/$name.results 2>/dev/null | grep -c "^REJECT" | sed 's/^/rejections (unique): /'
grep "^SUMMARY" $outdir/$name.results 2>/dev/null || echo "SUMMARY: none -- the sweep never reached the end"
[[ $finished -eq 0 && $attempt -eq $max_attempts ]] && echo "  (stopped after $max_attempts attempts; raise SWEEP_MAX_ATTEMPTS to go further)"
echo "CRASHED cases: ${#crashed[@]}"
for c in $crashed; do echo "  CRASH $c"; done
echo "HUNG cases: ${#hung[@]}"
for c in $hung; do echo "  HANG $c"; done
# A crashed cell is a P0 gap: the decline is in the wrong place. A hung cell is
# one too: nothing may take a GPU or the compiler down with it.
[[ ${#crashed[@]} -gt 0 || ${#hung[@]} -gt 0 ]] && exit 1
exit 0
