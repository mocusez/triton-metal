#!/bin/zsh
# Re-invoke a sweep until it completes, skipping whatever crashed the process.
#
# Needed because an in-conversion decline on this backend does not raise — it
# SIGSEGVs or SIGABRTs the process, so a plain run stops at the first such cell
# and every cell after it goes unmeasured.
#
# usage: drive.sh <sweep.py>
#        SWEEP_OUT_DIR=<dir> drive.sh <sweep.py>
script=$1
[[ -z "$script" ]] && { print -u2 "usage: drive.sh <sweep.py>"; exit 2 }
name=$(basename $script .py)

here=${0:A:h}
repo=${here:h:h:h}
outdir=${SWEEP_OUT_DIR:-$repo/build/metal-sweeps}
mkdir -p $outdir
export SWEEP_OUT_DIR=$outdir

py=${SWEEP_PY:-"pixi run python"}

rm -f $outdir/$name.results $outdir/$name.progress
skip=""
crashed=()
for attempt in {1..40}; do
  SWEEP_SKIP="$skip" ${=py} -u $script >/dev/null 2>&1
  code=$?
  if [ $code -eq 0 ]; then break; fi
  if [ -f $outdir/$name.progress ]; then
    tag=$(cat $outdir/$name.progress)
    crashed+=("$tag")
    if [ -z "$skip" ]; then skip="$tag"; else skip="$skip,$tag"; fi
    rm -f $outdir/$name.progress
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
echo "CRASHED cases: ${#crashed[@]}"
for c in $crashed; do echo "  CRASH $c"; done
# A crashed cell is a P0 gap: the decline is in the wrong place.
[[ ${#crashed[@]} -gt 0 ]] && exit 1
exit 0
