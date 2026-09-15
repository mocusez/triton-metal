#!/bin/zsh
# "Is this crash MINE, or was it already there?"
#
# An in-conversion decline on this backend does not raise, it takes the process
# down — and INTERMITTENTLY, so one run answers nothing and a retry can even
# look fine. Measured on the 128x32 masked-load gap: 6 runs of one unchanged
# shape gave 139 134 139 134 139 139 on one build and 139 139 139 1 1 1 on the
# next. Both are "~50% crash"; neither run alone says so.
#
# So before blaming a change for a crash, run the SAME script N times on each
# build and compare the histograms. Exit codes:
#   0    ran, answer checked by the script
#   1    clean Python-level rejection (what the doctrine wants)
#   134  SIGABRT   \  the conversion killed the process; the shape is a gap
#   139  SIGSEGV   /  AND the decline is in the wrong place
#
# usage: flaky.sh <script.py> [runs=30]
#        env vars are inherited, so parameterize with e.g. PROBE_D=24 flaky.sh ...
script=$1
n=${2:-30}
[[ -z "$script" ]] && { print -u2 "usage: flaky.sh <script.py> [runs]"; exit 2 }
py=${SWEEP_PY:-"pixi run python"}
typeset -A hist
codes=()
for i in $(seq 1 $n); do
  ${=py} -u $script >/dev/null 2>&1
  c=$?
  codes+=($c)
  hist[$c]=$(( ${hist[$c]:-0} + 1 ))
done
print -r -- "--- $(basename $script) x$n ---"
print -r -- "sequence: $codes"
kills=0
for c in ${(kon)hist}; do
  case $c in
    0)   what="ok" ;;
    1)   what="clean reject" ;;
    134) what="SIGABRT  <-- process kill"; kills=$(( kills + hist[$c] )) ;;
    139) what="SIGSEGV  <-- process kill"; kills=$(( kills + hist[$c] )) ;;
    *)   what="?"; kills=$(( kills + hist[$c] )) ;;
  esac
  printf "  exit %-4s %3d/%d  %s\n" $c $hist[$c] $n $what
done
exit $(( kills > 0 ? 1 : 0 ))
