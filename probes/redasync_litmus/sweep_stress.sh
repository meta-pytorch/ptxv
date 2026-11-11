#!/bin/bash
# The configurations in which the controls give stale reads (L2 thrash, a payload line per
# word): the form under test and the MEMBAR.ALL.CTA + REDG control, R rounds.
# Usage: sweep_stress.sh <binary> <dev> <out> [rounds]
B=$1; D=$2; OUT=$3; R=${4:-50}
for r in $(seq 1 $R); do
  for c in "pairs=64 iters=5000 nw=8 js=32 ps=512 stress=6 smode=1" "pairs=32 iters=5000 nw=8 js=32 ps=512 stress=8 smode=1 dmask=63" "pairs=64 iters=5000 nw=8 js=32 ps=512 stress=10 smode=1 dmask=15"; do
    for f in 0 3; do
      timeout 600 $B $f dev=$D seed=$((r + 100)) $c >> $OUT || echo "form=$f FAILED seed=$r $c" >> $OUT
    done
  done
done
