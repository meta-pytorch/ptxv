#!/bin/bash
# Sweep for redlit (README.md): every form under each configuration, several seeds.
# Usage: sweep.sh <binary> <dev> <out> [rounds]
B=$1; D=$2; OUT=$3; R=${4:-4}
CFGS=(
  "pairs=64 iters=20000 nw=4 ps=32 dmask=255"
  "pairs=64 iters=20000 nw=8 js=32 ps=512 dmask=1023"
  "pairs=64 iters=5000 nw=8 js=32 ps=512 stress=6 smode=1"
  "pairs=64 iters=5000 nw=8 js=32 ps=512 stress=6 smode=1 pp=1"
  "pairs=32 iters=5000 nw=8 js=32 ps=512 stress=8 smode=1 dmask=63"
  "pairs=64 iters=5000 nw=4 ps=32 stress=4 smode=2"
  "mode=1 pairs=64 iters=5000 nw=1024 ps=1024 js=0"
  "mode=1 pairs=64 iters=5000 nw=1024 ps=1024 js=1"
)
for r in $(seq 1 $R); do
  for c in "${CFGS[@]}"; do
    for f in 0 1 2 3; do
      timeout 600 $B $f dev=$D seed=$r $c >> $OUT || echo "form=$f FAILED seed=$r $c" >> $OUT
    done
  done
done
