#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# N-MUFU: the range of the 15 PTX f32 .approx ops on GB300 over all 2^32 operands each
# (sweep.cu). Compiles the sweep (nvcc of PTXV_CUDA_HOME, else on PATH), runs it on one GPU
# (the first of PTXV_GPUS, else CUDA_VISIBLE_DEVICES), writes its records, summary and SASS
# listing into OUT (default: a new temporary directory) and compares the summary and the sorted
# records with expected/. Exit 0: GB300 gives the recorded results; 3: it does not (N-MUFU's
# range statements rest on them); 2: tool failure. The SASS listing (expected/sweep_sass.txt:
# CUDA 13.0, sm_103a) is written for inspection, not compared.
#   PTXV_GPUS=<gpu index> [OUT=<dir>] bash probes/mufu_range/run.sh
set -u
here=$(cd "$(dirname "$0")" && pwd)
nvcc=${PTXV_CUDA_HOME:+$PTXV_CUDA_HOME/bin/}nvcc
out=${OUT:-$(mktemp -d)}
mkdir -p "$out" || exit 2
"$nvcc" -O2 -gencode arch=compute_103a,code=sm_103a -o "$out/sweep" "$here/sweep.cu" || exit 2
if [ -n "${PTXV_GPUS:-}" ]; then export CUDA_VISIBLE_DEVICES=${PTXV_GPUS%%,*}; fi
"$out/sweep" "$out/records.unsorted.csv" "$out/summary.csv" || exit 2
{ head -1 "$out/records.unsorted.csv"; tail -n +2 "$out/records.unsorted.csv" | LC_ALL=C sort; } > "$out/records.csv"
cuobjdump=${PTXV_CUDA_HOME:+$PTXV_CUDA_HOME/bin/}cuobjdump
"$cuobjdump" -sass "$out/sweep" | grep -E "Function|MUFU|FMUL|FSETP|FADD" > "$out/sweep_sass.txt"
rc=0
for f in summary.csv records.csv; do
  if ! cmp -s "$out/$f" "$here/expected/$f"; then
    echo "GB300's $f differs from expected/$f:"
    diff "$here/expected/$f" "$out/$f" | head -20
    rc=3
  fi
done
cat "$out/summary.csv"
echo "outputs in $out"
echo "exit $rc"
exit $rc
