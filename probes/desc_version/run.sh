#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Runs every config with descriptor bits 46-48 = (va, vb) for va = vb = 0..7 and the mixed
# pairs (0, 1), (1, 0), twice each, on CUDA_VISIBLE_DEVICES (an idle GPU), and compares each
# dump with the (1, 1) dump bitwise (compare.py).
set -e
cd "$(dirname "$0")"
OUT=$(mktemp -d)
nvcc -O3 -gencode arch=compute_103a,code=sm_103a -o "$OUT/dv" desc_version.cu
"$OUT/dv" 1 1 /dev/null list > "$OUT/configs.txt"
for p in "1 1" "0 0" "2 2" "3 3" "4 4" "5 5" "6 6" "7 7" "0 1" "1 0"; do
  set -- $p
  for r in 0 1; do
    "$OUT/dv" "$1" "$2" "$OUT/d_$1_$2_$r.bin" || echo "va=$1 vb=$2 run $r failed"
  done
done
"${PTXV_PYTHON:-python3}" compare.py "$OUT"
rm -rf "$OUT"
