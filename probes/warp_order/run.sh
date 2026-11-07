#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Builds the probe unrolled and with -DUNROLL1, prints the SASS of the store / load pairs and
# runs both on CUDA_VISIBLE_DEVICES (must be an idle GPU). Usage: run.sh [runs]
set -e
cd "$(dirname "$0")"
RUNS=${1:-50}
OUT=$(mktemp -d)
for v in un u1; do
  flags=""; [ $v = u1 ] && flags=-DUNROLL1
  nvcc -O3 -arch=sm_103a -diag-suppress 550 $flags -o "$OUT/wo_$v" warp_order.cu
  echo "== build $v: SASS shared-memory accesses of probe()"
  cuobjdump -sass "$OUT/wo_$v" | grep -E "STSM|LDSM|STS|LDS|WARPSYNC|BSSY|BSYNC" | sed 's@/\* 0x.*@@' | head -40
  "$OUT/wo_$v" "$RUNS" 32 296
  "$OUT/wo_$v" "$RUNS" 1 2368
done
rm -rf "$OUT"
