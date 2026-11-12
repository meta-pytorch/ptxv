#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Builds and runs the E-TMEM-QUEUE window calibration (window.cu) on CUDA_VISIBLE_DEVICES (an
# idle GPU); dumps the probe's TMEM accesses and flag stores first (check the order per pair).
#   window.sh [iterations per launch (100)] [launches per variant (1000)]
set -e
cd "$(dirname "$0")"
OUT=$(mktemp -d)
nvcc -O3 -gencode arch=compute_103a,code=sm_103a -o "$OUT/window" window.cu
cuobjdump -sass "$OUT/window" | grep -E "LDTM|STTM|FENCE.VIEW.ASYNC.T|STS|WARPSYNC" || true
"$OUT/window" "$@"
rm -rf "$OUT"
