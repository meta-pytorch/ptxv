#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Builds and runs the same-thread tcgen05.st -> wait::st -> mma probe on CUDA_VISIBLE_DEVICES
# (an idle GPU); prints per-mode counts of lane-0 words the MMA did not read as stored.
set -e
cd "$(dirname "$0")"
OUT=$(mktemp -d)
nvcc -O3 -gencode arch=compute_103a,code=sm_103a -o "$OUT/tc_st_mma" tc_st_mma.cu
cuobjdump -sass "$OUT/tc_st_mma" | grep -cE "FENCE|UTCHMMA|UTCBAR|STTM" || true
"$OUT/tc_st_mma" "$@"
rm -rf "$OUT"
