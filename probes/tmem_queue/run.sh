#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Builds and runs the same-thread LDTM -> STTM order probe (E-TMEM-QUEUE) on
# CUDA_VISIBLE_DEVICES (an idle GPU); dumps the SASS of the probe loop's TMEM accesses first.
set -e
cd "$(dirname "$0")"
OUT=$(mktemp -d)
nvcc -O3 -gencode arch=compute_103a,code=sm_103a -o "$OUT/tmem_queue" tmem_queue.cu
cuobjdump -sass "$OUT/tmem_queue" | grep -E "LDTM|STTM|FENCE.VIEW.ASYNC.T|DEPBAR|UTCHMMA" || true
"$OUT/tmem_queue" "$@"
rm -rf "$OUT"
