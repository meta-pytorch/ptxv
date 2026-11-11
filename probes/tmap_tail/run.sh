#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Builds and runs the tensor-map tail probe on CUDA_VISIBLE_DEVICES (an idle GPU); prints the
# per-variant count of loads whose shared tile differs from the encoded map's.
set -e
cd "$(dirname "$0")"
OUT=$(mktemp -d)
nvcc -O3 -gencode arch=compute_103a,code=sm_103a -o "$OUT/tmap_tail" tmap_tail.cu -lcuda
"$OUT/tmap_tail"
rm -rf "$OUT"
