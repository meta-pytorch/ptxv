#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# E-CPASYNC-PROXY probe matrix (cpx.cu): writer x fence per launch shape, 1000 launches per cell.
set -u
cd "${PROBE_DIR:-$(dirname "$0")}"
export PATH=/usr/local/cuda-13.0/bin:$PATH
N=${LAUNCHES:-1000}
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1
nvcc -gencode arch=compute_103a,code=sm_103a -O2 -o cpx cpx.cu -lcuda || exit 1
cuobjdump -sass cpx | grep -E 'LDGSTS|LDGDEPBAR|DEPBAR|ARRIVES|SYNCS|FENCE|MEMBAR|UBLKCP|UTMASTG|STS' | sed 's@/\*[0-9a-f]*\*/@@; s@/\*.*@@' | sort | uniq -c
cells() { # reader rounds W ctas_per_sm delay
  for wf in "0 0" "0 1" "0 2" "1 0" "1 1" "1 2" "2 0" "2 2"; do
    timeout 1800 ./cpx $wf "$@" $N
  done
}
# reader rounds W ctas_per_sm delay
cells 1 8 4096 1 0
cells 1 8 4096 4 2000
cells 1 8 1024 8 2000
cells 1 8 16384 2 0
cells 0 8 4096 4 2000
echo DONE
