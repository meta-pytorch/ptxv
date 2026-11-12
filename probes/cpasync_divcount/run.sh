#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Divergent LDGDEPBAR counting probe (divcount.cu): 6 modes, LAUNCHES each.
set -u
cd "${PROBE_DIR:-$(dirname "$0")}"
export PATH=/usr/local/cuda-13.0/bin:$PATH
N=${LAUNCHES:-1000}
NCTA=${NCTA:-592}
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1
nvcc -gencode arch=compute_103a,code=sm_103a -O2 -o divcount divcount.cu || exit 1
cuobjdump -sass divcount > divcount.sass
for k in divcountILi0E divcountILi1E divcountILi2E divcountILi3E divcountILi4E; do
  echo "== $k:"
  awk -v k="$k" '/Function :/ {p = index($0, k) > 0} p' divcount.sass | grep -E 'LDGSTS|LDGDEPBAR|DEPBAR|BSSY|BSYNC|WARPSYNC|LDS|BRA' | sed 's@/\*[0-9a-f]*\*/@@; s@ *;.*@@; s@^ *@  @'
done
for m in 0 1 2 3 4 5; do
  timeout 1800 ./divcount $m $NCTA $N
done
echo DONE
