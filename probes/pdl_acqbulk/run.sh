#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# griddepcontrol.wait (ACQBULK) stale-L1 probe (pdl.cu): modes 0-5 x 3 delays, modes 6-8 once,
# LAUNCHES each.
set -u
cd "${PROBE_DIR:-$(dirname "$0")}"
export PATH=/usr/local/cuda-13.0/bin:$PATH
N=${LAUNCHES:-1000}
NCTA=${NCTA:-512}
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1
nvcc -gencode arch=compute_103a,code=sm_103a -O2 -o pdl pdl.cu || exit 1
cuobjdump -sass pdl > pdl.sass
for k in _Z7primary primary_nop secondaryILi0E secondaryILi1E secondaryILi2E secondaryILi3E secondaryILi4E \
  secondaryILi5E lateILi6E lateILi7E lateILi8E; do
  echo "== $k: $(awk -v k="$k" '/Function :/ {p = index($0, k) > 0} p' pdl.sass | grep -oE 'ACQBULK|CCTL[.A-Z]*|MEMBAR[.A-Z]*|FENCE[.A-Z]*|LDG[.A-Z0-9]*|STG[.A-Z0-9]*|PREEXIT' | sort | uniq -c | tr '\n' ' ')"
done
for d in 2000 20000 100000; do
  for m in 0 1 2 3 4 5; do
    timeout 1800 ./pdl $m $d $NCTA $N
  done
done
for m in 6 7 8; do
  timeout 1800 ./pdl $m 0 $NCTA $N
done
echo DONE
