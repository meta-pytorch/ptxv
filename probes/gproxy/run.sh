#!/bin/bash
set -u
cd "${PROBE_DIR:-$(dirname "$0")}"
export PATH=/usr/local/cuda-13.0/bin:$PATH
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1
nvcc -gencode arch=compute_103a,code=sm_103a -O2 -o g2s g2s.cu || exit 1
cuobjdump -sass g2s | grep -E 'FENCE|MEMBAR|UBLKCP|STG' | sort | uniq -c
for v in 0 3 4 1 2; do
  for W in 4 64 1024; do
    timeout 900 ./g2s $v 2000 $W 10
  done
done
echo DONE
