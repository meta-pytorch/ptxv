#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Runs repro.py configurations one after another on GPUs 0,1. Before each run it waits while a
# process it did not start is on GPU 0 or 1. Usage: sweep.sh <plan file> <out file> [secs]
cd "$(dirname "$0")/../.." || exit
# shellcheck disable=SC1091  # env.sh at the repo root (cwd); not followed
source env.sh
TR=$(dirname "$PTXV_PYTHON")/torchrun
SECS=${3:-90}
busy() { nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep -E "GPU-1ced79bd|GPU-7910a120"; }
while read -r line; do
  [ -z "$line" ] && continue; [[ "$line" == \#* ]] && continue
  while [ -n "$(busy)" ]; do echo "waiting for: $(busy)" >&2; sleep 10; done
  {
    echo "=== $line"
    # shellcheck disable=SC2086  # $line is a list of repro.py arguments
    CUDA_VISIBLE_DEVICES=0,1 NCCL_DEBUG=WARN timeout $((SECS + 240)) "$TR" --nproc-per-node 2 --master-port 29613 \
        tools/deepep_repro/"${SCRIPT:-repro.py}" --secs "$SECS" $line 2>&1 | grep -E "MISMATCH|RESULT|Error|error|Traceback|assert" | head -40
    echo "rc=${PIPESTATUS[0]}"
  } >> "$2"
done < "$1"
