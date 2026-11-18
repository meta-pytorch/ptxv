#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# MISS 168 repro: GB300 vs ptxv for one STG.E.MMIO.SYS by every lane to one word.
# Usage: bash run.sh             (model claim against the recorded GB300 result)
#        PTXV_LIVE=1 bash run.sh (also re-runs GB300 on a free GPU)
# Exit 1 while ptxv's gb300 level calls TRAP31 clean (GB300 traps).
set -u
D=$(cd "$(dirname "$0")" && pwd)
PTXV=${PTXV:-ptxv}
T=$(mktemp -d)
echo "GB300 recorded (expected/gb300.txt): $(tr '\n' ' ' < $D/expected/gb300.txt)"
if [ "${PTXV_LIVE:-0}" = 1 ]; then
  mkdir -p $T/dump
  echo "GB300 live, good (100 runs; w[0] after the store):"
  PTXV_ROOT=$T PTXV_HW_DUMP=$T/dump $PTXV hw $D --variant good --runs 100 --scrub --out $T/hw 2>&1 | grep -E "^hw|rror" | cut -c1-200
  for f in $T/dump/*.bin; do [ -f "$f" ] && python3 -c "import struct,sys; b=open(sys.argv[1],'rb').read(); w=struct.unpack('<%dI'%(len(b)//4),b); print('  w[0] =', w[32], ' out words:', sorted(set(w[:32])))" "$f"; done
  echo "GB300 live, TRAP31 (one launch):"
  PTXV_ROOT=$T $PTXV hw $D --variant TRAP31 --runs 1 --scrub --out $T/hw31 2>&1 | grep -E "^hw|rror" | cut -c1-200
fi
claim=$(PTXV_ROOT=$T CUDA_VISIBLE_DEVICES= $PTXV check $D --variant TRAP31 --front sass 2>&1 | grep -E "^    sass_gb300:" | cut -d'(' -f1)
echo "ptxv TRAP31 (expected: trap at every level, GB300 traps in every launch):${claim}"
case "$claim" in
  *"sass_gb300: clean"*) echo "DEFECT PRESENT: E-HW-STG-LOWEST gives the MMIO store lane 0's value; GB300 keeps lane 31's"; exit 1 ;;
  *) echo "defect absent"; exit 0 ;;
esac
