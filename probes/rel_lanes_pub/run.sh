#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# MISS 223 repro: PTX lane orders and an equal-valued publishing store.
# 32 lanes st.release (good) or fence.acq_rel + st.relaxed (FENCE) the same value to flag; thread
# 32 acquires it and reads data[31]. PTX contract: race (the acquire may read any lane's store and
# synchronizes with that lane alone). GB300 cannot decide (expected/gb300.txt).
# Exit 1 while ptxv calls good or FENCE clean at the PTX contract.
#   PTXV=<ptxv binary> bash probes/rel_lanes_pub/run.sh
set -u
D=$(cd "$(dirname "$0")" && pwd)
PTXV=${PTXV:-ptxv}
T=$(mktemp -d)
echo "GB300 recorded (expected/gb300.txt): $(head -1 $D/expected/gb300.txt)"
rc=0
for v in good FENCE W0 ONE; do
  line=$(PTXV_ROOT=$T CUDA_VISIBLE_DEVICES= $PTXV check $D --variant $v --front ptx 2>&1 | grep -E "^(PASS|FAIL)")
  exp=$(awk -v id="$v" '/^\[\[variants\]\]/{inv=0} $0 ~ "^id = \""id"\""{inv=1} inv && /^expect = /{print $3,$4; exit}' $D/case.toml)
  echo "ptxv $v (PTX contract: $exp): $line"
  case "$v:$line" in
    good:*"[clean]"*|FENCE:*"[clean]"*) rc=1 ;;
  esac
done
[ $rc = 1 ] && echo "DEFECT PRESENT: equal-valued publishing stores run in lane order only" || echo "defect absent"
exit $rc
