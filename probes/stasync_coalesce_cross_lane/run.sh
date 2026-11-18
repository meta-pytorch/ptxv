#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Review repro (wrong-clean against the PTX memory model): probes/stasync_coalesce C0 on the
# PTX front. Lane 1's st.async of y is not ordered before rank 1's read of y under the PTX
# memory model (only lane 0 fences and releases; lane 1 never synchronizes after its st.async;
# rank 1 waits on the st.async's mbarrier M1 only after the read). Expected: the verdict
# contains race, for each explorer. Exits 1 while it does not (53ef07ac: [contract] three
# times; ebfacb16: [contract, race]).
# Usage: [PTXV=ptxv] [FIX=<case dir, default probes/stasync_coalesce>] run.sh
set -u
here="$(cd "$(dirname "$0")" && pwd)"
ptxv="${PTXV:-ptxv}"; fix="${FIX:-$here/../stasync_coalesce}"
rc=0
for env in "" PTXV_NO_LOCAL_POR=1 PTXV_EXPLORER=dpor; do
  out=$(env $env PTXV_THREADS=${PTXV_THREADS:-8} systemd-run --user --scope -q -p MemoryMax=16G -p MemorySwapMax=0 \
    "$ptxv" check "$fix" --variant C0 --front ptx --no-gb300 --hw-profile none --no-input-coverage 2>&1)
  cls=$(echo "$out" | awk '/^(PASS|FAIL) /{for(i=1;i<=NF;i++) if($i ~ /^\[/){print $i; exit}}')
  case "$cls" in *race*) ok=ok;; *) ok=WRONG-CLEAN; rc=1;; esac
  printf 'C0 ptx %-20s observed %-22s expected race  %s\n' "${env:-stateful}" "$cls" "$ok"
done
exit $rc
