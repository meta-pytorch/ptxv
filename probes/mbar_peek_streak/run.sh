#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# RMK277 F1 (MISS 277 follow-up): the deterministic runs past a warp stalled on a peek of an
# mbarrier another warp initializes later. Exits 0 when (a) one deterministic run
# (PTXV_RUN_ONCE) reaches the end with the write-write race, (b) the depth run after a search cut
# at 3 states reports the race, (c) --residency auto does not call the kernel a deadlock; 1
# otherwise. Simulator only.   PTXV=<ptxv binary> probes/mbar_peek_streak/run.sh
set -u
PTXV=${PTXV:-ptxv}
D=$(cd "$(dirname "$0")" && pwd)
export PTXV_THREADS=${PTXV_THREADS:-8} CUDA_VISIBLE_DEVICES=
O=$(mktemp -d); trap 'rm -rf "$O"' EXIT
fail=0
r1=$(PTXV_RUN_ONCE=1 timeout 120 "$PTXV" check "$D" --front ptx --no-gb300 --no-input-coverage --out "$O" 2>&1)
echo "$r1" | grep -E "RUN_ONCE"
echo "$r1" | grep -q "terminal=true" || { echo "(a) the deterministic run stopped before the end"; fail=1; }
echo "$r1" | grep -q "RACE: write-write" || { echo "(a) the deterministic run lost the write-write race"; fail=1; }
r2=$(timeout 120 "$PTXV" check "$D" --front ptx --max-states 3 --no-gb300 --no-input-coverage --out "$O" 2>&1)
echo "$r2" | grep -E "^(PASS|FAIL)|note: the bounded"
echo "$r2" | grep -q "RACE: write-write" || { echo "(b) the depth run lost the race"; fail=1; }
r3=$(timeout 120 "$PTXV" check "$D" --front ptx --residency auto --no-gb300 --no-input-coverage --out "$O" 2>&1)
echo "$r3" | grep -E "GPU 0:"
echo "$r3" | grep -q "deadlocks even with every CTA resident" && { echo "(c) --residency auto reports a deadlock"; fail=1; }
exit $fail
