#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# MISS 235 repro: the certificate's deterministic run must end after a step stops threads at an
# error. fixtures/r4c_nowrap B36 (SASS): notify_dispatch's barrier misses a signal, a dispatch
# warp's bulk copy then faults, and the model releases the named barrier the faulted warp held
# (exec.rs release_after_stop): the other dispatch warps go round it forever. Runs
# `ptxv check --certify` (PTXV_CERTIFY_ONLY: no exploration after a refusal) cut at CUT_S
# (default 120) and counts the certificate attempts that ended. Exits 1 while fewer than two
# ended (an attempt never ends and its memory grows), 0 otherwise. Simulator only (no GPU).
#   PTXV=<ptxv binary> probes/cert_fault_stop/run.sh [CUT_S]
set -u
PTXV=${PTXV:-ptxv}
CUT=${1:-120}
here=$(cd "$(dirname "$0")" && pwd)
CASE=$here/../../fixtures/r4c_nowrap
OUT=$(mktemp -d)
trap 'rm -rf "$OUT"' EXIT
s=$(date +%s.%N)
export CUDA_VISIBLE_DEVICES= PTXV_THREADS=1 PTXV_CERTIFY_ONLY=1
timeout "$CUT" "$PTXV" check "$CASE" --variant B36 --front sass --certify --no-input-coverage \
  --out "$OUT" > "$OUT/log" 2>&1
rc=$?
e=$(date +%s.%N)
n=$(grep -c '^certificate for launch .* does not hold' "$OUT/log")
echo "expected: two certificate attempts end within seconds, refused"
t=$(python3 -c "print(round($e - $s, 1))")
echo "observed: $n attempt(s) ended in $t s (rc $rc; 124 = cut at $CUT s)"
grep '^certificate for launch' "$OUT/log" | cut -c1-300
grep -E '^(PASS|FAIL) ' "$OUT/log" | cut -c1-200
[ "$n" -ge 2 ]
