#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Stateful-explorer repro (MISS 278, RDP2 finding G): a peek of an mbarrier phase after the
# thread's own arrival, against another warp's arrival that cannot complete the phase (subcases
# k0_p0, k0_p1_d, k1_p1_d: case.toml in each). For each line of expected/classes.txt (subcase,
# front, explorer, classes): the explorer's classes, expected vs observed. Explorers: stateful
# (the default), certify (--certify), dpor (PTXV_EXPLORER=dpor), unreduced (every reduction off).
# Exits 1 if any observed classes differ from the pinned ones.
# Usage: [PTXV=ptxv] probes/local_peek_arrival/run.sh
set -u
here="$(cd "$(dirname "$0")" && pwd)"
ptxv="${PTXV:-ptxv}"
classes() { # <subcase> <front> <explorer>
  local c=$1 f=$2 e=$3 envs=(X=1) flags=()
  case $e in
    dpor) envs=(PTXV_EXPLORER=dpor) ;;
    unreduced)
      envs=(PTXV_NO_TMA_POR=1 PTXV_NO_OWN_ASYNC=1 PTXV_NO_SYNC_POR=1 PTXV_NO_LOCAL_POR=1
        PTXV_NO_CTA_POR=1 PTXV_NO_BSYNC_POR=1 PTXV_NO_RELEASED=1 PTXV_NO_GATE=1 PTXV_NO_VALTRACK=1
        PTXV_NO_PHASE_CLIP=1 PTXV_NO_DEAD_SYNC=1 PTXV_NO_PEEKSKIP=1 PTXV_NO_PEEKWAIT=1
        PTXV_NO_VIP=1) ;;
    certify) flags=(--certify) ;;
  esac
  env "${envs[@]}" PTXV_THREADS="${PTXV_THREADS:-8}" systemd-run --user --scope -q \
    -p MemoryMax=16G -p MemorySwapMax=0 "$ptxv" check "$here/$c" --variant good --front "$f" \
    --no-gb300 --hw-profile none --no-input-coverage "${flags[@]}" 2>&1 |
    awk '/^(PASS|FAIL) /{for(i=1;i<=NF;i++) if($i ~ /^\[/){print $i; exit}}'
}
rc=0
while read -r c f e want; do
  case $c in ''|'#'*) continue ;; esac
  got=$(classes "$c" "$f" "$e")
  ok=ok
  [ "$got" = "$want" ] || { ok=MISMATCH; rc=1; }
  printf '%-8s %-4s %-9s expected %-26s observed %-26s %s\n' "$c" "$f" "$e" "$want" "${got:-none}" "$ok"
done < "$here/expected/classes.txt"
echo "rc=$rc"
exit $rc
