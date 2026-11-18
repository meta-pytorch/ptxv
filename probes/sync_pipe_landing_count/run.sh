#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Stateful-explorer repro (MISS 232): probes/sync_pipe_issue_order with a third, local arrival on
# M1, so neither remote landing completes the phase where the drain decides (case.toml). For each
# line of expected/classes.txt (variant, front, explorer, classes): the explorer's classes on the
# variant, expected vs observed. Explorers: stateful (the default), nosyncpor (PTXV_NO_SYNC_POR=1),
# dpor (PTXV_EXPLORER=dpor), certify (--certify), unreduced (every reduction off). Exits 1 if any
# observed classes differ from the pinned ones (not just from another explorer's).
# Usage: [PTXV=ptxv] probes/sync_pipe_landing_count/run.sh
set -u
here="$(cd "$(dirname "$0")" && pwd)"
ptxv="${PTXV:-ptxv}"
classes() { # <variant> <front> <explorer>
  local v=$1 f=$2 e=$3 envs=(X=1) flags=()
  case $e in
    dpor) envs=(PTXV_EXPLORER=dpor) ;;
    nosyncpor) envs=(PTXV_NO_SYNC_POR=1) ;;
    unreduced)
      envs=(PTXV_NO_TMA_POR=1 PTXV_NO_OWN_ASYNC=1 PTXV_NO_SYNC_POR=1 PTXV_NO_LOCAL_POR=1
        PTXV_NO_CTA_POR=1 PTXV_NO_BSYNC_POR=1 PTXV_NO_RELEASED=1 PTXV_NO_GATE=1 PTXV_NO_VALTRACK=1
        PTXV_NO_PHASE_CLIP=1 PTXV_NO_DEAD_SYNC=1 PTXV_NO_PEEKSKIP=1 PTXV_NO_PEEKWAIT=1
        PTXV_NO_VIP=1) ;;
    certify) flags=(--certify) ;;
  esac
  env "${envs[@]}" PTXV_THREADS="${PTXV_THREADS:-8}" systemd-run --user --scope -q \
    -p MemoryMax=16G -p MemorySwapMax=0 "$ptxv" check "$here" --variant "$v" --front "$f" \
    --no-gb300 --hw-profile none --no-input-coverage "${flags[@]}" 2>&1 |
    awk '/^(PASS|FAIL) /{for(i=1;i<=NF;i++) if($i ~ /^\[/){print $i; exit}}'
}
rc=0
while read -r v f e want; do
  case $v in ''|'#'*) continue ;; esac
  got=$(classes "$v" "$f" "$e")
  ok=ok
  [ "$got" = "$want" ] || { ok=MISMATCH; rc=1; }
  printf '%-5s %-4s %-8s expected %-26s observed %-26s %s\n' "$v" "$f" "$e" "$want" "${got:-none}" "$ok"
done < "$here/expected/classes.txt"
echo "rc=$rc"
exit $rc
