#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# DPOR repro (MISS 290): splits of one warp that share uniform registers. A plain BSYNC no
# longer holds the waiting lanes at the strict level (E-BSYNC), so the passing lanes run beside
# the elected lane, and a step of one split that writes a uniform register (UMOV, LDCU, S2UR)
# lands between another split's write and read of it. For each line of expected/classes.txt
# (case directory, variant, front, explorer, classes): the explorer's classes, expected vs
# observed. Explorers: stateful (the default), certify (--certify), dpor (PTXV_EXPLORER=dpor),
# unreduced (every reduction off). Exits 1 if any observed classes differ from the pinned ones.
# Usage: [PTXV=ptxv] probes/dpor_split_ur/run.sh
set -u
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
ptxv="${PTXV:-ptxv}"
classes() { # <case dir> <variant> <front> <explorer>
  local d=$1 c=$2 f=$3 e=$4 envs=(X=1) flags=()
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
    -p MemoryMax=16G -p MemorySwapMax=0 "$ptxv" check "$root/$d" --variant "$c" --front "$f" \
    --no-gb300 --hw-profile none --no-input-coverage "${flags[@]}" 2>&1 |
    awk '/^(PASS|FAIL) /{for(i=1;i<=NF;i++) if($i ~ /^\[/){print $i; exit}}'
}
rc=0
while read -r d c f e want; do
  case $d in ''|'#'*) continue ;; esac
  got=$(classes "$d" "$c" "$f" "$e")
  ok=ok
  [ "$got" = "$want" ] || { ok=MISMATCH; rc=1; }
  printf '%-22s %-5s %-4s %-9s expected %-28s observed %-28s %s\n' "$d" "$c" "$f" "$e" "$want" "${got:-none}" "$ok"
done < "$here/expected/classes.txt"
echo "rc=$rc"
exit $rc
