#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# DPOR repro (MISS 231): async tx decrements against an expect_tx of one phase (case.toml). For
# each line of expected/classes.txt (variant, front, explorer, classes): the explorer's
# classes on the variant, expected vs observed. Explorers: stateful (the default), dpor
# (PTXV_EXPLORER=dpor), certify (--certify). Exits 1 if any observed classes differ from the
# pinned ones (not just from another explorer's).
# Usage: [PTXV=ptxv] probes/dpor_tx_commute_async/run.sh
set -u
here="$(cd "$(dirname "$0")" && pwd)"
ptxv="${PTXV:-ptxv}"
classes() { # <variant> <front> <explorer>
  local v=$1 f=$2 e=$3 envs=(X=1) flags=()
  case $e in
    dpor) envs=(PTXV_EXPLORER=dpor) ;;
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
