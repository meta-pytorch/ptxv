#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Stateful-reduction repro (MISS 295): the synchronization reduction's learned sets took a
# value-insensitive poll's step (E-POLL-ANY) without its read. fixtures/e_peek_result ACQ SASS:
# warp 0 peeks an mbarrier (result unused), then reads a word with an acquiring LDS; warp 1
# stores the word and arrives. The LDS ran as a poll after the store (one value seen, 7), so
# its learned footprint had no read, and the store's ample check in the state before it saw
# nothing of warp 0's that conflicts: the read-before-write order, which reads the
# never-written word, was cut (default [clean] against [nondeterministic, violation]). Seed
# walks can hide it (PTXV_POR_SEED_WALKS; the default at one thread happened to learn the
# read), so the rows pin several. For each line of expected/classes.txt (case directory,
# variant, front, explorer, env, classes): the explorer's classes, expected vs observed.
# Explorers: stateful (the default), dpor (PTXV_EXPLORER=dpor), unreduced (every reduction
# off). env: extra VAR=VALUE settings, comma-separated, or "-". Exits 1 if any observed classes
# differ from the pinned ones.
# Usage: [PTXV=ptxv] probes/sync_vip_learned/run.sh
set -u
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
ptxv="${PTXV:-ptxv}"
classes() { # <case dir> <variant> <front> <explorer> <env>
  local d=$1 c=$2 f=$3 e=$4 x=$5 envs=(X=1)
  case $e in
    dpor) envs=(PTXV_EXPLORER=dpor) ;;
    unreduced)
      envs=(PTXV_NO_TMA_POR=1 PTXV_NO_OWN_ASYNC=1 PTXV_NO_SYNC_POR=1 PTXV_NO_LOCAL_POR=1
        PTXV_NO_CTA_POR=1 PTXV_NO_BSYNC_POR=1 PTXV_NO_RELEASED=1 PTXV_NO_GATE=1 PTXV_NO_VALTRACK=1
        PTXV_NO_PHASE_CLIP=1 PTXV_NO_DEAD_SYNC=1 PTXV_NO_PEEKSKIP=1 PTXV_NO_PEEKWAIT=1
        PTXV_NO_VIP=1) ;;
  esac
  [ "$x" = - ] || IFS=, read -r -a more <<< "$x"
  [ "$x" = - ] || envs+=("${more[@]}")
  env PTXV_THREADS=4 "${envs[@]}" systemd-run --user --scope -q -p MemoryMax=16G \
    -p MemorySwapMax=0 "$ptxv" check "$root/$d" --variant "$c" --front "$f" --no-gb300 \
    --hw-profile none --no-input-coverage 2>&1 |
    awk '/^(PASS|FAIL) /{for(i=1;i<=NF;i++) if($i ~ /^\[/){print $i; exit}}'
}
rc=0
while read -r d c f e x want; do
  case $d in ''|'#'*) continue ;; esac
  got=$(classes "$d" "$c" "$f" "$e" "$x")
  ok=ok
  [ "$got" = "$want" ] || { ok=MISMATCH; rc=1; }
  printf '%-27s %-5s %-4s %-9s %-26s expected %-28s observed %-28s %s\n' "$d" "$c" "$f" "$e" "$x" "$want" "${got:-none}" "$ok"
done < "$here/expected/classes.txt"
echo "rc=$rc"
exit $rc
