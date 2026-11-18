#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# DPOR liveness repro (MISS 274, RDP2 finding F): DPOR on probes/mbar_zero and mbar_reserved
# (one thread; an mbarrier word overwritten by a plain store, then test_waits that stay at the
# uninitialized mbarrier). The thread repeats one state; a stateless search with no cycle check
# runs that path without end (RSS 20-25 MB/s, past 16 GB before the 2M-state cap), where the
# stateful search gives [violation] in 4 states. For each line of expected/classes.txt (probe,
# variant, front, explorer, classes): the classes within 60 s and 6 GB, expected vs observed
# ("none": no verdict). Exits 1 if any differ.
# Usage: [PTXV=ptxv] probes/dpor_cycle/run.sh
set -u
here="$(cd "$(dirname "$0")" && pwd)"
ptxv="${PTXV:-ptxv}"
classes() { # <probe> <variant> <front> <explorer>
  local envs=(X=1)
  [ "$4" = dpor ] && envs=(PTXV_EXPLORER=dpor)
  env "${envs[@]}" PTXV_THREADS="${PTXV_THREADS:-8}" timeout 60 systemd-run --user --scope -q \
    -p MemoryMax=6G -p MemorySwapMax=0 "$ptxv" check "$here/../$1" --variant "$2" --front "$3" \
    --no-gb300 --hw-profile none --no-input-coverage 2>&1 |
    awk '/^(PASS|FAIL) /{for(i=1;i<=NF;i++) if($i ~ /^\[/){print $i; exit}}'
}
rc=0
while read -r p v f e want; do
  case $p in ''|'#'*) continue ;; esac
  got=$(classes "$p" "$v" "$f" "$e")
  ok=ok
  [ "$got" = "$want" ] || { ok=MISMATCH; rc=1; }
  printf '%-13s %-5s %-4s %-8s expected %-22s observed %-22s %s\n' "$p" "$v" "$f" "$e" "$want" \
    "${got:-none}" "$ok"
done < "$here/expected/classes.txt"
echo "rc=$rc"
exit $rc
