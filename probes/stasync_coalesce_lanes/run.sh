#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# S-REMOTE-INFLIGHT, the lanes of one st.async instruction and their own drains (review repros;
# gen.py writes cases/). In each case lanes 0..NL-1 of rank 0 run one st.async (lane k: word k of
# rank 1, complete_tx on M1); some lanes then fence.acq_rel.cluster and arrive .release.cluster
# on M2; rank 1 waits on M2, maybe test_waits M1 (TW), maybe reads a word (R), then waits M1.
# EXPECT is the PTX-model category of each case (race / norace / nondeterministic /
# deterministic); case.toml pins the classes. A lane's release orders only that lane's word,
# so another lane's word can still be in flight after it (L2_D1_TW*, and L2_D0_TW* where the
# draining lane issued the coalesced operation).
# For each case and front: the stateful verdict, the stateful verdict without local reduction
# and --certify's must equal case.toml's classes and fit EXPECT; exits 1 otherwise. DPOR is not
# run: it misses non-blocking poll outcomes (a separate known gap).
# Usage: [PTXV=ptxv] probes/stasync_coalesce_lanes/run.sh
set -u
here="$(cd "$(dirname "$0")" && pwd)"
ptxv="${PTXV:-ptxv}"
cls() { awk '/^(PASS|FAIL) /{for(i=1;i<=NF;i++) if($i ~ /^\[/){print $i; exit}}'; }
run() { # <case dir> <front> [env...] -- [flags...]
  local d=$1 f=$2; shift 2
  local envs=()
  while [ $# -gt 0 ] && [ "$1" != -- ]; do envs+=("$1"); shift; done
  shift
  env "${envs[@]}" PTXV_THREADS="${PTXV_THREADS:-8}" systemd-run --user --scope -q \
    -p MemoryMax=16G -p MemorySwapMax=0 "$ptxv" check "$d" --variant good --front "$f" \
    --no-gb300 --hw-profile none --no-input-coverage "$@" 2>&1 | cls
}
expected() { # <case dir> <front>
  python3 -I - "$1/case.toml" "$2" <<'PY'
import sys, tomllib
v = tomllib.load(open(sys.argv[1], "rb"))["variants"][0]
e = v.get("expect_sass", v["expect"]) if sys.argv[2] == "sass" else v["expect"]
print("[" + ",".join(sorted(e)) + "]")
PY
}
fits() { # <category> <classes>
  case "$1" in
    race) [[ "$2" == *race* ]] ;;
    norace) [[ -n "$2" && "$2" != *race* ]] ;;
    nondeterministic) [[ "$2" == *nondeterministic* ]] ;;
    deterministic) [[ -n "$2" && "$2" != *nondeterministic* ]] ;;
  esac
}
rc=0
for d in "$here"/cases/*/; do
  n=$(basename "$d"); cat=$(cat "$d/EXPECT")
  for f in ptx sass; do
    e=$(expected "$d" $f)
    s=$(run "$d" $f --)
    l=$(run "$d" $f PTXV_NO_LOCAL_POR=1 --)
    c=$(run "$d" $f -- --certify)
    ok=ok
    fits "$cat" "$e" || ok=BAD-EXPECTATION
    for x in "$s" "$l" "$c"; do [ "$x" = "$e" ] || ok=MISMATCH; done
    [ $ok = ok ] || rc=1
    printf '%-10s %-4s %-16s expected %-34s stateful %-34s no-local-por %-34s certify %-34s %s\n' \
      "$n" $f "$cat" "$e" "$s" "$l" "$c" $ok
  done
done
exit $rc
