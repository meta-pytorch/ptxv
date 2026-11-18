#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# S-REMOTE-INFLIGHT / S-CLUSTER-LIFE: a lane's drain lands only its own part of a coalesced
# st.async operation (case.toml). For each variant and front: the stateful verdict, the
# stateful verdict without local reduction (PTXV_NO_LOCAL_POR) and --certify's. Exits 1 unless
# every one equals the classes case.toml expects. DPOR is not run (non-blocking polls).
# Usage: [PTXV=ptxv] probes/stasync_lanes/run.sh
set -u
here="$(cd "$(dirname "$0")" && pwd)"
ptxv="${PTXV:-ptxv}"
cls() { awk '/^(PASS|FAIL) /{for(i=1;i<=NF;i++) if($i ~ /^\[/){print $i; exit}}'; }
run() { # <variant> <front> [env...] -- [flags...]
  local v=$1 f=$2; shift 2
  local envs=()
  while [ $# -gt 0 ] && [ "$1" != -- ]; do envs+=("$1"); shift; done
  shift
  env "${envs[@]}" PTXV_THREADS="${PTXV_THREADS:-8}" systemd-run --user --scope -q \
    -p MemoryMax=16G -p MemorySwapMax=0 "$ptxv" check "$here" --variant "$v" --front "$f" \
    --no-gb300 --hw-profile none --no-input-coverage "$@" 2>&1 | cls
}
expected() { # <variant> <front>
  python3 -I - "$here/case.toml" "$1" "$2" <<'PY'
import sys, tomllib
m = tomllib.load(open(sys.argv[1], "rb"))
v = next(x for x in m["variants"] if x["id"] == sys.argv[2])
e = v.get("expect_sass", v["expect"]) if sys.argv[3] == "sass" else v["expect"]
print("[" + ",".join(sorted(e)) + "]")
PY
}
rc=0
for v in D1R0 D1 D1R1 D0R1 D0 D0X D0XW D0XB; do
  for f in ptx sass; do
    e=$(expected $v $f)
    s=$(run $v $f --)
    l=$(run $v $f PTXV_NO_LOCAL_POR=1 --)
    c=$(run $v $f -- --certify)
    ok=ok
    for x in "$s" "$l" "$c"; do [ "$x" = "$e" ] || ok=MISMATCH; done
    [ $ok = ok ] || rc=1
    printf '%-5s %-4s expected %-34s stateful %-34s no-local-por %-34s certify %-34s %s\n' \
      $v $f "$e" "$s" "$l" "$c" $ok
  done
done
exit $rc
