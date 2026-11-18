#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# S-REMOTE-INFLIGHT, a remote st.async that lands before its issuer's drain (case.toml). For
# each variant and front: the stateful verdict (local reduction), the stateful verdict without
# it (PTXV_NO_LOCAL_POR), DPOR's and --certify's. Exits 1 unless every one equals the classes
# case.toml expects (so also while they disagree).
# Usage: [PTXV=ptxv] probes/stasync_early_land/run.sh
set -u
here="$(cd "$(dirname "$0")" && pwd)"
ptxv="${PTXV:-ptxv}"
classes() { # <variant> <front> [env...] -- [flags...]
  local v=$1 f=$2; shift 2
  local envs=()
  while [ $# -gt 0 ] && [ "$1" != -- ]; do envs+=("$1"); shift; done
  shift
  env "${envs[@]}" PTXV_THREADS="${PTXV_THREADS:-8}" systemd-run --user --scope -q \
    -p MemoryMax=16G -p MemorySwapMax=0 "$ptxv" check "$here" --variant "$v" --front "$f" \
    --no-gb300 --hw-profile none --no-input-coverage "$@" 2>&1 |
    awk '/^(PASS|FAIL) /{for(i=1;i<=NF;i++) if($i ~ /^\[/){print $i; exit}}'
}
# The verdict classes case.toml expects for <variant> <front> (expect_sass, else expect).
expected() {
  python3 -I - "$here/case.toml" "$1" "$2" <<'PY'
import sys, tomllib
m = tomllib.load(open(sys.argv[1], "rb"))
v = next(x for x in m["variants"] if x["id"] == sys.argv[2])
e = v.get("expect_sass") if sys.argv[3] == "sass" and "expect_sass" in v else v["expect"]
print("[" + ",".join(sorted(e)) + "]")
PY
}
rc=0
for v in FEN FENLD; do
  for f in ptx sass; do
    s=$(classes $v $f --)
    n=$(classes $v $f PTXV_NO_LOCAL_POR=1 --)
    d=$(classes $v $f PTXV_EXPLORER=dpor --)
    c=$(classes $v $f -- --certify)
    e=$(expected $v $f)
    ok=ok
    for x in "$s" "$n" "$d" "$c"; do [ "$x" = "$e" ] || { ok=MISMATCH; rc=1; }; done
    printf '%-6s %-4s expected %-18s stateful %-18s no-local-por %-18s dpor %-18s certify %-18s %s\n' \
      "$v" "$f" "$e" "$s" "$n" "$d" "$c" "$ok"
  done
done
exit $rc
