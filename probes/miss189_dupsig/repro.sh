#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# MISS 189 repro: a deadlock that the strict reading of an unknown launch order and the list's
# order both reach (case.toml). Checks variant good on both fronts and prints the deadlock
# findings and the E-LAUNCH-ORDER note. Expected: the strict reading's deadlock only, and a note
# that the list's order adds none. Observed before the fix: a second finding, `list-order:` plus
# the strict finding's signature, and the note "1 deadlock(s) under the list's order (...),
# which the strict reading of the unknown order does not reach". Exits 1 while a list-order
# finding repeats a strict finding's signature, 0 otherwise.
#   bash probes/miss189_dupsig/repro.sh [PTXV_BINARY]   (default: $PTXV_BIN, else ptxv)
set -u
bin=${1:-${PTXV_BIN:-ptxv}}
case $bin in */*) bin=$(cd "$(dirname "$bin")" && pwd)/$(basename "$bin") ;; esac
here=$(cd "$(dirname "$0")" && pwd)
t=$(mktemp -d)
dup=0
for front in ptx sass; do
  (cd "$t" && env -u PTXV_UNRECORDED_ORDER PTXV_NO_VCACHE=1 "$bin" check "$here" --variant good \
    --front $front --json > "$front.json" 2> "$front.log")
  out=$(python3 - "$t/$front.json" <<'PY'
import json, sys
d = None
for ln in open(sys.argv[1]):
    try:
        d = json.loads(ln)
    except ValueError:
        pass
v = ((d or {}).get("rows") or [{}])[0].get("verdict") or {}
sigs = [x.get("sig") for x in v.get("findings") or [] if x.get("kind") == "deadlock"]
for s in sigs:
    print("  deadlock", s)
for n in v.get("notes") or []:
    if "E-LAUNCH-ORDER" in n:
        print("  note", n)
if not sigs:
    print("  no deadlock finding (see the log)")
plain = {s for s in sigs if not s.startswith("list-order:")}
dup = any(s.startswith("list-order:") and s[len("list-order:"):] in plain for s in sigs)
print("DUP" if dup else "NODUP")
PY
)
  echo "$front:"
  echo "$out" | grep -v -x -e DUP -e NODUP
  echo "$out" | grep -q -x DUP && dup=1
done
echo "expected: one deadlock per front (the strict reading's) and a note that the list's order adds none"
echo "logs: $t"
if [ $dup = 1 ]; then
  echo "OBSERVED: a list-order deadlock repeats a strict-reading deadlock (same signature)"
  exit 1
fi
echo "OK: no list-order finding repeats a strict finding"
