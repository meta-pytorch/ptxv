#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# MISS 226 repro (one command): bash probes/race_dead_mixed/run.sh
# Checks each row of expected/verdicts.csv with `ptxv check` (PTXV=<binary>, default ptxv on PATH;
# the PTX front needs ptxas, no GPU) and prints the expected verdict next to ptxv's, with the
# kinds of its findings. Exits 1 while any differs: before the fix ptxv reports good and PLAIN
# clean (their races race-dead, E-RACE-DEAD), and the NIC rows (CASE:VARIANT, a variant of
# fixtures/CASE) too.
set -u
here=$(cd "$(dirname "$0")" && pwd)
PTXV=${PTXV:-ptxv}
rc=0
printf '%-18s %-5s %-9s %-26s %s\n' variant front expected ptxv result
while IFS=, read -r variant front want basis; do
  [ "$variant" = variant ] && continue
  case "$variant" in
    *:*) dir="$here/../../fixtures/${variant%%:*}"; v=${variant#*:} ;;
    *) dir=$here; v=$variant ;;
  esac
  out=$("$PTXV" check "$dir" --variant "$v" --front "$front" --no-input-coverage \
      --no-gb300 --json 2>/dev/null)
  got=$(printf '%s' "$out" | python3 -I -c '
import json, sys
try:
    rows = json.load(sys.stdin).get("rows") or []
    print("+".join(sorted(rows[0].get("got") or [])) or "clean" if rows else "no-result")
except Exception:
    print("no-result")
')
  kinds=$(printf '%s' "$out" | python3 -I -c '
import json, sys
try:
    rows = json.load(sys.stdin).get("rows") or []
    fs = ((rows[0].get("verdict") or {}).get("findings") or []) if rows else []
    print(",".join(sorted({f.get("kind", "") for f in fs})) or "-")
except Exception:
    print("-")
')
  if [ "$got" = "$want" ]; then res=ok; else res=DEFECT; rc=1; fi
  printf '%-18s %-5s %-9s %-26s %s\n' "$variant" "$front" "$want" "$got ($kinds)" "$res"
done < "$here/expected/verdicts.csv"
exit $rc
