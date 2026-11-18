#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# MISS 194 repro (one command): bash probes/nic_overlap/run.sh
# Checks each row of expected/verdicts.csv with `ptxv check` (PTXV=<binary>, default ptxv on PATH;
# the PTX front needs ptxas, no GPU) and prints the expected verdict next to ptxv's. Exits 1
# while any differs: before the fix ptxv reports PART, PART_OTHERCTA, WORDPART and CQEPART clean.
set -u
here=$(cd "$(dirname "$0")" && pwd)
PTXV=${PTXV:-ptxv}
rc=0
printf '%-14s %-5s %-12s %-12s %s\n' variant front expected ptxv result
while IFS=, read -r variant front want basis; do
  [ "$variant" = variant ] && continue
  got=$("$PTXV" check "$here" --variant "$variant" --front "$front" --no-input-coverage \
      --no-gb300 --json 2>/dev/null | python3 -I -c '
import json, sys
try:
    rows = json.load(sys.stdin).get("rows") or []
    print("+".join(sorted(rows[0].get("got") or [])) or "clean" if rows else "no-result")
except Exception:
    print("no-result")
')
  if [ "$got" = "$want" ]; then res=ok; else res=DEFECT; rc=1; fi
  printf '%-14s %-5s %-12s %-12s %s\n' "$variant" "$front" "$want" "$got" "$res"
done < "$here/expected/verdicts.csv"
exit $rc
