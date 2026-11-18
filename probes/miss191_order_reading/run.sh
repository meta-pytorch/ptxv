#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# MISS 191 repro: the reading of an unknown launch order (PTXV_UNRECORDED_ORDER: concurrent, the
# default, or ordered) changes `ptxv check`'s verdict but leaves no trace in its JSON, so a check
# run under an exported ordered reading (fb/kernel_db/run.py passes its environment on) cannot be
# told from a strict one. fixtures/e_equiv_order's UNK variant (launch order unknown), PTX
# front, under both readings: prints each run's classes and the row's `launch_order` field.
# Expected: the field names the case's order and the reading; observed: what the JSON holds.
# Exits 1 while a row lacks it.
#   PTXV_BIN=<ptxv binary> bash probes/miss191_order_reading/run.sh
set -u
here=$(cd "$(dirname "$0")" && pwd)
fx=$here/../../fixtures/e_equiv_order
bin=${PTXV_BIN:-ptxv}
t=$(mktemp -d)
cd "$t"  # ptxv writes ./out/<case>/
bad=0
for reading in concurrent ordered; do
  if [ $reading = concurrent ]; then unset PTXV_UNRECORDED_ORDER; else export PTXV_UNRECORDED_ORDER=ordered; fi
  PTXV_NO_VCACHE=1 "$bin" check "$fx" --variant UNK --front ptx --json > "$t/$reading.json" 2> "$t/$reading.log"
  got=$(python3 -c '
import json, sys
d = None
for ln in open(sys.argv[1]):
    try:
        d = json.loads(ln)
    except ValueError:
        continue
row = (d or {}).get("rows", [{}])[0]
print(",".join(row.get("got", [])), json.dumps(row.get("launch_order")))
' "$t/$reading.json")
  classes=${got%% *}
  field=${got#* }
  echo "reading $reading: classes [$classes]; expected launch_order {\"case\": \"unknown\", \"reading\": \"$reading\"}; observed $field"
  case "$field" in
    *"\"reading\": \"$reading\""*) ;;
    *) bad=1 ;;
  esac
done
exit $bad
