#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# RMK277 S1 (MISS 283): a complete exhaustive verdict must include the write-write race of the two
# stores, whichever order the peek and the init take. Exits 1 while a complete verdict lacks the
# race. Simulator only.   PTXV=<ptxv binary> probes/mbar_peek_selfloop/run.sh
set -u
PTXV=${PTXV:-ptxv}
D=$(cd "$(dirname "$0")" && pwd)
export PTXV_THREADS=${PTXV_THREADS:-8} CUDA_VISIBLE_DEVICES=
O=$(mktemp -d); trap 'rm -rf "$O"' EXIT
timeout 120 "$PTXV" check "$D" --front ptx --no-gb300 --no-input-coverage --out "$O" --json 2>/dev/null | python3 -I -c '
import json, sys
v = json.load(sys.stdin)["rows"][0]["verdict"]
ks = sorted({f["kind"] for f in v["findings"]})
print("complete=%s terminal_states=%d kinds=%s" % (v["complete"], v["terminal_states"], ks))
sys.exit(1 if v["complete"] and "race" not in ks else 0)'
