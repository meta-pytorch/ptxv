#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# MISS 220 repro (E-WARP-ORDER, PTX): an observer between two warp atomics. Lanes 31 and 1 of a
# converged warp add 1 and 2 to one word at two instructions, results dead; thread 32 reads the
# word once and traps on 2, which only lane 1's add first leaves (reachable under the PTX model).
# Prints, per variant and front, GB300's recorded output (expected/: `ptxv hw`, 1000 runs per
# variant), the expected classes and ptxv's verdict (`ptxv check`), and exits 1 if a verdict
# differs. good must stay non-clean (the adds commute, but the observer can tell the orders
# apart). While MISS 220 is open: SAME PTX nondeterministic (two adds of 1 take the word through
# the same values in either order) and RED PTX clean (two reductions were exempt whatever their
# operands: the trap goes unreported). An exemption for every commuting pair turns good clean.
# With --hw every variant also runs on a free GPU first (`ptxv hw`, PTXV_GPUS /
# CUDA_VISIBLE_DEVICES; RUNS runs, default 1000), compared with expected/ (exit 1 on a
# difference).
#   PTXV=<ptxv binary> bash probes/wo_commute/run.sh [--hw]
set -u
PTXV=${PTXV:-ptxv}
here=$(cd "$(dirname "$0")" && pwd)
OUT=$(mktemp -d)  # the check's and the GPU runs' outputs, kept
words() { od -An -tu4 -v "$1" | xargs; }
expected_bin() { # variant -> its recorded GB300 output
  if [ "$1" = good ]; then echo "$here/expected/out.bin"; else echo "$here/expected/$1.bin"; fi
}
variants=$(python3 -c 'import sys, tomllib
print(" ".join(v["id"] for v in tomllib.load(open(sys.argv[1], "rb"))["variants"]))' "$here/case.toml")
rc=0
if [ "${1:-}" = "--hw" ]; then
  echo "== GB300: ${RUNS:-1000} runs per variant"
  for v in $variants; do
    mkdir -p "$OUT/dump-$v"
    PTXV_HW_DUMP="$OUT/dump-$v" "$PTXV" hw "$here" --variant "$v" --runs "${RUNS:-1000}" \
      --out "$OUT/hw-$v" > "$OUT/hw-$v.log" 2>&1 || { cat "$OUT/hw-$v.log"; exit 2; }
    runs=$(sed -nE "s/^hw $v .*\(([0-9]+) runs, .*/\1/p" "$OUT/hw-$v.log" | head -1)
    n=$(find "$OUT/dump-$v" -name '*.bin' | wc -l)
    printf '%-9s %s runs, %s distinct output(s)\n' "$v" "$runs" "$n"
    for d in "$OUT/dump-$v"/*.bin; do
      [ -e "$d" ] || continue
      mark=
      cmp -s "$d" "$(expected_bin "$v")" || { mark=" <- differs from expected/"; rc=1; }
      printf '%-9s   out[] = %s%s\n' "$v" "$(words "$d")" "$mark"
    done
  done
fi
verdicts=$(CUDA_VISIBLE_DEVICES= "$PTXV" check "$here" --no-input-coverage --out "$OUT/check" \
  2>/dev/null | sed -nE 's/^(PASS|FAIL) +([^ ]+) +(ptx|sass) +\[([^]]*)\].*/\2 \3 \4/p')
[ -n "$verdicts" ] || { echo "no verdicts (PTXV=$PTXV)"; exit 2; }
printf '%-9s %-5s %-14s %-22s %s\n' variant front "GB300 out[]" expected ptxv
while read -r v f got; do
  want=$(python3 -c 'import sys, tomllib
v = next(x for x in tomllib.load(open(sys.argv[1], "rb"))["variants"] if x["id"] == sys.argv[2])
print(",".join(v.get("expect_" + sys.argv[3], v["expect"])))' "$here/case.toml" "$v" "$f")
  mark=
  if [ "$got" != "$want" ]; then mark=" <- differs"; rc=1; fi
  printf '%-9s %-5s %-14s %-22s %s%s\n' "$v" "$f" "$(words "$(expected_bin "$v")")" "$want" "$got" "$mark"
done <<<"$verdicts"
exit $rc
