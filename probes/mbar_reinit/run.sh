#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# MISS 203 repro (fixtures/e_mbar_reinit_drain): a wait on an mbarrier must not cover remote
# deliveries into the object that a later inval / init at the same address creates. Runs every
# variant on both fronts through `ptxv check` (exhaustive) and `ptxv check --certify`, prints
# both verdicts next to the expected one, and exits 1 if any differ. Simulator only (no GPU).
#   PTXV=<ptxv binary> probes/mbar_reinit/run.sh
set -u
PTXV=${PTXV:-ptxv}
here=$(cd "$(dirname "$0")" && pwd)
CASE=$here/../../fixtures/e_mbar_reinit_drain
OUT=$(mktemp -d)
trap 'rm -rf "$OUT"' EXIT
verdicts() { # <mode flag or empty>: "<variant> <front> <classes>" per line
  CUDA_VISIBLE_DEVICES= "$PTXV" check "$CASE" --no-input-coverage --out "$OUT" $1 2>/dev/null \
    | sed -nE 's/^(PASS|FAIL) +([^ ]+) +(ptx|sass) +\[([^]]*)\].*/\2 \3 \4/p'
}
exh=$(verdicts "")
cert=$(verdicts --certify)
[ -n "$exh" ] && [ -n "$cert" ] || { echo "no verdicts (PTXV=$PTXV)"; exit 2; }
rc=0
printf '%-8s %-5s %-18s %-18s %s\n' variant front expected exhaustive certify
while read -r v f e; do
  c=$(awk -v v="$v" -v f="$f" '$1 == v && $2 == f { print $3 }' <<<"$cert")
  want=$(python3 -c 'import sys, tomllib
m = tomllib.load(open(sys.argv[1], "rb"))
print(",".join(next(x for x in m["variants"] if x["id"] == sys.argv[2])["expect"]))' "$CASE/case.toml" "$v")
  mark=
  if [ "$e" != "$want" ] || [ "$c" != "$want" ]; then mark=" <- differs"; rc=1; fi
  printf '%-8s %-5s %-18s %-18s %s%s\n' "$v" "$f" "$want" "$e" "$c" "$mark"
done <<<"$exh"
exit $rc
