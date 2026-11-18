#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# MISS 218 repro (fixtures/e_mbar_sts_live; CASE=<fixture dir> runs another fixture the same way,
# probes/mbar_store_order): an mbarrier.init over a live mbarrier without an
# inval, which ptxas lowers to a plain STS.64 of the state word (+ SYNCS.CCTL.IVALL, no
# SYNCS.CCTL.IV) for a per-thread address, must re-initialize the object on the SASS front as it
# does on the PTX front; a plain store over a live mbarrier must not be clean. Runs every variant
# on both fronts through `ptxv check` (exhaustive) and `ptxv check --certify`, prints both
# verdicts next to the expected one (`expect`, `expect_sass` for SASS), and exits 1 if any
# differ. Simulator only (no GPU).
#   PTXV=<ptxv binary> probes/mbar_sts_live/run.sh
set -u
PTXV=${PTXV:-ptxv}
here=$(cd "$(dirname "$0")" && pwd)
CASE=${CASE:-$here/../../fixtures/e_mbar_sts_live}
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
# Every variant on both fronts must have a verdict in both modes: a variant that fails to build
# or errors prints no PASS / FAIL row and counts as a difference (it was skipped before).
for v in $(python3 -c 'import sys, tomllib
print(" ".join(x["id"] for x in tomllib.load(open(sys.argv[1], "rb"))["variants"]))' "$CASE/case.toml"); do
  for f in ptx sass; do
    for mode in exhaustive certify; do
      rows=$exh; [ $mode = certify ] && rows=$cert
      awk -v v="$v" -v f="$f" '$1 == v && $2 == f { n++ } END { exit n != 1 }' <<<"$rows" \
        || { echo "$v $f $mode: no verdict (the variant errored) <- differs"; rc=1; }
    done
  done
done
printf '%-9s %-5s %-26s %-26s %s\n' variant front expected exhaustive certify
while read -r v f e; do
  c=$(awk -v v="$v" -v f="$f" '$1 == v && $2 == f { print $3 }' <<<"$cert")
  want=$(python3 -c 'import sys, tomllib
m = tomllib.load(open(sys.argv[1], "rb"))
x = next(x for x in m["variants"] if x["id"] == sys.argv[2])
print(",".join(x.get("expect_sass", x["expect"]) if sys.argv[3] == "sass" else x["expect"]))' \
    "$CASE/case.toml" "$v" "$f")
  mark=
  if [ "$e" != "$want" ] || [ "$c" != "$want" ]; then mark=" <- differs"; rc=1; fi
  printf '%-9s %-5s %-26s %-26s %s%s\n' "$v" "$f" "$want" "$e" "$c" "$mark"
done <<<"$exh"
exit $rc
