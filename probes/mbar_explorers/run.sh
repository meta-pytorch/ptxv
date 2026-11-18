#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# The mbarrier memory-reuse fixtures under every explorer (MISS 218 / 241 / 242 follow-ups, RMK4's
# review): for each fixture, every variant on both fronts under the stateful explorer (default),
# DPOR (PTXV_EXPLORER=dpor), --certify and no reductions (PTXV_NO_LOCAL_POR=1
# PTXV_NO_SYNC_POR=1), one run each, expected (`expect`, `expect_sass` for SASS) next to observed. A row that
# differs, or a variant without a verdict, makes the exit status 1, unless OPEN below pins it
# (a registered, not yet fixed miss, with the row it gives meanwhile: printed as "open MISS n";
# STRICT=1 counts them too).
# Simulator only (no GPU).
#   PTXV=<ptxv binary> probes/mbar_explorers/run.sh [fixture ...]   (default: the list below)
set -u
PTXV=${PTXV:-ptxv}
here=$(cd "$(dirname "$0")" && pwd)
FX=$here/../../fixtures
FIXTURES=${*:-e_mbar_store_order e_mbar_hidden_write e_mbar_partial_store e_mbar_reuse_ordered
  e_mbar_inval_vis e_mbar_lane_inval e_mbar_double_sts_init e_mbar_resurrect e_mbar_init_restrict
  e_mbar_reinits e_mbar_discarded_poll}
# fixture:variant:front MISS — rows that differ for a registered miss not fixed yet.
# Each open row is pinned: "fixture:variant:front MISS exh|dpor|cert|nopor", the classes each
# explorer reports while the miss is open ("(none)": no verdict within TO). A row that matches
# its pin prints "<- open MISS n"; one that does not (a column the miss does not explain moved)
# is a difference.
# 265: two STS.64 inits by two warps clean on SASS. 266: an STS.64-initialized object revives
# after its inval (SASS). 267: a wait on an invalidated (or never initialized) object never
# finishes under DPOR or --certify. 272: a poll whose result is discarded (SYNCS.PHASECHK.TRANS64 PT) is lifted to a
# NOP on SASS.
OPEN="
e_mbar_inval_vis:TWAIT:ptx 267 violation|(none)|violation|violation
e_mbar_inval_vis:TWAIT:sass 272 clean|clean|clean|clean
e_mbar_inval_vis:TWAITS:ptx 267 violation|(none)|(none)|violation
e_mbar_inval_vis:TWAITS:sass 272 clean|clean|clean|clean
e_mbar_inval_vis:TRYW:ptx 267 violation|(none)|violation|violation
e_mbar_inval_vis:TRYW:sass 272 clean|clean|clean|clean
e_mbar_inval_vis:TRYWS:ptx 267 violation|(none)|(none)|violation
e_mbar_inval_vis:TRYWS:sass 272 clean|clean|clean|clean
e_mbar_inval_vis:TWAITU:ptx 267 violation|(none)|violation|violation
e_mbar_inval_vis:TWAITU:sass 267 violation|(none)|violation|violation
e_mbar_double_sts_init:good:sass 265 clean|clean|clean|clean
e_mbar_double_sts_init:PREINV:sass 265 clean|clean|clean|clean
e_mbar_resurrect:good:sass 266 clean|clean|clean|clean
e_mbar_resurrect:WAIT:ptx 267 violation|(none)|(none)|violation
e_mbar_resurrect:WAIT:sass 272 clean|clean|clean|clean
e_mbar_resurrect:INV2:sass 266 clean|clean|clean|clean
e_mbar_discarded_poll:good:ptx 267 violation|(none)|(none)|violation
e_mbar_discarded_poll:good:sass 272 clean|clean|clean|clean
e_mbar_discarded_poll:INV:ptx 267 violation|(none)|(none)|violation
e_mbar_discarded_poll:INV:sass 272 clean|clean|clean|clean
e_mbar_discarded_poll:USED:ptx 267 violation|(none)|(none)|violation
e_mbar_discarded_poll:USED:sass 267 violation|(none)|(none)|violation
"
OUT=$(mktemp -d)
trap 'rm -rf "$OUT"' EXIT
rc=0
for fx in $FIXTURES; do
  CASE=$FX/$fx
  VARS=$(python3 -c 'import sys, tomllib
print(" ".join(x["id"] for x in tomllib.load(open(sys.argv[1], "rb"))["variants"]))' "$CASE/case.toml")
  echo "== $fx"
  printf '%-10s %-5s %-26s %-26s %-26s %-26s %s\n' variant front expected exh dpor certify nopor
  for v in $VARS; do
    for f in ptx sass; do
      want=$(python3 -c 'import sys, tomllib
x = next(x for x in tomllib.load(open(sys.argv[1], "rb"))["variants"] if x["id"] == sys.argv[2])
print(",".join(x.get("expect_sass", x["expect"]) if sys.argv[3] == "sass" else x["expect"]))' "$CASE/case.toml" "$v" "$f")
      row=(); bad=
      # One run per variant, front and explorer (TO seconds each, default 300): a variant
      # that hangs under one explorer costs only its own row.
      for mode in exh dpor cert nopor; do
        case $mode in
          exh) envs=(); flags=() ;;
          dpor) envs=(PTXV_EXPLORER=dpor); flags=() ;;
          cert) envs=(); flags=(--certify) ;;
          nopor) envs=(PTXV_NO_LOCAL_POR=1 PTXV_NO_SYNC_POR=1); flags=() ;;
        esac
        g=$(env CUDA_VISIBLE_DEVICES= "${envs[@]}" timeout ${TO:-300} "$PTXV" check "$CASE" --variant "$v" \
          --front "$f" --no-input-coverage --out "$OUT/s" "${flags[@]}" </dev/null 2>/dev/null \
          | sed -nE 's/^(PASS|FAIL) +([^ ]+) +(ptx|sass) +\[([^]]*)\].*/\4/p')
        [ -n "$g" ] || g="(none)"
        row+=("$g"); [ "$g" = "$want" ] || bad=1
      done
      mark=
      if [ -n "$bad" ]; then
        miss=$(awk -v k="$fx:$v:$f" '$1 == k { print $2 }' <<<"$OPEN")
        pin=$(awk -v k="$fx:$v:$f" '$1 == k { print $3 }' <<<"$OPEN")
        got=$(IFS='|'; echo "${row[*]}")
        if [ -z "$miss" ]; then mark=" <- differs"; rc=1
        elif [ "$got" != "$pin" ]; then mark=" <- differs (open MISS $miss pinned $pin)"; rc=1
        elif [ -n "${STRICT:-}" ]; then mark=" <- differs (open MISS $miss)"; rc=1
        else mark=" <- open MISS $miss"; fi
      fi
      printf '%-10s %-5s %-26s %-26s %-26s %-26s %s%s\n' "$v" "$f" "$want" "${row[@]}" "$mark"
    done
  done
done
exit $rc
