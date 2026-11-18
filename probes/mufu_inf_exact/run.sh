#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# MISS 258 repro (simulator only, no GPU): lg2, rsqrt and rcp .approx (with and without .ftz) of x = ((a + b) + c) + e,
# GB300 x = 1 where the exact x is 0 (lanes 0..15), alone and under max / min / ex2 / rcp / add.
# The true values are -inf, +inf, a pole and what follows (max(-inf, -1000) = -1000, ex2(-inf) = 0);
# num_mufu makes each of them the exact value 0 (`fin`), and out_rec's MISS 258 check flags only NaN true values.
# Runs `ptxv num` on r8z_under/ and checks every element against exact values (check.py, Decimal;
# manifest.json holds the inputs). Exits 1 while an element is claimed whose chain has a MUFU result
# with no finite true value at its exact operand (N-NUM-NONFINITE: not analyzed), or whose claim or
# bound is below its true error; 2 on a tool failure. GB300's output of the case equals the
# simulator's (expected/r8z_under.gb300.txt).
#   PTXV=<ptxv> bash run.sh
set -u
HERE=$(dirname "$(readlink -f "$0")")
PTXV=$(command -v "${PTXV:-ptxv}") || { echo "no ptxv binary"; exit 2; }
w=$(mktemp -d "${TMPDIR:-/tmp}/mufu_inf_exact.XXXXXX")
(cd "$w" && CUDA_VISIBLE_DEVICES= "$PTXV" num "$HERE/r8z_under" --out "$w" > "$w/num.out" 2>&1) || { cat "$w/num.out"; exit 2; }
mkdir "$w/csv"
for f in "$w"/good.*.num.csv; do cp "$f" "$w/csv/r8z_under.$(basename "$f")"; done
grep -E '^num good' "$w/num.out"
python3 -I "$HERE/check.py" "$w/csv" "$HERE/manifest.json" "$w/check.csv"
rc=$?
echo "per element: $w/check.csv"
echo "exit $rc"
exit $rc
