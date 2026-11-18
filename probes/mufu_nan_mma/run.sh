#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# MISS 258 repro (simulator only, no GPU): sqrt.approx.ftz of a negative subnormal (GB300 -0, no real true value) in lane 0,
# converted to bf16 and summed by mma.sync m16n8k16 with B = bf16 ones (rows 0 and 8 of D, out[0..16)).
# out_rec takes an MMA's products as terms and never walks its factor nodes, so the MISS 258 check
# does not see the MUFU under the factor, whose exact value num_mufu made up (`fin`).
# Runs `ptxv num` on r8m_mma/ and checks every element against exact values (check.py, Decimal;
# manifest.json holds the inputs). Exits 1 while an element is claimed whose chain has a MUFU result
# with no finite true value at its exact operand (N-NUM-NONFINITE: not analyzed), or whose claim or
# bound is below its true error; 2 on a tool failure. GB300's output of the case equals the
# simulator's (expected/r8m_mma.gb300.txt).
#   PTXV=<ptxv> bash run.sh
set -u
HERE=$(dirname "$(readlink -f "$0")")
PTXV=$(command -v "${PTXV:-ptxv}") || { echo "no ptxv binary"; exit 2; }
w=$(mktemp -d "${TMPDIR:-/tmp}/mufu_nan_mma.XXXXXX")
(cd "$w" && CUDA_VISIBLE_DEVICES= "$PTXV" num "$HERE/r8m_mma" --out "$w" > "$w/num.out" 2>&1) || { cat "$w/num.out"; exit 2; }
mkdir "$w/csv"
for f in "$w"/good.*.num.csv; do cp "$f" "$w/csv/r8m_mma.$(basename "$f")"; done
grep -E '^num good' "$w/num.out"
python3 -I "$HERE/check.py" "$w/csv" "$HERE/manifest.json" "$w/check.csv"
rc=$?
echo "per element: $w/check.csv"
echo "exit $rc"
exit $rc
