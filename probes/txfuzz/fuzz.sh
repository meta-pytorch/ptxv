#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# mbarrier tx fuzz driver (MISS 273): generates N cases from SEED (gen.py; 400 cases from seed 2
# by default, the set RDP2 ran: 1 miss at the last-arrival levels' i < d rule, case c2_0282 =
# probes/tx_level_po) and runs run.sh on each, JOBS at a time. Prints the MISS lines and a
# tally; exits 1 if any case misses a class of the unreduced search.
# Usage: PTXV=<ptxv> [FRONT=ptx] [SEED=2] [N=400] [JOBS=8] probes/txfuzz/fuzz.sh [OUTDIR]
set -u
here="$(cd "$(dirname "$0")" && pwd)"
: "${PTXV:?}"
out=${1:-$(mktemp -d)}
python3 "$here/gen.py" "$out/cases" "${SEED:-2}" "${N:-400}"
ls -d "$out"/cases/* | xargs -P "${JOBS:-8}" -n 4 "$here/run.sh" > "$out/results-${FRONT:-ptx}.txt" 2>&1
n=$(grep -c . "$out/results-${FRONT:-ptx}.txt")
m=$(grep -c MISS "$out/results-${FRONT:-ptx}.txt")
grep MISS "$out/results-${FRONT:-ptx}.txt"
echo "${FRONT:-ptx}: $m of $n cases miss (results in $out)"
[ "$m" = 0 ]
