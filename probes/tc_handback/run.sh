#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# GB300 runs of the tc_handback probe: 74 pairs (148 CTAs), ROUNDS rounds per launch, RUNS
# launches per variant, through `ptxv hw` (bitwise against all-zero counts).
#   PTXV=<ptxv binary> ./run.sh [RUNS] [ROUNDS]
set -eu
RUNS=${1:-20000}; ROUNDS=${2:-1000}
D=$(mktemp -d)
cp "$(dirname "$0")"/kernel.ptx "$D"/
sed -e 's/^grid = \[2, 1, 1\]/grid = [148, 1, 1]/' -e 's/^shape = \[256\]/shape = [18944]/' \
    -e "s/^value = 2$/value = $ROUNDS/" "$(dirname "$0")"/case.toml > "$D"/case.toml
mkdir -p "$D"/expected
head -c $((18944 * 4)) /dev/zero > "$D"/expected/out.bin
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1
for v in ${VARIANTS:-good RCTA EARLY EARLYD NOWST}; do
  "$PTXV" hw "$D" --variant $v --runs "$RUNS" --scrub --no-num --timeout 900 || echo "hw $v: exit $?"
done
rm -rf "$D"
