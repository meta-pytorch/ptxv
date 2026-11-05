#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Rung 1e hardware evidence at scale: fixtures/r1e_pdl with N CTAs per kernel (default 512),
# too large for the model's reference run, checked against the closed form instead:
# out[i] = 2 * (3 * i + 1) + 100. Usage: tools/r1e_scale.sh [RUNS] [CTAS]
set -euo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
runs=${1:-5}; n=${2:-512}
d=$(mktemp -d /tmp/r1e_scale.XXXX)
cp "$here/fixtures/r1e_pdl/kernel.ptx" "$d/"
sed -e "s/^grid = \[2, 1, 1\]/grid = [$n, 1, 1]/" -e "s/^shape = \[64\]/shape = [$((32 * n))]/" \
    "$here/fixtures/r1e_pdl/case.toml" > "$d/case.toml"
for v in good B18 B55; do
  for i in $(seq "$runs"); do
    rm -f "$d/out/$v.probe.bin"
    msg=$("$CARGO_TARGET_DIR/release/ptxv" probe "$d" --variant "$v" --out "$d/out" 2>&1 \
          | grep -oE "(ERROR|HANG).{0,90}" | head -1 || true)
    bad=$(python3 - "$d/out/$v.probe.bin" "$n" <<'PY'
import sys, struct
try: b = open(sys.argv[1], "rb").read()
except OSError: print("no output"); sys.exit()
m = 32 * int(sys.argv[2])
w = struct.unpack(f"{2*m}I", b[:8*m])          # buf, then out
print(sum(w[m + i] != 2 * (3 * i + 1) + 100 for i in range(m)), "of", m, "outputs wrong")
PY
)
    echo "$v run $i: ${msg:-ok} | $bad"
  done
done
