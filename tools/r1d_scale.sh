#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Rung 1d hardware evidence: at the fixture's grid of 4 every CTA is resident at once on GB300,
# so try_cancel never succeeds. This reruns fixtures/r1d_clc at 8192 CTAs (most CTAs are not
# launched yet when the first requests arrive) and counts wrong or missing tiles per variant.
# Usage: tools/r1d_scale.sh [RUNS]   (ptxv picks a GPU with no foreign processes)
set -euo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
runs=${1:-3}; n=8192
d=$(mktemp -d /tmp/r1d_scale.XXXX)
cp "$here/fixtures/r1d_clc/kernel.ptx" "$d/"
sed -e "s/^grid = .*/grid = [$n, 1, 1]/" -e "s/^shape = \[128\]/shape = [$((32 * n))]/" \
    "$here/fixtures/r1d_clc/case.toml" > "$d/case.toml"
for v in good B17 B53 B54; do
  for i in $(seq "$runs"); do
    rm -f "$d/out/$v.probe.bin"
    # `ptxv probe` compiles the variant, runs it once on a free GPU and dumps the output.
    msg=$("$CARGO_TARGET_DIR/release/ptxv" probe "$d" --variant "$v" \
          --out "$d/out" 2>&1 | grep -oE "(ERROR|HANG).{0,90}" | head -1 || true)
    bad=$(python3 - "$d/out" "$v" "$n" <<'PY'
import sys, glob, struct
out, v, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
try: b = open(f"{out}/{v}.probe.bin", "rb").read()
except OSError: print("no output"); sys.exit()
w = struct.unpack(f"{32*n}I", b[:128*n])
print(sum(any(w[32*t+l] != 1000*t+l for l in range(32)) for t in range(n)), "of", n, "tiles wrong")
PY
)
    echo "$v run $i: ${msg:-ok} | $bad"
  done
done
