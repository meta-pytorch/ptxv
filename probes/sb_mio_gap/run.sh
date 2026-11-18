#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# probes/sb_mio_gap end to end: ptxas, patch.py, the expected outputs, then RUNS scrubbed GB300
# runs of each window (ptxv hw-worker). Usage:
#   PTXV_GPUS=<idle gpu> [REGIME=occ|sat] [NSAT=n] [VARIANTS="..."] probes/sb_mio_gap/run.sh [RUNS] > results
# REGIME=sat runs sat/case.toml with nsat = NSAT (default 8). CAL=1: the cal_<G> variants of
# kernel_cal.ptx (mkcal.py), printing the window's measured cycles (first run's threads).
set -eu
here="$(cd "$(dirname "$0")" && pwd)"
runs="${1:-1000}"
ptxv="${PTXV:-ptxv}"
regime="${REGIME:-occ}"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
case_dir="$here"
if [ "$regime" = sat ]; then
  case_dir="$work/case"
  mkdir -p "$case_dir"
  sed "s/^value = NSAT$/value = ${NSAT:-8}/; s|^ptx = .*|ptx = \"$here/kernel.ptx\"|" "$here/sat/case.toml" > "$case_dir/case.toml"
fi
ptx="$here/kernel.ptx"
if [ "${CAL:-0}" = 1 ]; then
  python3 "$here/mkcal.py" "$ptx" > "$work/kernel_cal.ptx"
  ptx="$work/kernel_cal.ptx"
fi
"$PTXV_CUDA_HOME/bin/ptxas" -arch=sm_103a -o "$work/kernel.cubin" "$ptx"
SB_MIO_ONLY="${VARIANTS:-}" python3 "$here/patch.py" "$work/kernel.cubin" "$work/v" > "$work/patch.log"
python3 - "$work" <<'PY'
import struct, sys
# Thread t's {region 0, 1, 2, x, W[0..3]} per expected-output tag (patch.py).
for tag in ("a", "u8", "u16", "w64", "w128", "stsm", "aux", "cal"):
    v = []
    for c in range(1184):
        for t in range(256):
            m = 0x5A00C580 + t
            s0 = {"a": m, "u8": m & 0xFF, "u16": m & 0xFFFF}.get(tag, 0)
            x = m if tag == "aux" else 0
            row = [0x5A00C580 + (t & ~31) + 4 * (t % 8) + k for k in range(4)]
            w = {"w64": [m, m, 0, 0], "w128": [m] * 4, "stsm": row}.get(tag, [0] * 4)
            v += [s0, 0, 0, x, *w]
    open(f"{sys.argv[1]}/expected_{tag}.bin", "wb").write(struct.pack(f"<{len(v)}I", *v))
PY
all=$(sed -n 's/^\([a-z0-9_]*_g[0-9]*\) .*/\1/p' "$work/patch.log")
for v in ${VARIANTS:-$all}; do
  printf '%-18s ' "$v"
  # the worker refuses any image but this one (FNV-1a of the patched cubin)
  fnv=$(python3 -c 'import sys
h = 0xcbf29ce484222325
for b in open(sys.argv[1], "rb").read():
    h = ((h ^ b) * 0x100000001b3) & 0xffffffffffffffff
print("%016x" % h)' "$work/v/$v.cubin")
  if [ "${CAL:-0}" = 1 ]; then
    "$ptxv" hw-worker "$case_dir" "$work/v/$v.cubin" --runs "$runs" --scrub --dump "$work/dump.bin" --image-fnv "$fnv" > /dev/null 2>&1
    python3 -c 'import struct, sys
b = open(sys.argv[1], "rb").read()
w = struct.unpack(f"<{len(b) // 4}I", b)
d = sorted(w[3::8])
print("window cycles over %d threads (first run): min %d, median %d, max %d" % (len(d), d[0], d[len(d) // 2], d[-1]))' "$work/dump.bin"
    continue
  fi
  tag=$(awk -v v="$v" '$1 == v {print $2}' "$work/patch.log")
  "$ptxv" hw-worker "$case_dir" "$work/v/$v.cubin" --runs "$runs" --scrub --expected "$work/expected_$tag.bin" --image-fnv "$fnv" 2>&1 | tail -1 |
    python3 -c 'import json, sys
s = sys.stdin.read()
try:
    r = json.loads(s)
except ValueError:
    sys.exit("hw-worker failed: " + s.strip()[-300:])
w = [d[1] for d in r["distinct"] if d[1]] or [0]
print("%d/%d runs wrong; words wrong per failing run: %d..%d" % (r["mismatches"], r["runs"], min(w), max(w)))'
done
