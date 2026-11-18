#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# probes/sb_mio_order end to end: ptxas, patch.py, the expected outputs, then RUNS scrubbed
# GB300 runs of each window (ptxv hw-worker). Usage: PTXV_GPUS=<idle gpu> probes/sb_mio_order/run.sh [RUNS] > results.txt
set -eu
here="$(cd "$(dirname "$0")" && pwd)"
runs="${1:-1000}"
ptxv="${PTXV:-ptxv}"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
"$PTXV_CUDA_HOME/bin/ptxas" -arch=sm_103a -o "$work/kernel.cubin" "$here/kernel.ptx"
python3 "$here/patch.py" "$work/kernel.cubin" "$work/v"
python3 - "$work" <<'PY'
import struct, sys
for name, b in (("ab", 1), ("a", 0)):
    v = []
    for c in range(1184):
        for t in range(256):
            x = 0x5A00C580 + t
            v += [x, x * b, 0, 0]
    open(f"{sys.argv[1]}/expected_{name}.bin", "wb").write(struct.pack(f"<{len(v)}I", *v))
PY
for v in ${VARIANTS:-inorder inorder_ldg rd_on_ldg rd_on_lds lds_nowait rd_on_a nowait}; do
  exp="$work/expected_ab.bin"
  case "$v" in rd_on_ldg|rd_on_lds*|lds_nowait*|wcov*|qwar*|rvs_shfl*|rvs_qctl*|dq_*) exp="$work/expected_a.bin" ;; esac
  n="$runs"
  case "$v" in rd_on_a|nowait|lds_nowait*|wcov_nowait*|qwar_ctl|dq_*ctl*) n=200 ;; esac
  printf '%-12s ' "$v"
  # the worker refuses any image but this one (FNV-1a of the patched cubin)
  fnv=$(python3 -c 'import sys
h = 0xcbf29ce484222325
for b in open(sys.argv[1], "rb").read():
    h = ((h ^ b) * 0x100000001b3) & 0xffffffffffffffff
print("%016x" % h)' "$work/v/$v.cubin")
  "$ptxv" hw-worker "$here" "$work/v/$v.cubin" --runs "$n" --scrub --expected "$exp" --image-fnv "$fnv" 2>&1 | tail -1 |
    python3 -c 'import json, sys
r = json.loads(sys.stdin.read())
w = [d[1] for d in r["distinct"] if d[1]] or [0]
print("%d/%d runs wrong; words wrong per failing run: %d..%d" % (r["mismatches"], r["runs"], min(w), max(w)))'
done
