# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Copy a small replay sample of `mm ldr` / `mm sr3` dumps into results/ (the full dumps are
# ~1.5 GB): the first BYTES bytes of every member input and of member 0's first result, and
# the sr3 rows k in {0, 63, 64, 127} of every section and frac.
#   python extract.py RUNDIR_LDR RUNDIR_SR3
import os
import shutil
import sys

import numpy as np

BYTES = 8192
HERE = os.path.dirname(os.path.abspath(__file__))
ldr, sr3 = sys.argv[1], sys.argv[2]
out = os.path.join(HERE, "results", "ldr")
os.makedirs(out, exist_ok=True)
shutil.copy(os.path.join(ldr, "meta.txt"), os.path.join(out, "meta.txt"))
for f in sorted(os.listdir(ldr)):
    if not f.endswith(".bin"):
        continue
    t, kind = f.split(".")[0], f.split(".")[1]
    if kind.startswith("in") or f.endswith(".out0.0.bin"):
        b = np.fromfile(os.path.join(ldr, f), dtype=np.uint8)[:BYTES]
        name = f if kind.startswith("in") else f"{t}.out.bin"
        b.tofile(os.path.join(out, name))
o3 = os.path.join(HERE, "results", "sr3")
os.makedirs(o3, exist_ok=True)
rows = [
    s * 16384 + f * 128 + k
    for s in range(4)
    for f in range(8)
    for k in (0, 63, 64, 127)
]
for tn in ("bf16x8a", "bf16x8", "f16x8a"):
    for part in [f"in{m}" for m in range(4)] + ["r0"]:
        a = np.fromfile(
            os.path.join(sr3, f"sr3_{tn}.{part}.bin"), dtype=np.uint16
        ).reshape(-1, 8)
        a[rows].tofile(os.path.join(o3, f"{tn}.{part}.bin"))
