# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Compare desc_version dumps (run.sh) with the documented (1, 1) dump, config by config."""

import glob
import os
import sys

import numpy as np

d = sys.argv[1]
cfgs = [l.split(" ", 1)[1].strip() for l in open(os.path.join(d, "configs.txt"))]
n = 128 * 64
ref = np.fromfile(os.path.join(d, "d_1_1_0.bin"), np.uint32).reshape(len(cfgs), n)
# The reference itself: the MMA wrote every cell (no 0xffffffff fill left), configs differ.
print(f"{len(cfgs)} configs x {n} f32 outputs")
print("reference (1, 1): cells left at the fill value:", int((ref == 0xFFFFFFFF).sum()))
print(
    "reference (1, 1): distinct dumps among configs:", len({r.tobytes() for r in ref})
)
for f in sorted(glob.glob(os.path.join(d, "d_*.bin"))):
    va, vb, r = os.path.basename(f)[2:-4].split("_")
    x = np.fromfile(f, np.uint32)
    if x.size != ref.size:
        print(
            f"va={va} vb={vb} run {r}: {x.size // n} of {len(cfgs)} configs (launch failed)"
        )
        continue
    x = x.reshape(len(cfgs), n)
    bad = [i for i in range(len(cfgs)) if not np.array_equal(x[i], ref[i])]
    print(
        f"va={va} vb={vb} run {r}: {len(cfgs) - len(bad)}/{len(cfgs)} configs bitwise equal to (1, 1)"
        + ("".join(f"\n    differs: {cfgs[i]}" for i in bad))
    )
