# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Classify each element of each run of probes/tma_red_order: a0 first, a1 first, single
rounding, or other. usage: analyze.py DIR (g.bin, a.bin, out/m{0,1}_*.bin)"""

import collections
import glob
import struct
import sys

d = sys.argv[1]
sys.path.insert(0, d)
from gen import f, rn  # noqa: E402  (gen.py runs nothing on import)

g = struct.unpack("<4096H", open(f"{d}/g.bin", "rb").read())
a = struct.unpack("<8192H", open(f"{d}/a.bin", "rb").read())
p0 = [rn(f(rn(f(g[i]) + f(a[i]))) + f(a[4096 + i])) for i in range(4096)]
p1 = [rn(f(rn(f(g[i]) + f(a[4096 + i]))) + f(a[i])) for i in range(4096)]
s1 = [rn(f(g[i]) + f(a[i]) + f(a[4096 + i])) for i in range(4096)]
for m in (0, 1):
    runs = collections.Counter()
    tot = collections.Counter()
    for fn in sorted(glob.glob(f"{d}/out/m{m}_*.bin")):
        o = struct.unpack("<4096H", open(fn, "rb").read())
        c = collections.Counter(
            "a0"
            if o[i] == p0[i]
            else "a1"
            if o[i] == p1[i]
            else "fused"
            if o[i] == s1[i]
            else "other"
            for i in range(4096)
        )
        tot.update(c)
        runs["mixed" if len(c) > 1 else next(iter(c))] += 1
    print(f"mode {m}: runs {dict(runs)}; elements {dict(tot)}")
