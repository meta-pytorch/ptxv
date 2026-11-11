# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Per cubin / on: runs and threads whose out[t] != {v, ~v} (v = t ^ 0x5a5a5a5a) for
probes/stg_guard_war. usage: analyze.py DIR"""

import collections
import glob
import sys

import numpy as np

d = sys.argv[1]
n = 1024 * 256
t = np.arange(n, dtype=np.uint32)
v = t ^ np.uint32(0x5A5A5A5A)
res = collections.defaultdict(lambda: [0, 0, 0])
for f in sorted(glob.glob(f"{d}/out/*.bin")):
    key = f.rsplit("/", 1)[1].rsplit("_", 1)[0]
    o = np.fromfile(f, np.uint32, count=2 * n).reshape(n, 2)
    bad = (o[:, 0] != v) | (o[:, 1] != ~v)
    r = res[key]
    r[0] += 1
    r[1] += int(bad.any())
    r[2] += int(bad.sum())
for k, (runs, badruns, badthreads) in sorted(res.items()):
    print(f"{k}: {runs} runs, {badruns} with a wrong A, {badthreads} threads")
