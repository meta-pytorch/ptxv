"""probes/lds_waw: per variant, launches and threads whose second-load word (out[3g+2]) or first
word (out[3g]) differ from the expected value. usage: analyze.py DIR"""

import collections
import glob
import sys

import numpy as np

d = sys.argv[1]
S = {"bb": (0, 0), "cb": (128, 0), "cc": (128, 128), "cb_alu": (128, None)}
n = 1024 * 128
g = np.arange(n, dtype=np.uint32)
tid = g % 128
warp, lane = tid // 32, tid % 32
for v, (s1, s2) in S.items():
    l1 = lane if s1 else 0 * lane
    first = (warp << 20) + (l1 << 12)
    if s2 is None:
        second = np.full(n, 0x5A5A5A5A, np.uint32)
    else:
        l2 = lane if s2 else 0 * lane
        second = (warp << 20) + (l2 << 12) + 4
    stale = (warp << 20) + (l1 << 12) + 2  # the 128-bit load's word 2
    runs = bad = 0
    kinds = collections.Counter()
    for f in sorted(glob.glob(f"{d}/out/{v}_[0-9]*.bin")):
        o = np.fromfile(f, np.uint32).reshape(n, 3)
        runs += 1
        w = o[:, 2] != second
        f0 = o[:, 0] != first
        bad += int(w.any() or f0.any())
        kinds["second wrong"] += int(w.sum())
        kinds["second = stale word 2"] += int((o[:, 2] == stale).sum())
        kinds["first wrong"] += int(f0.sum())
    print(f"{v}: {runs} launches, {bad} with a wrong word; {dict(kinds)}")
