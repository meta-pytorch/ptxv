"""probes/stg_na: per variant, launches and threads whose second load returned the first value
(a stale L1 line) or anything other than first + 0x1000. usage: analyze.py DIR"""

import glob
import sys

import numpy as np

d = sys.argv[1]
for v in ("na", "plain", "remote"):
    runs = stale = other = 0
    for f in sorted(glob.glob(f"{d}/out/{v}_*.bin")):
        o = np.fromfile(f, np.uint32).reshape(-1, 2)
        runs += 1
        s = o[:, 1] == o[:, 0]
        ok = o[:, 1] == o[:, 0] + 0x1000
        stale += int(s.sum())
        other += int((~s & ~ok).sum())
    print(f"{v}: {runs} launches, {stale} stale second loads, {other} other")
