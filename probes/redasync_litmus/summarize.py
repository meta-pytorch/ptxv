# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Aggregates redlit result lines (sweep.sh output) by form and configuration.
import collections
import re
import sys

FORMS = {
    0: "red.async.release.gpu",
    1: "red.relaxed.gpu",
    2: "red.release.gpu",
    3: "fence.acq_rel.cta + red.relaxed.gpu",
    4: "st.relaxed.gpu",
    5: "st.release.gpu",
}
tot = collections.defaultdict(lambda: [0, 0, 0, 0, 0])
for path in sys.argv[1:]:
    for line in open(path):
        if "FAILED" in line or not line.startswith("form="):
            continue
        kv = dict(x.split("=", 1) for x in line.split())
        cfg = " ".join(
            f"{k}={kv[k]}"
            for k in ("mode", "nw", "js", "ps", "stress", "smode", "pp", "dmask")
        )
        t = tot[(int(kv["form"]), cfg)]
        t[0] += 1
        for i, k in enumerate(("ok", "stale", "other", "timeout"), 1):
            t[i] += int(kv[k])
print(
    f"{'form':40} {'config':55} {'runs':>4} {'reads':>14} {'stale':>10} {'other':>6} {'timeout':>7}"
)
for (f, cfg), t in sorted(tot.items()):
    print(
        f"{FORMS[f]:40} {cfg:55} {t[0]:4} {t[1] + t[2] + t[3]:14} {t[2]:10} {t[3]:6} {t[4]:7}"
    )
byf = collections.defaultdict(lambda: [0, 0])
for (f, _), t in tot.items():
    byf[f][0] += t[1] + t[2] + t[3]
    byf[f][1] += t[2]
for f, (n, s) in sorted(byf.items()):
    print(f"total {FORMS[f]:40} reads {n} stale {s}")
