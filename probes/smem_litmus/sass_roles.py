#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Print, in program order, the memory/fence SASS of every litmus<T,M,G,WF,RF> kernel minus the
harness's own accesses (smem init `STS [..], RZ`, stress scratch `+0x8000]`, evaluator
`LDG.E.STRONG.SYS` / `ATOMS` / `RED`).  What remains is the litmus role code plus, for global
tests, the plain `STG.E` result stores.  Runs of identical lines are folded as `(xN)`."""

import re
import subprocess
import sys

TN = [
    "MP",
    "MPS",
    "SB",
    "LB",
    "CORR",
    "COWR",
    "IRIW",
    "TEAR32",
    "TEAR64",
    "TEAR128",
    "MPV",
    "MPG",
    "MPL1",
    "MPGN",
]
MN = ["shv", "shw", "glw", "gls", "glv", "dsm", "dsr"]
GN = ["cta", "clu"]
binary = sys.argv[1] if len(sys.argv) > 1 else "litmus"
sass = subprocess.run(
    ["/usr/local/cuda-13.0/bin/cuobjdump", "-sass", binary],
    capture_output=True,
    text=True,
).stdout
res = []
for fn in re.split(r"\n\s*Function : ", sass)[1:]:
    name = fn.split("\n", 1)[0].strip()
    m = re.match(r"_Z6litmusILi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)E", name)
    if not m:
        continue
    t, mm, g, wf, rf = map(int, m.groups())
    ins = [
        re.sub(r"\s*/\*[0-9a-f]+\*/\s*", "", ln).split(";")[0].strip()
        for ln in fn.split("\n")
        if re.search(r"/\*[0-9a-f]{4}\*/", ln)
    ]
    out = []
    for i in ins:
        if not re.search(
            r"\b(LDS|STS|LDG|STG|LD|ST|MEMBAR|FENCE|CCTL|ERRBAR)\b|\bLDS\.|\bSTS\.|\bLD\.|\bST\.|MEMBAR|FENCE|CCTL",
            i,
        ):
            continue
        if re.search(
            r"STS(\.\d+)? \[[^\]]*\], RZ$|\+0x8000\]|ATOMS|^RED|LDG\.E\.STRONG\.SYS", i
        ):
            continue
        i = re.sub(r"R\d+", "R", i)
        i = re.sub(r"UR\d+", "UR", i)
        i = re.sub(r"P\d", "P", i)
        if out and out[-1][0] == i:
            out[-1][1] += 1
        else:
            out.append([i, 1])
    res.append((t, mm, g, wf, rf, out))
for t, mm, g, wf, rf, out in sorted(res):
    print(f"== {TN[t]} {MN[mm]} {GN[g]} wf={wf} rf={rf}")
    for o, n in out:
        print("   ", o + (f"   (x{n})" if n > 1 else ""))
