#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Show the role instructions of one litmus kernel with decoded control bits
(stall, wbar=scoreboard set, wait mask) so back-to-back LDS without a scoreboard wait is visible.
usage: ctrl.py litmus.sass <mangled-substring> <regex of instructions to show>"""

import re
import sys

sass, key, pat = open(sys.argv[1]).read(), sys.argv[2], sys.argv[3]
fn = [f for f in re.split(r"\n\s*Function : ", sass) if f.startswith(key)][0]
lines = fn.split("\n")
for i, l in enumerate(lines):
    m = re.match(r"\s*/\*([0-9a-f]{4})\*/\s*(.*?);\s*/\* 0x([0-9a-f]{16}) \*/", l)
    if not m:
        continue
    hi = int(re.search(r"0x([0-9a-f]{16})", lines[i + 1]).group(1), 16)
    c = hi >> 41
    stall, wbar, rbar, wait = c & 15, (c >> 5) & 7, (c >> 8) & 7, (c >> 11) & 63
    if re.search(pat, m.group(2)):
        print(
            f"{m.group(1)}  {m.group(2):45s} stall={stall:2d} wbar={'-' if wbar == 7 else 'SB%d' % wbar} rbar={'-' if rbar == 7 else 'SB%d' % rbar} wait={'{' + ','.join('SB%d' % b for b in range(6) if wait >> b & 1) + '}'}"
        )
