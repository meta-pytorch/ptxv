# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Marks kernel.ptx (as captured by capture.py) with `// @TAG` comments and writes the
# [[variants]] section of case.toml. Idempotent.
#   python3 variants.py
import re

src = [
    re.sub(r"\s*// @[A-Z0-9_]+$", "", ln)
    for ln in open("kernel.ptx").read().split("\n")
]
mmas = [i for i, ln in enumerate(src) if "tcgen05.mma" in ln]
assert len(mmas) == 4, mmas
# The idesc of the k-tile's first MMA (K-block 0): `or.b32 %rX, %r32, %r16` puts sf_id 0
# (from the scale-factor TMEM addresses) into the descriptor; the next MMA's uses %r19
# (sf_id 1).
ids = [
    max(i for i in range(m) if re.match(r"^\s*or\.b32\s+%r\d+, %r\d+, %r\d+;", src[i]))
    for m in mmas
]
sf = [re.search(r"(%r\d+);", src[i]).group(1) for i in ids]
src[ids[0]] += " // @SF0"
open("kernel.ptx", "w").write("\n".join(src))

variants = [
    ("good", "as compiled by CuTeDSL 4.6.1", "", '["race", "violation"]', '["clean"]'),
    (
        "B30",
        "B30: each k-tile's first MMA (K-block 0) uses the scale factors of K-block 1 (sf_id 1)",
        f'marker = "@SF0"\nfind = "{sf[0]};"\nreplace = "{sf[1]};"\n',
        '["race", "violation"]',
        '["clean"]',
    ),
]
case = open("case.toml").read()
case = case[: case.index("[[variants]]")] if "[[variants]]" in case else case
for vid, summary, edit, ptx, sass in variants:
    case += f'[[variants]]\nid = "{vid}"\nsummary = "{summary}"\n{edit}expect = {ptx}\nexpect_sass = {sass}\n\n'
open("case.toml", "w").write(case.rstrip("\n") + "\n")
print("SF ids", sf)
