# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Marks kernel.ptx (as captured by capture.py) with `// @TAG` comments and writes the
# [[variants]] section of case.toml. Idempotent: old tags and variants are replaced.
#   python3 variants.py
import re

DESC = 4611686293372403712  # Triton's smem descriptor constant: bits 46-48 = 0b000
DESC_FIXED = DESC | (
    1 << 46
)  # the fixed version field 0b001 (PTX ISA tcgen05 descriptor)
BEFORE = "tcgen05.fence::before_thread_sync;"
AFTER = "tcgen05.fence::after_thread_sync;"

src = [
    re.sub(r"\s*// @[A-Z0-9_]+$", "", ln)
    for ln in open("kernel.ptx").read().split("\n")
]


def find(pat, start=0):
    for i in range(start, len(src)):
        if re.search(pat, src[i]):
            return i
    raise SystemExit(f"pattern not found: {pat}")


def tag(i, t):
    src[i] += f" // @{t}"


alloc = find(r"tcgen05\.alloc")
tag(find(r"^\s*bar\.sync", alloc), "ALLOC_BAR")
zero = find(r"tcgen05\.wait::st")
tag(find(r"^\s*bar\.sync", zero), "ZERO_BAR")
mmas = [i for i, ln in enumerate(src) if "tcgen05.mma" in ln]
groups = [mmas[0]] + [b for a, b in zip(mmas, mmas[1:]) if b - a > 20]
for k, i in enumerate(groups):
    tag(i, f"MMA{k}")
lds = [i for i, ln in enumerate(src) if re.search(r"\btcgen05\.ld\.", ln)]
for k, i in enumerate(lds):
    tag(i, f"LD{k}")
    tag(find(r"tcgen05\.wait::ld", i), f"LDW{k}")
tag(find(r"tcgen05\.dealloc"), "DEALLOC")
descs = [i for i, ln in enumerate(src) if str(DESC) in ln]
for k, i in enumerate(descs):
    tag(i, f"D{k:02}")
# Mutation sites: the TMA-ready wait before the loop MMA, the accumulator-ready wait before
# the epilogue tcgen05.ld, the bulk-store drain, and the proxy fence before the TMA store.
waits = [i for i, ln in enumerate(src) if "mbarrier.try_wait" in ln]
mma_wait = max(w for w in waits if w < groups[1])
acc_wait = max(w for w in waits if w < lds[0])
for w, t in ((mma_wait, "LOADWAIT"), (acc_wait, "ACCWAIT")):
    tag(w, t)
    tag(w + 1, t)  # its `@!complete bra.uni waitLoop`
# The drain before the peeled last epilogue: at this grid each CTA's loop epilogue runs once
# (with nothing in flight), so the drain that matters is the one protecting the staging buffer
# from the second tile's st.shared.
drains = [i for i, ln in enumerate(src) if "cp.async.bulk.wait_group.read" in ln]
tag(max(i for i in drains if lds[0] < i < lds[1]), "DRAIN")
tag(find(r"fence\.proxy\.async", lds[0]), "PROXY")
open("kernel.ptx", "w").write("\n".join(src))


def edit(marker, find, replace):
    return f'{{ marker = "@{marker}", find = "{find}", replace = "{replace}" }}'


fenced = [  # one edit per marked line
    edit("ALLOC_BAR", "bar.sync \\t0;", f"{BEFORE} bar.sync 0; {AFTER}"),
    edit("ZERO_BAR", "bar.sync \\t0;", f"{BEFORE} bar.sync 0; {AFTER}"),
    *[edit(f"MMA{k}", "@", f"{AFTER} @") for k in range(len(groups))],
    *[edit(f"LD{k}", "tcgen05.ld", f"{AFTER} tcgen05.ld") for k in range(len(lds))],
    *[edit(f"LDW{k}", "aligned;", f"aligned; {BEFORE}") for k in range(len(lds))],
    edit("DEALLOC", "@", f"{AFTER} @"),
    *[edit(f"D{k:02}", str(DESC), str(DESC_FIXED)) for k in range(len(descs))],
]


def deletes(*markers):
    return [f'{{ marker = "@{m}", replace = "<delete>" }}' for m in markers]


variants = [
    ("good", "as compiled by Triton 3.5", [], '["race", "violation"]', '["violation"]'),
    (
        "fenced",
        "Triton's PTX with the tcgen05 fences it omits and the descriptor version fixed",
        fenced,
        '["clean"]',
        '["clean"]',
    ),
    (
        "M1",
        "fenced, minus the TMA-ready wait before the loop's MMAs",
        fenced + deletes("LOADWAIT"),
        '["nondeterministic", "race", "violation"]',
        '["nondeterministic", "race", "violation"]',
    ),
    (
        "M2",
        "fenced, minus the accumulator-ready wait before the epilogue tcgen05.ld",
        fenced + deletes("ACCWAIT"),
        '["nondeterministic", "race", "violation"]',
        '["nondeterministic", "race", "violation"]',
    ),
    (
        "M3",
        "fenced, minus the bulk-store drain before the last tile's epilogue reuses the staging smem",
        fenced + deletes("DRAIN"),
        '["race"]',
        '["race"]',
    ),
    (
        "M4",
        "fenced, minus the generic->async proxy fence before the TMA store",
        fenced + deletes("PROXY"),
        '["race"]',
        '["race"]',
    ),
]
case = open("case.toml").read()
case = case[: case.index("[[variants]]")] if "[[variants]]" in case else case
for vid, summary, edits, ptx, sass in variants:
    case += f'[[variants]]\nid = "{vid}"\nsummary = "{summary}"\n'
    if edits:
        case += "edits = [\n  " + ",\n  ".join(edits) + ",\n]\n"
    case += f"expect = {ptx}\nexpect_sass = {sass}\n\n"
open("case.toml", "w").write(case.rstrip("\n") + "\n")
print(
    f"tagged: {len(groups)} MMA groups, {len(lds)} tcgen05.ld, {len(descs)} descriptors"
)
