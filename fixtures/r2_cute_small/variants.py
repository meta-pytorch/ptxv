# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Marks kernel.ptx (as captured by capture.py) with `// @TAG` comments and writes the
# [[variants]] section of case.toml. Idempotent: old tags and variants are replaced.
#   python3 variants.py
import re

BEFORE = "tcgen05.fence::before_thread_sync;"
AFTER = "tcgen05.fence::after_thread_sync;"

src = [
    re.sub(r"\s*// @[A-Z0-9_]+$", "", ln)
    for ln in open("kernel.ptx").read().split("\n")
]


def find(pat, start=0, end=None):
    for i in range(start, end if end is not None else len(src)):
        if re.search(pat, src[i]):
            return i
    raise SystemExit(f"pattern not found: {pat}")


def tag(i, t):
    src[i] += f" // @{t}"


# TMEM allocation (epilogue warp) -> named barrier 2 -> the MMA warp reads the address.
alloc = find(r"tcgen05\.alloc")
tag(find(r"^\s*bar\.sync\s+2, 160", alloc), "ALLOC_BAR")
tag(find(r"^\s*bar\.sync\s+2, 160"), "MMA_BAR")
# First MMA of each k-block group, the epilogue's tcgen05.ld, the barrier after the subtile
# stores (before the accumulator-empty arrive), the dealloc.
mmas = [i for i, ln in enumerate(src) if "tcgen05.mma" in ln]
groups = [mmas[0]] + [b for a, b in zip(mmas, mmas[1:]) if b - a > 20]
for k, i in enumerate(groups):
    tag(i, f"MMA{k}")
ld = find(r"\btcgen05\.ld\.")
tag(ld, "LD")
arrive = find(r"mbarrier\.arrive\.shared\.b64 %rd\d+, \[%r\d+\+112\]")
tag(arrive, "ACC_EMPTY")
tag(
    max(i for i in range(ld, arrive) if re.match(r"^\s*bar\.sync\s+1, 128", src[i])),
    "SUB_BAR",
)
tag(find(r"tcgen05\.dealloc"), "DEALLOC")
# The epilogue's TMA store (issued by every lane of epilogue warp 0 as captured).
tag(find(r"cp\.async\.bulk\.tensor\.3d\.global\.shared::cta"), "STORE")
# Mutation sites.
w = max(i for i in range(ld) if "mbarrier.try_wait" in src[i])
tag(w, "ACCWAIT")
tag(w + 1, "ACCWAIT")  # `@P1 bra.uni DONE`
tag(w + 2, "ACCWAIT")  # `bra.uni LAB_WAIT`
tag(find(r"fence\.proxy\.async", ld), "PROXY")
tag(find(r"mbarrier\.arrive\.expect_tx"), "EXPECT_TX")
open("kernel.ptx", "w").write("\n".join(src))


def edit(marker, find, replace):
    return f'{{ marker = "@{marker}", find = "{find}", replace = "{replace}" }}'


STORE = "{\\n\\t.reg .pred %pst;\\n\\telect.sync _|%pst, -1;\\n\\t@%pst cp.async.bulk.tensor"
fenced = [
    edit("ALLOC_BAR", "bar.sync", f"{BEFORE} bar.sync"),
    edit("MMA_BAR", "bar.sync \\t2, 160;", f"bar.sync 2, 160; {AFTER}"),
    *[
        edit(f"MMA{k}", "tcgen05.mma", f"{AFTER} tcgen05.mma")
        for k in range(len(groups))
    ],
    edit("LD", "tcgen05.ld", f"{AFTER} tcgen05.ld"),
    edit("LD", "];", "]; tcgen05.wait::ld.sync.aligned;"),
    edit("SUB_BAR", "bar.sync", f"{BEFORE} bar.sync"),
    edit("DEALLOC", "tcgen05.dealloc", f"{AFTER} tcgen05.dealloc"),
    edit("STORE", "cp.async.bulk.tensor", STORE),
]
# The STORE replacement opens a scope; close it after the store.
fenced.append(edit("STORE", "%rd41;", "%rd41;\\n\\t}"))


def deletes(*markers):
    return [f'{{ marker = "@{m}", replace = "<delete>" }}' for m in markers]


variants = [
    ("good", "as compiled by CuTeDSL 4.6.1", [], '["race", "violation"]', '["clean"]'),
    (
        "fenced",
        "the PTX with tcgen05.wait::ld, the tcgen05 thread-sync fences and an elected TMA store",
        fenced,
        '["clean"]',
        '["clean"]',
    ),
    (
        "CM1",
        "fenced, minus the accumulator-full wait before the epilogue tcgen05.ld",
        fenced + deletes("ACCWAIT"),
        '["deadlock", "race", "violation"]',
        '["deadlock", "race", "violation"]',
    ),
    (
        "CM2",
        "fenced, minus the epilogue's accumulator-empty arrive",
        fenced + deletes("ACC_EMPTY"),
        '["deadlock"]',
        '["deadlock"]',
    ),
    (
        "CM3",
        "fenced, minus the generic->async proxy fence before the TMA store",
        fenced + deletes("PROXY"),
        '["race"]',
        '["race"]',
    ),
    (
        "CM4",
        "fenced, with expect_tx of half the bytes the two TMA loads deliver",
        fenced + [edit("EXPECT_TX", "%r109;", "16384;")],
        '["nondeterministic", "race", "violation"]',
        '["nondeterministic", "race", "violation"]',
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
print(f"tagged: {len(groups)} MMA groups")
