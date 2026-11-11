# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Generates case.toml (3 ranks x 2 calls; too repetitive to write by hand).
R, C = 3, 2
out = [
    '''name = "r1g_allreduce"
description = """
Rung 1g. One-shot all-reduce over symmetric memory on 3 GPUs, two back-to-back calls per rank:
stage into the symmetric buffer, signal/wait barrier, reduce in rank order, closing barrier."""
ptx = "kernel.ptx"
entry = "allreduce"
grid = [1, 1, 1]
block = [1, 1, 1]'''
]
outs = [f"out_{r}_{c}" for c in range(C) for r in range(R)]
out.append("outputs = [" + ", ".join(f'"{o}"' for o in outs) + "]")
out.append(
    "replicas = ["
    + ", ".join(
        "[" + ", ".join(f'"out_{r}_{c}"' for r in range(R)) + "]" for c in range(C)
    )
    + "]"
)


def buf(name, dtype, n, init, dev, seed=0):
    s = f'\n[[buffers]]\nname = "{name}"\ndtype = "{dtype}"\nshape = [{n}]\ninit = "{init}"\ndevice = {dev}'
    return s + (f"\nseed = {seed}" if seed else "")


for r in range(R):
    out.append(buf(f"buf{r}", "f32", 32, "zeros", r))
    out.append(buf(f"sig{r}", "u32", 3, "zeros", r))
for c in range(C):
    for r in range(R):
        out.append(buf(f"in_{r}_{c}", "f32", 32, "normal", r, seed=100 * c + r + 1))
        out.append(buf(f"out_{r}_{c}", "f32", 32, "zeros", r))


def params(r, c):
    p = [("u32", r), ("u32", c + 1), ("ptr", f"in_{r}_{c}"), ("ptr", f"out_{r}_{c}")]
    p += [("ptr", f"buf{q}") for q in range(R)] + [("ptr", f"sig{q}") for q in range(R)]
    return p


def fmt(k, v):
    return (
        f'{{ kind = "u32", value = {v} }}'
        if k == "u32"
        else f'{{ kind = "ptr", buffer = "{v}" }}'
    )


for k, v in params(0, 0):
    out.append(
        f'\n[[params]]\nkind = "{k}"\n'
        + (f"value = {v}" if k == "u32" else f'buffer = "{v}"')
    )
for c in range(C):
    for r in range(R):
        if (r, c) == (0, 0):
            continue
        ps = ",\n    ".join(fmt(k, v) for k, v in params(r, c))
        out.append(
            f'\n[[launches]]\nentry = "allreduce"\ngrid = [1, 1, 1]\nblock = [1, 1, 1]\ndevice = {r}\nparams = [\n    {ps},\n]'
        )
out.append("""
[[variants]]
id = "good"
summary = "correct all-reduce"
expect = ["clean"]

[[variants]]
id = "B21"
summary = "the barrier signals a constant 1 and resets each flag after seeing it (no epoch): a fast peer's next signal can be wiped by the reset"
marker = "@B21"
find = "add.u32         %r21, %r21, %r20;"
replace = "mov.u32         %r21, 1;"
edits = [{ marker = "@RESET", find = "// (B21 reset slot)", replace = "st.relaxed.sys.global.u32 [%rd32], 0;" }]
expect = ["deadlock"]

[[variants]]
id = "B22"
summary = "no closing barrier: a rank restages its buffer for the next call while peers still read it"
marker = "@B22"
replace = "<delete>"
# The race's consequences are explored too: a rank can sum a peer's next-call input.
expect = ["divergent", "nondeterministic", "race"]

[[variants]]
id = "B24"
summary = "each rank sums starting from its own buffer: ranks hold different bits"
marker = "@B24"
find = "mov.u32         %r30, 0;"
replace = "mov.u32         %r30, %r1;"
expect = ["divergent"]
""")
open("case.toml", "w").write("\n".join(out) + "\n")
