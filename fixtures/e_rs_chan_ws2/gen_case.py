# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Write case.toml for the signal-pad channel case.

(after the reporters' gen_case.py; arch sm_103a)
usage: gen_case.py <case_dir> <name> <ws> <n_ring_calls> <triton_barriers_per_call> <description>

Per-rank launch sequence (stream order on that rank's GPU; ranks are unordered):
  triton RS #1 : cas_barrier(ch 1) [x triton_barriers_per_call: start barrier, all_to_all closing barrier]
  ring RS x N  : for i in range(ws): i > 0: ring_wait(own slot i); i < ws-1: ring_write(right peer, slot i+1)
                 cas_barrier(ch 0)                       (end-of-ring hdl.barrier())
  triton RS #2 : cas_barrier(ch 1) [x triton_barriers_per_call]
"""

import sys

case_dir, name = sys.argv[1], sys.argv[2]
WS, n_ring, n_tb, desc = (
    int(sys.argv[3]),
    int(sys.argv[4]),
    int(sys.argv[5]),
    sys.argv[6],
)
assert 2 <= WS <= 4
pads = [
    f"pad{r % WS}" if r < WS else "pad0" for r in range(4)
]  # unused pointer slots: pad0


def ptr(b):
    return f'{{ kind = "ptr", buffer = "{b}" }}'


def u32(v):
    return f'{{ kind = "u32", value = {v} }}'


def barrier(r, ch):
    return (
        "cas_barrier",
        [WS, 1, 1],
        [u32(r), u32(ch), u32(WS)] + [ptr(p) for p in pads],
    )


def ring(r):
    out = []
    for i in range(WS):
        if i > 0:
            out.append(
                (
                    "ring_wait",
                    [1, 1, 1],
                    [ptr(f"pad{r}"), ptr(f"ctr{r}"), u32(WS), u32(i)],
                )
            )
        if i < WS - 1:
            out.append(
                (
                    "ring_write",
                    [1, 1, 1],
                    [ptr(f"pad{(r + 1) % WS}"), u32(WS), u32(i + 1)],
                )
            )
    out.append(barrier(r, 0))
    return out


def seq(r):
    s = [barrier(r, 1) for _ in range(n_tb)]
    for _ in range(n_ring):
        s += ring(r)
    s += [barrier(r, 1) for _ in range(n_tb)]
    return s


per_rank = [seq(r) for r in range(WS)]
launches = [(r, per_rank[r][k]) for k in range(len(per_rank[0])) for r in range(WS)]

out = [
    f'name = "{name}"',
    f'description = """\n{desc}"""',
    'arch = "sm_103a"',
    'ptx = "kernel.ptx"',
]
r0, (e0, b0, p0) = launches[0]
out += [f'entry = "{e0}"', "grid = [1, 1, 1]", f"block = {b0}"]
out.append("outputs = [" + ", ".join(f'"pad{r}"' for r in range(WS)) + "]\n")
for r in range(WS):
    out.append(
        f'[[buffers]]\nname = "pad{r}"\ndtype = "u32"\nshape = [{3 * WS}]\ninit = "zeros"\ndevice = {r}\n'
    )
    out.append(
        f'[[buffers]]\nname = "ctr{r}"\ndtype = "u32"\nshape = [{WS - 1}]\ninit = "zeros"\ndevice = {r}\n'
    )
for p in p0:
    out.append("[[params]]\n" + p.strip("{} ").replace(", ", "\n") + "\n")
for r, (e, b, ps) in launches[1:]:
    out += [
        "[[launches]]",
        f'entry = "{e}"',
        "grid = [1, 1, 1]",
        f"block = {b}",
        f"device = {r}",
    ]
    out.append("params = [\n    " + ",\n    ".join(ps) + ",\n]\n")

out.append("""[[variants]]
id = "good"
summary = "fix: ring flags in channel 2 (DistGemmPadChannel.RING_FLAGS), disjoint from the triton RS CAS barrier channel 1"
expect = ["clean"]

[[variants]]
id = "prefix"
summary = "pre-fix: ring flags in channel 1 (_FLAG_CHANNEL = 1): monotonic ring flags share slots 1..ws-1 with the CAS barrier signals of channel 1"
marker = "@CHW"
find = "mov.u32         %r1, 2;"
replace = "mov.u32         %r1, 1;"
edits = [{ marker = "@CHR", find = "mov.u32         %r1, 2;", replace = "mov.u32         %r1, 1;" }]
expect = ["deadlock"]
""")
open(f"{case_dir}/case.toml", "w").write("\n".join(out))
print(
    f"wrote {case_dir}/case.toml: ws={WS} ring_calls={n_ring} triton_barriers={n_tb} launches={len(launches)}"
)
