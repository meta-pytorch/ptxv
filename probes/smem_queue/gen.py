# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Variants of probes/smem_queue: completion order across shared-memory opcodes.

ptxas (sm_103a, CUDA 13.0) compiles kernel.ptx; each variant rewrites the slots after the
BAR.SYNC up to the EXIT with the compiled instructions in another order and with other
control bits (`bits` patches of whole instructions):

    address arithmetic, the MUFU's input           (as compiled)
    the producers not under test                   write scoreboard SB1, stall 2
    FIRST  (a slow op)                             waits on every scoreboard; stall G; no write
                                                   scoreboard (OWN: SB2)
    SECOND (a fast op)                             stall 2, write scoreboard SB3
    NOP ...                                        padding: FIRST's first reader D stall cycles after FIRST
    FIRST's readers (LOP3, read at issue)          the first waits on SB3 (COVER), nothing (NOWAIT), SB2 (OWN)
    the other readers (the first waits on every scoreboard), LDCU, the four STG.E.128

Slow ops: `LDSM.16.M88.4 R4, [R4+UR4]` (the 8 rows of each matrix in 4 banks: 8-way bank
conflicts) and `LDS R2, [R2]` (one bank per warp: 32-way). Fast ops (conflict-free):
`LDSM.16.M88.4 R8, [R18+0x4000]`, `LDSM.16.MT88.4 R12, [R18+0x4200]`, `LDS R0, [R22+0x4000]`,
`SHFL.BFLY`; `MUFU.EX2` is outside the shared-memory unit (CTL: the control a wait on a fast op
of another unit gives). LAT_<op>_D<d>: the op alone, its reader D stall cycles later, no wait
(the op's latency in this layout). An LDSM that is neither FIRST nor SECOND has no LOP3
readers in a variant (its registers are stored as loaded), so variants compute something else
than good: PTXV_HW_EXPECT=variant (each against its own simulator run).

    python3 gen.py > case.toml                        (block 256: 8 warps)
    python3 gen.py w1 > ../smem_queue_w1/case.toml    (block 32: one warp, ../smem_queue_w1/kernel.ptx)

In the one-warp layout no other warp's conflicted traffic queues in the unit, so the
conflict-free op completes sooner than the conflicted one (the LAT variants measure both): there
COVER discriminates on its own.
"""

import os
import re
import struct
import subprocess
import sys
import tempfile

CUDA = os.environ.get("PTXV_CUDA_HOME", "/usr/local/cuda-13.0")
HERE = os.path.dirname(os.path.abspath(__file__))
ENTRY = "smem_queue"


def run(*a):
    return subprocess.run(a, capture_output=True, text=True, check=True).stdout


def insts(cubin):
    d = run(f"{CUDA}/bin/nvdisasm", "-c", "-hex", cubin).split("\n")
    r = []
    for i, ln in enumerate(d):
        m = re.match(r"\s+/\*([0-9a-f]{4})\*/\s+(.*?)\s*;\s*/\* (0x[0-9a-f]{16}) \*/", ln)
        if m:
            h = re.search(r"/\* (0x[0-9a-f]{16}) \*/", d[i + 1])
            r.append((int(m.group(1), 16), m.group(2), int(m.group(3), 16), int(h.group(1), 16)))
    return r


def ctrl(hi, stall, wbar=7, rbar=7, wait=0, yld=None):
    if yld is None:
        yld = 1 if stall <= 11 else 0  # stalls above 11 encode with the yield bit clear
    c = stall | yld << 4 | wbar << 5 | rbar << 8 | wait << 11
    return (hi & ((1 << 41) - 1)) | c << 41


OPS = {  # name: (the producer as compiled, the number of result registers from the first)
    "LDSMs": (r"LDSM\.16\.M88\.4 R(\d+), \[R\d+\+UR4\]", 4),
    "LDSs": (r"LDS R(\d+), \[R\d+\]", 1),
    "LDSMf": (r"LDSM\.16\.M88\.4 R(\d+), \[R\d+\+0x4000\]", 4),
    "MT88f": (r"LDSM\.16\.MT88\.4 R(\d+), \[R\d+\+0x4200\]", 4),
    "LDSf": (r"LDS R(\d+), \[R\d+\+0x4000\]", 1),
    "SHFL": (r"SHFL\.BFLY PT, R(\d+), R\d+, 0x1, 0x1f", 1),
    "MUFU": (r"MUFU\.EX2 R(\d+), R\d+", 1),
}


def build(lines):
    def find(pat, ls=None):
        x = [l for l in (ls or win) if re.fullmatch(pat, l[1])]
        assert len(x) == 1, (pat, x)
        return x[0]

    bar = find(r"BAR\.SYNC\.DEFER_BLOCKING 0x0", lines)
    ex = find(r"EXIT", lines)
    win = [l for l in lines if bar[0] < l[0] < ex[0]]
    slots = [l[0] for l in win]
    prod = {n: find(p) for n, (p, _) in OPS.items()}
    regs = {}
    for n, (p, k) in OPS.items():
        r0 = int(re.fullmatch(p, prod[n][1]).group(1))
        regs[n] = list(range(r0, r0 + k))
    readers = {
        n: [find(rf"LOP3\.LUT R\d+, R{r}, UR8, RZ, 0x3c, !PT") for r in rs]
        for n, rs in regs.items()
    }
    # The op whose result register LDSf's reader overwrites (as allocated by ptxas): that op's
    # reader goes first.
    ldsf_dst = int(re.match(r"LOP3\.LUT R(\d+),", readers["LDSf"][0][1]).group(1))
    victim = next(n for n, rs in regs.items() if n != "LDSf" and ldsf_dst in rs)
    pro = [find(r"FMUL R\d+, R3, 0\.0625"), find(r"IMAD\.WIDE\.U32 R2, R2, 0x50, R18")]
    viadd = find(r"VIADD R12, R12, UR4")
    stgs = [l for l in win if l[1].startswith("STG.E.128")]
    nop = [l for l in lines if l[1] == "NOP"][-1]
    used = {x[0] for x in pro + [viadd] + stgs + list(prod.values())}
    used |= {r[0] for v in readers.values() for r in v}
    assert used == set(slots), sorted(set(slots) - used)

    def I(l, stall, **k):
        return l[2], ctrl(l[3], stall, **k), l[1]

    def NOP(stall, wait=0):
        return nop[2], ctrl(nop[3], stall, wait=wait), "NOP"

    def window(first, second, g, d, kind):
        test = [first] + ([second] if second else [])
        # LDS fast's reader overwrites the victim's register (the MUFU's at block 256, the
        # SHFL's at block 32): after the victim's reader. An LDSM neither FIRST nor SECOND keeps
        # its raw result (no LOP3).
        late = ["LDSf"] if victim in test and "LDSf" not in test else []
        pre = [n for n in OPS if n not in test and n not in late and not n.startswith(("LDSM", "MT88"))]
        if "LDSf" in pre and victim in pre:  # the victim's reader first
            pre.remove("LDSf")
            pre.append("LDSf")
        post = (([second] if second else []) + [n for n in ("MUFU",) if n in test and n != first]
                + late)
        post = list(dict.fromkeys(post))
        w = [I(pro[0], 5, wait=0x3F), I(pro[1], 2, wait=0x3F)]
        for n in OPS:
            if n not in test:
                w.append(I(prod[n], 2, wbar=1, rbar=5 if n == "LDSMs" else 7))
        rd = [r for n in pre for r in readers[n]]
        for j, r in enumerate(rd):
            w.append(I(r, 2, wait=0x3F if j == 0 else 0))
        # FIRST issues after every scoreboard is released
        # (the slow LDSM's address register is rewritten later: read scoreboard SB5, as compiled)
        rb = 5 if first == "LDSMs" else 7
        w.append(I(prod[first], g if second else 2, wbar=2 if kind == "OWN" else 7, rbar=rb, wait=0x3F))
        cyc = g if second else 2
        if second:
            w.append(I(prod[second], 2, wbar=3, rbar=5 if second == "LDSMs" else 7))
            cyc += 2
        left = d - cyc
        assert left >= 0, (first, second, g, d)
        while left > 0:
            s = min(left, 15)
            if left - s == 1:
                s -= 1
            w.append(NOP(s))
            left -= s
        wt = {"COVER": 1 << 3, "NOWAIT": 0, "OWN": 1 << 2}[kind]
        for j, r in enumerate(readers[first]):
            w.append(I(r, 2, wait=wt if j == 0 else 0))
        rd = [r for n in post for r in readers[n]]
        # (fixed-latency results: stall 2 between them and 8 before the stores read them)
        w.append(I(viadd, 2, wait=0x3F))
        for r in rd:
            w.append(I(r, 2))
        w.append(NOP(8))
        for s in stgs:
            w.append(I(s, 1))
        assert len(w) <= len(slots), (first, second, g, d, len(w), len(slots))
        while len(w) < len(slots):
            w.append(NOP(1))
        return w

    return slots, window


PAIRS = [
    ("LDSMs", "LDSf"),
    ("LDSMs", "LDSMf"),
    ("LDSMs", "MT88f"),
    ("LDSMs", "SHFL"),
    ("LDSs", "LDSMf"),
    ("LDSs", "MT88f"),
    ("LDSMs", "MUFU"),  # CTL
    ("LDSs", "MUFU"),  # CTL
]


def variants(w1=False):
    if w1:
        return variants_w1()
    out = []
    for f, s in PAIRS:
        kind_c = "CTL" if s == "MUFU" else "COVER"
        for g in (2, 4, 8):
            out.append((f"{f}_{s}_G{g}_{kind_c}", f, s, g, g + 2, "COVER"))
            out.append((f"{f}_{s}_G{g}_NOWAIT", f, s, g, g + 2, "NOWAIT"))
        out.append((f"{f}_{s}_G2_OWN", f, s, 2, 4, "OWN"))
    for f in ("LDSMs", "LDSs"):
        for d in (8, 16, 24, 32, 48, 64, 80, 96) + ((112, 128) if f == "LDSs" else ()):
            out.append((f"LAT_{f}_D{d}", f, None, 2, d, "NOWAIT"))
        out.append((f"LAT_{f}_D4_OWN", f, None, 2, 4, "OWN"))
    for f in ("LDSf", "LDSMf", "MT88f", "SHFL", "MUFU"):
        for d in (4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 80, 96):
            out.append((f"LAT_{f}_D{d}", f, None, 2, d, "NOWAIT"))
    return out


def variants_w1():
    """One warp: the pairs at stalls 2 / 4 / 8 (COVER / CTL, NOWAIT, OWN); each op's latency
    (LAT, finer and longer than at block 256); DELAY_<pair>_G<g>_D<d>: NOWAIT with the reader at
    the distance where the fast SECOND op is complete (the slow FIRST op still outstanding
    there makes COVER discriminating)."""
    out = []
    for f, s in PAIRS:
        kind_c = "CTL" if s == "MUFU" else "COVER"
        for g in (2, 4, 8):
            out.append((f"{f}_{s}_G{g}_{kind_c}", f, s, g, g + 2, "COVER"))
            out.append((f"{f}_{s}_G{g}_NOWAIT", f, s, g, g + 2, "NOWAIT"))
        out.append((f"{f}_{s}_G2_OWN", f, s, 2, 4, "OWN"))
        # every fast op is complete 24 stall cycles after issue in this layout (LAT)
        for g in (2, 8):
            out.append((f"DELAY_{f}_{s}_G{g}_D{g + 26}", f, s, g, g + 26, "NOWAIT"))
    for f in ("LDSMs", "LDSs", "LDSf", "LDSMf", "MT88f", "SHFL", "MUFU"):
        # (the window's slots fit the NOP padding up to D = 96 with an LDSM's four readers)
        for d in (4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 48, 56, 64, 72, 80, 96) + (() if f.startswith(("LDSM", "MT88")) else (112,)):
            out.append((f"LAT_{f}_D{d}", f, None, 2, d, "NOWAIT"))
        out.append((f"LAT_{f}_D4_OWN", f, None, 2, 4, "OWN"))
    return out


HEADER = r'''# Generated by gen.py (python3 gen.py{arg} > case.toml); see gen.py for the window.
name = "probe_{name}"
description = """
Probe (L-OPCLASS, E-HW-SB-QUEUE): completion order across shared-memory opcodes on GB300. Each
variant reorders the compiled instructions after the BAR.SYNC (gen.py): a slow, bank-conflicted
FIRST op (LDSMs: LDSM.16.M88.4, 8-way; LDSs: LDS, 32-way) with no write scoreboard, G stall
cycles later a conflict-free SECOND op with SB3 (LDSf: LDS, LDSMf: LDSM.16.M88.4, MT88f:
LDSM.16.MT88.4, SHFL: SHFL.BFLY; MUFU: MUFU.EX2, outside the unit), then FIRST's LOP3 readers.
COVER: the first reader waits on SB3 only (correct iff FIRST completes no later than SECOND);
CTL: the same with the MUFU as SECOND (control); NOWAIT: no wait (positive control); OWN: on
FIRST's own SB2 (reference). LAT_<op>_D<d>: one op, its reader d stall cycles later, no wait.
Variants store the LDSMs not under test without the LOP3: PTXV_HW_EXPECT=variant. GB300 counts in
results.txt."""
ptx = "kernel.ptx"
entry = "smem_queue"
grid = [1, 1, 1]
block = [{block}, 1, 1]
outputs = ["out"]

[[buffers]]
name = "out"
dtype = "u32"
shape = [{words}]
init = "zeros"

[[params]]
kind = "ptr"
buffer = "out"

[[params]]
kind = "u32"
value = 2654435761

[[variants]]
id = "good"
expect = ["clean"]
'''


def main():
    w1 = len(sys.argv) > 1 and sys.argv[1] == "w1"
    kdir = os.path.join(HERE, "..", "smem_queue_w1") if w1 else HERE
    block = 32 if w1 else 256
    tmp = tempfile.mkdtemp()
    cubin = os.path.join(tmp, "k.cubin")
    run(f"{CUDA}/bin/ptxas", "-arch=sm_103a", "-o", cubin, os.path.join(kdir, "kernel.ptx"))
    lines = insts(cubin)
    slots, window = build(lines)
    img0 = open(cubin, "rb").read()
    shoff = struct.unpack_from("<Q", img0, 0x28)[0]
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", img0, 0x3A)
    sh = lambda i: struct.unpack_from("<IIQQQQIIQQ", img0, shoff + i * shentsize)
    stroff = sh(shstrndx)[4]
    nm = lambda i: img0[stroff + sh(i)[0] : img0.index(0, stroff + sh(i)[0])].decode()
    text = next(sh(i)[4] for i in range(shnum) if nm(i) == f".text.{ENTRY}")
    hdr = HEADER.format(arg=" w1" if w1 else "", name="smem_queue_w1" if w1 else "smem_queue", block=block, words=20 * block)
    if w1:
        hdr = hdr.replace("Probe (L-OPCLASS, E-HW-SB-QUEUE): completion order across shared-memory opcodes on GB300. Each",
                          "Probe (L-OPCLASS, E-HW-SB-QUEUE): completion order across shared-memory opcodes on GB300, one\nwarp (probes/smem_queue's window at block 32: the conflict-free op is the faster). Each")
    toml = [hdr.strip()]
    for name, f, s, g, d, kind in variants(w1):
        w = window(f, s, g, d, kind)
        img = bytearray(img0)
        for a, (lo, hi, _) in zip(slots, w):
            struct.pack_into("<QQ", img, text + a, lo, hi)
        p = os.path.join(tmp, "p.cubin")
        open(p, "wb").write(img)
        got = {a: t for a, t, *_ in insts(p)}
        toml.append(f'\n[[variants]]\nid = "{name}"')
        toml.append(
            f'summary = "FIRST {f}, SECOND {s or "-"} {g} stall cycles later, FIRST\'s reader {d} stall cycles after it ({kind})"'
        )
        toml.append("patch = [")
        for a, (lo, hi, want) in zip(slots, w):
            assert lo < 1 << 63 and hi < 1 << 63
            if got[a] != want:
                sys.exit(f"{name} @{a:#06x}: nvdisasm `{got[a]}`, intended `{want}`")
            toml.append(
                f'    {{ at = "@{a:#06x}", bits = [[0, 64, {lo:#x}], [64, 64, {hi:#x}]], expect = "{want}" }},'
            )
        toml.append("]")
        toml.append('expect = ["clean"]')
    sys.stdout.write("\n".join(toml) + "\n")


if __name__ == "__main__":
    main()
