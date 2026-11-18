# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Variants of probes/f64_lat (and f64_lat_occ): the result latency of a DADD with no write scoreboard (MISS 151:
F64_LAT trusts 128 stall cycles from ptxas's schedules).

ptxas (sm_103a, CUDA 13.0) compiles kernel.ptx; each variant rewrites the slots from the first
DADD up to the EXIT with whole instructions copied from ptxas's own encodings:

    LOP3.LUT R5, RZ, 0x0, RZ, 0xfc, !PT   R5 = 0 (the stale value a too-early read sees)
    DADD R4, R2, R2                        stall 2, no write scoreboard (OWN: SB2)
    K (<= 7) x DADD R<6+2k>, R2, <imm>     stall F each, write scoreboard SB3 (the followers)
    NOP ...                                padding: the reader D stall cycles after the DADD
    LOP3.LUT R5, R5, UR6, RZ, 0x3c, !PT    the reader (hi word; _LO: R4, the lo word; ALU, reads at
                                           issue; stall 6). NOWAIT:
                                           no wait; COVER: SB3; OWN: SB2
    STG.E desc[UR4][R20.64], R5            waits on every scoreboard; then K x STG.E.64
    NOP ...                                up to the EXIT

D is the stall cycles from the DADD to the reader (a lower bound on elapsed cycles). Variants
compute something else than good: PTXV_HW_EXPECT=variant.

    python3 gen.py > case.toml; python3 gen.py occ > ../f64_lat_occ/case.toml (the same kernel.ptx)
    python3 gen.py sat > ../f64_lat_sat/case.toml (from that directory's kernel.ptx)

sat (probes/f64_lat_sat): the probe pool's window has the same shape with the registers ptxas
gives it there (the DADD R6, the reader on R7, the stores at R4), and after the follower stores
`CS2R.32 R28, SR_CLOCKLO` (the clock after the window; the clk store past the window reads R28).
_Q variants patch the pool selector's mask from 1 (the odd warps saturate) to 4 (warps 4..7 of
every 8).
"""

import os
import re
import struct
import subprocess
import sys
import tempfile

CUDA = os.environ.get("PTXV_CUDA_HOME", "/usr/local/cuda-13.0")
HERE = os.path.dirname(os.path.abspath(__file__))
ENTRY = "f64l"


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


def field(w, start, width, v):
    m = ((1 << width) - 1) << start
    return (w & ~m) | ((v << start) & m)


def ctrl(hi, stall, wbar=7, rbar=7, wait=0, yld=None):
    if yld is None:
        yld = 1 if stall <= 11 else 0
    c = stall | yld << 4 | wbar << 5 | rbar << 8 | wait << 11
    return (hi & ((1 << 41) - 1)) | c << 41


# The registers ptxas gives the window: (DADD destination lo word, its source, the hi-word OR's
# destination, the first follower's destination, the reader's uniform register, the store
# descriptor, the output address pair).
REGS = (4, 2, 3, 6, "UR6", "UR4", "R20")
REGS_SAT = (6, 12, 13, 14, None, "UR4", "R4")


def build(lines, sat=False):
    def find(pat):
        return next(x for x in lines if re.fullmatch(pat, x[1]))

    lo_r, src, orr, f0, ur, desc, addr = REGS_SAT if sat else REGS
    hi_r = lo_r + 1
    dadd = find(rf"DADD R{lo_r}, R{src}, R{src}")
    lop_or = find(rf"LOP3\.LUT R{orr}, R0, 0x3ff00000, RZ, 0xfc, !PT")
    if sat:  # the seed's uniform register: as ptxas allocates it
        ur = re.fullmatch(r".*, (UR\d+), .*", find(rf"LOP3\.LUT R{hi_r}, R{hi_r}, UR\d+, RZ, 0x3c, !PT")[1]).group(1)
    lop_rd = find(rf"LOP3\.LUT R{hi_r}, R{hi_r}, {ur}, RZ, 0x3c, !PT")
    stg = find(rf"STG\.E desc\[{desc}\]\[{addr}\.64\], R{hi_r}")
    fol = [find(rf"DADD R{f0 + 2 * k}, R{src}, {k + 1}") for k in range(7)]
    fst = [find(rf"STG\.E\.64 desc\[{desc}\]\[{addr}\.64\+{8 * (k + 1):#x}\], R{f0 + 2 * k}") for k in range(7)]
    nop = [x for x in lines if x[1] == "NOP"][-1]
    s2r = find(r"S2R R\d+, SR_TID\.X")
    exit_ = find(r"EXIT")
    if sat:
        # the window ends at the first EXIT (`@P1 EXIT`: lanes 1..31 leave, lane 0 stores clk)
        exit_ = next(x for x in lines if x[1].endswith("EXIT"))
        cs2r = find(r"CS2R\.32 R5, SR_CLOCKLO")
        clkst = find(rf"STG\.E desc\[{desc}\]\[R2\.64\+0x10\], R5")
        sel = find(r"LOP3\.LUT P0, RZ, R(\d+), 0x1, RZ, 0xc0, !PT")
    slots = [a for a, *_ in lines if dadd[0] <= a < exit_[0]]
    nwin = len(slots)
    if sat:
        slots += [clkst[0], sel[0]]

    def I(t, stall, **k):
        return t[2], ctrl(t[3], stall, **k), t[1]

    def ZERO():
        lo = field(field(field(lop_or[2], 16, 8, hi_r), 24, 8, 255), 32, 32, 0)
        return lo, ctrl(lop_or[3], 6), f"LOP3.LUT R{hi_r}, RZ, 0x0, RZ, 0xfc, !PT"

    def NOP(stall):
        return nop[2], ctrl(nop[3], stall), "NOP"

    def READ(wait, lo):
        if not lo:
            return I(lop_rd, 6, wait=wait)
        enc = field(field(lop_rd[2], 16, 8, lo_r), 24, 8, lo_r)
        return enc, ctrl(lop_rd[3], 6, wait=wait), f"LOP3.LUT R{lo_r}, R{lo_r}, {ur}, RZ, 0x3c, !PT"

    def STORE(lo):
        if not lo:
            return I(stg, 1, wait=0x3F)
        return field(stg[2], 32, 8, lo_r), ctrl(stg[3], 1, wait=0x3F), f"STG.E desc[{desc}][{addr}.64], R{lo_r}"

    def src_window(kind):
        # MISS 181, a 64-bit source: `S2R R5` (no write scoreboard; OWN: SB2), then a DADD
        # reading R4:R5 (no wait; OWN: SB2), its result stored.
        z4 = field(field(field(lop_or[2], 16, 8, 4), 24, 8, 255), 32, 32, 0)
        w = [(z4, ctrl(lop_or[3], 6), "LOP3.LUT R4, RZ, 0x0, RZ, 0xfc, !PT")]
        w.append((field(s2r[2], 16, 8, 5), ctrl(s2r[3], 2, wbar=2 if kind == "OWN" else 7), "S2R R5, SR_TID.X"))
        da = field(field(dadd[2], 16, 8, 8), 24, 8, 4)  # Rb: bits 0..8 of the high word
        w.append((da, ctrl(field(dadd[3], 0, 8, 4), 2, wbar=3, wait=(1 << 2) if kind == "OWN" else 0), "DADD R8, R4, R4"))
        w.append(I(fst[1], 1, wait=0x3F))
        while len(w) < len(slots):
            w.append(NOP(1))
        return w

    def window(k, d, kind, fs=2, lo=False, q=False):
        if kind.startswith("SRC_"):
            return src_window(kind[4:])
        w = [ZERO(), I(dadd, fs if k else 2, wbar=2 if kind == "OWN" else 7)]
        cyc = fs if k else 2
        for j in range(k):
            w.append(I(fol[j], fs, wbar=3))
            cyc += fs
        left = d - cyc
        assert left >= 0, (k, d)
        while left > 0:
            s = min(left, 15)
            if left - s == 1:
                s -= 1
            w.append(NOP(s))
            left -= s
        w.append(READ({"NOWAIT": 0, "COVER": 1 << 3, "OWN": 1 << 2}[kind], lo))
        w.append(STORE(lo))
        for j in range(k):
            w.append(I(fst[j], 1))
        if sat:
            w.append((field(cs2r[2], 16, 8, 28), cs2r[3], "CS2R.32 R28, SR_CLOCKLO"))
        assert len(w) <= nwin, (k, d, len(w), nwin)
        while len(w) < nwin:
            w.append(NOP(1))
        if sat:
            w.append((field(clkst[2], 32, 8, 28), clkst[3], f"STG.E desc[{desc}][R2.64+0x10], R28"))
            m = 4 if q else 1
            pr = re.fullmatch(r"LOP3\.LUT P0, RZ, (R\d+), 0x1, RZ, 0xc0, !PT", sel[1]).group(1)
            w.append((field(sel[2], 32, 32, m), sel[3], f"LOP3.LUT P0, RZ, {pr}, {m:#x}, RZ, 0xc0, !PT"))
        return w

    return slots, window


SMS = 152
HEADER = r'''# Generated by gen.py (python3 gen.py{arg} > case.toml); see gen.py for the window.
name = "probe_{name}"
description = """
Probe (L-OPCLASS, MISS 151) on GB300: the result latency of a DADD with no write scoreboard
(F64_LAT: 128 stall cycles, from ptxas's schedules). Each variant rewrites the window from the
DADD (gen.py): R5 zeroed, the DADD (no write scoreboard), K independent DADD followers (stall F,
SB3), NOP padding, a LOP3 reading the DADD's hi word D stall cycles after it. NOWAIT: no wait;
COVER: on the followers' SB3; OWN: on the DADD's own SB2 (reference). Grid {gx} x {bx}.
Variants compute something else than good: PTXV_HW_EXPECT=variant. GB300 counts in results.txt."""
ptx = "kernel.ptx"
entry = "f64l"
grid = [{gx}, 1, 1]
block = [{bx}, 1, 1]
outputs = ["out"]

[[buffers]]
name = "out"
dtype = "u32"
shape = [{n}]
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


def vid(k, d, kind, fs, lo=False, q=False):
    if kind.startswith("SRC_"):
        return kind
    return f"K{k}_D{d}_{kind}" + (f"_F{fs}" if fs != 2 else "") + ("_LO" if lo else "") + ("_Q" if q else "")


# A saturated f64 pipe (probes/f64_lat_sat): full occupancy, the saturating pool (the odd warps;
# _Q: warps 4..7 of every 8) running SAT_ITERS rounds of 4 DADD + 4 DFMA while the probe pool
# runs the window after sleeping SAT_DELAY ns.
SAT_DS = (16, 128, 256, 384, 512, 640, 768, 896, 1024, 1152, 1280, 1408, 1536, 288, 320, 352, 416, 448, 480, 544, 576, 608, 704)
SAT = (
    [(0, 8, "OWN", 2, False, q) for q in (False, True)]
    + [(1, 8, "OWN", 2, False, q) for q in (False, True)]
    + [(k, d, "NOWAIT", 2, False, q) for q in (False, True) for k in (0, 1) for d in SAT_DS]
    + [(1, d, "COVER", 2, False, q) for q in (False, True) for d in (8, 256, 1024)]
)
SAT_ITERS = 128
SAT_DELAY = 60000

SAT_HEADER = r'''# Generated by gen.py (python3 gen.py sat > ../f64_lat_sat/case.toml); see gen.py for the window.
name = "probe_f64_lat_sat"
description = """
Probe (L-OPCLASS, MISS 151) on GB300: the result latency of a DADD with no write scoreboard under
a saturated f64 pipe. Grid {gx} x {bx} threads (full occupancy: 64 warps per SM). The saturating
pool (the odd warps of every CTA; _Q variants: warps 4..7 of every 8) runs {iters} rounds of 4
DADD + 4 DFMA (8 independent chains); the probe pool sleeps {delay} ns (nanosleep), then runs
probes/f64_lat's window (gen.py sat): R7 zeroed, the DADD R6 (no write scoreboard), K DADD
followers (SB3), NOP padding, a LOP3 reading the DADD's hi word D stall cycles after it, the
stores, the clock. NOWAIT: no wait; COVER: on the followers' SB3; OWN: on the DADD's own SB2
(reference). clk (not compared) holds each warp's SM id, warp slot and clocks (the overlap check
in results.txt, satclk.py). Variants compute something else than good: PTXV_HW_EXPECT=variant.
GB300 counts in results.txt."""
ptx = "kernel.ptx"
entry = "f64s"
grid = [{gx}, 1, 1]
block = [{bx}, 1, 1]
outputs = ["out"]

[[buffers]]
name = "out"
dtype = "u32"
shape = [{n}]
init = "zeros"

[[buffers]]
name = "clk"
dtype = "u32"
shape = [{nclk}]
init = "zeros"

[[params]]
kind = "ptr"
buffer = "out"

[[params]]
kind = "u32"
value = 2654435761

[[params]]
kind = "u32"
value = {iters}   # iters: the saturating pool's rounds

[[params]]
kind = "u32"
value = {delay}   # delay: the probe pool's nanosleep (ns) before the window

[[params]]
kind = "ptr"
buffer = "clk"

[[variants]]
id = "good"
expect = ["clean"]
'''


def variants(occ):
    if occ:
        ds = (16, 64, 128, 192, 256, 320, 384, 448, 512, 640, 352, 416, 480)
        return ([(0, 8, "OWN", 2), (1, 8, "OWN", 2)]
                + [(0, d, "NOWAIT", 2) for d in ds]
                + [(1, d, "NOWAIT", 2) for d in ds])
    out = []
    for k in (0, 1, 2, 4, 7):
        lo = 2 + 2 * k
        ds = [d for d in (4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32, 40, 48, 64, 96, 128) if d >= lo]
        out.append((k, ds[0], "OWN", 2))
        for d in ds:
            out.append((k, d, "NOWAIT", 2))
        if k:
            for d in ds[:6]:
                out.append((k, d, "COVER", 2))
    for k in (1, 2, 4):
        for d in (8 * k + 10, 8 * k + 14, 8 * k + 18, 8 * k + 24, 64, 128):
            out.append((k, d, "NOWAIT", 8))
    # (second pass) past F64_LAT, and the lo word read (LO)
    for d in (160, 192, 256, 384, 512):
        out.append((0, d, "NOWAIT", 2))
        out.append((1, d, "NOWAIT", 2))
    out.append((0, 4, "OWN", 2, True))
    out += [(0, 0, "SRC_NOWAIT", 2), (0, 0, "SRC_OWN", 2)]
    for d in (8, 32, 64, 96, 128, 160, 192, 256, 384, 512):
        out.append((0, d, "NOWAIT", 2, True))
    return out


def main():
    occ = len(sys.argv) > 1 and sys.argv[1] == "occ"
    sat = len(sys.argv) > 1 and sys.argv[1] == "sat"
    tmp = tempfile.mkdtemp()
    cubin = os.path.join(tmp, "k.cubin")
    kdir = os.path.join(HERE, "..", "f64_lat_sat") if sat else HERE
    run(f"{CUDA}/bin/ptxas", "-arch=sm_103a", "-o", cubin, os.path.join(kdir, "kernel.ptx"))
    lines = insts(cubin)
    slots, window = build(lines, sat)
    img0 = open(cubin, "rb").read()
    shoff = struct.unpack_from("<Q", img0, 0x28)[0]
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", img0, 0x3A)
    sh = lambda i: struct.unpack_from("<IIQQQQIIQQ", img0, shoff + i * shentsize)
    stroff = sh(shstrndx)[4]
    nm = lambda i: img0[stroff + sh(i)[0] : img0.index(0, stroff + sh(i)[0])].decode()
    text = next(sh(i)[4] for i in range(shnum) if nm(i) == f".text.{'f64s' if sat else ENTRY}")
    gx, bx = (2 * SMS, 1024) if occ or sat else (1, 256)
    toml = [HEADER.format(arg=" occ" if occ else "", name="f64_lat_occ" if occ else "f64_lat", gx=gx, bx=bx, n=20 * gx * bx).strip()]
    if sat:
        nth = gx * bx
        toml = [SAT_HEADER.format(gx=gx, bx=bx, n=64 * nth, nclk=8 * nth // 32, iters=SAT_ITERS, delay=SAT_DELAY).strip()]
    for v in SAT if sat else variants(occ):
        k, d, kind, fs = v[:4]
        w = window(*v)
        img = bytearray(img0)
        for a, (lo, hi, _) in zip(slots, w):
            struct.pack_into("<QQ", img, text + a, lo, hi)
        p = os.path.join(tmp, "p.cubin")
        open(p, "wb").write(img)
        got = {a: t for a, t, *_ in insts(p)}
        name = vid(*v)
        toml.append(f'\n[[variants]]\nid = "{name}"')
        toml.append(f'summary = "{k} DADD follower(s) at stall {fs}; the reader {d} stall cycles after the DADD ({kind})"')
        toml.append("patch = [")
        for a, (lo, hi, want) in zip(slots, w):
            assert lo < 1 << 63 and hi < 1 << 63, (name, hex(a))
            if got[a] != want:
                sys.exit(f"{name} @{a:#06x}: nvdisasm `{got[a]}`, intended `{want}`")
            toml.append(f'    {{ at = "@{a:#06x}", bits = [[0, 64, {lo:#x}], [64, 64, {hi:#x}]], expect = "{want}" }},')
        toml.append("]")
        toml.append('expect = ["clean"]')
    sys.stdout.write("\n".join(toml) + "\n")


if __name__ == "__main__":
    main()
