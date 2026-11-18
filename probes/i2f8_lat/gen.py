# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Variants of probes/i2f8_lat: the latency of an I2F.S8 with no write scoreboard.

ptxas (sm_103a, CUDA 13.0) compiles kernel.ptx; each variant rewrites the slots from the first
`SHF.R.U32.HI` up to the EXIT with whole instructions (`bits` patches) copied from ptxas's own
encodings with other registers / control bits:

    SHF.R.U32.HI R4, RZ, 0x3, R0      stall 1
    SHF.R.U32.HI R6, RZ, 0x4, R0      stall 4 (the SHF results are in their registers)
    I2F.S8 R4, R4                     stall 2, no write scoreboard (OWN: SB2)
    K x I2F.S16 R<8+k>, R6            stall 2 each, write scoreboard SB3 (the followers)
    NOP ...                           stalls padding the distance to D
    FMUL R4, R4, UR7                  NOWAIT: waits on nothing; COVER: SB3; OWN: SB2
                                      (WAW: FMUL R4, R6, UR7, no wait: overwrites the result)
    STG.E desc[UR4][R2.64], R4        waits on SB0, SB2, SB3; then K x STG of the followers
    NOP ...                           up to the EXIT

D is the stall cycles from the I2F.S8 to the FMUL (a lower bound on elapsed cycles); the FMUL
reads its operands at issue. The I2F.S8 overwrites its own source, so a read before its write
sees the integer bits, not the float (scale = 1.0). out[16 t] = the FMUL, out[16 t + 1 + k] =
follower k (I2F.S16 of tid * seed >> 4). Variants compute something else than good:
PTXV_HW_EXPECT=variant.

    python3 gen.py [occ|sat] writes case.toml (occ: probes/i2f8_lat_occ/case.toml; sat:
                             probes/i2f8_lat_sat/case.toml, from that directory's kernel.ptx)
"""

import os
import re
import struct
import subprocess
import sys
import tempfile

CUDA = os.environ.get("PTXV_CUDA_HOME", "/usr/local/cuda-13.0")
HERE = os.path.dirname(os.path.abspath(__file__))
ENTRY = "i2f8l"


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
    c = hi >> 41
    y = (c >> 4) & 1 if yld is None else yld
    c = stall | y << 4 | wbar << 5 | rbar << 8 | wait << 11
    return (hi & ((1 << 41) - 1)) | c << 41


def build(lines, areg=2):
    """`areg`: the output address register pair (R2 as compiled from kernel.ptx, R8 from
    probes/i2f8_lat_sat/kernel.ptx). The window is the slots from the first `SHF.R.U32.HI ..,
    R0` of the probe's shifts up to the first EXIT (in i2f8_lat_sat: the probe pool's path,
    which ptxas places first)."""

    def find(pat):
        return next(x for x in lines if re.fullmatch(pat, x[1]))

    exit_ = find(r"EXIT")
    shf = find(r"SHF\.R\.U32\.HI R\d+, RZ, 0x3, R0(\.reuse)?")
    i2f8 = find(r"I2F\.S8 R\d+, R\d+")
    i2f16 = find(r"I2F\.S16 R\d+, R\d+")
    fmul = find(r"FMUL R\d+, R\d+, UR7")
    stg = find(rf"STG\.E desc\[UR4\]\[R{areg}\.64\], R\d+")
    nop = [x for x in lines if x[1] == "NOP"][-1]
    # (after the last write of R0 before the window: the probe's input)
    r0def = max(x[0] for x in lines if x[0] < shf[0] and re.match(r"(@!?P\d )?[A-Z][A-Z0-9.]* R0,", x[1]))
    shifts = [x for x in lines if r0def < x[0] < exit_[0] and re.fullmatch(r"SHF\.R\.U32\.HI R\d+, RZ, 0x[3-9ab], R0(\.reuse)?", x[1])]
    start = min(shf[0], *(x[0] for x in shifts))
    assert start < exit_[0]
    slots = [a for a, *_ in lines if start <= a < exit_[0]]

    def SHF(rd, sh, stall):
        lo = field(field(shf[2], 16, 8, rd), 32, 32, sh)
        return lo, ctrl(field(shf[3], 0, 8, 0), stall), f"SHF.R.U32.HI R{rd}, RZ, {sh:#x}, R0"

    def I2F(t, rd, rs, stall, wbar):
        lo = field(field(t[2], 16, 8, rd), 32, 8, rs)
        op = "I2F.S8" if t is i2f8 else "I2F.S16"
        return lo, ctrl(t[3], stall, wbar=wbar), f"{op} R{rd}, R{rs}"

    def NOP(stall, wait=0):
        return nop[2], ctrl(nop[3], stall, wait=wait, yld=1), "NOP"

    def FMUL(wait, rs=4):
        lo = field(field(fmul[2], 16, 8, 4), 24, 8, rs)
        return lo, ctrl(fmul[3], 6, wait=wait), f"FMUL R4, R{rs}, UR7"

    def STG(rv, off, wait):
        lo = field(field(stg[2], 32, 8, rv), 40, 24, off)
        t = f"STG.E desc[UR4][R{areg}.64+{off:#x}], R{rv}" if off else f"STG.E desc[UR4][R{areg}.64], R{rv}"
        return lo, ctrl(stg[3], 1, wait=wait), t

    # The followers' registers: R8.. as compiled from kernel.ptx (address R2), R10.. when the
    # address is R8 (i2f8_lat_sat).
    fr = 10 if areg == 8 else 8

    def window(k, d, kind, fs=2, ft="S16", g=2):
        w = [SHF(4, 3, 1), SHF(6, 4, 4)]
        w.append(I2F(i2f8, 4, 4, g if k else 2, 2 if kind == "OWN" else 7))
        cyc = g if k else 2
        for j in range(k):
            w.append(I2F(i2f8 if ft == "S8" else i2f16, fr + j, 6, fs, 3))
            cyc += fs
        left = d - cyc
        assert left >= 0, (k, d)
        while left > 0:
            s = min(left, 11)
            if left - s == 1:
                s -= 1
            w.append(NOP(s))
            left -= s
        # WAW: the FMUL overwrites the I2F.S8's register from R6 (no read of it, no wait): the
        # I2F's write landing after it is what the STG stores.
        w.append(FMUL({"NOWAIT": 0, "COVER": 1 << 3, "OWN": 1 << 2, "WAW": 0}[kind], 6 if kind == "WAW" else 4))
        w.append(STG(4, 0, 0b1101))
        for j in range(k):
            w.append(STG(fr + j, 4 + 4 * j, 0))
        assert len(w) <= len(slots), (k, d, len(w), len(slots))
        while len(w) < len(slots):
            w.append(NOP(1))
        return w

    return slots, window


# GB300: 152 SMs (2 CTAs of 1024 threads each: full occupancy, 64 warps per SM).
SMS = 152

HEADER = r'''
# Generated by gen.py (python3 gen.py{occ_arg} > case.toml); see gen.py for the window.
name = "probe_{name}"
description = """
Probe (L-OPCLASS): the result latency of an I2F.S8 with no write scoreboard. Each variant
rewrites the window after the address arithmetic (gen.py): the I2F.S8 (overwriting its own
source), K = 0, 1, 2, 4, 8 independent I2F.S16 followers (stall 2 each, write scoreboard SB3),
NOP padding, then an FMUL reading the I2F.S8's result D stall cycles after it. NOWAIT: the FMUL
waits on nothing (positive control at short D); COVER: on the followers' SB3; OWN: on the
I2F.S8's own SB2 (reference, same layout). Grid {gx} x {bx} threads. Variants compute something
else than good: PTXV_HW_EXPECT=variant. GB300 counts in results.txt."""
ptx = "kernel.ptx"
entry = "i2f8l"
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

[[params]]
kind = "u32"
value = 1065353216   # scale = 1.0f

[[variants]]
id = "good"
expect = ["clean"]
'''

DISTS = [6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 33, 40]


def vid(k, d, kind, fs=2, ft="S16", g=2):
    x = f"K{k}_D{d}_{kind}"
    if fs != 2:
        x += f"_F{fs}"
    if g != 2:
        x += f"_G{g}"
    if ft != "S16":
        x += f"_{ft}"
    return x


def grid_variants():
    """(k, d, kind, follower stall, follower opcode, I2F.S8 stall = gap to the first follower)"""
    out = []
    # A: dense I2F.S16 followers (stall 2)
    for k in (0, 1, 2, 4, 8):
        lo = 2 + 2 * k
        ds = [d for d in DISTS if d >= lo]
        out.append((k, ds[0], "OWN", 2, "S16", 2))
        for d in ds:
            out.append((k, d, "NOWAIT", 2, "S16", 2))
            if k:
                out.append((k, d, "COVER", 2, "S16", 2))
    # B: one follower, finer distances
    for d in (17, 19, 21, 22, 23, 26, 30):
        out.append((1, d, "NOWAIT", 2, "S16", 2))
    # C: spaced followers (stall 4, 8)
    for fs in (4, 8):
        for k in (1, 2, 4):
            lo = 2 + fs * k
            for d in [x for x in DISTS + [48] if x >= lo]:
                out.append((k, d, "NOWAIT", fs, "S16", 2))
            out.append((k, lo, "COVER", fs, "S16", 2))
    # D: one follower issued g stall cycles after the I2F.S8
    for g in (4, 6, 8, 10):
        for d in (16, 18, 20, 22, 24, 28):
            if d >= g + 2:
                out.append((1, d, "NOWAIT", 2, "S16", g))
                out.append((1, d, "COVER", 2, "S16", g))
    # F: the boundary (spaced followers keep the reader's issue unthrottled)
    for k, fs, ds in ((2, 8, (21, 22, 23, 24, 26, 28)), (3, 8, (26, 28, 30, 32, 36)), (4, 8, (34, 36, 38))):
        for d in ds:
            out.append((k, d, "NOWAIT", fs, "S16", 2))
    for d in (21, 23):
        out.append((1, d, "NOWAIT", 2, "S16", 4))
    for k, fs, ds in ((2, 6, (14, 16, 18, 20, 22, 24, 26)), (3, 6, (20, 22, 24, 26, 28)), (2, 4, (22, 24))):
        for d in ds:
            if d >= 2 + k * fs:
                out.append((k, d, "NOWAIT", fs, "S16", 2))
    # G: WAW (the FMUL writes the I2F.S8's register without reading it), no follower and one
    for k in (0, 1):
        for d in (6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 33):
            if d >= 2 + 2 * k:
                out.append((k, d, "WAW", 2, "S16", 2))
    # E: I2F.S8 followers
    for k in (1, 2, 4):
        for d in [x for x in DISTS if x >= 2 + 2 * k]:
            out.append((k, d, "NOWAIT", 2, "S8", 2))
    return out


# Full occupancy (grid 2 x SMs of 1024 threads), every warp issuing the window.
OCC = (
    [(0, 12, "NOWAIT", 2, "S16", 2), (0, 12, "OWN", 2, "S16", 2), (1, 12, "OWN", 2, "S16", 2)]
    + [(0, d, "NOWAIT", 2, "S16", 2) for d in (16, 18, 20, 24, 28, 33, 40, 48, 64)]
    + [(1, d, "NOWAIT", 2, "S16", 2) for d in (12, 18, 20, 24, 28, 33, 40, 48, 64)]
    + [(1, d, "COVER", 2, "S16", 2) for d in (12, 20, 33)]
    + [(4, d, "NOWAIT", 4, "S16", 2) for d in (18, 24, 33, 40, 48, 64)]
    + [(8, d, "NOWAIT", 2, "S16", 2) for d in (18, 24, 33, 40, 48, 64)]
)


# A saturated conversion pipe (probes/i2f8_lat_sat): full occupancy, the odd warps of every CTA
# running SAT_ITERS rounds of 16 narrow I2F while the even warps run the window after SAT_DELAY
# dependent IMADs.
SAT_DS = (16, 20, 24, 28, 32, 40, 48, 56, 64)
SAT = (
    [(0, 16, "OWN", 2, "S16", 2), (1, 16, "OWN", 2, "S16", 2), (8, 24, "OWN", 2, "S16", 2)]
    + [(0, d, "NOWAIT", 2, "S16", 2) for d in SAT_DS]
    + [(1, d, "NOWAIT", 2, "S16", 2) for d in SAT_DS]
    + [(1, d, "NOWAIT", 2, "S16", 4) for d in SAT_DS]
    + [(2, d, "NOWAIT", 8, "S16", 2) for d in SAT_DS if d >= 18]
    + [(4, d, "NOWAIT", 4, "S16", 2) for d in SAT_DS if d >= 18]
    + [(8, d, "NOWAIT", 2, "S16", 2) for d in SAT_DS if d >= 18]
    + [(1, d, "COVER", 2, "S16", 2) for d in (16, 24, 48)]
    # the boundary (after the first pass: K0 wrong at 32, 0 at 40)
    + [(0, d, "NOWAIT", 2, "S16", 2) for d in (34, 36, 38)]
    + [(1, 30, "NOWAIT", 2, "S16", 4)]
    # the boundary at 1000 runs and the threshold's static classes (MISS 186: CONV_LAT_CYC = 50)
    + [(0, d, "NOWAIT", 2, "S16", 2) for d in (33, 47, 49, 35, 37, 39, 50)]
    # WAW (the FMUL overwrites the I2F.S8's register, no read, no wait) under saturation
    + [(k, d, "WAW", 2, "S16", 2) for k in (0, 1) for d in (12, 16, 20, 24, 28, 32, 40, 48, 50, 56, 64)]
)
SAT_ITERS = 32
SAT_DELAY = 256

SAT_HEADER = r'''
# Generated by gen.py (python3 gen.py sat > ../i2f8_lat_sat/case.toml); see gen.py for the window.
name = "probe_i2f8_lat_sat"
description = """
Probe (L-OPCLASS): the result latency of an I2F.S8 with no write scoreboard under a saturated
conversion pipe. Grid {gx} x {bx} threads (full occupancy: 64 warps per SM). The odd warps of
every CTA (the saturating pool) run {iters} rounds of 16 independent I2F.S16 / I2F.S8; the even
warps (the probe pool) spin {delay} dependent IMADs, then run probes/i2f8_lat's window (gen.py
sat): the I2F.S8 (no write scoreboard), K I2F.S16 followers (SB3), NOP padding, the FMUL reading
the I2F.S8's result D stall cycles after it. NOWAIT: the FMUL waits on nothing; COVER: on the
followers' SB3; OWN: on the I2F.S8's own SB2 (reference). clk (not compared) holds each warp's
SM id and clocks (the overlap check in results.txt). Variants compute something else than good:
PTXV_HW_EXPECT=variant. GB300 counts in results.txt."""
ptx = "kernel.ptx"
entry = "i2f8s"
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
value = 1065353216   # scale = 1.0f

[[params]]
kind = "u32"
value = {iters}   # iters: the saturating pool's rounds

[[params]]
kind = "u32"
value = {delay}   # delay: the probe pool's IMADs before the window

[[params]]
kind = "ptr"
buffer = "clk"

[[variants]]
id = "good"
expect = ["clean"]
'''


def main():
    occ = len(sys.argv) > 1 and sys.argv[1] == "occ"
    sat = len(sys.argv) > 1 and sys.argv[1] == "sat"
    tmp = tempfile.mkdtemp()
    cubin = os.path.join(tmp, "k.cubin")
    kdir = os.path.join(HERE, "..", "i2f8_lat_sat") if sat else HERE
    run(f"{CUDA}/bin/ptxas", "-arch=sm_103a", "-o", cubin, os.path.join(kdir, "kernel.ptx"))
    lines = insts(cubin)
    slots, window = build(lines, areg=8 if sat else 2)
    img0 = open(cubin, "rb").read()
    # .text.<entry> file offset
    shoff = struct.unpack_from("<Q", img0, 0x28)[0]
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", img0, 0x3A)
    sh = lambda i: struct.unpack_from("<IIQQQQIIQQ", img0, shoff + i * shentsize)
    stroff = sh(shstrndx)[4]
    nm = lambda i: img0[stroff + sh(i)[0] : img0.index(0, stroff + sh(i)[0])].decode()
    text = next(sh(i)[4] for i in range(shnum) if nm(i) == f".text.{'i2f8s' if sat else ENTRY}")
    vs = grid_variants()
    if occ:
        vs = OCC
    if occ:
        grid, block, name = [SMS * 2, 1, 1], [1024, 1, 1], "i2f8_lat_occ"
    else:
        grid, block, name = [1, 1, 1], [256, 1, 1], "i2f8_lat"
    toml = [HEADER.format(occ_arg=" occ" if occ else "", name=name, gx=grid[0], bx=block[0], n=16 * grid[0] * block[0]).strip()]
    if sat:
        vs = SAT
        grid, block = [SMS * 2, 1, 1], [1024, 1, 1]
        nth = grid[0] * block[0]
        toml = [SAT_HEADER.format(gx=grid[0], bx=block[0], n=16 * nth, nclk=4 * nth // 32, iters=SAT_ITERS, delay=SAT_DELAY).strip()]
    for v in vs:
        k, d, kind, fs, ft, g = v
        w = window(*v)
        img = bytearray(img0)
        for a, (lo, hi, _) in zip(slots, w):
            struct.pack_into("<QQ", img, text + a, lo, hi)
        p = os.path.join(tmp, "p.cubin")
        open(p, "wb").write(img)
        got = {a: t for a, t, *_ in insts(p)}
        name = vid(*v)
        toml.append(f'\n[[variants]]\nid = "{name}"')
        toml.append(
            f'summary = "{k} I2F.{ft} follower(s) at stall {fs}, the first {g if k else "-"} stall cycles after the I2F.S8; the FMUL {d} stall cycles after the I2F.S8 ({kind})"'
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
