# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Build the long-gap windows of probes/sb_mio_gap: ptxas compiles kernel.ptx (160 placeholder
# `STS [Ra+0x3000], Rv` after the saturating stores), then each variant overwrites the first
# placeholders by copying ptxas's own encodings with other registers / offsets / control bits
# (Rt, Rg, Rd and the quads Rq, Rs: registers dead over the window; Rw: the thread's W slot
# base, Raux: its aux word's address; c[0x0][0x390] is the `bogus` parameter):
#   MOV Rt, Ra                      (Ra: the slot address, Rv: the marker)
#   [MOV Rg / Rq.., Rv]             (A stores a copy of the marker)
#   A: STS [Rt], Rv | STS [Rt], Rg | STS.64 / STS.128 [Rw+0xc00], Rq | STG.E [Raux.64], Rg,
#      its read scoreboard never waited on
#   LOP3.LUT Rd, Rd, 0x1, RZ, 0x3c  x n, stall 15 each: the delay chain (dependent ALU ops)
#   the later op G stall cycles after A (G = A's stall + the chain's), the window under test:
#     a memory op whose read / write scoreboard the next op waits on before rewriting A's
#     register (R10 / R11 covers), or a load / shuffle writing A's register with no wait (R12);
#     the controls put an ALU write (or the rewrite with no wait) in the same slot
#   STS [Ra+0x2c00], Rv             tail: the same opcode's read order completes A's reads
#                                   (an STG / STSM A: first STG.E [Rp.64+0x3000], Rv, to
#                                   in[3072 + t], / STSM.16.M88.4 [Rw+0x1c00], Rq, then the STS)
#   (next placeholder, waiting for every scoreboard the window set)
# and checks nvdisasm prints each patched instruction as intended.
#   python3 patch.py kernel.cubin OUTDIR [fixture.toml]
# Prints one line per variant: name, expected-output tag (a / u8 / u16: slot 0 = v / its low
# byte / its low half; w64 / w128: W slot = {v, v, 0, 0} / {v, v, v, v}; stsm: W slot of lane L
# = the markers of lanes 4 (L % 8) .. + 3; aux: aux word = v), slots, instructions.
# With a third argument, also writes the windows as `patch = [...]` variants for
# probes/sb_mio_gap/fixture (the static scoreboard check, tests/fixtures.rs).
# A cubin of kernel_cal.ptx (mkcal.py: the same kernel with a %clock read before and after the
# placeholder run, their difference in the fourth output word) gets the cal_<G> variants only:
# the two CS2R moved to right after A and to the later op's slot.
import os
import re
import struct
import subprocess
import sys

NVDISASM = os.path.join(
    os.environ.get("PTXV_CUDA_HOME", "/usr/local/cuda-13.0"), "bin", "nvdisasm"
)
src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
b0 = open(src, "rb").read()


def dis(f):
    return subprocess.run(
        [NVDISASM, "-c", "-hex", f], capture_output=True, text=True, check=True
    ).stdout


(shoff,) = struct.unpack_from("<Q", b0, 0x28)
shentsize, shnum, shstrndx = struct.unpack_from("<HHH", b0, 0x3A)


def sh(i):
    return struct.unpack_from("<IIQQQQIIQQ", b0, shoff + i * shentsize)


stroff = sh(shstrndx)[4]


def name(i):
    o = stroff + sh(i)[0]
    return b0[o : b0.index(0, o)].decode()


text = next(sh(i)[4] for i in range(shnum) if name(i) == ".text.sb_mio_gap")


def insts(f):
    r = []
    for ln in dis(f).split("\n"):
        m = re.match(r"\s+/\*([0-9a-f]{4})\*/\s+(.*?)\s*;", ln)
        if m:
            r.append((int(m.group(1), 16), m.group(2).strip()))
    return r


lines = insts(src)
# The placeholders `STS [Ra+0x3000], Rv`: the window starts at the first one followed by
# NSLOT - 1 more with nothing in between (ptxas may hoist a load into the run's head).
PH = re.compile(r"STS \[R(\d+)\+0x3[0-9a-f]{3}\], R(\d+)")
NSLOT = 150
first = next(
    i
    for i in range(len(lines))
    if all(PH.fullmatch(t) for _, t in lines[i : i + NSLOT])
)
ra, rv = map(int, PH.fullmatch(lines[first][1]).groups())
slots = [a for a, _ in lines[first : first + NSLOT]]
assert all(
    t.startswith(f"STS [R{ra}+0x") and t.endswith(f"], R{rv}")
    for _, t in lines[first : first + NSLOT]
), "placeholder run interrupted"
# The kernel's later `LDG.E Rx, desc[URu][Rp.64+0x4]`: its address pair is live over the window.
ur, rp = map(
    int,
    next(
        re.fullmatch(r"LDG\.E R\d+, desc\[UR(\d+)\]\[R(\d+)\.64\+0x4\]", t).groups()
        for a, t in lines
        if a > slots[-1]
        and re.fullmatch(r"LDG\.E R\d+, desc\[UR\d+\]\[R\d+\.64\+0x4\]", t)
    ),
)
# kernel_cal.ptx: its two clock reads (`[@P] CS2R.32 Rc, SR_CLOCKLO`).
clk = [(a, t) for a, t in lines if re.fullmatch(r"(@!?P\d )?(S2UR UR|CS2R(\.32)? R)\d+, SR_CLOCKLO", t)]
CAL = bool(clk)


# The W slot's base (`LDS.128 Rx, [Rw+0xc00]` after the window) and the aux word's address
# (`STG.E desc[URu][Raux.64], RZ` before it): both live over the window.
rw = next(
    int(re.fullmatch(r"LDS\.128 R\d+, \[R(\d+)\+0xc00\]", t).group(1))
    for a, t in lines
    if a > slots[-1] and re.fullmatch(r"LDS\.128 R\d+, \[R\d+\+0xc00\]", t)
)
uaux, raux = map(
    int,
    next(
        re.fullmatch(r"STG\.E desc\[UR(\d+)\]\[R(\d+)\.64\], RZ", t).groups()
        for a, t in lines
        if a < slots[0] and re.fullmatch(r"STG\.E desc\[UR\d+\]\[R\d+\.64\], RZ", t)
    ),
)
def regs(t):
    """(registers written, registers read) of an instruction (straight-line code only)."""
    op, _, rest = t.partition(" ")
    if op.startswith("@"):
        op, _, rest = rest.partition(" ")
    w = 4 if ".128" in op else 2 if ".64" in op else 1
    ops = [o.strip() for o in rest.split(",")]
    dst, rd = set(), set()
    for k, o in enumerate(ops):
        for r, pair in re.findall(r"\bR(\d+)(\.64)?\b", o):
            n = range(int(r), int(r) + (2 if pair else 1))
            if (
                k == 0
                and not op.startswith(("ST", "BAR", "BRA", "EXIT"))
                and "[" not in o
            ):
                dst |= set(range(int(r), int(r) + w))
            else:
                rd |= set(n)
    return dst, rd


cuobjdump = os.path.join(os.path.dirname(NVDISASM), "cuobjdump")
res = subprocess.run(
    [cuobjdump, "-res-usage", src], capture_output=True, text=True
).stdout
# GB300 traps (illegal instruction) on the top two of the reported registers.
nreg = int(re.search(r"REG:(\d+)", res).group(1)) - 2
# Registers dead over the window: from the window on, never accessed or written before read.
# Not read just before it either (a memory instruction may read its operands late), and not
# the clock registers (kernel_cal.ptx).
recent = set().union(*(regs(t)[1] for _, t in lines[first - 16 : first]))
recent |= set().union(set(), *(regs(t)[0] for _, t in clk))
free = []
for r in range(2, nreg):
    if r in recent:
        continue
    for _, t in lines[first:]:
        d, u = regs(t)
        if r in u:
            break
        if r in d:
            free.append(r)
            break
    else:
        free.append(r)
# Two 4-aligned quads (the wide stores' data, the wide loads' destinations), then singles.
quads = [q for q in range(0, nreg, 4) if all(q + i in free for i in range(4))]
assert len(quads) >= 2, (free, quads)
rq, rs = quads[:2]
single = [r for r in free if not (rq <= r < rq + 4 or rs <= r < rs + 4)]
assert len(single) >= 3, (free, quads)
rt, rg, rd = single[:3]


def ctrl(hi, stall, wb=7, rb=7, wait=0):
    c = stall | 0 << 4 | wb << 5 | rb << 8 | wait << 11
    return (hi & ((1 << 41) - 1)) | c << 41


# Encodings (low word, high word without control bits) as ptxas emits them, from
# probes/sb_mio_order/patch.py: Rd at bit 16, Ra at 24, Rb / UR at 32, offset at 40.
MOV = (rt << 16 | ra << 32 | 0x7202, 0xF00, f"MOV R{rt}, R{ra}")
STS_A = (rv << 32 | rt << 24 | 0x7388, 0x800, f"STS [R{rt}], R{rv}")
LDS = (0x800 << 40 | ra << 24 | rg << 16 | 0x7984, 0x800, f"LDS R{rg}, [R{ra}+0x800]")
LDS_T = (0x800 << 40 | ra << 24 | rt << 16 | 0x7984, 0x800, f"LDS R{rt}, [R{ra}+0x800]")
LDG = (
    ur << 32 | rp << 24 | rg << 16 | 0x7981,
    0x0C1E1900,
    f"LDG.E R{rg}, desc[UR{ur}][R{rp}.64]",
)
LDG_T = (
    ur << 32 | rp << 24 | rt << 16 | 0x7981,
    0x0C1E1900,
    f"LDG.E R{rt}, desc[UR{ur}][R{rp}.64]",
)
MOVZ = (rt << 16 | 0xFF << 32 | 0x7202, 0xF00, f"MOV R{rt}, RZ")
LDC = (
    (0x390 // 4) << 40 | 0xFF << 24 | rt << 16 | 0x7B82,
    0x800,
    f"LDC R{rt}, c[0x0][0x390]",
)


def mov(d, s):
    """`MOV Rd, Rs` (s = 0xff: RZ)."""
    return (d << 16 | s << 32 | 0x7202, 0xF00, f"MOV R{d}, {'RZ' if s == 0xFF else f'R{s}'}")


def shfl(r):
    return (0x0C201F0000007F89 | r << 24 | r << 16, 0xE0000, f"SHFL.BFLY PT, R{r}, R{r}, 0x1, 0x1f")


MOVG = mov(rg, 0xFF)
MOVGV = mov(rg, rv)
SHFL_G = shfl(rg)
STS_AG = (rg << 32 | rt << 24 | 0x7388, 0x800, f"STS [R{rt}], R{rg}")
# The Z row (zeros, never written) as the wide loads' source; the W slot as the wide stores'
# target. Size field (high word): 32-bit 0x800, .64 0xa00, .128 0xc00. `LDSM.16.M88(.4)` =
# 0x783b / 0x0 (.4: 0x200); `STG.E desc[UR4][R2.64], R5` = 0x0000000502007986 / 0x0c101904.
LDSM_G = (0x1C00 << 40 | rw << 24 | rg << 16 | 0x783B, 0x0, f"LDSM.16.M88 R{rg}, [R{rw}+0x1c00]")


def lds_z(d, size):
    hi, sfx = {64: (0xA00, ".64"), 128: (0xC00, ".128")}[size]
    return (0x1C00 << 40 | rw << 24 | d << 16 | 0x7984, hi, f"LDS{sfx} R{d}, [R{rw}+0x1c00]")


def sts_w(q, size):
    hi, sfx = {64: (0xA00, ".64"), 128: (0xC00, ".128")}[size]
    return (0xC00 << 40 | q << 32 | rw << 24 | 0x7388, hi, f"STS{sfx} [R{rw}+0xc00], R{q}")


LDSM4_S = (0x1C00 << 40 | rw << 24 | rs << 16 | 0x783B, 0x200, f"LDSM.16.M88.4 R{rs}, [R{rw}+0x1c00]")
# `SHFL.IDX PT, Rd, Ra, 0x0, 0x1f` (every lane reads lane 0's), the BFLY form's mode field 0.
def shfl_idx(r):
    return (0x00001F0000007F89 | r << 24 | r << 16, 0xE0000, f"SHFL.IDX PT, R{r}, R{r}, 0x0, 0x1f")


# `STS.U8 / .U16 [Rt], Rv` (size field 0x000 / 0x400); `STSM.16.M88.4 [Rw+0xc00], Rq` = 0x7844 / 0x200
# (lane L's row: its own W slot; word c of row r = lane 4r + c's register of matrix L / 8).
STS_U8 = (rv << 32 | rt << 24 | 0x7388, 0x000, f"STS.U8 [R{rt}], R{rv}")
STS_U16 = (rv << 32 | rt << 24 | 0x7388, 0x400, f"STS.U16 [R{rt}], R{rv}")
STS_U8G = (rg << 32 | rt << 24 | 0x7388, 0x000, f"STS.U8 [R{rt}], R{rg}")
STS_U16G = (rg << 32 | rt << 24 | 0x7388, 0x400, f"STS.U16 [R{rt}], R{rg}")
STSM_Q = (0xC00 << 40 | rq << 32 | rw << 24 | 0x7844, 0x200, f"STSM.16.M88.4 [R{rw}+0xc00], R{rq}")
STG_AQ = (rq << 32 | raux << 24 | 0x7986, 0x0C101900 | uaux, f"STG.E desc[UR{uaux}][R{raux}.64], R{rq}")
LDG_S = (ur << 32 | rp << 24 | rs << 16 | 0x7981, 0x0C1E1900, f"LDG.E R{rs}, desc[UR{ur}][R{rp}.64]")


def sts_wr(a, q, size):
    """`STS.64 / .128 [Ra+0xc00], Rq` (A's address in Ra, a copy of Rw)."""
    hi, sfx = {64: (0xA00, ".64"), 128: (0xC00, ".128")}[size]
    return (0xC00 << 40 | q << 32 | a << 24 | 0x7388, hi, f"STS{sfx} [R{a}+0xc00], R{q}")


# `STG.E desc[URu][Rs.64], Rg` (A's address in the pair Rs, a copy of Raux's), and the control's
# `LOP3.LUT Rs, Rs, 0x4, RZ, 0x3c, !PT` (the neighbour's aux word: a valid address).
STG_AS = (rg << 32 | rs << 24 | 0x7986, 0x0C101900 | uaux, f"STG.E desc[UR{uaux}][R{rs}.64], R{rg}")
XOR4_S = (4 << 32 | rs << 24 | rs << 16 | 0x7812, 0x78E3CFF, f"LOP3.LUT R{rs}, R{rs}, 0x4, RZ, 0x3c, !PT")
LDSM4_Q = (0x1C00 << 40 | rw << 24 | rq << 16 | 0x783B, 0x200, f"LDSM.16.M88.4 R{rq}, [R{rw}+0x1c00]")
STS_AQ = (rq << 32 | rt << 24 | 0x7388, 0x800, f"STS [R{rt}], R{rq}")
STG_AG = (rg << 32 | raux << 24 | 0x7986, 0x0C101900 | uaux, f"STG.E desc[UR{uaux}][R{raux}.64], R{rg}")
LDS_S = (0x800 << 40 | ra << 24 | rs << 16 | 0x7984, 0x800, f"LDS R{rs}, [R{ra}+0x800]")
# The delay chain: `LOP3.LUT Rd, RZ, 0x1, RZ, 0x3c, !PT` (defines Rd), then `Rd ^= 1` (each
# reads the one before; stall 15 each, above the ALU latency).
CHAIN0 = (1 << 32 | 0xFF << 24 | rd << 16 | 0x7812, 0x78E3CFF, f"LOP3.LUT R{rd}, RZ, 0x1, RZ, 0x3c, !PT")
CHAIN = (1 << 32 | rd << 24 | rd << 16 | 0x7812, 0x78E3CFF, f"LOP3.LUT R{rd}, R{rd}, 0x1, RZ, 0x3c, !PT")
# The tail: a later STS with a read scoreboard the next slot waits on (the same opcode's read
# order, the strict rules' queue) completes A's reads before the kernel reuses its registers.
STS_TAIL = (0x2C00 << 40 | rv << 32 | ra << 24 | 0x7388, 0x800, f"STS [R{ra}+0x2c00], R{rv}")
# An STSM A's tail: an STSM over the Z rows (the window's loads from them are issued before it).
STSM_TAIL = (0x1C00 << 40 | rq << 32 | rw << 24 | 0x7844, 0x200, f"STSM.16.M88.4 [R{rw}+0x1c00], R{rq}")
# An STG A's tail: an STG of the marker to in[3072 + t] (never read), the same opcode's order.
STG_TAIL = (0x3000 << 40 | rv << 32 | rp << 24 | 0x7986, 0x0C101900 | ur, f"STG.E desc[UR{ur}][R{rp}.64+0x3000], R{rv}")

SB = lambda *k: sum(1 << x for x in k)
KEEP = (None, None, None)
TAIL = [(STS_TAIL, 1, 7, 4), (KEEP, 4, 7, 7, SB(0, 2, 4, 5))]
# An STG or STSM A: a tail of its own opcode, then the STS one (the placeholders' reads).
TAIL_STG = [(STG_TAIL, 1, 7, 4), (STS_TAIL, 1, 7, 3), (KEEP, 4, 7, 7, SB(0, 2, 3, 4, 5))]
# (The STSM tail reads Rq: it waits for the window's own writes of it first.)
TAIL_STSM = [(STSM_TAIL, 1, 7, 4, SB(0, 2, 5)), (STS_TAIL, 1, 7, 3), (KEEP, 4, 7, 7, SB(0, 2, 3, 4, 5))]
GAPS = tuple(
    int(g) for g in os.environ.get("SB_MIO_GAPS", "4 8 16 22 32 45 64 128 256 512 768 1024 2048").split()
)


def delay(g):
    """A's stall and the chain after it: G stall cycles from A to the later op."""
    k = (g - 1) // 15
    return g - 15 * k, ([(CHAIN0, 15)] + [(CHAIN, 15)] * (k - 1)) if k else []


def window(g, pre, a, later):
    s, ch = delay(g)
    tail = {"STG": TAIL_STG, "STSM": TAIL_STSM}.get(a[2].split(".")[0], TAIL)
    return [(MOV, 11), *pre, (a, s), *ch, *later, *tail]


Q4 = [(mov(rq + i, rv), 4) for i in range(4)]
Q2 = Q4[:2]
# Families: name -> (pre, A, later ops, expected-output tag). `_nw` / `ctl`: the controls.
FAMILIES = {
    # R10 / R11: A's address read covered by a later op's read / write scoreboard, waited on
    # before the LDC rewrites Rt; the controls rewrite with no wait.
    "rd_on_lds": ([], STS_A, [(LDS, 4, 0, 4), (LDC, 1, 5, 7, SB(4))], "a"),
    "lds_nowait": ([], STS_A, [(LDS, 4, 0, 4), (LDC, 1, 5)], "a"),
    "wcov": ([], STS_A, [(LDS, 4, 2, 7), (LDC, 1, 5, 7, SB(2))], "a"),
    "wcov_nowait": ([], STS_A, [(LDS, 4, 2, 7), (LDC, 1, 5)], "a"),
    "cov_ldg": ([], STS_A, [(LDG, 4, 2, 7), (LDC, 1, 5, 7, SB(2))], "a"),
    "cov_ldg_nw": ([], STS_A, [(LDG, 4, 2, 7), (LDC, 1, 5)], "a"),
    "cov_shfl": ([], STS_A, [(SHFL_G, 4, 2, 7), (LDC, 1, 5, 7, SB(2))], "a"),
    "cov_shfl_nw": ([], STS_A, [(SHFL_G, 4, 2, 7), (LDC, 1, 5)], "a"),
    "cov_l128": ([], STS_A, [(lds_z(rs, 128), 4, 2, 7), (LDC, 1, 5, 7, SB(2))], "a"),
    "cov_l128_nw": ([], STS_A, [(lds_z(rs, 128), 4, 2, 7), (LDC, 1, 5)], "a"),
    # ... A's data read (the W slot's stores, the aux STG), rewritten by an ALU MOV.
    "cov_s64_l64": (Q2, sts_w(rq, 64), [(lds_z(rs, 64), 4, 2, 7), (mov(rq + 1, 0xFF), 1, 7, 7, SB(2))], "w64"),
    "cov_s64_l64_nw": (Q2, sts_w(rq, 64), [(lds_z(rs, 64), 4, 2, 7), (mov(rq + 1, 0xFF), 1)], "w64"),
    "cov_s128_l128": (Q4, sts_w(rq, 128), [(lds_z(rs, 128), 4, 2, 7), (mov(rq + 3, 0xFF), 1, 7, 7, SB(2))], "w128"),
    "cov_s128_l128_nw": (Q4, sts_w(rq, 128), [(lds_z(rs, 128), 4, 2, 7), (mov(rq + 3, 0xFF), 1)], "w128"),
    "cov_s128_lds": (Q4, sts_w(rq, 128), [(LDS_S, 4, 2, 7), (mov(rq + 3, 0xFF), 1, 7, 7, SB(2))], "w128"),
    "cov_s128_lds_nw": (Q4, sts_w(rq, 128), [(LDS_S, 4, 2, 7), (mov(rq + 3, 0xFF), 1)], "w128"),
    "cov_stg_lds": ([(MOVGV, 4)], STG_AG, [(LDS_S, 4, 2, 7), (MOVG, 1, 7, 7, SB(2))], "aux"),
    "cov_stg_lds_nw": ([(MOVGV, 4)], STG_AG, [(LDS_S, 4, 2, 7), (MOVG, 1)], "aux"),
    # The same covers through the later op's read scoreboard (R10; rd=SB4, its write one SB2).
    "cov_ldg_rd": ([], STS_A, [(LDG, 4, 2, 4), (LDC, 1, 5, 7, SB(4))], "a"),
    "cov_ldg_rd_nw": ([], STS_A, [(LDG, 4, 2, 4), (LDC, 1, 5)], "a"),
    "cov_shfl_rd": ([], STS_A, [(SHFL_G, 4, 2, 4), (LDC, 1, 5, 7, SB(4))], "a"),
    "cov_shfl_rd_nw": ([], STS_A, [(SHFL_G, 4, 2, 4), (LDC, 1, 5)], "a"),
    "cov_l128_rd": ([], STS_A, [(lds_z(rs, 128), 4, 2, 4), (LDC, 1, 5, 7, SB(4))], "a"),
    "cov_l128_rd_nw": ([], STS_A, [(lds_z(rs, 128), 4, 2, 4), (LDC, 1, 5)], "a"),
    "cov_s64_l64_rd": (Q2, sts_w(rq, 64), [(lds_z(rs, 64), 4, 2, 4), (mov(rq + 1, 0xFF), 1, 7, 7, SB(4))], "w64"),
    "cov_s64_l64_rd_nw": (Q2, sts_w(rq, 64), [(lds_z(rs, 64), 4, 2, 4), (mov(rq + 1, 0xFF), 1)], "w64"),
    "cov_s128_l128_rd": (Q4, sts_w(rq, 128), [(lds_z(rs, 128), 4, 2, 4), (mov(rq + 3, 0xFF), 1, 7, 7, SB(4))], "w128"),
    "cov_s128_l128_rd_nw": (Q4, sts_w(rq, 128), [(lds_z(rs, 128), 4, 2, 4), (mov(rq + 3, 0xFF), 1)], "w128"),
    "cov_s128_lds_rd": (Q4, sts_w(rq, 128), [(LDS_S, 4, 2, 4), (mov(rq + 3, 0xFF), 1, 7, 7, SB(4))], "w128"),
    "cov_s128_lds_rd_nw": (Q4, sts_w(rq, 128), [(LDS_S, 4, 2, 4), (mov(rq + 3, 0xFF), 1)], "w128"),
    "cov_stg_lds_rd": ([(MOVGV, 4)], STG_AG, [(LDS_S, 4, 2, 4), (MOVG, 1, 7, 7, SB(4))], "aux"),
    "cov_stg_lds_rd_nw": ([(MOVGV, 4)], STG_AG, [(LDS_S, 4, 2, 4), (MOVG, 1)], "aux"),
    # R12, A's address register: an LDS / LDG writes Rt with no wait; control an ALU MOV.
    "qwar_lds": ([], STS_A, [(LDS_T, 4, 2, 7)], "a"),
    "qwar_ldg": ([], STS_A, [(LDG_T, 4, 2, 7)], "a"),
    "qctl": ([], STS_A, [(MOVZ, 4)], "a"),
    # R12, A's data register(s): a load / shuffle writes one with no wait; control an ALU MOV.
    "dq_lds": ([(MOVGV, 4)], STS_AG, [(LDS, 4, 2, 7)], "a"),
    "dq_ldg": ([(MOVGV, 4)], STS_AG, [(LDG, 4, 2, 7)], "a"),
    "dq_shfl": ([(MOVGV, 4)], STS_AG, [(SHFL_G, 4, 2, 7)], "a"),
    "dq_ldsm": ([(MOVGV, 4)], STS_AG, [(LDSM_G, 4, 2, 7)], "a"),
    "dq_ctl": ([(MOVGV, 4)], STS_AG, [(MOVG, 4)], "a"),
    "dq_q_l128": (Q4[:1], STS_AQ, [(lds_z(rq, 128), 4, 2, 7)], "a"),
    "dq_q_ctl": (Q4[:1], STS_AQ, [(mov(rq, 0xFF), 4)], "a"),
    "dq_s64_l64": (Q2, sts_w(rq, 64), [(lds_z(rq, 64), 4, 2, 7)], "w64"),
    "dq_s64_shfl": (Q2, sts_w(rq, 64), [(shfl(rq + 1), 4, 2, 7)], "w64"),
    "dq_s64_ctl": (Q2, sts_w(rq, 64), [(mov(rq + 1, 0xFF), 4)], "w64"),
    "dq_s128_l128": (Q4, sts_w(rq, 128), [(lds_z(rq, 128), 4, 2, 7)], "w128"),
    "dq_s128_ldsm4": (Q4, sts_w(rq, 128), [(LDSM4_Q, 4, 2, 7)], "w128"),
    "dq_s128_shfl": (Q4, sts_w(rq, 128), [(shfl(rq + 2), 4, 2, 7)], "w128"),
    "dq_s128_ctl": (Q4, sts_w(rq, 128), [(mov(rq + 3, 0xFF), 4)], "w128"),
    "dq_stg_lds": ([(MOVGV, 4)], STG_AG, [(LDS, 4, 2, 7)], "aux"),
    "dq_stg_shfl": ([(MOVGV, 4)], STG_AG, [(SHFL_G, 4, 2, 7)], "aux"),
    "dq_stg_ctl": ([(MOVGV, 4)], STG_AG, [(MOVG, 4)], "aux"),
}
# More pairs (the forms the certified kernels hold): covers through the write scoreboard,
# data and address registers written with no wait, each with its control.
W2 = lambda r: (mov(r, 0xFF), 1, 7, 7, SB(2))
N2 = lambda r: (mov(r, 0xFF), 1)
RS2 = [(mov(rs, rw), 4)]
RSAUX = [(mov(rs, raux), 4), (mov(rs + 1, raux + 1), 4)]
FAMILIES.update({
    "c_s64_shfl": (Q2, sts_w(rq, 64), [(SHFL_G, 4, 2, 7), W2(rq + 1)], "w64"),
    "c_s64_shfl_nw": (Q2, sts_w(rq, 64), [(SHFL_G, 4, 2, 7), N2(rq + 1)], "w64"),
    "c_s128_shfl": (Q4, sts_w(rq, 128), [(SHFL_G, 4, 2, 7), W2(rq + 3)], "w128"),
    "c_s128_shfl_nw": (Q4, sts_w(rq, 128), [(SHFL_G, 4, 2, 7), N2(rq + 3)], "w128"),
    "c_s128_ldsm4": (Q4, sts_w(rq, 128), [(LDSM4_S, 4, 2, 7), W2(rq + 3)], "w128"),
    "c_s128_ldsm4_nw": (Q4, sts_w(rq, 128), [(LDSM4_S, 4, 2, 7), N2(rq + 3)], "w128"),
    "c_s128_sidx": (Q4, sts_w(rq, 128), [(shfl_idx(rg), 4, 2, 7), W2(rq + 3)], "w128"),
    "c_s128_sidx_nw": (Q4, sts_w(rq, 128), [(shfl_idx(rg), 4, 2, 7), N2(rq + 3)], "w128"),
    "c_s_sidx": ([], STS_A, [(shfl_idx(rg), 4, 2, 7), (LDC, 1, 5, 7, SB(2))], "a"),
    "c_s_sidx_nw": ([], STS_A, [(shfl_idx(rg), 4, 2, 7), (LDC, 1, 5)], "a"),
    "c_u8_lds": ([], STS_U8, [(LDS, 4, 2, 7), (LDC, 1, 5, 7, SB(2))], "u8"),
    "c_u8_lds_nw": ([], STS_U8, [(LDS, 4, 2, 7), (LDC, 1, 5)], "u8"),
    "c_u16_lds": ([], STS_U16, [(LDS, 4, 2, 7), (LDC, 1, 5, 7, SB(2))], "u16"),
    "c_u16_lds_nw": ([], STS_U16, [(LDS, 4, 2, 7), (LDC, 1, 5)], "u16"),
    "c_stg_shfl": ([(MOVGV, 4)], STG_AG, [(shfl(rs), 4, 2, 7), W2(rg)], "aux"),
    "c_stg_shfl_nw": ([(MOVGV, 4)], STG_AG, [(shfl(rs), 4, 2, 7), N2(rg)], "aux"),
    "c_stg_ldg": ([(MOVGV, 4)], STG_AG, [(LDG_S, 4, 2, 7), W2(rg)], "aux"),
    "c_stg_ldg_nw": ([(MOVGV, 4)], STG_AG, [(LDG_S, 4, 2, 7), N2(rg)], "aux"),
    "c_stg_l128": ([(MOVGV, 4)], STG_AG, [(lds_z(rs, 128), 4, 2, 7), W2(rg)], "aux"),
    "c_stg_l128_nw": ([(MOVGV, 4)], STG_AG, [(lds_z(rs, 128), 4, 2, 7), N2(rg)], "aux"),
    "c_stsm_l128": (Q4, STSM_Q, [(lds_z(rs, 128), 4, 2, 7), W2(rq + 3)], "stsm"),
    "c_stsm_l128_nw": (Q4, STSM_Q, [(lds_z(rs, 128), 4, 2, 7), N2(rq + 3)], "stsm"),
    "c_stsm_lds": (Q4, STSM_Q, [(LDS_S, 4, 2, 7), W2(rq + 3)], "stsm"),
    "c_stsm_lds_nw": (Q4, STSM_Q, [(LDS_S, 4, 2, 7), N2(rq + 3)], "stsm"),
    # Data registers.
    "d_s_sidx": ([(MOVGV, 4)], STS_AG, [(shfl_idx(rg), 4, 2, 7)], "a"),
    "d_s128_sidx": (Q4, sts_w(rq, 128), [(shfl_idx(rq + 2), 4, 2, 7)], "w128"),
    "d_s128_lds": (Q4, sts_w(rq, 128), [((0x800 << 40 | ra << 24 | (rq + 1) << 16 | 0x7984, 0x800, f"LDS R{rq + 1}, [R{ra}+0x800]"), 4, 2, 7)], "w128"),
    "d_stg_ldg": ([(MOVGV, 4)], STG_AG, [(LDG, 4, 2, 7)], "aux"),
    "d_stg_l128": (Q4[:1], STG_AQ, [(lds_z(rq, 128), 4, 2, 7)], "aux"),
    "d_stgq_ctl": (Q4[:1], STG_AQ, [(mov(rq, 0xFF), 4)], "aux"),
    "d_u8_lds": ([(MOVGV, 4)], STS_U8G, [(LDS, 4, 2, 7)], "u8"),
    "d_u8_ctl": ([(MOVGV, 4)], STS_U8G, [(MOVG, 4)], "u8"),
    "d_u16_lds": ([(MOVGV, 4)], STS_U16G, [(LDS, 4, 2, 7)], "u16"),
    "d_u16_ctl": ([(MOVGV, 4)], STS_U16G, [(MOVG, 4)], "u16"),
    "d_stsm_l128": (Q4, STSM_Q, [(lds_z(rq, 128), 4, 2, 7)], "stsm"),
    "d_stsm_ldsm4": (Q4, STSM_Q, [(LDSM4_Q, 4, 2, 7)], "stsm"),
    "d_stsm_lds": (Q4, STSM_Q, [((0x800 << 40 | ra << 24 | (rq + 1) << 16 | 0x7984, 0x800, f"LDS R{rq + 1}, [R{ra}+0x800]"), 4, 2, 7)], "stsm"),
    "d_stsm_ctl": (Q4, STSM_Q, [(mov(rq + 3, 0xFF), 4)], "stsm"),
    # Address registers (a shuffle writes the lane-xor-1 neighbour's address, a load zero).
    "a_s_shfl": ([], STS_A, [(shfl(rt), 4, 2, 7)], "a"),
    "a_s64_l64": (Q2 + RS2, sts_wr(rs, rq, 64), [(lds_z(rs, 64), 4, 2, 7)], "w64"),
    "a_s64_shfl": (Q2 + RS2, sts_wr(rs, rq, 64), [(shfl(rs), 4, 2, 7)], "w64"),
    "a_s64_ctl": (Q2 + RS2, sts_wr(rs, rq, 64), [(mov(rs, 0xFF), 4)], "w64"),
    "a_s128_l128": (Q4 + RS2, sts_wr(rs, rq, 128), [(lds_z(rs, 128), 4, 2, 7)], "w128"),
    "a_s128_lds": (Q4 + RS2, sts_wr(rs, rq, 128), [((0x800 << 40 | ra << 24 | rs << 16 | 0x7984, 0x800, f"LDS R{rs}, [R{ra}+0x800]"), 4, 2, 7)], "w128"),
    "a_s128_shfl": (Q4 + RS2, sts_wr(rs, rq, 128), [(shfl(rs), 4, 2, 7)], "w128"),
    "a_s128_ctl": (Q4 + RS2, sts_wr(rs, rq, 128), [(mov(rs, 0xFF), 4)], "w128"),
    "a_u8_lds": ([], STS_U8, [(LDS_T, 4, 2, 7)], "u8"),
    "a_u8_ctl": ([], STS_U8, [(MOVZ, 4)], "u8"),
    "a_u16_lds": ([], STS_U16, [(LDS_T, 4, 2, 7)], "u16"),
    "a_u16_ctl": ([], STS_U16, [(MOVZ, 4)], "u16"),
    "a_stg_shfl": ([(MOVGV, 4)] + RSAUX, STG_AS, [(shfl(rs), 4, 2, 7)], "aux"),
    "a_stg_ctl": ([(MOVGV, 4)] + RSAUX, STG_AS, [(XOR4_S, 4)], "aux"),
})
# Review additions: controls on the exact register a data window writes, and covers measured on
# the register role the families above leave out (results.txt, the 512 / 768 / 1024 rows).
FAMILIES.update({
    # STS.128 data controls on rq+1 / rq+2 (windows d_s128_lds write rq+1; dq_s128_shfl, d_s128_sidx rq+2)
    "x_s128_ctl1": (Q4, sts_w(rq, 128), [(mov(rq + 1, 0xFF), 4)], "w128"),
    "x_s128_ctl2": (Q4, sts_w(rq, 128), [(mov(rq + 2, 0xFF), 4)], "w128"),
    # STSM data control on rq+1 (window d_stsm_lds writes rq+1)
    "x_stsm_ctl1": (Q4, STSM_Q, [(mov(rq + 1, 0xFF), 4)], "stsm"),
    # 32-bit STS cover by LDS on A's DATA register (the families above: its address only)
    "x_cov_lds_d": ([(MOVGV, 4)], STS_AG, [(LDS_S, 4, 2, 7), (MOVG, 1, 7, 7, SB(2))], "a"),
    "x_cov_lds_d_nw": ([(MOVGV, 4)], STS_AG, [(LDS_S, 4, 2, 7), (MOVG, 1)], "a"),
    # STS.128 cover by LDS on A's ADDRESS register (above: its data rq+3 only)
    "x_cov_s128_lds_a": (Q4 + RS2, sts_wr(rs, rq, 128), [(LDS, 4, 2, 7), (mov(rs, 0xFF), 1, 7, 7, SB(2))], "w128"),
    "x_cov_s128_lds_a_nw": (Q4 + RS2, sts_wr(rs, rq, 128), [(LDS, 4, 2, 7), (mov(rs, 0xFF), 1)], "w128"),
    # STG.E cover by LDS on A's ADDRESS pair (above: its data only)
    "x_cov_stg_lds_a": ([(MOVGV, 4)] + RSAUX, STG_AS, [(LDS_T, 4, 2, 7), (XOR4_S, 1, 7, 7, SB(2))], "aux"),
    "x_cov_stg_lds_a_nw": ([(MOVGV, 4)] + RSAUX, STG_AS, [(LDS_T, 4, 2, 7), (XOR4_S, 1)], "aux"),
})
CTL = {n for n in FAMILIES if n.endswith(("_nw", "_nowait", "ctl", "ctl1", "ctl2"))}
# The v3 families run at these gaps (results.txt).
VARIANTS, TAG = {}, {}
for g in GAPS:
    for n, (pre, a, later, tag) in FAMILIES.items():
        VARIANTS[f"{n}_g{g}"] = window(g, pre, a, later)
        TAG[f"{n}_g{g}"] = tag
# Static only (MISS 250, not run): an STS with no read scoreboard, attached to a later LDS's read
# scoreboard (rb = SB4) G1 stall cycles after it (an R10 cover inside its window), then an LDS
# rewriting the STS's address register with no wait G2 cycles after the attach. The R12 window
# is the gap from the STS, G1 + 4 + G2: past 1024 an sb-unmeasured: contract like the direct
# window at that gap, not a gap counted from the attach.
def chain(n):
    k = n // 15
    return ([(CHAIN0, 15)] + [(CHAIN, 15)] * (k - 1)) if k else []


ATTACH = {(1000, 1005): "contract", (1000, 300): "contract", (16, 1005): "contract", (16, 300): "clean"}
for (g1, g2), _ in ATTACH.items():
    s, ch = delay(g1)
    VARIANTS[f"att_{g1}_{g2}"] = [(MOV, 11), (STS_A, s), *ch, (LDS, 4, 0, 4), *chain(g2), (LDS_T, 4, 2, 7), *TAIL]
    TAG[f"att_{g1}_{g2}"] = "a"
# The windows clean at the gb300 level (strict: sb-hwq: contracts) once measured, their controls
# violations. EXPECT / EXPECT_SASS: what the fixture asserts (see tests/fixtures.rs).
# Past the measured windows (2048: no control is wrong there) a window is an sb-unmeasured:
# contract at every level.
EXPECT = {}
EXPECT_SASS = {}
for v in VARIANTS:
    n, _, g = v.rpartition("_g")
    if v.startswith("att_"):
        g1, g2 = map(int, v.split("_")[1:])
        EXPECT[v] = ATTACH[(g1, g2)]
        EXPECT_SASS[v] = '["contract"]'
    elif n in CTL:
        # (strict: also the post-window placeholders' reads of the marker register, rewritten by
        # the kernel's later LDS.128: an sb-hwq: contract in every variant)
        EXPECT[v] = "violation"
        EXPECT_SASS[v] = '["contract", "violation"]'
    elif int(g) > 1024:
        EXPECT[v] = "contract"
        EXPECT_SASS[v] = '["contract"]'
    else:
        EXPECT[v] = "clean"
        EXPECT_SASS[v] = '["contract"]'
# The fixture (tests/fixtures.rs): every family at gap 64, a few windows and controls at 1024,
# and two windows past the measured range (the full set of runs is in results.txt).
FIXTURE = {f"{n}_g64" for n in FAMILIES}
FIXTURE |= {
    f"{n}_g1024"
    for n in ("rd_on_lds", "lds_nowait", "dq_s128_l128", "dq_s128_ctl", "cov_stg_lds", "cov_stg_lds_nw")
}
FIXTURE |= {"rd_on_lds_g2048", "dq_lds_g2048"}
FIXTURE |= {f"att_{g1}_{g2}" for g1, g2 in ATTACH}

if CAL:
    # The two CS2R (their own destination registers, live to the output) moved into the window:
    # the first right after A (the chain's first slot), the second in the later op's slot (an
    # ALU write, as qctl_<G>): the output's fourth word is the chain's cycles, G - A's stall.
    # A window with no chain (G < 16) has no cal variant of its own: cal_g4 / cal_g8 still put
    # the 15-cycle CS2R slot after A and time that slot (results.txt).
    assert len(clk) == 2 and all(t.split()[-2].startswith("R") for _, t in clk), clk
    PRED = ~(0xF << 12) & ((1 << 64) - 1)
    enc = []
    for at, t in clk:
        lo, hi = struct.unpack_from("<QQ", b0, text + at)
        enc.append((at, lo, hi & ((1 << 41) - 1), t.split(" ", 1)[1] if t.startswith("@") else t))
    CS0, CS1 = ((lo & PRED | 0x7 << 12, hi, t) for _, lo, hi, t in enc)
    # The originals guarded off (@!PT: predicate field bits 12..15 = 0xf).
    OFF = {at: (lo | 0xF << 12, hi, "@!PT " + t) for at, lo, hi, t in enc}
    VARIANTS = {}
    for g in GAPS:
        s, ch = delay(g)
        VARIANTS[f"cal_g{g}"] = [(MOV, 11), (STS_A, s), (CS0, 15), *ch[1:], (CS1, 1), (MOVZ, 4), *TAIL]
        TAG[f"cal_g{g}"] = "cal"

# SB_MIO_ONLY="<variant> ...": build only these (run.sh passes its VARIANTS).
only = os.environ.get("SB_MIO_ONLY", "").split()
if only:
    VARIANTS = {v: w for v, w in VARIANTS.items() if v in only}
PREDICTED = "# expect: model-predicted (not run on GB300 at this gap)\n"
toml = []
for v, win in VARIANTS.items():
    b = bytearray(b0)
    patches = list(zip(slots, win))
    for slot, (enc, *c) in patches:
        lo, hi_bits, _ = enc
        lo0, hi0 = struct.unpack_from("<QQ", b, text + slot)
        if enc is KEEP:
            lo, hi_bits = lo0, hi0 & ((1 << 41) - 1)
        struct.pack_into("<QQ", b, text + slot, lo, ctrl(hi_bits, *c))
    if CAL:
        for at, (lo, hi_bits, _) in OFF.items():
            hi0 = struct.unpack_from("<QQ", b, text + at)[1]
            struct.pack_into("<QQ", b, text + at, lo, hi_bits | (hi0 & ~((1 << 41) - 1)))
    f = os.path.join(out, f"{v}.cubin")
    open(f, "wb").write(b)
    got = dict(insts(f))
    for slot, (enc, *_) in patches:
        assert got[slot] == (enc[2] or dict(lines)[slot]), (v, hex(slot), got[slot])
    if CAL:
        for at, (_, _, want) in OFF.items():
            assert got[at] == want, (v, hex(at), got[at])
    print(v, TAG[v], len(win), "slots;", "; ".join(got[s] for s in slots[: min(len(win), 4)]), "...",
          "; ".join(got[s] for s in slots[max(0, len(win) - 4) : len(win)]))
    if CAL or v not in FIXTURE:
        continue
    sass = f"expect_sass = {EXPECT_SASS[v]}\n" if v in EXPECT_SASS else ""
    exp = EXPECT[v] if isinstance(EXPECT[v], list) else [EXPECT[v]]
    exp = ", ".join(f'"{e}"' for e in exp)
    # `expect` is what GB300 did where it ran the variant; the x_ families ran at 512 / 768 /
    # 1024 only and the att_ windows not at all: there it is the model's prediction.
    note = PREDICTED if v.startswith("att_") or v.startswith("x_") and v.endswith("_g64") else ""
    toml.append(f'[[variants]]\nid = "{v}"\n{note}expect = [{exp}]\n{sass}')
    for slot, (enc, *c) in patches:
        _, _, want = enc
        stall, wb, rb, wait = list(c) + [None, 7, 7, 0][len(c) :]
        # The whole 128 bits as in the GB300 image (the yield bit cleared too: kept set, it
        # reads as an opex field of the ALU ops).
        lo, hi = struct.unpack_from("<QQ", b, text + slot)
        bits = f"bits = [[0, 64, {lo:#x}], [64, 64, {hi:#x}]]\n"
        toml.append(
            f'[[variants.patch]]\nat = "@{slot:#06x}"\n{bits}'
            f'stall = {stall}\nwbar = {wb}\nrbar = {rb}\nwait = {wait}\nexpect = "{want or dict(lines)[slot]}"\n'
        )
# The fixture's case: one CTA of the full-occupancy layout, the skeleton compiled from ../kernel.ptx.
HEADER = f'''name = "probe_sb_mio_gap_fixture"
description = """
The windows of probes/sb_mio_gap as control-bit / encoding patches of the ptxas cubin of
../kernel.ptx (written by `patch.py <cubin> <dir> fixture/case.toml`), one CTA: the static
scoreboard check must accept at the gb300 level the windows GB300 runs correctly with a
discriminating control (results.txt; strict: sb-hwq: contracts), flag the controls, and leave
the windows past the measured gaps (2048) sb-unmeasured: contracts."""
ptx = "../kernel.ptx"
entry = "sb_mio_gap"
grid = [1, 1, 1]
block = [256, 1, 1]
dynamic_smem = 18432
outputs = ["out"]

[[buffers]]
name = "out"
dtype = "u32"
shape = [2048]
init = "zeros"

[[buffers]]
name = "in"
dtype = "u32"
shape = [4096]
init = "zeros"

[[buffers]]
name = "aux"
dtype = "u32"
shape = [256]
init = "zeros"

[[params]]
kind = "ptr"
buffer = "out"

[[params]]
kind = "ptr"
buffer = "in"

[[params]]
kind = "u32"
value = 12288

[[params]]
kind = "u32"
value = 0

[[params]]
kind = "ptr"
buffer = "aux"

[[variants]]
id = "good"
summary = "the unpatched skeleton"
expect = ["clean"]

'''
if len(sys.argv) > 3:
    open(sys.argv[3], "w").write(HEADER + "\n".join(toml))
