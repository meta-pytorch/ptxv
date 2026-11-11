# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Build the probe windows of probes/sb_mio_order: ptxas compiles kernel.ptx (a placeholder run
# of `STS [R0+0x4000..], R29` after three LDS.128), then each variant overwrites the first
# placeholders, by copying ptxas's own encodings with other registers / offsets / control
# bits (Rt, Rg: registers the skeleton never uses; c[0x0][0x390] is the `bogus` parameter):
#   MOV Rt, Ra                 (Ra: the slot address, Rv: the marker)
#   STS [Rt], Rv               A
#   [LDG.E Rg, desc[URu][Rp.64]]   (variant inorder_ldg: rmsnorm_bwd's shape)
#   STS [Rt+0x400], Rv         B
#   LDC Rt, c[0x0][0x390]      overwrites the address of A and B
# and checks nvdisasm prints each patched instruction as intended.
#   python3 patch.py kernel.cubin OUTDIR [fixture.toml]
# With a third argument, also writes the windows as `patch = [...]` variants for
# probes/sb_mio_order/fixture (the static scoreboard check, tests/fixtures.rs).
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


text = next(sh(i)[4] for i in range(shnum) if name(i) == ".text.sb_mio_order")


def insts(f):
    r = []
    for ln in dis(f).split("\n"):
        m = re.match(r"\s+/\*([0-9a-f]{4})\*/\s+(.*?)\s*;", ln)
        if m:
            r.append((int(m.group(1), 16), m.group(2).strip()))
    return r


lines = insts(src)
# The placeholder run right after the LDS.128 burst: `STS [Ra+0x4000], Rv`, ...
first = next(
    i
    for i, (_, t) in enumerate(lines)
    if re.fullmatch(r"STS \[R\d+\+0x4000\], R\d+", t)
    and lines[i - 1][1].startswith("LDS.128")
)
ra, rv = map(
    int, re.fullmatch(r"STS \[R(\d+)\+0x4000\], R(\d+)", lines[first][1]).groups()
)
slots = [a for a, _ in lines[first : first + 6]]
assert all(
    t.startswith(f"STS [R{ra}+0x") and t.endswith(f"], R{rv}")
    for _, t in lines[first : first + 6]
)
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


def regs(t):
    """(registers written, registers read) of an instruction (straight-line code only)."""
    op, _, rest = t.partition(" ")
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
# Not read just before it either (a memory instruction may read its operands late).
recent = set().union(*(regs(t)[1] for _, t in lines[first - 16 : first]))
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
rt, rg = free[:2]


def ctrl(hi, stall, wb=7, rb=7, wait=0):
    c = stall | 0 << 4 | wb << 5 | rb << 8 | wait << 11
    return (hi & ((1 << 41) - 1)) | c << 41


# Encodings (low word, high word without control bits) as ptxas emits them: `MOV R17, R14` =
# 0x0000000e00117202 / 0x...0f00, `STS [R0+0x4000], R29` = 0x0040001d00007388 / 0x...0800,
# `LDG.E R0, desc[UR6][R24.64]` = 0x0000000618007981 / 0x...0c1e1900, `LDC R1, c[0x0][0x37c]`
# = 0x0000df00ff017b82 / 0x...0800: Rd at bit 16, Ra at 24, Rb / UR at 32, offset at 40.
MOV = (rt << 16 | ra << 32 | 0x7202, 0xF00, f"MOV R{rt}, R{ra}")
STS_A = (rv << 32 | rt << 24 | 0x7388, 0x800, f"STS [R{rt}], R{rv}")
STS_B = (0x400 << 40 | rv << 32 | rt << 24 | 0x7388, 0x800, f"STS [R{rt}+0x400], R{rv}")
LDG = (
    ur << 32 | rp << 24 | rg << 16 | 0x7981,
    0x0C1E1900,
    f"LDG.E R{rg}, desc[UR{ur}][R{rp}.64]",
)
LDC = (
    (0x390 // 4) << 40 | 0xFF << 24 | rt << 16 | 0x7B82,
    0x800,
    f"LDC R{rt}, c[0x0][0x390]",
)

SB = lambda *k: sum(1 << x for x in k)
# The next placeholder STS, unchanged, waiting for the window's own LDG / LDC writes (SB0, SB5).
KEEP = (None, None, None)
DRAIN = (KEEP, 4, 7, 7, SB(0, 5))
VARIANTS = {
    # ptxas's own pattern: the read scoreboard on the last STS only, the LDC waits for it.
    "inorder": [(MOV, 11), (STS_A, 4), (STS_B, 4, 7, 4), (LDC, 1, 5, 7, SB(4)), DRAIN],
    # rmsnorm_bwd N=256: an LDG with its own read scoreboard between A and B.
    "inorder_ldg": [
        (MOV, 11),
        (STS_A, 4),
        (LDG, 4, 0, 2),
        (STS_B, 4, 7, 4),
        (LDC, 1, 5, 7, SB(4)),
        DRAIN,
    ],
    # The LDG's read scoreboard as the only cover of A (no B: its slot stays 0).
    "rd_on_ldg": [(MOV, 11), (STS_A, 4), (LDG, 4, 0, 4), (LDC, 1, 5, 7, SB(4)), DRAIN],
    # Controls: the scoreboard on A only (B's read uncovered), and no wait at all.
    "rd_on_a": [(MOV, 11), (STS_A, 4, 7, 4), (STS_B, 4), (LDC, 1, 5, 7, SB(4)), DRAIN],
    "nowait": [(MOV, 11), (STS_A, 4), (STS_B, 4), (LDC, 1, 5), DRAIN],
}
EXPECT = {
    "inorder": "clean",
    "inorder_ldg": "clean",
    "rd_on_ldg": "clean",
    "rd_on_a": "violation",
    "nowait": "violation",
}
toml = []
for v, win in VARIANTS.items():
    b = bytearray(b0)
    for slot, (enc, *c) in zip(slots, win):
        lo, hi_bits, _ = enc
        lo0, hi0 = struct.unpack_from("<QQ", b, text + slot)
        if enc is KEEP:
            lo, hi_bits = lo0, hi0 & ((1 << 41) - 1)
        struct.pack_into("<QQ", b, text + slot, lo, ctrl(hi_bits, *c))
    f = os.path.join(out, f"{v}.cubin")
    open(f, "wb").write(b)
    got = dict(insts(f))
    for slot, (enc, *_) in zip(slots, win):
        assert got[slot] == (enc[2] or dict(lines)[slot]), (v, hex(slot), got[slot])
    print(v, "; ".join(got[s] for s in slots[: len(win)]))
    toml.append(f'[[variants]]\nid = "{v}"\nexpect = ["{EXPECT[v]}"]\n')
    for slot, (enc, *c) in zip(slots, win):
        lo, hi_bits, want = enc
        stall, wb, rb, wait = list(c) + [None, 7, 7, 0][len(c) :]
        bits = (
            ""
            if enc is KEEP
            else f"bits = [[0, 64, {lo:#x}], [64, 41, {hi_bits:#x}]]\n"
        )
        toml.append(
            f'[[variants.patch]]\nat = "@{slot:#06x}"\n{bits}'
            f'stall = {stall}\nwbar = {wb}\nrbar = {rb}\nwait = {wait}\nexpect = "{want or dict(lines)[slot]}"\n'
        )
if len(sys.argv) > 3:
    open(sys.argv[3], "w").write("\n".join(toml))
