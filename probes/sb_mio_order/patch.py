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
# `LDS R6, [R0+0x800]` = 0x0008000000067984 / 0x...0800: a load of region 2 (never written).
LDS = (0x800 << 40 | ra << 24 | rg << 16 | 0x7984, 0x800, f"LDS R{rg}, [R{ra}+0x800]")
# The same load into A's address register (a load writing a register an earlier queue op reads),
# an LDG into it, and `MOV Rt, RZ` (an ALU write, the control).
LDS_T = (0x800 << 40 | ra << 24 | rt << 16 | 0x7984, 0x800, f"LDS R{rt}, [R{ra}+0x800]")
LDG_T = (
    ur << 32 | rp << 24 | rt << 16 | 0x7981,
    0x0C1E1900,
    f"LDG.E R{rt}, desc[UR{ur}][R{rp}.64]",
)
MOVZ = (rt << 16 | 0xFF << 32 | 0x7202, 0xF00, f"MOV R{rt}, RZ")
# B guarded off (@!PT), an ALU write of the free register Rg, and a SHFL on Rg (the adversarial
# review's windows).
STS_B_OFF = (0x400 << 40 | rv << 32 | rt << 24 | 0xF388, 0x800, f"@!PT STS [R{rt}+0x400], R{rv}")
MOVG = (rg << 16 | 0xFF << 32 | 0x7202, 0xF00, f"MOV R{rg}, RZ")
SHFL_G = (0x0C201F0000007F89 | rg << 24 | rg << 16, 0xE0000, f"SHFL.BFLY PT, R{rg}, R{rg}, 0x1, 0x1f")
LDC = (
    (0x390 // 4) << 40 | 0xFF << 24 | rt << 16 | 0x7B82,
    0x800,
    f"LDC R{rt}, c[0x0][0x390]",
)

# R12 data-register windows (SND3, MISS 170): A stores Rg (a copy of the marker), then a load writes
# Rg with no wait. Rx: a third free register (LDSM's 16-byte-aligned row address).
rx = free[2]
MOVGV = (rg << 16 | rv << 32 | 0x7202, 0xF00, f"MOV R{rg}, R{rv}")
STS_AG = (rg << 32 | rt << 24 | 0x7388, 0x800, f"STS [R{rt}], R{rg}")
# `LOP3.LUT R0, R5, 0xfffffff0, RZ, 0xc0, !PT` = 0xfffffff005007812 / 0x...078ec0ff (mask 0x7ffffff0
# here: the immediate's top bit clear), `LDSM.16.M88 R6, [R0+0x2000]` = 0x002000000006783b / 0,
# `ATOMS.EXCH RZ, [R5], R6` = 0x0000000605ff738c / 0x...04000000.
LOPX = (0x7FFFFFF0 << 32 | ra << 24 | rx << 16 | 0x7812, 0x78EC0FF, f"LOP3.LUT R{rx}, R{ra}, 0x7ffffff0, RZ, 0xc0, !PT")
LDSM_G = (0x2000 << 40 | rx << 24 | rg << 16 | 0x783B, 0x0, f"LDSM.16.M88 R{rg}, [R{rx}+0x2000]")
ATOMS_AG = (rg << 32 | rt << 24 | 0xFF << 16 | 0x738C, 0x4000000, f"ATOMS.EXCH RZ, [R{rt}], R{rg}")
MOVX = (rx << 16 | 0xFF << 32 | 0x7202, 0xF00, f"MOV R{rx}, RZ")

SB = lambda *k: sum(1 << x for x in k)
# The next placeholder STS, unchanged, waiting for the window's own LDG / LDC writes (SB0, SB5).
KEEP = (None, None, None)
DRAIN = (KEEP, 4, 7, 7, SB(0, 5))
# The qwar windows' loads write Rt on SB2: drained too (the kernel writes Rt again later).
DRAIN2 = (KEEP, 4, 7, 7, SB(0, 2, 5))
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
    # A shared load's read scoreboard as the only cover of A (another opcode of the queue), and
    # the same window with no wait (its positive control).
    "rd_on_lds": [(MOV, 11), (STS_A, 4), (LDS, 4, 0, 4), (LDC, 1, 5, 7, SB(4)), DRAIN],
    "lds_nowait": [(MOV, 11), (STS_A, 4), (LDS, 4, 0, 4), (LDC, 1, 5), DRAIN],
    # wcov: A's read (rd=SB1, never waited) covered by a later LDS's *write* scoreboard; and the
    # same with no wait (its positive control).
    "wcov": [(MOV, 11), (STS_A, 4, 7, 1), (LDS, 4, 2, 7), (LDC, 1, 5, 7, SB(2)), DRAIN],
    "wcov_nowait": [(MOV, 11), (STS_A, 4, 7, 1), (LDS, 4, 2, 7), (LDC, 1, 5), DRAIN],
    # qwar: a load writes A's address register with no wait after A (untracked): an LDS, an
    # LDG; the control writes it with an ALU MOV.
    "qwar_lds": [(MOV, 11), (STS_A, 4), (LDS_T, 4, 2, 7), DRAIN2],
    "qwar_ldg": [(MOV, 11), (STS_A, 4), (LDG_T, 4, 2, 7), DRAIN2],
    "qwar_ctl": [(MOV, 11), (STS_A, 4), (MOVZ, 4), DRAIN2],
    # Adversarial review: A covered by a guarded-off STS's read scoreboard (and no wait); a
    # SHFL between A and the rewrite of A's address register, and an ALU MOV in its slot
    # (MISS 128: the SHFL orders nothing); delay controls, the rewrite 8 to 33 cycles after A.
    "rvs_inorder_off": [(MOV, 11), (STS_A, 4), (STS_B_OFF, 4, 7, 4), (LDC, 1, 5, 7, SB(4)), DRAIN],
    "rvs_inorder_off_nowait": [(MOV, 11), (STS_A, 4), (STS_B_OFF, 4, 7, 4), (LDC, 1, 5), DRAIN],
    "rvs_shfl_retain": [(MOV, 11), (STS_A, 4), (SHFL_G, 4, 2, 7), (MOVZ, 4), DRAIN2],
    "rvs_shfl_retain_ctl": [(MOV, 11), (STS_A, 4), (MOVG, 4), (MOVZ, 4), DRAIN2],
    "rvs_qctl_8": [(MOV, 11), (STS_A, 8), (MOVZ, 4), DRAIN2],
    "rvs_qctl_11": [(MOV, 11), (STS_A, 11), (MOVZ, 4), DRAIN2],
    "rvs_qctl_22": [(MOV, 11), (STS_A, 11), (MOVG, 11), (MOVZ, 4), DRAIN2],
    "rvs_qctl_33": [(MOV, 11), (STS_A, 11), (MOVG, 11), (MOVG, 11), (MOVZ, 4), DRAIN2],
    # R12 data register (SND3): a load writing the register A stores, no wait (an LDS, LDG, SHFL,
    # LDSM; A an ATOMS.EXCH too); dq_ctl*: an ALU write in its slot, right after A and 8 / 22 / 33
    # cycles after (discriminating controls); dq_own: A's read scoreboard waited on (reference).
    "dq_lds": [(MOV, 11), (MOVGV, 4), (STS_AG, 4), (LDS, 4, 2, 7), DRAIN2],
    "dq_ldg": [(MOV, 11), (MOVGV, 4), (STS_AG, 4), (LDG, 4, 2, 7), DRAIN2],
    "dq_shfl": [(MOV, 11), (MOVGV, 4), (STS_AG, 4), (SHFL_G, 4, 2, 7), DRAIN2],
    "dq_ldsm": [(MOV, 11), (MOVGV, 4), (LOPX, 4), (STS_AG, 4), (LDSM_G, 4, 2, 7), DRAIN2],
    "dq_atoms_lds": [(MOV, 11), (MOVGV, 4), (ATOMS_AG, 4), (LDS, 4, 2, 7), DRAIN2],
    "dq_ctl": [(MOV, 11), (MOVGV, 4), (STS_AG, 4), (MOVG, 4), DRAIN2],
    "dq_ctl_8": [(MOV, 11), (MOVGV, 4), (STS_AG, 8), (MOVG, 4), DRAIN2],
    "dq_ctl_22": [(MOV, 11), (MOVGV, 4), (STS_AG, 11), (MOVX, 11), (MOVG, 4), DRAIN2],
    "dq_ctl_33": [(MOV, 11), (MOVGV, 4), (STS_AG, 11), (MOVX, 11), (MOVX, 11), (MOVG, 4)],
    "dq_atoms_ctl": [(MOV, 11), (MOVGV, 4), (ATOMS_AG, 4), (MOVG, 4), DRAIN2],
    "dq_own": [(MOV, 11), (MOVGV, 4), (STS_AG, 4, 7, 4), (LDS, 4, 2, 7, SB(4)), DRAIN2],
    # Gap windows (MISS 170, the R10 to R12 recovery): the same covers and queued WARs with the
    # later op 8, 11 and 22 stall cycles after A (22: A's stall 11, then an ALU op of 11), each
    # with its control in the same layout (no wait, or an ALU write in the load's slot).
    **{
        f"{n}_g{g}": w
        for g, pre in ((8, [(STS_A, 8)]), (11, [(STS_A, 11)]), (22, [(STS_A, 11), (MOVG, 11)]))
        for n, w in (
            ("rd_on_lds", [(MOV, 11), *pre, (LDS, 4, 0, 4), (LDC, 1, 5, 7, SB(4)), DRAIN]),
            ("lds_nowait", [(MOV, 11), *pre, (LDS, 4, 0, 4), (LDC, 1, 5), DRAIN]),
            ("qwar_lds", [(MOV, 11), *pre, (LDS_T, 4, 2, 7), DRAIN2]),
            ("qwar_ldg", [(MOV, 11), *pre, (LDG_T, 4, 2, 7), DRAIN2]),
        )
    },
    **{
        f"wcov{x}_g{g}": [(MOV, 11), (STS_A, a, 7, 1), *mid, (LDS, 4, 2, 7), ldc, DRAIN]
        for g, a, mid in ((8, 8, []), (11, 11, []), (22, 11, [(MOVG, 11)]))
        for x, ldc in (("", (LDC, 1, 5, 7, SB(2))), ("_nowait", (LDC, 1, 5)))
    },
    **{
        f"dq_{n}_g{g}": [(MOV, 11), (MOVGV, 4), *lop, (STS_AG, a), *mid, load, DRAIN2]
        for g, a, mid in ((8, 8, []), (11, 11, []), (22, 11, [(MOVX, 11)]))
        for n, lop, load in (
            ("lds", [], (LDS, 4, 2, 7)),
            ("ldg", [], (LDG, 4, 2, 7)),
            ("shfl", [], (SHFL_G, 4, 2, 7)),
            ("ldsm", [(LOPX, 4)], (LDSM_G, 4, 2, 7)),
        )
        if not (n == "ldsm" and g == 22)
    },
    "dq_ctl_11": [(MOV, 11), (MOVGV, 4), (STS_AG, 11), (MOVG, 4), DRAIN2],
    # Controls: the scoreboard on A only (B's read uncovered), and no wait at all.
    "rd_on_a": [(MOV, 11), (STS_A, 4, 7, 4), (STS_B, 4), (LDC, 1, 5, 7, SB(4)), DRAIN],
    "nowait": [(MOV, 11), (STS_A, 4), (STS_B, 4), (LDC, 1, 5), DRAIN],
}
EXPECT = {
    "inorder": "clean",
    "inorder_ldg": "clean",
    # (an STS by a later LDG's read scoreboard: measured in probes/sb_mio_gap cov_ldg_rd)
    "rd_on_ldg": "clean",
    "rd_on_lds": "clean",
    "lds_nowait": "violation",
    "wcov": "clean",
    "wcov_nowait": "violation",
    "qwar_lds": "clean",
    "qwar_ldg": "clean",
    "qwar_ctl": "violation",
    "rvs_inorder_off": "clean",
    "rvs_inorder_off_nowait": "violation",
    "rvs_shfl_retain": "violation",
    "rvs_shfl_retain_ctl": "violation",
    "rvs_qctl_8": "violation",
    "rvs_qctl_11": "violation",
    "rvs_qctl_22": "violation",
    "rvs_qctl_33": "violation",
    "dq_lds": "clean",
    # (dq_ldg / dq_shfl / dq_ldsm: the window's Rt stays read by A until the skeleton's later
    # `LDG Rt`, an STS's address register written by an LDG.E inside R12's measured gaps, 4 to
    # 1024 stall cycles: probes/sb_mio_gap; strict: an sb-hwq: contract)
    "dq_ldg": "clean",
    "dq_shfl": "clean",
    "dq_ldsm": "clean",
    "dq_atoms_lds": "clean",
    "dq_ctl": "violation",
    "dq_ctl_8": "violation",
    "dq_ctl_22": "violation",
    "dq_ctl_33": "violation",
    "dq_atoms_ctl": "violation",
    "dq_own": "clean",
    "rd_on_a": "violation",
    "nowait": "violation",
}
# The strict level where it differs (E-HW-SB-QUEUE): another opcode's read or write scoreboard
# covering A, and another opcode's load waiting for A's read, hold at the gb300 level only:
# those windows are strict sb-hwq: contracts.
EXPECT_SASS = {
    "rd_on_ldg": '["contract"]',
    "rd_on_lds": '["contract"]',
    "lds_nowait": '["violation"]',
    "wcov": '["contract"]',
    "qwar_lds": '["contract"]',
    "qwar_ldg": '["contract"]',
    "dq_lds": '["contract"]',
    "dq_ldg": '["contract"]',
    "dq_shfl": '["contract"]',
    "dq_ldsm": '["contract"]',
    # (the window's Rt stays read by A until the kernel's later `LDG Rt`: a strict sb-hwq: WAR)
    "dq_ctl": '["contract", "violation"]',
    "dq_ctl_8": '["contract", "violation"]',
    "dq_ctl_22": '["contract", "violation"]',
    "dq_ctl_33": '["contract", "violation"]',
}
# The gap windows: the classes of their 4-cycle counterparts (the cover / WAR windows clean at
# gb300, strict sb-hwq: contracts; the no-wait and ALU controls violations).
GAP_CLASS = {
    "rd_on_lds": ("clean", '["contract"]'),
    "lds_nowait": ("violation", None),
    "qwar_lds": ("clean", '["contract"]'),
    "qwar_ldg": ("clean", '["contract"]'),
    "wcov": ("clean", '["contract"]'),
    "wcov_nowait": ("violation", None),
    "dq_lds": ("clean", '["contract"]'),
    "dq_ldg": ("clean", '["contract"]'),
    "dq_shfl": ("clean", '["contract"]'),
    "dq_ldsm": ("clean", '["contract"]'),
}
for v in VARIANTS:
    n, _, g = v.rpartition("_g")
    if v not in EXPECT and n in GAP_CLASS and g in ("8", "11", "22"):
        EXPECT[v], sass = GAP_CLASS[n]
        if sass:
            EXPECT_SASS[v] = sass
EXPECT.setdefault("dq_ctl_11", "violation")
EXPECT_SASS.setdefault("dq_ctl_11", '["contract", "violation"]')
# Hardware only (run.sh): `ATOMS.EXCH` is not lifted (modset.rs), so no fixture variant.
NO_FIXTURE = {"dq_atoms_lds", "dq_atoms_ctl"}
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
    if v in NO_FIXTURE:
        continue
    sass = f"expect_sass = {EXPECT_SASS[v]}\n" if v in EXPECT_SASS else ""
    exp = EXPECT[v] if isinstance(EXPECT[v], list) else [EXPECT[v]]
    exp = ", ".join(f'"{e}"' for e in exp)
    toml.append(f'[[variants]]\nid = "{v}"\nexpect = [{exp}]\n{sass}')
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
