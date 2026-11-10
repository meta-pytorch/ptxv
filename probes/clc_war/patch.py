"""Mutants of base.cubin (probes/clc_war): `st1`: UGETNEXTWORKID with stall 1, then
`UIADD3 UR6, UPT, UPT, UR6, 0x10, URZ` (the response address register moved 16 bytes, for 15
cycles), `UIADD3 UR6, UPT, UPT, UR6, -0x10, URZ` (restored for the later LDS), then the loop's
`@P0 PLOP3` (the `PLOP3 P0` reset and the loop branch dropped: one thread runs the loop once). `st1_rb`: the same with read scoreboard SB2 on UGETNEXTWORKID, never waited on (as
ptxas emits it in TLX CLC kernels). `st1_bar`: as `st1` on the mbarrier operand, UR7 moved
16 bytes down (onto resp[16..32), zeros) for 15 cycles and restored; a late read would leave the
real mbarrier incomplete (the kernel hangs). usage: patch.py base.cubin OUTDIR"""

import struct
import sys

b = bytearray(open(sys.argv[1], "rb").read())
GET = 0x00000000060083CA
UIADD3 = 0x0000001006067890  # UIADD3 UR6, UPT, UPT, UR6, 0x10, URZ
UIADD3_CTL = 0x000FE2000FFFE0FF
UNDO = 0xFFFFFFF006067890  # UIADD3 UR6, UPT, UPT, UR6, -0x10, URZ


def at(off):
    return struct.unpack_from("<QQ", b, off)


offs = [
    o for o in range(0, len(b) - 16, 16) if struct.unpack_from("<Q", b, o)[0] == GET
]
assert len(offs) == 1, offs
g = offs[0]


def ctl(hi, st=None, rb=None):
    c = hi >> 41
    if st is not None:
        c = (c & ~0xF) | st
    if rb is not None:
        c = (c & ~(7 << 8)) | (rb << 8)
    return (hi & ((1 << 41) - 1)) | (c << 41)


BAR_DOWN = 0xFFFFFFF007077890  # UIADD3 UR7, UPT, UPT, UR7, -0x10, URZ
BAR_UP = 0x0000001007077890  # UIADD3 UR7, UPT, UPT, UR7, 0x10, URZ
for name, rb, (mv, undo) in (
    ("st1", None, (UIADD3, UNDO)),
    ("st1_rb", 2, (UIADD3, UNDO)),
    ("st1_bar", None, (BAR_DOWN, BAR_UP)),
):
    m = bytearray(b)
    lo, hi = at(g)
    struct.pack_into("<QQ", m, g, lo, ctl(hi, st=1, rb=rb))
    nlo, nhi = at(g + 16)
    struct.pack_into("<QQ", m, g + 16, mv, ctl(UIADD3_CTL, st=15) & ~(1 << 45))
    struct.pack_into("<QQ", m, g + 32, undo, UIADD3_CTL)
    struct.pack_into("<QQ", m, g + 48, nlo, nhi)
    open(f"{sys.argv[2]}/{name}.cubin", "wb").write(m)
    print(name)
