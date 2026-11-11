# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Mutants of base.cubin (probes/stg_guard_war). Store A `STG.E.64 [R2.64], R14` keeps no read
scoreboard; store B `@P0 STG.E [R4.64+0x4], R13` gets read scoreboard SB0; the next instruction
rewrites one of A's operands: `data`: `LOP3.LUT R14, RZ, R14, RZ, 0x33, !PT` (R14 = ~R14);
`addr`: `IADD3 R2, P1, PT, R2, 0x8, RZ`. `_w0`: the rewrite waits on SB0 (B's read scoreboard)
only; `_nw`: on nothing (sensitivity control). usage: patch.py base.cubin OUTDIR"""

import struct
import sys

b = bytearray(open(sys.argv[1], "rb").read())
A = 0x0000000E02007986
B = 0x0000040D04000986
offs = [o for o in range(0, len(b) - 16, 16) if struct.unpack_from("<Q", b, o)[0] == A]
assert len(offs) == 1, offs
a = offs[0]
assert struct.unpack_from("<Q", b, a + 16)[0] == B
REW = {
    "data": (0x0000000EFF0E7212, 0x000FE200078E33FF),
    "addr": (0x0000000802027810, 0x000FE20007F3E0FF),
}


def ctl(hi, rb=None, wait=None):
    c = hi >> 41
    if rb is not None:
        c = (c & ~(7 << 8)) | (rb << 8)
    if wait is not None:
        c = (c & ~(63 << 11)) | (wait << 11)
    return (hi & ((1 << 41) - 1)) | (c << 41)


for kind, (lo, hi) in REW.items():
    for w, mask in (("w0", 1), ("nw", 0)):
        m = bytearray(b)
        blo, bhi = struct.unpack_from("<QQ", m, a + 16)
        struct.pack_into("<QQ", m, a + 16, blo, ctl(bhi, rb=0))
        struct.pack_into("<QQ", m, a + 32, lo, ctl(hi, wait=mask))
        open(f"{sys.argv[2]}/{kind}_{w}.cubin", "wb").write(m)
        print(kind, w)
