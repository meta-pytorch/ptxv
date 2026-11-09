# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Rewrite each `IMAD.X Rd, Rs, 0x1, ~Ry, Pk` of the probe (the high word of
# `(sext(a) << s) - y`, after `LEA Rlo, Pk, Ra, -Rx, s`) into `LEA.HI.X.SX32 Rd, Ra, ~Ry, s, Pk`,
# which computes the same word. Encoding from ptxas's `LEA.HI.X.SX32 R10, R2, ~R5, 0x1, P0`
# (triton _triton_fsdp_chunk_cat: 0x80000005020a7211 / 0x000fe200000f0eff): Rd bits 16..23,
# Ra 24..31, Rb 32..39, ~Rb bit 63; in the high word the shift is bits 11..15 and the carry-in
# predicate bits 23..25. The control bits (hi >> 41) are the IMAD.X's.
#   python3 patch.py in.cubin out.cubin
import re
import struct
import subprocess
import sys

src, dst = sys.argv[1], sys.argv[2]
b = bytearray(open(src, "rb").read())
dis = subprocess.run(
    ["/usr/local/cuda-13.0/bin/nvdisasm", "-c", src],
    capture_output=True,
    text=True,
    check=True,
).stdout
(shoff,) = struct.unpack_from("<Q", b, 0x28)
shentsize, shnum, shstrndx = struct.unpack_from("<HHH", b, 0x3A)


def sh(i):
    return struct.unpack_from("<IIQQQQIIQQ", b, shoff + i * shentsize)


stroff = sh(shstrndx)[4]


def name(i):
    o = stroff + sh(i)[0]
    return b[o : b.index(0, o)].decode()


text = next(sh(i)[4] for i in range(shnum) if name(i) == ".text.int16x2")
lea = {}
n = 0
for ln in dis.split("\n"):
    m = re.match(r"\s+/\*([0-9a-f]{4})\*/\s+(.*?)\s*;", ln)
    if not m:
        continue
    addr, ins = int(m.group(1), 16), m.group(2).replace(".reuse", "")
    if mm := re.fullmatch(r"LEA R\d+, P(\d), R(\d+), -R\d+, (0x[0-9a-f]+)", ins):
        lea[int(mm.group(1))] = (int(mm.group(2)), int(mm.group(3), 16))
    elif mm := re.fullmatch(r"IMAD\.X R(\d+), R\d+, 0x1, ~R(\d+), P(\d)", ins):
        d, y, p = map(int, mm.groups())
        a, s = lea[p]
        lo, hi = struct.unpack_from("<QQ", b, text + addr)
        new_lo = 0x8000000000007211 | y << 32 | a << 24 | d << 16
        new_hi = (hi >> 41) << 41 | 0x0F06FF | s << 11 | p << 23
        struct.pack_into("<QQ", b, text + addr, new_lo, new_hi)
        print(f"{addr:#06x}: {ins} -> LEA.HI.X.SX32 R{d}, R{a}, ~R{y}, {s:#x}, P{p}")
        n += 1
assert n == 2, n
open(dst, "wb").write(b)
