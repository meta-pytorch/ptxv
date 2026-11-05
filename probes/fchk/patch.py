# Rewrite the probe's `FSETP.GTU.AND P0, PT, Ra, Rb, PT` into `FCHK P0, Ra, Rb` (encoding from
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# ptxas's div.rn.f32 sequence: opcode 0x302, Ra bits 24..31, Rb bits 32..39, Pd bits 81..83),
# give it write scoreboard SB5 (FCHK is variable-latency; ptxas scoreboards it) and make the
# SEL that reads the predicate wait on SB5.
#   python3 patch.py fchk.cubin out.cubin
import re
import struct
import subprocess
import sys

src, dst = sys.argv[1], sys.argv[2]
b = bytearray(open(src, "rb").read())
dis = subprocess.run(
    ["/usr/local/cuda-13.0/bin/nvdisasm", "-c", "-hex", src],
    capture_output=True,
    text=True,
    check=True,
).stdout
# .text.fchk file offset (section header table).
(shoff,) = struct.unpack_from("<Q", b, 0x28)
shentsize, shnum, shstrndx = struct.unpack_from("<HHH", b, 0x3A)


def sh(i):
    return struct.unpack_from("<IIQQQQIIQQ", b, shoff + i * shentsize)


stroff = sh(shstrndx)[4]


def name(i):
    o = stroff + sh(i)[0]
    return b[o : b.index(0, o)].decode()


text = next(sh(i)[4] for i in range(shnum) if name(i) == ".text.fchk")
lines = [ln for ln in dis.split("\n") if re.match(r"\s+/\*[0-9a-f]{4}\*/", ln)]
fsetp = next(ln for ln in lines if "FSETP.GTU.AND P0, PT," in ln)
addr = int(re.match(r"\s+/\*([0-9a-f]+)\*/", fsetp).group(1), 16)
ra, rb = map(int, re.search(r"PT, R(\d+), R(\d+), PT", fsetp).groups())
lo, hi = struct.unpack_from("<QQ", b, text + addr)
ctrl = hi >> 41
ctrl = (ctrl & ~(7 << 5)) | (5 << 5)  # wbar = SB5
new_lo = 0x7302 | ra << 24 | rb << 32
new_hi = ctrl << 41  # Pd = P0, no other fields
struct.pack_into("<QQ", b, text + addr, new_lo, new_hi)
# Everything from the next instruction to the SEL reading P0 waits on SB5 (so FCHK has read its
# operands before the MOV after it overwrites one).
sel = next(
    ln
    for ln in lines
    if "SEL" in ln and int(re.match(r"\s+/\*([0-9a-f]+)\*/", ln).group(1), 16) > addr
)
sa = int(re.match(r"\s+/\*([0-9a-f]+)\*/", sel).group(1), 16)
for a in range(addr + 16, sa + 16, 16):
    lo2, hi2 = struct.unpack_from("<QQ", b, text + a)
    hi2 |= (1 << 5) << (41 + 11)
    struct.pack_into("<QQ", b, text + a, lo2, hi2)
open(dst, "wb").write(b)
print(f"patched {addr:#x} -> FCHK P0, R{ra}, R{rb}; wait SB5 at {sa:#x}")
