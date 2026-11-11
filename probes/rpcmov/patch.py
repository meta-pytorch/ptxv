# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Insert `RPCMOV.32 Rpc.LO, Ra` after the probe's last LDG (Ra = the register loaded from
# [R.64], waiting on every scoreboard first) and `RPCMOV.32 Rx, Rpc.LO` after
# `IADD3 Rx, PT, PT, Ra, 0x1234, RZ`; later instructions move down one slot per insertion into
# the NOP padding (the kernel has no branches but the self-loop after EXIT). Encodings from
# ptxas's `RPCMOV.32 Rpc.LO, R2` = 0x0000000200007352 (source bits 32..39) and
# `RPCMOV.32 R2, Rpc.LO` = 0x0000000000027353 (destination bits 16..23). Both get stall 12,
# no scoreboards (as ptxas's).
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


sec = next(i for i in range(shnum) if name(i) == ".text.rpcmov")
text, size = sh(sec)[4], sh(sec)[5]
ins = [struct.unpack_from("<QQ", b, text + o) for o in range(0, size, 16)]
txt = {}
for ln in dis.split("\n"):
    m = re.match(r"\s+/\*([0-9a-f]{4})\*/\s+(.*?)\s*;", ln)
    if m:
        txt[int(m.group(1), 16) // 16] = m.group(2)
a = int(re.match(r"LDG\.E R(\d+), desc\[UR\d+\]\[R\d+\.64\]$", txt[9]).group(1))
x_at = next(
    k
    for k, t in txt.items()
    if re.fullmatch(rf"IADD3 R\d+, PT, PT, R{a}, 0x1234, RZ", t)
)
x = int(re.match(r"IADD3 R(\d+)", txt[x_at]).group(1))


def ctrl(wait):
    return (12 | 7 << 5 | 7 << 8 | wait << 11) << 41


save = (0x7352 | a << 32, ctrl(0x3F))
restore = (0x7353 | x << 16, ctrl(0))
out = ins[:10] + [save] + ins[10 : x_at + 1] + [restore] + ins[x_at + 1 :]
assert all(lo == 0x7918 for lo, _ in out[len(ins) :]), "no NOP padding to absorb"
for k, (lo, hi) in enumerate(out[: len(ins)]):
    struct.pack_into("<QQ", b, text + 16 * k, lo, hi)
open(dst, "wb").write(b)
print(f"saved R{a} after 0x90, restored into R{x} after {x_at * 16:#x}")
