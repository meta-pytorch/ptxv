# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Set the dividend's abs bits (bit 73: bit 9 of the high word, and bit 62 of the low word, as in
# ptxas's `FCHK P1, |R2|, |R15|` = 0x4000000f02007302 / 0x...0200) on the FCHK of
# probes/fchk/fchk.cubin, giving `FCHK P0, |Ra|, |Rb|`.
#   python3 patch.py ../fchk/fchk.cubin fchk_abs.cubin
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
(shoff,) = struct.unpack_from("<Q", b, 0x28)
shentsize, shnum, shstrndx = struct.unpack_from("<HHH", b, 0x3A)


def sh(i):
    return struct.unpack_from("<IIQQQQIIQQ", b, shoff + i * shentsize)


stroff = sh(shstrndx)[4]


def name(i):
    o = stroff + sh(i)[0]
    return b[o : b.index(0, o)].decode()


text = next(sh(i)[4] for i in range(shnum) if name(i) == ".text.fchk")
line = next(
    ln for ln in dis.split("\n") if re.match(r"\s+/\*[0-9a-f]{4}\*/\s+FCHK", ln)
)
addr = int(re.match(r"\s+/\*([0-9a-f]+)\*/", line).group(1), 16)
lo, hi = struct.unpack_from("<QQ", b, text + addr)
struct.pack_into("<QQ", b, text + addr, lo | 1 << 62, hi | 1 << 9)
open(dst, "wb").write(b)
print(f"patched {addr:#x}: {line.split(';')[0].strip()} -> |a|, |b|")
