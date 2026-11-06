# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Turn the probe's `LDC Rx, c[0x0][Rx+0x2a8]` (the %cluster_nctaid.z read) into a NOP that keeps
# the LDC's control bits (hi bits 41-63: stall, yield, scoreboards), so that Rx keeps
# `SR_CgaSize * -0xa0` from the IMAD before it and word 0 of each CTA's record is the raw
# special register.
#   python3 patch.py kernel.cubin patched.cubin
import re
import struct
import subprocess
import sys

src, out = sys.argv[1], sys.argv[2]
b = bytearray(open(src, "rb").read())


def dis(f):
    return subprocess.run(
        ["/usr/local/cuda-13.0/bin/nvdisasm", "-c", "-hex", f],
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


text = next(sh(i)[4] for i in range(shnum) if name(i) == ".text.k")
lines = [ln for ln in dis(src).split("\n") if re.match(r"\s+/\*[0-9a-f]{4}\*/", ln)]
ldc = [ln for ln in lines if re.search(r"LDC R\d+, c\[0x0\]\[R\d+\+0x2a8\]", ln)]
assert len(ldc) == 1, ldc
reg = re.search(r"LDC (R\d+),", ldc[0]).group(1)
addr = int(re.match(r"\s+/\*([0-9a-f]+)\*/", ldc[0]).group(1), 16)
prev = lines[lines.index(ldc[0]) - 1]
assert f"IMAD {reg}, {reg}, -0xa0, RZ" in prev, prev
nop = next(ln for ln in lines if re.search(r"\bNOP\b", ln))
naddr = int(re.match(r"\s+/\*([0-9a-f]+)\*/", nop).group(1), 16)
nlo, nhi = struct.unpack_from("<QQ", b, text + naddr)
lo, hi = struct.unpack_from("<QQ", b, text + addr)
ctrl = ((1 << 23) - 1) << 41
struct.pack_into("<QQ", b, text + addr, nlo, (nhi & ~ctrl) | (hi & ctrl))
open(out, "wb").write(b)
got = next(ln for ln in dis(out).split("\n") if f"/*{addr:04x}*/" in ln)
assert "NOP" in got, got
print(
    f"{addr:#06x}: {got.split('*/')[1].split(';')[0].strip()}  ({reg} = SR_CgaSize * -0xa0)"
)
