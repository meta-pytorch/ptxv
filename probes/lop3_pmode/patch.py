# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Rewrite the probe's `LOP3.LUT P0, RZ, a, b, RZ, 0x30, !PT` predicate operand: bit 80 = the
# `.PAND` mode (clear: the default, an OR), bits 87-89 = the predicate, bit 90 = its negation
# (encoding read off cuBLAS's `LOP3.LUT.PAND P0, RZ, R3, UR4, RZ, 0x30, P0` and ptxas's `!PT`
# forms). Writes one cubin per form and checks nvdisasm prints it.
#   python3 patch.py kernel.cubin OUTDIR
import os
import re
import struct
import subprocess
import sys

src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
b0 = open(src, "rb").read()


def dis(f):
    return subprocess.run(
        ["/usr/local/cuda-13.0/bin/nvdisasm", "-c", "-hex", f],
        capture_output=True,
        text=True,
        check=True,
    ).stdout


(shoff,) = struct.unpack_from("<Q", b0, 0x28)
shentsize, shnum, shstrndx = struct.unpack_from("<HHH", b0, 0x3A)


def sh(i):
    return struct.unpack_from("<IIQQQQIIQQ", b0, shoff + i * shentsize)


stroff = sh(shstrndx)[4]


def name(i):
    o = stroff + sh(i)[0]
    return b0[o : b0.index(0, o)].decode()


text = next(sh(i)[4] for i in range(shnum) if name(i) == ".text.lop3_pmode")
lines = [ln for ln in dis(src).split("\n") if re.match(r"\s+/\*[0-9a-f]{4}\*/", ln)]
lop = next(ln for ln in lines if "LOP3.LUT P0, RZ," in ln and "!PT" in ln)
assert any("ISETP.GE.AND P1, PT," in ln for ln in lines[: lines.index(lop)])
addr = int(re.match(r"\s+/\*([0-9a-f]+)\*/", lop).group(1), 16)
for form, pand, neg in [
    ("pand_p1", 1, 0),
    ("pand_np1", 1, 1),
    ("por_p1", 0, 0),
    ("por_np1", 0, 1),
]:
    b = bytearray(b0)
    lo, hi = struct.unpack_from("<QQ", b, text + addr)
    hi &= ~((1 << 16) | (0xF << 23))
    hi |= pand << 16 | (1 | neg << 3) << 23
    struct.pack_into("<QQ", b, text + addr, lo, hi)
    f = os.path.join(out, f"{form}.cubin")
    open(f, "wb").write(b)
    got = next(ln for ln in dis(f).split("\n") if f"/*{addr:04x}*/" in ln)
    want = ("LOP3.LUT.PAND" if pand else "LOP3.LUT") + " P0, RZ,"
    tail = ("!P1" if neg else "P1") + " ;"
    assert want in got and tail in got, got
    print(form, got.split("/*")[1].split("*/")[1].strip())
