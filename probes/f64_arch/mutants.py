# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The cubins/ mutants of probes/f64_arch (straight-line code).

usage: mutants.py IN OUT MODE
  st<N>     every f64 op (DFMA / DADD / DMUL / DSETP) stalls N cycles
  nosb<N>   f64 write scoreboards and the waits on them removed; f64 ops and the NOP padding
            stall N cycles
  tk<A>     the toolkit note's `-arch sm_*` replaced by `-arch A` (same length)

sm100f_st1 = st1 of sm100f, sm103a_nosb = nosb1 of sm103a, *_tk* = tk of those."""

import re
import struct
import subprocess
import sys

src, dst, mode = sys.argv[1:4]
img = bytearray(open(src, "rb").read())
if mode.startswith("tk"):
    i = img.index(b"-arch sm_") + len(b"-arch ")
    new = mode[2:].encode()
    assert len(new) == len(b"sm_100f")
    img[i : i + len(new)] = new
    open(dst, "wb").write(img)
    sys.exit(0)
# ELF64 section headers
shoff = struct.unpack_from("<Q", img, 0x28)[0]
shentsz, shnum, shstrndx = struct.unpack_from("<HHH", img, 0x3A)


def sh(i):
    return struct.unpack_from("<IIQQQQIIQQ", img, shoff + i * shentsz)


strtab_off = sh(shstrndx)[4]


def name(o):
    e = img.index(b"\0", strtab_off + o)
    return img[strtab_off + o : e].decode()


text = [sh(i) for i in range(shnum) if name(sh(i)[0]).startswith(".text.")]
assert len(text) == 1
toff, tsize = text[0][4], text[0][5]
dis = subprocess.run(
    ["/usr/local/cuda-13.0/bin/nvdisasm", "-c", src], capture_output=True, text=True
).stdout
ops = {}
for m in re.finditer(r"/\*([0-9a-f]{4})\*/\s+(\S+)", dis):
    ops[int(m.group(1), 16)] = m.group(2)
F64 = ("DFMA", "DADD", "DMUL", "DSETP")
last_writer = {}
nf = 0
for a in range(0, tsize, 16):
    op = ops.get(a, "")
    hoff = toff + a + 8
    hi = struct.unpack_from("<Q", img, hoff)[0]
    c = hi >> 41
    stall, yld, wbar, rbar, wait = (
        c & 15,
        (c >> 4) & 1,
        (c >> 5) & 7,
        (c >> 8) & 7,
        (c >> 11) & 63,
    )
    isf = op.split(".")[0] in F64
    if mode.startswith("st") and isf:
        stall = int(mode[2:])
        nf += 1
    if mode.startswith("nosb"):
        n = int(mode[4:])
        for k in range(6):
            if wait >> k & 1 and last_writer.get(k):
                wait &= ~(1 << k)
        if isf:
            if wbar != 7:
                last_writer[wbar] = True
            wbar = 7
            stall = n
            nf += 1
        elif wbar != 7:
            last_writer[wbar] = False
        if op == "NOP" and a < max(k for k, v in ops.items() if v == "EXIT"):
            stall = n
    c = stall | yld << 4 | wbar << 5 | rbar << 8 | wait << 11 | (c & ~((1 << 17) - 1))
    hi = (hi & ((1 << 41) - 1)) | c << 41
    struct.pack_into("<Q", img, hoff, hi)
open(dst, "wb").write(img)
print(f"{dst}: {nf} f64 ops patched")
