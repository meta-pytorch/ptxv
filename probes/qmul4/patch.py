# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Patch the two LOP3 placeholders of base.cubin (usage: patch.py base.cubin k.cubin) into QMUL4.E4M3x4.E2M1x4.E4M3x4
R6, R2.N3210, R3.B0 and R7, R2.N7654, R3.B0 (encodings from a trtllm-gen sm_103a cubin, with
the register fields set), stall 15 each, the placeholder's wait mask, no reuse."""

import struct
import sys

src, dst = sys.argv[1], sys.argv[2]
img = bytearray(open(src, "rb").read())
shoff = struct.unpack_from("<Q", img, 0x28)[0]
es, n, si = struct.unpack_from("<HHH", img, 0x3A)
sh = lambda i: struct.unpack_from("<IIQQQQIIQQ", img, shoff + i * es)
so = sh(si)[4]
toff = next(
    sh(i)[4]
    for i in range(n)
    if img[so + sh(i)[0] : img.index(b"\0", so + sh(i)[0])] == b".text.qmul4"
)


def q(at, rd, ra, rb, sel_hi):
    o = toff + at
    old_hi = struct.unpack_from("<Q", img, o + 8)[0]
    c = old_hi >> 41
    wait = (c >> 11) & 63
    ctrl = 15 | (0 << 4) | (7 << 5) | (7 << 8) | (wait << 11)
    lo = 0x2000000000000000 | (rb << 32) | (ra << 24) | (rd << 16) | 0x727C
    hi = (0x0101B400 if sel_hi else 0x0101B800) | (ctrl << 41)
    struct.pack_into("<QQ", img, o, lo, hi)


q(0x0B0, 6, 2, 3, False)
q(0x0C0, 7, 2, 3, True)
open(dst, "wb").write(img)
