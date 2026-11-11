# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Inputs for probes/mma_mxf8_tmem: A[128][128] and B[128][128] e4m3 codes (random, NaN codes
0x7f / 0xff excluded), SFA[128][16] and SFB[128][16] words of four ue8m0 scale bytes each
(codes 120..134). Writes A.bin, B.bin, SFA.bin, SFB.bin."""

import random
import struct

r = random.Random(20261108)


def code():
    while True:
        c = r.getrandbits(8)
        if c & 0x7F != 0x7F:
            return c


open("A.bin", "wb").write(bytes(code() for _ in range(128 * 128)))
open("B.bin", "wb").write(bytes(code() for _ in range(128 * 128)))
for name in ("SFA.bin", "SFB.bin"):
    w = [sum(r.randint(120, 134) << (8 * i) for i in range(4)) for _ in range(128 * 16)]
    open(name, "wb").write(struct.pack("<%dI" % len(w), *w))
