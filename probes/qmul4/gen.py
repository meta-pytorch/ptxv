# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Inputs of the QMUL4 probe: 4096 threads (a, b). Thread t: scale s = t mod 256 in byte 0 of b
(bytes 1..3 a hash of t, to show they are ignored); group g = t / 256: nibble p of a holds
8 (g mod 2) + (p + g / 2) mod 8, so every e2m1 code meets every e4m3 scale at every nibble
position."""

import struct

out = []
for t in range(4096):
    s, g = t & 255, t >> 8
    a = 0
    for p in range(8):
        a |= (8 * (g & 1) + (p + (g >> 1)) % 8) << (4 * p)
    h = (t * 0x9E3779B1) & 0xFFFFFF00
    out.append(struct.pack("<II", a, h | s))
open("in.bin", "wb").write(b"".join(out))
