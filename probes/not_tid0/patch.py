# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Turn the probe's `BRA.U UP0, SKIP` into `BRA.U.NOT_TID0 SKIP` (argv[1] -> argv[2]): opcode
# 0x547 -> 0x947 in the low word, and the high word's flags 0x0b800000 (uniform predicate UP0)
# -> 0x03880000 (as in cuBLAS's `BRA.U.NOT_TID0`, encoding 0x000fea0003880000); offset and
# control bits kept.
import struct
import sys

b = bytearray(open(sys.argv[1], "rb").read())
hits = 0
for i in range(0, len(b) - 15, 16):
    lo, hi = struct.unpack_from("<QQ", b, i)
    if lo & 0xFFF == 0x547 and hi & 0xFFFFFFFF == 0x0B800000:
        lo = (lo & ~0xFFF) | 0x947
        hi = (hi & ~0xFFFFFFFF) | 0x03880000
        struct.pack_into("<QQ", b, i, lo, hi)
        hits += 1
assert hits == 1, hits
open(sys.argv[2], "wb").write(b)
