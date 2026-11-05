# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Set bit 63 of the two mbarrier.init words of kernel.ptx's cubin (argv[1] -> argv[2]): the
# `USHF.L.U32 URhi, URx, 0xb, URZ` computing each high word becomes `UMOV URhi, imm` with the
# count-n high word | 0x80000000 (UMOV encoding and control word copied from the
# `UMOV UR5, 0x1` ptxas emits).
import struct
import sys

b = bytearray(open(sys.argv[1], "rb").read())


def find(lo):
    i = b.find(struct.pack("<Q", lo))
    assert i >= 0 and i % 8 == 0, hex(lo)
    assert b.find(struct.pack("<Q", lo), i + 1) < 0
    return i


umov = find(0x0000000100057882)  # UMOV UR5, 0x1
hi = b[umov + 8 : umov + 16]
for shf, d, n in ((0x0000000B04057899, 5, 1), (0x0000000B06077899, 7, 2)):
    i = find(shf)
    word = ((0x100000 - n) << 11 | 0x80000000) & 0xFFFFFFFF
    b[i : i + 8] = struct.pack("<Q", word << 32 | d << 16 | 0x7882)
    b[i + 8 : i + 16] = hi
open(sys.argv[2], "wb").write(b)
