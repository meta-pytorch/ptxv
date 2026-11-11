# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# probes/tc_mma_ovf inputs: A [128 x 16] and B [256 x 16] bf16 (deterministic), A mostly and
# B all of magnitude 2^[58, 65): many dot products reach 2^128 (finite overflow).
import os
import random
import struct
import sys

r = random.Random(3)


def bf(v):
    x = struct.unpack("<I", struct.pack("<f", v))[0]
    return ((x + 0x7FFF + ((x >> 16) & 1)) >> 16) & 0xFFFF


A = [
    bf(r.choice([1, -1]) * (1 + r.random()) * 2.0 ** r.randint(58, 64))
    if r.random() < 0.8
    else bf(r.uniform(-2, 2))
    for _ in range(128 * 16)
]
B = [
    bf(r.choice([1, 1, 1, -1]) * (1 + r.random()) * 2.0 ** r.randint(58, 64))
    for _ in range(256 * 16)
]
d = sys.argv[1] if len(sys.argv) > 1 else "data"
open(os.path.join(d, "A.bin"), "wb").write(struct.pack("<%dH" % len(A), *A))
open(os.path.join(d, "B.bin"), "wb").write(struct.pack("<%dH" % len(B), *B))
