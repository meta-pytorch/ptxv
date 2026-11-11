# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# probes/float_lit input: 32 f32 values (deterministic): 0, +-20 and its neighbors, 1e-300's
# tiny f32 neighbors, random values in [-40, 40].
import random
import struct
import sys

r = random.Random(5)


def bits(v):
    return struct.unpack("<I", struct.pack("<f", v))[0]


ws = [
    0,
    bits(20.0),
    bits(20.0) + 1,
    bits(20.0) - 1,
    bits(-20.0),
    1,
    0x80000001,
    bits(1.0),
]
ws += [bits(r.uniform(-40, 40)) for _ in range(24)]
open(sys.argv[1] if len(sys.argv) > 1 else "in.bin", "wb").write(
    struct.pack("<32I", *ws)
)
