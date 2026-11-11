# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# probes/f64_misc input: 256 (a, b) f64 pairs (deterministic): all pairs of 15 specials
# (signed zeros, infinities, quiet / signaling / negative NaNs with payloads, subnormals,
# +-1, a large normal), then random words and random finite values.
import random
import struct
import sys

r = random.Random(23)


def bits(v):
    return struct.unpack("<Q", struct.pack("<d", v))[0]


sp = [
    0,
    1 << 63,
    bits(float("inf")),
    bits(-float("inf")),
    0x7FF8000000000000,
    0x7FF8000000001234,
    0x7FF0000000000001,
    0xFFF8000000000005,
    0xFFF4000000000000,
    1,
    (1 << 63) | 0x000FFFFFFFFFFFFF,
    bits(1.0),
    bits(-1.0),
    bits(1.5e300),
    bits(-2.5),
]
ws = [x for a in sp for b in sp for x in (a, b)]
while len(ws) < 512:
    if len(ws) % 4:
        ws += [r.getrandbits(64), r.getrandbits(64)]
    else:
        ws += [bits(r.uniform(-4, 4)), bits(r.uniform(-4, 4))]
open(sys.argv[1] if len(sys.argv) > 1 else "in.bin", "wb").write(
    struct.pack("<512Q", *ws[:512])
)
