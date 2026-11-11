# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# probes/cvt_ftz input: 64 f32 words (deterministic): signed zeros, smallest / largest
# subnormals, smallest normals, NaN payloads (quiet, signaling, negative), infinities, 1 and
# its neighbors, values in (0, 1), above 1, below 0.
import random
import struct
import sys

r = random.Random(11)


def bits(v):
    return struct.unpack("<I", struct.pack("<f", v))[0]


ws = [
    0,
    0x80000000,
    1,
    0x80000001,
    0x007FFFFF,
    0x807FFFFF,
    0x00400000,
    0x00800000,
    0x80800000,
    0x7FC00000,
    0x7F800001,
    0xFFC00001,
    0x7FFFFFFF,
    0x7F800000,
    0xFF800000,
    bits(1.0),
    bits(1.0) + 1,
    bits(1.0) - 1,
    bits(-1.0),
    bits(0.5),
    0x3F7FFFFF,
    bits(2.0),
    bits(1e-38),
    bits(-1e-38),
    0x7F7FFFFF,
    0xFF7FFFFF,
]
while len(ws) < 64:
    ws.append(bits(r.uniform(-2, 2)) if len(ws) % 3 else r.getrandbits(32))
open(sys.argv[1] if len(sys.argv) > 1 else "in.bin", "wb").write(
    struct.pack("<64I", *ws)
)
