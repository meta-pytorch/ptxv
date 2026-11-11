# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Inputs of probes/hmnmx2: every f16 pattern in lane 0 of a against each edge
pattern of E in lane 0 of b; lane 1 of a is (p * 40503) mod 2^16 (a bijection), lane 1 of b
the next edge of E (2,621,440 threads, the exhaustive run). With `small`: every pair of
the 40 edges, then random pairs, 2048 threads (in.bin). usage: gen.py OUT [small]"""

import sys

import numpy as np

E = [
    0x0000,
    0x8000,
    0x0001,
    0x8001,
    0x03FF,
    0x83FF,
    0x0400,
    0x8400,
    0x3C00,
    0xBC00,
    0x3C01,
    0xBBFF,
    0x3800,
    0xB800,
    0x4000,
    0xC000,
    0x7BFF,
    0xFBFF,
    0x7C00,
    0xFC00,
    0x7E00,
    0xFE00,
    0x7C01,
    0xFC01,
    0x7FFF,
    0xFFFF,
    0x7D00,
    0x5640,
    0xD640,
    0x0200,
    0x8200,
    0x1234,
    0x9234,
    0x6543,
    0xE543,
    0x3555,
    0x2E66,
    0xAE66,
    0x7A00,
    0xFA00,
]
E = np.array(E, np.uint32)
out = sys.argv[1]
small = len(sys.argv) > 2
if small:
    # every edge pair, then 1024 - 1600 % 1024 random pairs: 2048 threads
    a0 = np.repeat(E, len(E))
    b0 = np.tile(E, len(E))
    rng = np.random.default_rng(3)
    k = 2048 - len(a0)
    a0 = np.concatenate([a0, rng.integers(0, 1 << 16, k, dtype=np.uint32)])
    b0 = np.concatenate([b0, rng.integers(0, 1 << 16, k, dtype=np.uint32)])
else:
    a0 = np.repeat(np.arange(1 << 16, dtype=np.uint32), len(E))
    b0 = np.tile(E, 1 << 16)
a1 = (a0 * 40503) & 0xFFFF
eidx = {int(e): i for i, e in enumerate(E)}
b1 = (
    np.array(
        [E[(eidx.get(int(x), int(x) % len(E)) + 1) % len(E)] for x in b0], np.uint32
    )
    if small
    else np.tile(np.roll(E, -1), 1 << 16)
)
np.stack([a0 | a1 << 16, b0 | b1 << 16], 1).astype(np.uint32).tofile(out)
print(out, len(a0))
