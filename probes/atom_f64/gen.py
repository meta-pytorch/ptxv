# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# mem.bin: acc (256 f64) then zeros for old (256) and sum (4); b.bin: 256 f64. Pairs: special pairs (NaN, inf - inf, subnormal sums, ties at the
# last bit, overflow) then random values of nearby and distant magnitudes, seed 13.
import random
import struct

random.seed(13)
inf = float("inf")
P = [
    (1.0, 2.0**-53),
    (1.0, 3 * 2.0**-53),
    (inf, -inf),
    (float("nan"), 1.0),
    (5e-324, 5e-324),
    (-5e-324, 5e-324),
    (1.7976931348623157e308, 1e292),
    (0.0, -0.0),
    (-0.0, -0.0),
    (2.0**-1022, -(2.0**-1023)),
]
a = [p[0] for p in P]
b = [p[1] for p in P]
while len(a) < 256:
    e = random.randint(-60, 60)
    a.append(random.choice([-1, 1]) * random.uniform(1, 2) * 2.0**e)
    b.append(
        random.choice([-1, 1])
        * random.uniform(1, 2)
        * 2.0 ** (e + random.randint(-60, 3))
    )
open("mem.bin", "wb").write(struct.pack("<516d", *(a + [0.0] * 260)))
open("b.bin", "wb").write(struct.pack("<256d", *b))
