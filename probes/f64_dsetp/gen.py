# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# probes/f64_dsetp input: 32 x 4 f64 (deterministic): normals, equal pairs, specials.
import random
import struct
import sys

r = random.Random(5)
xs = [r.gauss(0, 1) for _ in range(128)]
sp = [0.0, -0.0, float("inf"), -float("inf"), float("nan"), 1.0]
for t in range(32):
    if t % 4 == 1:
        xs[4 * t + 1] = xs[4 * t]
    elif t % 4 == 2:
        xs[4 * t] = sp[t % 6]
        xs[4 * t + 1] = sp[(t // 6) % 6]
open(sys.argv[1] if len(sys.argv) > 1 else "in.bin", "wb").write(
    struct.pack("<128d", *xs)
)
