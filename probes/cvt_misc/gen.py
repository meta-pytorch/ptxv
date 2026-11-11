# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# d.bin: 1024 f64 words, f.bin: 1024 f32 words: special patterns (zeros, subnormals, f32 range
# edges, ties, NaN, inf) then random values scaled by 2^k. Seed 9.
import random
import struct

random.seed(9)
sd = [
    0.0,
    -0.0,
    1e-310,
    -1e-310,
    1e-45,
    1.4e-45,
    3.4028235e38,
    3.4028236e38,
    3.5e38,
    1e300,
    -1e300,
    float("inf"),
    -float("inf"),
    float("nan"),
    1.0000000596046448,
    2.5,
    -2.5,
]
sf = [
    0.0,
    -0.0,
    1e-40,
    -1e-40,
    0.5,
    1.5,
    2.5,
    -0.5,
    -1.5,
    0.49999997,
    1.0,
    1.0000001,
    -1.0,
    2147483520.0,
    2147483648.0,
    -2147483904.0,
    4294967296.0,
    float("inf"),
    -float("inf"),
    float("nan"),
    3.4e38,
    -3.4e38,
    1e-38,
]
d = sd + [
    random.gauss(0, 1) * 2.0 ** random.randint(-140, 140) for _ in range(1024 - len(sd))
]
f = sf + [
    random.gauss(0, 1) * 2.0 ** random.randint(-30, 34) for _ in range(1024 - len(sf))
]
open("d.bin", "wb").write(struct.pack("<1024d", *d))
open("f.bin", "wb").write(
    b"".join(
        struct.pack("<f", x)
        if abs(x) < 3.5e38 or x != x or abs(x) == float("inf")
        else struct.pack("<I", 0x7F7FFFFF)
        for x in f
    )
)
