# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# in.bin: 1024 u64: f64 specials (+-0, +-inf, NaNs, subnormals, near the overflow /
# underflow edges of 1 / x, int64 min / max) then random bits (1/2) and random f64 of
# magnitude 2^-1030..2^1030, seed 11.
import random
import struct

random.seed(11)
f = lambda x: struct.unpack("<Q", struct.pack("<d", x))[0]
E = [
    0,
    1 << 63,
    0x7FF0000000000000,
    0xFFF0000000000000,
    0x7FF8000000000000,
    0x7FF0000000000001,
    1,
    (1 << 63) | 1,
    0x000FFFFFFFFFFFFF,
    0x0010000000000000,
    0x0004000000000000,
    0x0003FFFFFFFFFFFF,
    f(2.0**1023),
    f(2.0**-1022),
    f(2.0**1022),
    f(1.0),
    f(-3.0),
    f(1.7976931348623157e308),
    (1 << 63) - 1,
    0x0002000000000001,
]
w = list(E)
while len(w) < 1024:
    if random.random() < 0.5:
        w.append(random.getrandbits(64))
    else:
        w.append(
            f(
                random.choice([-1, 1])
                * random.uniform(1, 2)
                * 2.0 ** random.randint(-1030, 1022)
            )
        )
open("in.bin", "wb").write(struct.pack("<1024Q", *w))
