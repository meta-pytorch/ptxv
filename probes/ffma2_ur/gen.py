# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# in.bin: 512 f32 words: specials then random normals of magnitude 2^-20..2^20, seed 12.
import random
import struct

random.seed(12)
S = [
    0.0,
    -0.0,
    float("inf"),
    -float("inf"),
    float("nan"),
    1e-40,
    -1e-40,
    3.0e38,
    1.0,
    -1.0,
]
w = S + [
    random.choice([-1, 1]) * random.uniform(1, 2) * 2.0 ** random.randint(-20, 20)
    for _ in range(512 - len(S))
]
open("in.bin", "wb").write(struct.pack("<512f", *w))
