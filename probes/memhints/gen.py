# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# in.bin: 256 threads x 32 u32 words of random.getrandbits(32), seed 7.
import random
import struct

random.seed(7)
n = 256 * 32
open("in.bin", "wb").write(
    struct.pack("<%dI" % n, *[random.getrandbits(32) for _ in range(n)])
)
