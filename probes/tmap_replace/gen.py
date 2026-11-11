# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# probes/tmap_replace input: 32 maps x 32 random words, then 32 x 16 random operand words.
import random
import struct
import sys

r = random.Random(17)
xs = [r.getrandbits(32) for _ in range(32 * 32 + 32 * 16)]
open(sys.argv[1] if len(sys.argv) > 1 else "in.bin", "wb").write(
    struct.pack("<%dI" % len(xs), *xs)
)
