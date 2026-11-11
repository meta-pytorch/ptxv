# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# probes/local_vec input: 128 random words (deterministic).
import random
import struct
import sys

r = random.Random(47)
open(sys.argv[1] if len(sys.argv) > 1 else "in.bin", "wb").write(
    struct.pack("<128I", *[r.getrandbits(32) for _ in range(128)])
)
