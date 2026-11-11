# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# probes/applypriority input: 64 random u32 words (deterministic).
import random
import struct
import sys

r = random.Random(11)
open(sys.argv[1] if len(sys.argv) > 1 else "in.bin", "wb").write(
    struct.pack("<64I", *[r.getrandbits(32) for _ in range(64)])
)
