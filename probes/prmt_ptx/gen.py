# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# in.bin: 768 words of random.getrandbits(32), seed 4.
import random
import struct

random.seed(4)
open("in.bin", "wb").write(
    struct.pack("<768I", *[random.getrandbits(32) for _ in range(768)])
)
