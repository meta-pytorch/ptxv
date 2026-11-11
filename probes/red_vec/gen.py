# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# probes/red_vec input: 512 f32, small integers and halves (+-), so sums are exact in any order.
import random
import struct
import sys

r = random.Random(17)
xs = [r.randint(-64, 64) / 2.0 for _ in range(512)]
open(sys.argv[1] if len(sys.argv) > 1 else "in.bin", "wb").write(
    struct.pack("<512f", *xs)
)
