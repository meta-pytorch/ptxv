# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# data/ of c_num_tma16_f32 (python3 gen.py data): x in [1, 2).
import random
import struct
import sys

f32 = lambda v: struct.unpack("<f", struct.pack("<f", v))[0]
r = random.Random(1150)
xs = [f32(r.uniform(1.0, 2.0)) for _ in range(32)]
open(f"{sys.argv[1]}/x.bin", "wb").write(struct.pack("<32f", *xs))
