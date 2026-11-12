# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# data/ of c_num_fold_add_f32 (python3 gen.py data): rows k = 0, 1: x uniform in [-1.2, 1];
# rows k = 2, 3: x = t 2^-20 - c, c the folded constant of row k - 2 (rn(0.1f + 1) and
# rn(1 + 2^-24) = 1), so the sums cancel to t 2^-20.
import random
import struct
import sys

f32 = lambda v: struct.unpack("<f", struct.pack("<f", v))[0]
r = random.Random(108)
xs = [f32(r.uniform(-1.2, 1.0)) for _ in range(64)]
for c in [f32(f32(0.1) + 1.0), 1.0]:
    xs += [f32(t * 2.0**-20 - c) for t in range(32)]
open(f"{sys.argv[1]}/x.bin", "wb").write(struct.pack("<128f", *xs))
