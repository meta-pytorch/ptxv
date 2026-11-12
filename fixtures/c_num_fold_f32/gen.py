# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# data/ of c_num_fold_f32 (python3 gen.py data), the independent review's probe generator
# (seed 11; its first 32 draws are other probes' x): rows k = 0..2: x uniform in [-1.2, 1];
# rows k = 3..5: x = t 2^-20 - c_(k-3), c the folded constants rn(0.1f * 3), rn(0.1f * 0.1f)
# and ex2.approx.ftz(0.1f) = 1.0717734098434448, so the sums cancel to t 2^-20.
import random
import struct
import sys

f32 = lambda v: struct.unpack("<f", struct.pack("<f", v))[0]
r = random.Random(11)
for _ in range(32):
    r.uniform(1.0, 2.0)
c = [f32(f32(0.1) * 3.0), f32(f32(0.1) * f32(0.1)), f32(1.0717734098434448242)]
xs = [f32(r.uniform(-1.2, 1.0)) for _ in range(96)]
for k in range(3):
    xs += [f32(-(c[k] - t * 2.0**-20)) for t in range(32)]
open(f"{sys.argv[1]}/x.bin", "wb").write(struct.pack("<192f", *xs))
