# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# fixtures/c_num_nan2_f32 input (deterministic; the independent review's probe nan2).
# out[t] = x + y (32 lanes) and out[32 + t] = (x + y) + (u + v). Even lanes: x = y = 3e38, u = v = -3e38
# (x + y overflows to +inf; (x + y) + (u + v) = inf - inf = NaN; the exact values are 6e38 and 0).
# Odd lanes: uniform in [1, 2) (finite sums with rounding error).
import random, struct, sys
f32 = lambda v: struct.unpack("<f", struct.pack("<f", v))[0]
r = random.Random(11)
cols = {n: [] for n in "xyuv"}
for t in range(32):
    for n in "xyuv":
        if t % 2 == 0:
            cols[n].append(f32(3e38) if n in "xy" else f32(-3e38))
        else:
            cols[n].append(f32(r.uniform(1, 2)))
for n, v in cols.items():
    open(f"{sys.argv[1] if len(sys.argv) > 1 else 'data'}/{n}.bin", "wb").write(struct.pack("<32f", *v))
