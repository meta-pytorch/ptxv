# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# data/ of c_num_vote_f32 (python3 gen.py data): x, y with x*y just below 1 and rn(x*y) = 1 in
# every lane; z in [4, 8).
import random, struct, sys
f32 = lambda v: struct.unpack("<f", struct.pack("<f", v))[0]
r = random.Random(7)
xs, ys = [], []
for t in range(32):
    while True:
        x = f32(r.uniform(1.0, 2.0)); y = f32((1.0 - 2.0 ** -26) / x)
        p = x * y  # exact in double
        if p < 1.0 and f32(p) == 1.0:
            break
    xs.append(x); ys.append(y)
zs = [f32(r.uniform(4.0, 8.0)) for _ in range(32)]
d = sys.argv[1]
for n, v in (("x", xs), ("y", ys), ("z", zs)):
    open(f"{d}/{n}.bin", "wb").write(struct.pack("<32f", *v))
