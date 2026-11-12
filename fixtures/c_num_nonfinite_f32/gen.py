# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# fixtures/c_num_nonfinite_f32 input (deterministic). Even lanes overflow: x = y = 3e38,
# u = v = -3e38, w2 = -inf. Odd lanes: x = 1 + t 2^-20, y = (t + 1) 2^-30 (below half an ulp
# of x), u = -x, v = -y, w2 = 0.25. w by t mod 4: +inf, -inf, NaN, 1.5 + t / 8. z = 0.5 + t / 32.
import struct
import sys

d = sys.argv[1] if len(sys.argv) > 1 else "data"
inf, nan = float("inf"), struct.unpack("<f", struct.pack("<I", 0x7FC00000))[0]
big = struct.unpack("<f", struct.pack("<f", 3e38))[0]
cols = {n: [] for n in ("x", "y", "u", "v", "w", "z", "w2")}
for t in range(32):
    if t % 2 == 0:
        x, y, w2 = big, big, -inf
    else:
        x, y, w2 = 1 + t * 2.0**-20, (t + 1) * 2.0**-30, 0.25
    cols["x"].append(x)
    cols["y"].append(y)
    cols["u"].append(-x)
    cols["v"].append(-y)
    cols["w"].append([inf, -inf, nan, 1.5 + t / 8][t % 4])
    cols["z"].append(0.5 + t / 32)
    cols["w2"].append(w2)
for n, vals in cols.items():
    open(f"{d}/{n}.bin", "wb").write(struct.pack("<32f", *vals))
