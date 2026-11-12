# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# fixtures/c_num_udiv_f32 input (deterministic): x, y in [0.5, 2) (f32), so that x + y is exact
# in f64.
import random
import struct
import sys

r = random.Random(64)
d = sys.argv[1] if len(sys.argv) > 1 else "data"
for name in ("x", "y"):
    v = [struct.unpack("<f", struct.pack("<f", r.uniform(0.5, 2)))[0] for _ in range(128)]
    open(f"{d}/{name}.bin", "wb").write(struct.pack("<128f", *v))
