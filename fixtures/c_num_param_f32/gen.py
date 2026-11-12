# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# fixtures/c_num_param_f32 input: x[t] = 0 for even t (the sums are then the rounded values
# themselves), uniform in [-1, 1) for odd t (deterministic).
import random
import struct
import sys

r = random.Random(36)
xs = [0.0 if t % 2 == 0 else struct.unpack("<f", struct.pack("<f", r.uniform(-1, 1)))[0] for t in range(32)]
open(sys.argv[1] if len(sys.argv) > 1 else "data/x.bin", "wb").write(struct.pack("<32f", *xs))
