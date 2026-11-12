# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# fixtures/c_num_udiv_data_f32 input (deterministic): x in [0.5, 2) (f32); idx: 32 u32 indices
# in [0, 2^20).
import random
import struct
import sys

r = random.Random(65)
d = sys.argv[1] if len(sys.argv) > 1 else "data"
x = [struct.unpack("<f", struct.pack("<f", r.uniform(0.5, 2)))[0] for _ in range(128)]
idx = [r.randrange(1 << 20) for _ in range(32)]
open(f"{d}/x.bin", "wb").write(struct.pack("<128f", *x))
open(f"{d}/idx.bin", "wb").write(struct.pack("<32I", *idx))
