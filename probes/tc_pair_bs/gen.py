# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Inputs for probes/tc_pair_bs (mma_pair): random e2m1 codes (A 256 x 64, B 64 x 64, two per
byte, low nibble first) and random nonzero ue4m3 scale words per (CTA, lane, column)
(codes 0x30..0x47: 2^-1 .. 1.875 x 2)."""

import numpy as np

rng = np.random.default_rng(103)
rng.integers(0, 256, (256, 32), dtype=np.uint8).tofile("data/A.bin")
rng.integers(0, 256, (64, 32), dtype=np.uint8).tofile("data/B.bin")
rng.integers(0x30, 0x48, (2, 128, 4, 4), dtype=np.uint8).tofile("data/SFA.bin")
rng.integers(0x30, 0x48, (2, 128, 4, 4), dtype=np.uint8).tofile("data/SFB.bin")
