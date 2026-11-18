# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Data for fixtures/e_atom_half: 64 threads (two warps); lanes 3 and 4 of warp 0 add to
elements 0 and 1 (both halves of word 0) and lane 8 of warp 1 (thread 40) to element 0; the
packed adds all go to word 0; every other thread is inactive (index -1). The addends (1, 0.5,
0.25) and the initial ones sum exactly in any order, so the outputs do not depend on the
schedule."""

import numpy as np

idx = np.full(64, -1, np.int32)
vh = np.zeros(64, np.uint16)
vb = np.zeros(64, np.uint16)
for lane, i, h, b in [(3, 0, 0x3C00, 0x3F80), (4, 1, 0x3800, 0x3F00), (40, 0, 0x3400, 0x3E80)]:
    idx[lane], vh[lane], vb[lane] = i, h, b
idx.tofile("data/idx.bin")
vh.tofile("data/vh.bin")
vb.tofile("data/vb.bin")
np.array([0x3C00] * 3, np.uint16).tofile("data/h1.bin")
np.array([0x3F80] * 3, np.uint16).tofile("data/b1.bin")
np.array([0x3C003C00], np.uint32).tofile("data/h2.bin")
np.array([0x3F803F80], np.uint32).tofile("data/b2.bin")
