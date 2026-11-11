# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Inputs for probes/div_ftz (same as probes/fmnmx_xorsign): every ordered pair (a, b) of 32 f32 edge patterns (in.bin,
1024 x 2 words): signed zeros, subnormals, the normal range ends, 448 and its neighbours,
infinities and NaNs with several payloads and signs."""

import numpy as np

E = [
    0x00000000,
    0x80000000,
    0x00000001,
    0x80000001,
    0x007FFFFF,
    0x807FFFFF,
    0x00800000,
    0x80800000,
    0x3F800000,
    0xBF800000,
    0x43E00000,
    0xC3E00000,
    0x43DFFFFF,
    0xC3E00001,
    0x3FC00000,
    0xC0200000,
    0x7F7FFFFF,
    0xFF7FFFFF,
    0x7F800000,
    0xFF800000,
    0x7FC00000,
    0xFFC00000,
    0x7F800001,
    0xFF800001,
    0x7FFFFFFF,
    0xFF812345,
    0x7FA00000,
    0x3EAAAAAB,
    0x447A0000,
    0xC2F60000,
    0x00400000,
    0x80000002,
]
assert len(E) == 32
a = np.array([x for x in E for _ in E], np.uint32)
b = np.array([y for _ in E for y in E], np.uint32)
np.stack([a, b], 1).tofile("in.bin")
