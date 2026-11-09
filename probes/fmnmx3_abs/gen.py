# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Inputs for probes/fmnmx3_abs: every ordered triple (a, b, c) of 16 f32 edge patterns (in.bin,
4096 x 4 words, the 4th unused): signed zeros, subnormals, ones, the normal range ends,
infinities and NaNs with several payloads and signs."""

import struct

E = [
    0x00000000,
    0x80000000,
    0x00000001,
    0x807FFFFF,
    0x00800000,
    0x3F800000,
    0xBF800000,
    0xC0200000,
    0x7F7FFFFF,
    0xFF7FFFFF,
    0x7F800000,
    0xFF800000,
    0x7FC00000,
    0xFFC00000,
    0x7F800001,
    0xFF812345,
]
assert len(E) == 16
w = [v for a in E for b in E for c in E for v in (a, b, c, 0)]
open("in.bin", "wb").write(struct.pack("<%dI" % len(w), *w))
