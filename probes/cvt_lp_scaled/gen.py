# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""in.bin for probes/cvt_lp_scaled: (a, s) pairs; a = i (every code pair), s = scale pair: lane bytes from a
mix (0x7f = 1.0, 0, 0xfe, 0xff = NaN scale, 0x80, 0x7e, random)."""

import random
import struct
import sys

r = random.Random(53)
SP = [0x7F, 0x00, 0x01, 0x7E, 0x80, 0x81, 0xFE, 0xFF, 0x40, 0xC0, 0x86, 0x77]
xs = []
for i in range(65536):
    lo = SP[i % len(SP)] if r.random() < 0.5 else r.getrandbits(8)
    hi = SP[(i // 7) % len(SP)] if r.random() < 0.5 else r.getrandbits(8)
    xs += [i, (hi << 8) | lo]
open(sys.argv[1] if len(sys.argv) > 1 else "in.bin", "wb").write(
    struct.pack(f"<{len(xs)}I", *xs)
)
