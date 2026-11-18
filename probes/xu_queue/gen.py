# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Inputs for probes/xu_queue: 256 threads x 4 floats (a, b, c, d), a away from 0.
import struct
vals = []
for t in range(256):
    a = 1.0 + (t % 97) * 0.37
    vals += [a, t * 0.75 - 40.3, 0.5 + t * 1.25, 3.0 + (t % 13)]
open("in.bin", "wb").write(struct.pack("<%df" % len(vals), *vals))
