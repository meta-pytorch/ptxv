# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Inputs for probes/redux_queue: 256 threads x 4 words.
import struct
vals = []
for t in range(256):
    vals += [t * 7 + 1, (t * 13) ^ 0x55, 1000 - t, t * t % 977]
open("in.bin", "wb").write(struct.pack("<%di" % len(vals), *vals))
