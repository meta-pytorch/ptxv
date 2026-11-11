# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""check.py RUN.bin...: per run, threads and the stale A / stale B reads summed over them
(out = {stale A, stale B} per thread, `iters` rounds each; warp 0's own reads included)."""

import struct
import sys

for f in sys.argv[1:]:
    d = open(f, "rb").read()
    w = struct.unpack("<%dI" % (len(d) // 4), d)
    print(f"{f}: {len(w) // 2} threads; stale A {sum(w[0::2])}; stale B {sum(w[1::2])}")
