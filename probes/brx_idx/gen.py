# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# probes/brx_idx input: 128 u32 (deterministic).
import struct
import sys

xs = [(0x9E3779B9 * (i + 5)) & 0xFFFFFFFF for i in range(128)]
open(sys.argv[1] if len(sys.argv) > 1 else "in.bin", "wb").write(
    struct.pack("<128I", *xs)
)
