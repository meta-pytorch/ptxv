# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Reads driver constant-bank words that PTX cannot name: k.cu's `LDC R7, c[0x0][0x388]` (the
# parameter v) is patched to c[0x0][0x2d0] / 0x2d8 / 0x2dc (offset/4 in byte 5 of the word).
#   nvcc -cubin -gencode arch=compute_103a,code=sm_103a -o k.cubin k.cu
#   python3 patch.py k.cubin && nvcc -O2 -o run run.cu -lcuda && ./run
import sys

b = bytearray(open(sys.argv[1], "rb").read())
pat = bytes.fromhex("827b07ff00e20000")  # LDC R7, c[0x0][0x388]
i = b.find(pat)
assert i >= 0 and b.find(pat, i + 1) < 0
for off in (0x2D0, 0x2D8, 0x2DC):
    c = bytearray(b)
    c[i + 5] = off >> 2
    open(f"k{off:x}.cubin", "wb").write(c)
