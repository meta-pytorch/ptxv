# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# The captured inputs of ballot/ and plain/: x, w, v ~ N(0, 1) as f32 (seed 65), every |x| < 3
# (x x < 9), so the captured ballot is 0. Run from this directory: python3 gen.py
import random, struct
r = random.Random(65)
f32 = lambda v: struct.unpack("<f", struct.pack("<f", v))[0]
for n in ("x", "w", "v"):
    vals = [f32(r.gauss(0.0, 1.0)) for _ in range(32)]
    assert n != "x" or max(abs(v) for v in vals) < 3.0
    open(f"data/{n}.bin", "wb").write(struct.pack("<32f", *vals))
