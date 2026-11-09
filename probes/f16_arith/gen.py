# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Inputs for probes/f16_arith: 1024 rows of three words (in.bin). The first 512 rows combine
f16 classes in every half: +-0, +-min subnormal, +-max subnormal, +-1, +-max normal, +-Inf,
quiet / signaling NaNs, 2^-14, 1.5, 65504 / 2; the rest are random words."""

import numpy as np

H = [
    0x0000,
    0x8000,
    0x0001,
    0x8001,
    0x03FF,
    0x83FF,
    0x3C00,
    0xBC00,
    0x7BFF,
    0xFBFF,
    0x7C00,
    0xFC00,
    0x7E00,
    0xFD01,
    0x0400,
    0x3E00,
]
rng = np.random.default_rng(19)
rows = []
for i in range(1024):
    if i < 512:
        w = [H[(i + 5 * k) % 16] | (H[(i // 16 + 3 * k) % 16] << 16) for k in range(3)]
    else:
        w = list(rng.integers(0, 2**32, 3, dtype=np.uint64))
    rows.append(w)
np.array(rows, dtype=np.uint64).astype(np.uint32).tofile("in.bin")
