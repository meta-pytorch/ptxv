# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Inputs for probes/bf16_abs: 1024 words (in.bin) whose halves pair the bf16 classes +-0,
+-subnormal, +-normal, +-Inf, quiet / signaling NaNs of both signs, then random words."""

import numpy as np

H = [
    0x0000,
    0x8000,
    0x0001,
    0x8001,
    0x007F,
    0x807F,
    0x3F80,
    0xBF80,
    0x7F7F,
    0xFF7F,
    0x7F80,
    0xFF80,
    0x7FC0,
    0xFFC0,
    0x7F81,
    0xFF81,
]
rng = np.random.default_rng(17)
w = [H[i % 16] | (H[(i // 16) % 16] << 16) for i in range(256)]
w += list(rng.integers(0, 2**32, 768, dtype=np.uint64))
np.array(w, dtype=np.uint64).astype(np.uint32).tofile("in.bin")
