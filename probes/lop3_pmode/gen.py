# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Inputs for probes/lop3_pmode: 1024 (a, b, c) u32 triples; half with a & ~b == 0, c of
random sign."""

import numpy as np

rng = np.random.default_rng(7)
a = rng.integers(0, 2**32, 1024, dtype=np.uint64).astype(np.uint32)
b = rng.integers(0, 2**32, 1024, dtype=np.uint64).astype(np.uint32)
b[::2] |= a[::2]
c = rng.integers(0, 2**32, 1024, dtype=np.uint64).astype(np.uint32)
np.stack([a, b, c], 1).tofile("in.bin")
