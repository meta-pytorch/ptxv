# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Inputs for probes/ldmatrix_ptx: a 32 x 32 b16 tile (in.bin), element (r, c) = r << 8 | c."""

import numpy as np

np.array(
    [[(r << 8) | c for c in range(32)] for r in range(32)], dtype=np.uint16
).tofile("in.bin")
