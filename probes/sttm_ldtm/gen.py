# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Inputs for probes/sttm_ldtm: 128 threads x 32 random words (in.bin; the kernel reads the
first 16 of each row)."""

import numpy as np

np.random.default_rng(13).integers(0, 2**32, size=(128, 32), dtype=np.uint64).astype(
    np.uint32
).tofile("in.bin")
