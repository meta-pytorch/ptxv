# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# A capture tools/capture.py refuses (MISS 173): kernel 1 (arange) writes x, a host upload then
# overwrites it, and kernel 2 reads it. x changes between two kept launches that reference it,
# by something that is not a kernel, which a case cannot express.
import torch


def run():
    x = torch.arange(64, dtype=torch.float32, device="cuda")
    x.copy_(torch.linspace(1.0, 2.0, 64))
    return x * 2
