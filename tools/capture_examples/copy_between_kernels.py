# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# fixtures/c_copy_between_kernels (MISS 173), for tools/capture.py --file
# copy_between_kernels.py: kernel 1 (arange) writes x inside the call; a device copy
# (x.clone()) and a host upload (a CPU linspace, .cuda()) fill y and t between the kernels;
# kernel 2 (addcmul) reads all three: out = x + y * t.
import torch


def run():
    x = torch.arange(64, dtype=torch.float32, device="cuda")
    y = x.clone()
    t = torch.linspace(1.0, 2.0, 64).cuda()
    return torch.addcmul(x, y, t)
