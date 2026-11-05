# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# F.scaled_dot_product_attention through the flash backend (pytorch_flash::flash_fwd_kernel).
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend


def inputs():
    g = torch.Generator().manual_seed(0)
    return {n: torch.randn(1, 1, 128, 64, generator=g).bfloat16().cuda() for n in "qkv"}


def run(q, k, v):
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        return F.scaled_dot_product_attention(q, k, v)
