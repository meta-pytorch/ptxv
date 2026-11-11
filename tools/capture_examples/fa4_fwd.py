# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# FlashAttention-4 (CuTeDSL, flash_attn.cute) forward, bf16, 1 x FA4_SEQ x 1 x FA4_HDIM
# (default 128 x 64: fixtures/c_fa4_fwd; FA4_SEQ=256 FA4_HDIM=128: fixtures/c_fa4_hd128).
# FA4_MASK=causal / local (window (64, 0)): fixtures/c_fa4_causal / c_fa4_local.
import os

import torch
from flash_attn.cute.interface import flash_attn_func


def inputs():
    g = torch.Generator().manual_seed(0)
    mk = lambda *s: torch.randn(*s, generator=g).to(torch.bfloat16).cuda()
    t, d = int(os.environ.get("FA4_SEQ", 128)), int(os.environ.get("FA4_HDIM", 64))
    return {"q": mk(1, t, 1, d), "k": mk(1, t, 1, d), "v": mk(1, t, 1, d)}


def run(q, k, v):
    mask = os.environ.get("FA4_MASK", "")
    if mask == "causal":
        return flash_attn_func(q, k, v, causal=True)
    if mask == "local":
        return flash_attn_func(q, k, v, window_size=(64, 0))
    return flash_attn_func(q, k, v)
