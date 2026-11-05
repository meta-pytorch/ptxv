# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Rung 3b: flashinfer 0.5.3's trtllm-gen fp8 block-scale MoE (DeepSeek-style: e4m3 data, f32
# scales per 128 x 128 weight block and per 128 hidden channels of a token). Five launches:
# routing, the gemm1 batched GEMM (`bmm_E4m3_E4m3E4m3_Fp32_..._dsFp8_..._bN_tma_..._dynBatch`:
# routed tokens gathered with TMA gather4, software promotion per 128-K block, e4m3 output with
# per-token amax scales), activation, gemm2, finalize. Keep only gemm1; the launch recorder
# snapshots its buffers (tools/capture.py, --kernels):
#   CUDA_VISIBLE_DEVICES=3 $PTXV_PYTHON tools/capture.py --file tools/capture_examples/trtllm_fp8_moe.py \
#       --kernels '^bmm_E4m3' --out fixtures/r3b_trtllm_bmm
# T = 16 tokens, top-2 of E = 4 experts, routed so that every expert gets exactly 8 tokens (one
# full 8-token tile each). PTXV_MOE_PARTIAL=1: T = 8 with random routing (partial tiles).
import os

import torch
from flashinfer.fused_moe import trtllm_fp8_block_scale_moe

PARTIAL = os.environ.get("PTXV_MOE_PARTIAL") == "1"
# PTXV_MOE_H: hidden size (gemm1's K); 2048 = 16 K-blocks, so gemm1's 8 smem stages wrap.
T, H, I, E, K = (
    (8 if PARTIAL else 16),
    int(os.environ.get("PTXV_MOE_H", 256)),
    128,
    4,
    2,
)


def inputs():
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(T, E, generator=g) * (1.0 if PARTIAL else 0.1)
    if not PARTIAL:
        for t in range(T):
            logits[t, t % E] += 4.0
            logits[t, (t + 1) % E] += 3.0
    d = "cuda"
    return {
        "logits": logits.to(torch.bfloat16).to(d),
        "hs": torch.randn(T, H, generator=g).to(torch.float8_e4m3fn).to(d),
        "hss": (torch.rand(H // 128, T, generator=g) + 0.5).to(d),
        "w1": torch.randn(E, 2 * I, H, generator=g).to(torch.float8_e4m3fn).to(d),
        "w1s": (torch.rand(E, 2 * I // 128, H // 128, generator=g) + 0.5).to(d),
        "w2": torch.randn(E, H, I, generator=g).to(torch.float8_e4m3fn).to(d),
        "w2s": (torch.rand(E, H // 128, I // 128, generator=g) + 0.5).to(d),
    }


def run(logits, hs, hss, w1, w1s, w2, w2s):
    # routing_method_type 1 = Renormalize (softmax over the top-k).
    return trtllm_fp8_block_scale_moe(
        logits,
        None,
        hs,
        hss,
        w1,
        w1s,
        w2,
        w2s,
        E,
        K,
        None,
        None,
        I,
        0,
        E,
        None,
        routing_method_type=1,
        enable_pdl=False,
        tune_max_num_tokens=T,
    )
