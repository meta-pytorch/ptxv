# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# DeepEP intranode dispatch -> combine on 2 ranks, at the smallest size that still exercises
# the channel ring (docs/rung4.md, "Rung 4c"), for tools/capture_dist.py. The same DeepEP
# kernels as rung 4b (moe_deepep.py) without the torch expert computation in between: combine
# sums the dispatched rows themselves.
#
#   CUDA_VISIBLE_DEVICES=0,1 DEEPEP_RECV=2 $PTXV_PYTHON tools/capture_dist.py --nproc 2 \
#       --file tools/capture_examples/deepep_ring.py --out fixtures/r4c_ring_wrap
#
# The protocol does not depend on the payload, so everything is shrunk:
# - DEEPEP_T tokens of width DEEPEP_H (bf16) per rank. Routing: E = 2 experts (one per rank),
#   top-2, so every token goes to both ranks and each of the 4 (source, destination) flows of a
#   channel carries every token of its channel.
# - DEEPEP_NSMS SMs: NSMS / 2 channels (sender CTA 2k, receiver CTA 2k + 1).
# - Ring: DEEPEP_RECV slots per (channel, source rank), chunks of DEEPEP_SEND tokens. With more
#   tokens per channel than slots the senders wrap around and wait for the receivers' heads.
#   DEEPEP_CRECV (default DEEPEP_RECV) is combine's ring. Combine cannot wrap at this size:
#   each of its 23 receiver warps holds the ring head at its previous token while it waits
#   for its next one, so a ring smaller than min(23, tokens per channel) deadlocks (GB300: the
#   combine senders and receivers time out, DEEPEP_RECV=2 with T = 3).
import os

import deep_ep
import torch


def env(k, d):
    return int(os.environ.get(k, d))


T, H = env("DEEPEP_T", 3), env("DEEPEP_H", 16)
NSMS, SEND, RECV = env("DEEPEP_NSMS", 2), env("DEEPEP_SEND", 1), env("DEEPEP_RECV", 2)
CRECV = env("DEEPEP_CRECV", RECV)
E, K = 2, 2


def setup(rank, world, group):
    import torch.distributed as dist

    deep_ep.Buffer.set_num_sms(NSMS)
    cfg = deep_ep.Config(NSMS, SEND, RECV, 6, 128)
    ccfg = deep_ep.Config(NSMS, SEND, CRECV, 6, 128)
    nvl = max(c.get_nvl_buffer_size_hint(H * 2, world) for c in (cfg, ccfg))
    buf = deep_ep.Buffer(dist.group.WORLD, nvl, 0)
    g = torch.Generator().manual_seed(500 + rank)
    x = torch.randn(T, H, generator=g).bfloat16().cuda()
    topk = torch.stack([torch.randperm(E, generator=g)[:K] for _ in range(T)]).cuda()
    w = torch.rand(T, K, generator=g).cuda()
    return {"x": x, "topk": topk, "w": w, "ep": buf, "cfg": cfg, "ccfg": ccfg}


def run(x, topk, w, ep, cfg, ccfg):
    ntr, _, nte, itr, _ = ep.get_dispatch_layout(topk, E)
    rx, _, _, _, handle, _ = ep.dispatch(
        x,
        topk_idx=topk,
        topk_weights=w,
        num_tokens_per_rank=ntr,
        is_token_in_rank=itr,
        num_tokens_per_expert=nte,
        config=cfg,
    )
    y, _, _ = ep.combine(rx, handle, config=ccfg)
    return {"y": y}
