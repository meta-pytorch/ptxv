# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Expert-parallel MoE layer (rung 4b) for tools/capture_dist.py: DeepEP intranode dispatch ->
# per-rank grouped GEMM (torch._grouped_mm over the local experts) -> DeepEP combine.
# DeepEP's intranode kernels exist for 2, 4 and 8 ranks only; with one of 4 GPUs busy that is
# 2 ranks: CUDA_VISIBLE_DEVICES=0,1.
#
# Per rank: T tokens of width H (bf16), each routed to K of E experts (E / ranks local ones).
# After dispatch the host builds the expert-major row order from the received top-k indices
# (DeepEP's non-cached dispatch already synchronizes with the CPU for the received counts; its
# received top-k indices are local expert ids) and
# uploads it; the device gathers rows per local expert, runs one grouped GEMM, scales each row
# by its routing weight, scatters the rows back per received token (index_add_ into f32), and
# combine sums every token's expert outputs over the ranks.
import deep_ep
import torch

T, H, E, K = 8, 128, 4, 2
NSMS = 2


def setup(rank, world, group):
    import torch.distributed as dist

    deep_ep.Buffer.set_num_sms(NSMS)
    # The CUTLASS grouped GEMM is persistent: torch launches one CTA per SM (152 x 384
    # threads, almost all idle at this size). torch's matmul SM carveout (the knob for
    # overlapping communication with GEMMs) leaves it 2 SMs: one 2-CTA cluster.
    torch._C._set_sm_carveout_experimental(
        torch.cuda.get_device_properties(0).multi_processor_count - 2
    )
    cfg = deep_ep.Config(NSMS, 6, 256, 6, 128)
    buf = deep_ep.Buffer(
        dist.group.WORLD, cfg.get_nvl_buffer_size_hint(H * 2, world), 0
    )
    g = torch.Generator().manual_seed(300 + rank)
    x = torch.randn(T, H, generator=g).bfloat16().cuda()
    topk = torch.stack([torch.randperm(E, generator=g)[:K] for _ in range(T)]).cuda()
    w = torch.rand(T, K, generator=g).cuda()
    gw = torch.Generator().manual_seed(
        400
    )  # every rank holds all experts' weights; uses its own
    W = (torch.randn(E, H, H, generator=gw) / H**0.5).bfloat16()
    nloc = E // world
    Wl = (
        W[rank * nloc : (rank + 1) * nloc]
        .transpose(1, 2)
        .contiguous()
        .transpose(1, 2)
        .cuda()
    )
    return {
        "x": x,
        "topk": topk,
        "w": w,
        "Wl": Wl,
        "ep": buf,
        "cfg": cfg,
        "rank": rank,
        "nloc": nloc,
    }


def run(x, topk, w, Wl, ep, cfg, rank, nloc):
    ntr, _, nte, itr, _ = ep.get_dispatch_layout(topk, E)
    rx, rtopk, rw, cnt, handle, _ = ep.dispatch(
        x,
        topk_idx=topk,
        topk_weights=w,
        num_tokens_per_rank=ntr,
        is_token_in_rank=itr,
        num_tokens_per_expert=nte,
        config=cfg,
    )
    # Host: expert-major order of (received row, k) pairs routed to a local expert.
    rt = rtopk.cpu()
    rows, ks, offs = [], [], []
    for e in range(nloc):
        for t in range(rt.shape[0]):
            for k in range(rt.shape[1]):
                if rt[t, k] == e:  # DeepEP hands back local expert ids (-1: not here)
                    rows.append(t)
                    ks.append(k)
        offs.append(len(rows))
    idx = torch.tensor(rows, dtype=torch.int64).cuda()
    kidx = torch.tensor(ks, dtype=torch.int64).cuda()
    off = torch.tensor(offs, dtype=torch.int32).cuda()
    xs = rx.index_select(0, idx)
    ys = torch._grouped_mm(xs, Wl, offs=off)
    wsel = rw[idx, kidx]
    yw = ys.float() * wsel[:, None]
    acc = torch.zeros(rx.shape[0], H, dtype=torch.float32, device=x.device)
    acc.index_add_(0, idx, yw)
    y, _, _ = ep.combine(acc.bfloat16(), handle, config=cfg)
    return {"y": y}
