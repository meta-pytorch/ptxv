# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Stress test of unmodified DeepEP intranode dispatch / combine against an all_gather reference.
#
#   CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc-per-node 2 tools/deepep_repro/repro.py [options]
#
# Each iteration draws a new routing and new token values, runs DeepEP exactly as an MoE layer
# would (get_dispatch_layout -> dispatch -> combine, optionally the cached-handle dispatch and
# num_worst_tokens paths), and checks every output against a reference built from all_gather'ed
# inputs:
#   - notify:   num_recv_tokens_per_expert_list (the counts that barrier_block hands over in
#               notify_dispatch), the received token count, rank_prefix_matrix;
#   - dispatch: every received row of x (and fp8 scales), its source index, top-k ids, weights;
#   - combine:  out[t] = x[t] * sum_{ranks r that got t} (r + 1)  (each rank sends recv_x * (r+1));
#               combined top-k weights.
# Values are small integers so every bf16 sum is exact; any mismatch is a corruption.
#
# Realistic co-running work: --overlap N enqueues N bf16 GEMMs on a side stream before every
# DeepEP call (MoE compute/communication overlap); --async uses previous_event / async_finish /
# allocate_on_comm_stream as in DeepEP's overlap examples.
import argparse
import os
import random
import sys
import time

import deep_ep
import torch
import torch.distributed as dist


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--secs", type=float, default=60)
    p.add_argument("--iters", type=int, default=10**9)
    p.add_argument("--hidden", type=str, default="7168")
    p.add_argument(
        "--tokens", type=str, default="1-4096"
    )  # range lo-hi, drawn per iteration
    p.add_argument("--experts", type=int, default=64)
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--num-sms", type=str, default="20")
    p.add_argument(
        "--dcfg", type=str, default="24,256"
    )  # dispatch send chunk, recv buffer tokens
    p.add_argument(
        "--ccfg", type=str, default="10,256"
    )  # combine send chunk, recv buffer tokens
    p.add_argument("--async", dest="asyn", action="store_true")
    p.add_argument("--overlap", type=int, default=0)
    p.add_argument("--gemm", type=int, default=4096)
    p.add_argument(
        "--cached",
        action="store_true",
        help="also run cached-handle dispatch + combine",
    )
    p.add_argument(
        "--worst", action="store_true", help="also run num_worst_tokens dispatch"
    )
    p.add_argument("--fp8", action="store_true", help="dispatch fp8 x with scales")
    p.add_argument(
        "--drop", type=float, default=0.05, help="fraction of top-k slots set to -1"
    )
    p.add_argument(
        "--skew",
        type=float,
        default=0.0,
        help="fraction of tokens routed only to rank 0",
    )
    p.add_argument("--seed", type=int, default=1)
    p.add_argument(
        "--sync-every", type=int, default=1, help="host sync + check every N iterations"
    )
    return p.parse_args()


def rng_range(s):
    lo, hi = (int(v) for v in s.split("-")) if "-" in s else (int(s), int(s))
    return lo, hi


def main():  # noqa: C901 (repro script: one sequential experiment driver)
    a = parse()
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    dev = torch.device("cuda")
    hiddens = [int(h) for h in a.hidden.split(",")]
    smss = [int(s) for s in a.num_sms.split(",")]
    tlo, thi = rng_range(a.tokens)
    dchunk, dbuf = (int(v) for v in a.dcfg.split(","))
    cchunk, cbuf = (int(v) for v in a.ccfg.split(","))
    E, K = a.experts, a.topk
    nloc = E // world

    nvl = 0
    for s in smss:
        for h in hiddens:
            for c in (
                deep_ep.Config(s, dchunk, dbuf, 6, 128),
                deep_ep.Config(s, cchunk, cbuf, 6, 128),
            ):
                nvl = max(nvl, c.get_nvl_buffer_size_hint(h * 2, world))
    buf = deep_ep.Buffer(dist.group.WORLD, nvl, 0)
    side = torch.cuda.Stream()
    ga = torch.randn(a.gemm, a.gemm, device=dev, dtype=torch.bfloat16)
    gb = torch.randn(a.gemm, a.gemm, device=dev, dtype=torch.bfloat16)

    def overlap():
        if a.overlap:
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(a.overlap):
                    torch.mm(ga, gb)

    random.seed(a.seed * 1000 + rank)
    g = torch.Generator(device="cpu").manual_seed(a.seed * 7919 + rank)
    t0 = time.time()
    it = 0
    stats = {"iters": 0, "tokens": 0, "recv_rows": 0, "bad": 0}
    bad_log = []

    def fail(what, detail):
        stats["bad"] += 1
        if len(bad_log) < 20:
            bad_log.append(f"iter {it} {what}: {detail}")
            print(f"[rank {rank}] MISMATCH iter {it} {what}: {detail}", flush=True)

    while it < a.iters:
        # both ranks agree on when to stop
        flag = torch.tensor([1 if time.time() - t0 < a.secs else 0], device=dev)
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        if flag.item() == 0:
            break
        H = hiddens[it % len(hiddens)]
        S = smss[(it // len(hiddens)) % len(smss)]
        deep_ep.Buffer.set_num_sms(S)
        dcfg = deep_ep.Config(S, dchunk, dbuf, 6, 128)
        ccfg = deep_ep.Config(S, cchunk, cbuf, 6, 128)
        T = random.randint(tlo, thi)
        # routing: K distinct experts per token, some slots dropped (-1)
        scores = torch.rand(T, E, generator=g)
        if a.skew > 0:
            nskew = int(T * a.skew)
            scores[:nskew, nloc:] = -1.0
        topk = scores.topk(K, dim=1).indices
        topk[torch.rand(T, K, generator=g) < a.drop] = -1
        w = torch.rand(T, K, generator=g)
        x = torch.randint(-8, 9, (T, H), generator=g).to(torch.bfloat16)
        topk_d, w_d, x_d = topk.to(dev), w.to(dev), x.to(dev)
        if a.fp8:
            # exact in e4m3: small integers; scales are distinct per token / block
            x8 = x_d.to(torch.float8_e4m3fn)
            sc = (torch.randint(1, 64, (T, H // 128), generator=g).float() / 8).to(dev)

        # reference inputs
        sizes = [torch.zeros(1, dtype=torch.long, device=dev) for _ in range(world)]
        dist.all_gather(sizes, torch.tensor([T], device=dev))
        sizes = [int(s.item()) for s in sizes]

        def gather(t):
            out = [
                torch.empty((n,) + tuple(t.shape[1:]), dtype=t.dtype, device=dev)
                for n in sizes  # noqa: B023 (closure used only within this iteration)
            ]
            dist.all_gather(out, t.contiguous())
            return [o.cpu() for o in out]

        X, TK, W = gather(x_d), gather(topk_d), gather(w_d)
        SC = gather(sc) if a.fp8 else None

        # expected receive on this rank: (src, t) in source-rank order, token order
        lo_e, hi_e = rank * nloc, (rank + 1) * nloc
        rows, rtk, rw, rsrc, rsc = [], [], [], [], []
        for s in range(world):
            inr = ((TK[s] >= lo_e) & (TK[s] < hi_e)).any(dim=1)
            idx = inr.nonzero().flatten()
            rows.append(X[s][idx])
            loc = torch.where(
                (TK[s][idx] >= lo_e) & (TK[s][idx] < hi_e),
                TK[s][idx] - lo_e,
                torch.full_like(TK[s][idx], -1),
            )
            rtk.append(loc)
            rw.append(torch.where(loc >= 0, W[s][idx], torch.zeros_like(W[s][idx])))
            rsrc.append(idx.int())
            if a.fp8:
                rsc.append(SC[s][idx])
        ref_x, ref_tk, ref_w, ref_src = (
            torch.cat(rows),
            torch.cat(rtk),
            torch.cat(rw),
            torch.cat(rsrc),
        )
        ref_cnt = [int((ref_tk == e).sum().item()) for e in range(nloc)]
        # expected combine on this rank
        mult = torch.zeros(T)
        for r in range(world):
            inr = ((topk >= r * nloc) & (topk < (r + 1) * nloc)).any(dim=1)
            mult += inr.float() * (r + 1)
        ref_comb = (x.float() * mult[:, None]).to(torch.bfloat16)
        # combined weights: sum over ranks (ascending) of that rank's masked weights
        ref_cw = torch.zeros(T, K)
        for r in range(world):
            m = (topk >= r * nloc) & (topk < (r + 1) * nloc)
            inr = m.any(dim=1, keepdim=True)
            ref_cw = ref_cw + torch.where(m & inr, w, torch.zeros_like(w))

        # ---- DeepEP, as an MoE layer calls it
        overlap()
        prev = deep_ep.Buffer.capture() if a.asyn else None
        ntr, _, nte, itr, ev = buf.get_dispatch_layout(
            topk_d,
            E,
            previous_event=prev,
            async_finish=a.asyn,
            allocate_on_comm_stream=a.asyn,
        )
        overlap()
        xin = (x8, sc) if a.fp8 else x_d
        rx, rtopk, rwt, cnt, handle, ev = buf.dispatch(
            xin,
            num_tokens_per_rank=ntr,
            is_token_in_rank=itr,
            num_tokens_per_expert=nte,
            topk_idx=topk_d,
            topk_weights=w_d,
            config=dcfg,
            previous_event=ev if a.asyn else None,
            async_finish=a.asyn,
            allocate_on_comm_stream=a.asyn,
        )
        if a.asyn:
            ev.current_stream_wait()
        rx_x = rx[0] if a.fp8 else rx
        y = (rx_x.to(torch.bfloat16).float() * (rank + 1)).to(torch.bfloat16)
        overlap()
        prev = deep_ep.Buffer.capture() if a.asyn else None
        comb, cw, ev = buf.combine(
            y,
            handle,
            topk_weights=rwt,
            config=ccfg,
            previous_event=prev,
            async_finish=a.asyn,
            allocate_on_comm_stream=a.asyn,
        )
        if a.asyn:
            ev.current_stream_wait()
        extra = []
        if a.cached:
            x2 = (x_d.float() * 2).to(torch.bfloat16)  # new values, same routing
            overlap()
            rx2, _, _, _, _, ev2 = buf.dispatch(x2, handle=handle, config=dcfg)
            y2 = (rx2.float() * (rank + 1)).to(torch.bfloat16)
            overlap()
            comb2, _, _ = buf.combine(y2, handle, config=ccfg)
            extra.append((rx2, comb2))
        if a.worst:
            nw = sum(sizes)  # worst case: every token of every rank
            overlap()
            rx3, rtopk3, rwt3, cnt3, handle3, _ = buf.dispatch(
                x_d,
                num_tokens_per_rank=ntr,
                is_token_in_rank=itr,
                num_tokens_per_expert=nte,
                topk_idx=topk_d,
                topk_weights=w_d,
                config=dcfg,
                num_worst_tokens=nw,
            )
            extra.append((rx3, rtopk3, rwt3, cnt3))

        # ---- check
        torch.cuda.synchronize()
        n = ref_x.shape[0]
        if cnt != ref_cnt:
            fail("notify counts", f"got {cnt} want {ref_cnt}")
        if rx_x.shape[0] != n:
            fail("recv count", f"got {rx_x.shape[0]} want {n}")
        else:
            src = handle[3].cpu()
            if not torch.equal(src, ref_src):
                bad = (src != ref_src).nonzero().flatten()[:8].tolist()
                fail("recv_src_idx", f"{len(bad)}+ rows differ, first {bad}")
            gx = rx_x.float().cpu() if not a.fp8 else rx_x.float().cpu()
            d = (gx != ref_x.float()).any(dim=1)
            if d.any():
                bad = d.nonzero().flatten()
                fail(
                    "recv_x",
                    f"{bad.numel()} rows differ, first {bad[:8].tolist()} (H={H}, sms={S})",
                )
            if a.fp8:
                d = (rx[1].cpu() != torch.cat(rsc)).any(dim=1)
                if d.any():
                    fail("recv_x_scales", f"{int(d.sum())} rows differ")
            if not torch.equal(rtopk.cpu(), ref_tk):
                fail(
                    "recv_topk_idx",
                    f"{int((rtopk.cpu() != ref_tk).any(dim=1).sum())} rows differ",
                )
            if not torch.equal(rwt.cpu(), ref_w):
                fail(
                    "recv_topk_weights",
                    f"{int((rwt.cpu() != ref_w).any(dim=1).sum())} rows differ",
                )
            rpm = handle[0].cpu()
            want_rpm = torch.tensor([[0] * world] * world)
            # rank_prefix_matrix[i][j]: tokens from ranks <= i to rank j
            for i in range(world):
                for j in range(world):
                    want_rpm[i][j] = sum(
                        int(
                            (
                                ((TK[s] >= j * nloc) & (TK[s] < (j + 1) * nloc)).any(
                                    dim=1
                                )
                            ).sum()
                        )
                        for s in range(i + 1)
                    )
            # only column `rank` is written and used by this DeepEP version (the other columns
            # hold leftover buffer words; notify writes per_rank_buffer[rank * R + thread] only)
            if not torch.equal(
                rpm.view(world, world).long()[:, rank], want_rpm[:, rank]
            ):
                fail(
                    "rank_prefix_matrix",
                    f"got {rpm.tolist()} want column {rank} of {want_rpm.tolist()}",
                )
        d = (comb.cpu().float() != ref_comb.float()).any(dim=1)
        if d.any():
            fail(
                "combine x",
                f"{int(d.sum())} of {T} tokens differ, first {d.nonzero().flatten()[:8].tolist()} (H={H}, sms={S})",
            )
        if cw is not None and not torch.equal(cw.cpu(), ref_cw):
            dd = (cw.cpu() != ref_cw).any(dim=1)
            fail("combine weights", f"{int(dd.sum())} tokens differ")
        if a.cached:
            rx2, comb2 = extra[0]
            if rx2.shape[0] != n or not torch.equal(
                rx2.float().cpu(), ref_x.float() * 2
            ):
                fail("cached dispatch x", f"shape {tuple(rx2.shape)} vs {n}")
            if not torch.equal(comb2.cpu().float(), ref_comb.float() * 2):
                fail("cached combine x", "differs")
        if a.worst:
            rx3, rtopk3, rwt3, cnt3 = extra[-1]
            if not torch.equal(rx3[:n].float().cpu(), ref_x.float()):
                fail("worst dispatch x", "differs")
            if not torch.equal(rtopk3[:n].cpu(), ref_tk) or (rtopk3[n:] != -1).any():
                fail("worst topk", "differs")
        stats["iters"] += 1
        stats["tokens"] += T
        stats["recv_rows"] += n
        it += 1

    tot = torch.tensor(
        [stats["iters"], stats["tokens"], stats["recv_rows"], stats["bad"]], device=dev
    )
    dist.all_reduce(tot)
    if rank == 0:
        print(
            f"RESULT args={' '.join(sys.argv[1:])} iters={stats['iters']} tokens_sent={int(tot[1])} "
            f"rows_received={int(tot[2])} mismatches={int(tot[3])} secs={time.time() - t0:.0f}",
            flush=True,
        )
    buf.destroy() if hasattr(buf, "destroy") and getattr(
        buf, "explicitly_destroy", False
    ) else None
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
