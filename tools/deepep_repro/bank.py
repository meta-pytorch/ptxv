# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# High-throughput variant of repro.py: unmodified DeepEP intranode dispatch / combine, checked
# on the GPU against a precomputed bank of all_gather references.
#
#   CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc-per-node 2 tools/deepep_repro/bank.py [options]
#
# K cases (routing, token values, hidden size, num_sms) are built up front with their
# references (see repro.py for what is checked). The timed loop then replays cases in a random
# order shared by both ranks, so consecutive DeepEP calls always change routing, counts and
# buffer contents, as in training; every output is compared with torch.equal on the GPU.
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
    p.add_argument("--bank", type=int, default=24)
    p.add_argument("--hidden", type=str, default="7168")
    p.add_argument("--tokens", type=str, default="1-4096")
    p.add_argument("--experts", type=int, default=64)
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--num-sms", type=str, default="20")
    p.add_argument("--dcfg", type=str, default="24,256")
    p.add_argument("--ccfg", type=str, default="10,256")
    p.add_argument("--async", dest="asyn", action="store_true")
    p.add_argument(
        "--overlap",
        type=int,
        default=0,
        help="bf16 GEMMs on a side stream per DeepEP call",
    )
    p.add_argument("--gemm", type=int, default=4096)
    p.add_argument("--cached", action="store_true")
    p.add_argument("--worst", action="store_true")
    p.add_argument(
        "--no-sync",
        action="store_true",
        help="only num_worst_tokens dispatch (no CPU sync)",
    )
    p.add_argument("--drop", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=1)
    return p.parse_args()


def main():  # noqa: C901 (repro script: one sequential experiment driver)
    a = parse()
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    dev = torch.device("cuda")
    hiddens = [int(h) for h in a.hidden.split(",")]
    smss = [int(s) for s in a.num_sms.split(",")]
    tlo, thi = (
        (int(v) for v in a.tokens.split("-"))
        if "-" in a.tokens
        else (int(a.tokens),) * 2
    )
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

    rnd = random.Random(a.seed * 1000 + rank)
    g = torch.Generator(device="cpu").manual_seed(a.seed * 7919 + rank)
    shared = random.Random(a.seed)  # same sequence on both ranks

    def gather(t, sizes):
        out = [
            torch.empty((n,) + tuple(t.shape[1:]), dtype=t.dtype, device=dev)
            for n in sizes
        ]
        dist.all_gather(out, t.contiguous())
        return [o.cpu() for o in out]

    cases = []
    for ci in range(a.bank):
        H = hiddens[ci % len(hiddens)]
        S = smss[(ci // len(hiddens)) % len(smss)]
        T = rnd.randint(tlo, thi)
        scores = torch.rand(T, E, generator=g)
        topk = scores.topk(K, dim=1).indices
        topk[torch.rand(T, K, generator=g) < a.drop] = -1
        w = torch.rand(T, K, generator=g)
        x = torch.randint(-8, 9, (T, H), generator=g).to(torch.bfloat16)
        sz = [torch.zeros(1, dtype=torch.long, device=dev) for _ in range(world)]
        dist.all_gather(sz, torch.tensor([T], device=dev))
        sizes = [int(s.item()) for s in sz]
        X, TK, W = (
            gather(x.to(dev), sizes),
            gather(topk.to(dev), sizes),
            gather(w.to(dev), sizes),
        )
        lo_e, hi_e = rank * nloc, (rank + 1) * nloc
        rows, rtk, rw, rsrc = [], [], [], []
        for s in range(world):
            idx = ((TK[s] >= lo_e) & (TK[s] < hi_e)).any(dim=1).nonzero().flatten()
            sel = TK[s][idx]
            loc = torch.where(
                (sel >= lo_e) & (sel < hi_e), sel - lo_e, torch.full_like(sel, -1)
            )
            rows.append(X[s][idx])
            rtk.append(loc)
            rsrc.append(idx.int())
            rw.append(torch.where(loc >= 0, W[s][idx], torch.zeros_like(W[s][idx])))
        ref_tk = torch.cat(rtk)
        mult = torch.zeros(T)
        cw = torch.zeros(T, K)
        for r in range(world):
            m = (topk >= r * nloc) & (topk < (r + 1) * nloc)
            mult += m.any(dim=1).float() * (r + 1)
            cw = cw + torch.where(m, w, torch.zeros_like(w))
        rpm_col = []
        for i in range(world):
            rpm_col.append(
                sum(
                    int(((TK[s] >= lo_e) & (TK[s] < hi_e)).any(dim=1).sum())
                    for s in range(i + 1)
                )
            )
        cases.append(
            {
                "H": H,
                "S": S,
                "T": T,
                "nw": sum(sizes),
                "topk": topk.to(dev),
                "w": w.to(dev),
                "x": x.to(dev),
                "x2": (x.float() * 2).to(torch.bfloat16).to(dev),
                "ref_x": torch.cat(rows).to(dev),
                "ref_tk": ref_tk.to(dev),
                "ref_w": torch.cat(rw).to(dev),
                "ref_src": torch.cat(rsrc).to(dev),
                "ref_cnt": [int((ref_tk == e).sum()) for e in range(nloc)],
                "ref_comb": (x.float() * mult[:, None]).to(torch.bfloat16).to(dev),
                "ref_cw": cw.to(dev),
                "ref_rpm": torch.tensor(rpm_col, dtype=torch.int32, device=dev),
            }
        )
    dist.barrier()

    st = {"calls": 0, "rows": 0, "bad": 0}

    def fail(it, ci, what):
        st["bad"] += 1
        if st["bad"] <= 20:
            print(
                f"[rank {rank}] MISMATCH rep {it} case {ci} (H={cases[ci]['H']} sms={cases[ci]['S']}): {what}",
                flush=True,
            )

    t0 = time.time()
    it = 0
    while True:
        if it % 16 == 0:
            flag = torch.tensor([1 if time.time() - t0 < a.secs else 0], device=dev)
            dist.all_reduce(flag, op=dist.ReduceOp.MIN)
            if flag.item() == 0:
                break
        ci = shared.randrange(len(cases))
        c = cases[ci]
        deep_ep.Buffer.set_num_sms(c["S"])
        dcfg = deep_ep.Config(c["S"], dchunk, dbuf, 6, 128)
        ccfg = deep_ep.Config(c["S"], cchunk, cbuf, 6, 128)
        overlap()
        prev = deep_ep.Buffer.capture() if a.asyn else None
        ntr, _, nte, itr, ev = buf.get_dispatch_layout(
            c["topk"],
            E,
            previous_event=prev,
            async_finish=a.asyn,
            allocate_on_comm_stream=a.asyn,
        )
        overlap()
        n = c["ref_x"].shape[0]
        if a.no_sync:
            rx, rtk, rw, cnt, h, ev = buf.dispatch(
                c["x"],
                num_tokens_per_rank=ntr,
                is_token_in_rank=itr,
                num_tokens_per_expert=nte,
                topk_idx=c["topk"],
                topk_weights=c["w"],
                config=dcfg,
                num_worst_tokens=c["nw"],
                previous_event=ev if a.asyn else None,
                async_finish=a.asyn,
                allocate_on_comm_stream=a.asyn,
            )
            if a.asyn:
                ev.current_stream_wait()
            if not torch.equal(rx[:n], c["ref_x"]):
                fail(it, ci, "worst recv_x")
            if not torch.equal(rtk[:n], c["ref_tk"]) or (rtk[n:] != -1).any():
                fail(it, ci, "worst topk")
            if not torch.equal(rw[:n], c["ref_w"]):
                fail(it, ci, "worst weights")
            st["calls"] += 1
            st["rows"] += n
            it += 1
            continue
        rx, rtk, rw, cnt, h, ev = buf.dispatch(
            c["x"],
            num_tokens_per_rank=ntr,
            is_token_in_rank=itr,
            num_tokens_per_expert=nte,
            topk_idx=c["topk"],
            topk_weights=c["w"],
            config=dcfg,
            previous_event=ev if a.asyn else None,
            async_finish=a.asyn,
            allocate_on_comm_stream=a.asyn,
        )
        if a.asyn:
            ev.current_stream_wait()
        if cnt != c["ref_cnt"]:
            fail(it, ci, f"notify counts {cnt} != {c['ref_cnt']}")
        if rx.shape[0] != n:
            fail(it, ci, f"recv count {rx.shape[0]} != {n}")
        else:
            if not torch.equal(rx, c["ref_x"]):
                fail(it, ci, f"recv_x: {int((rx != c['ref_x']).any(dim=1).sum())} rows")
            if not torch.equal(rtk, c["ref_tk"]):
                fail(it, ci, "recv_topk_idx")
            if not torch.equal(rw, c["ref_w"]):
                fail(it, ci, "recv_topk_weights")
            if not torch.equal(h[3], c["ref_src"]):
                fail(it, ci, "recv_src_idx")
            if not torch.equal(h[0].view(world, world)[:, rank], c["ref_rpm"]):
                fail(it, ci, "rank_prefix_matrix")
        y = (rx.float() * (rank + 1)).to(torch.bfloat16)
        overlap()
        prev = deep_ep.Buffer.capture() if a.asyn else None
        comb, cw, ev = buf.combine(
            y,
            h,
            topk_weights=rw,
            config=ccfg,
            previous_event=prev,
            async_finish=a.asyn,
            allocate_on_comm_stream=a.asyn,
        )
        if a.asyn:
            ev.current_stream_wait()
        if not torch.equal(comb, c["ref_comb"]):
            fail(
                it,
                ci,
                f"combine x: {int((comb != c['ref_comb']).any(dim=1).sum())} tokens",
            )
        if not torch.equal(cw, c["ref_cw"]):
            fail(it, ci, "combine weights")
        st["calls"] += 2
        if a.cached:
            overlap()
            rx2, _, _, _, _, _ = buf.dispatch(c["x2"], handle=h, config=dcfg)
            if not torch.equal(rx2, c["ref_x"].float().mul(2).to(torch.bfloat16)):
                fail(it, ci, "cached dispatch x")
            y2 = (rx2.float() * (rank + 1)).to(torch.bfloat16)
            overlap()
            comb2, _, _ = buf.combine(y2, h, config=ccfg)
            if not torch.equal(comb2, c["ref_comb"].float().mul(2).to(torch.bfloat16)):
                fail(it, ci, "cached combine x")
            st["calls"] += 2
        if a.worst:
            overlap()
            rx3, rtk3, _, _, _, _ = buf.dispatch(
                c["x"],
                num_tokens_per_rank=ntr,
                is_token_in_rank=itr,
                num_tokens_per_expert=nte,
                topk_idx=c["topk"],
                topk_weights=c["w"],
                config=dcfg,
                num_worst_tokens=c["nw"],
            )
            if (
                not torch.equal(rx3[:n], c["ref_x"])
                or not torch.equal(rtk3[:n], c["ref_tk"])
                or (rtk3[n:] != -1).any()
            ):
                fail(it, ci, "worst dispatch")
            st["calls"] += 1
        st["rows"] += n
        it += 1
    torch.cuda.synchronize()
    tot = torch.tensor([it, st["calls"], st["rows"], st["bad"]], device=dev)
    dist.all_reduce(tot)
    if rank == 0:
        print(
            f"RESULT bank args={' '.join(sys.argv[1:])} reps={it} deepep_calls={int(tot[1])} "
            f"rows_received={int(tot[2])} mismatches={int(tot[3])} secs={time.time() - t0:.0f}",
            flush=True,
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
