# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Check `ptxv equiv`'s proven per-output bounds on the GPU (docs/equiv-reassoc.md): run each
# torch reference and Triton candidate of fixtures/c_equiv_{sum,cumsum,softmax} on N random
# inputs (even trials normal, odd uniform in [1, 2)), and report, per pair, the largest
# |C - R| / bound(x) over every output and trial (must stay <= 1). The atomic row sum also
# exercises GB300's own atomic orders. Bounds come from the equiv.json files `ptxv equiv
# --out DIR` writes (per output: k u sum|terms| for sums, rel u y + add u + abs for softmax).
#   CUDA_VISIBLE_DEVICES=3 $PTXV_PYTHON tools/equiv_gpu_check.py --trials 2000 \
#       --sum /tmp/e-sum/equiv.json --cumsum /tmp/e-cumsum/equiv.json \
#       --softmax /tmp/e-softmax/equiv.json
# Layer norm, RMS norm and the attention core (fixtures/c_equiv_layernorm, c_equiv_rmsnorm,
# c_equiv_attn): their bounds are
# evaluated exactly per input by ptxv itself, so --dump DIR writes the GPU's inputs and outputs
# (DIR/<candidate>/t<T>.<buffer>.bin, t<T>.ref.bin, t<T>.cand.bin), and
#   PTXV_EQUIV_HW_TRIALS=DIR ptxv equiv fixtures/c_equiv_layernorm/equiv.toml --no-hw
# checks them (its "trials" line then counts GB300 runs).
#   CUDA_VISIBLE_DEVICES=3 $PTXV_PYTHON tools/equiv_gpu_check.py --trials 2000 --dump /tmp/e-hw
# The bf16 tensor-core pairs (fixtures/c_equiv_mm_bf16, c_equiv_sdpa; docs/equiv-mma.md) the same
# way, bf16 values widened to f32:
#   CUDA_VISIBLE_DEVICES=1 $PTXV_PYTHON tools/equiv_gpu_check.py --trials 2000 --dump-mma /tmp/e-mma
#   PTXV_EQUIV_HW_TRIALS=/tmp/e-mma ptxv equiv fixtures/c_equiv_mm_bf16/equiv.toml --no-hw
import argparse
import importlib.util
import json
import os

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
U = 2.0**-24


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, path))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def inputs(t, g):
    if t % 2 == 0:
        return torch.randn(4, 128, generator=g).cuda()
    return (1.0 + torch.rand(4, 128, generator=g)).cuda()


def per_output(path, cand):
    for c in json.load(open(path)):
        if c["candidate"].startswith(cand):
            return c["report"]["per_output"]
    raise SystemExit(f"{cand} not in {path}")


def dump_norms(root, trials):
    """GPU inputs and outputs of the norm and attention-core pairs, for `ptxv equiv`
    (PTXV_EQUIV_HW_TRIALS)."""
    import torch.nn.functional as F

    ln = load("capture_examples/triton_layernorm_f32.py", "ln")
    rms = load("capture_examples/triton_rmsnorm_f32.py", "rms")
    aref = load("capture_examples/triton_attn_core_ref.py", "aref")
    aflash = load("capture_examples/triton_attn_core_flash.py", "aflash")
    fref = load("capture_examples/triton_attn_f32_ref.py", "fref")
    fflash = load("capture_examples/triton_attn_f32_flash.py", "fflash")
    pairs = [
        (
            "c_layernorm_triton_f32",
            ["x", "w", "b"],
            lambda x, w, b: F.layer_norm(x, (128,), w, b, 1e-5),
            ln.run,
        ),
        (
            "c_rmsnorm_triton_f32",
            ["x", "w"],
            lambda x, w: F.rms_norm(x, (128,), w, 1e-6),
            rms.run,
        ),
    ]
    pairs.append(("c_attn_core_flash", ["s", "v"], aref.run, aflash.run))
    pairs.append(("c_attn_f32_flash", ["q", "k", "v"], fref.run, fflash.run))
    shapes = {
        "x": (4, 128),
        "w": (128,),
        "b": (128,),
        "s": (16, 64),
        "v": (64, 16),
        "q": (16, 16),
        "k": (64, 16),
    }
    for name, bufs, ref, cand in pairs:
        d = os.path.join(root, name)
        os.makedirs(d, exist_ok=True)
        g = torch.Generator().manual_seed(2)
        for t in range(trials):
            ins = {}
            for b in bufs:
                if t % 2 == 0:
                    ins[b] = torch.randn(*shapes[b], generator=g).cuda()
                else:
                    ins[b] = (1.0 + torch.rand(*shapes[b], generator=g)).cuda()
            r = ref(**ins)
            c = cand(**ins)
            torch.cuda.synchronize()
            for b, v in ins.items():
                v.cpu().numpy().tofile(os.path.join(d, f"t{t}.{b}.bin"))
            r.cpu().numpy().tofile(os.path.join(d, f"t{t}.ref.bin"))
            c.cpu().numpy().tofile(os.path.join(d, f"t{t}.cand.bin"))
        print(f"{name}: {trials} GPU trials written to {d}")


def dump_mma(root, trials):
    """GPU inputs and outputs of the bf16 tensor-core pairs (docs/equiv-mma.md): torch.mm vs
    the Triton matmuls (fixtures/c_equiv_mm_bf16), torch SDPA flash vs Triton flash attention
    (fixtures/c_equiv_sdpa). bf16 values are written widened to f32."""
    import torch.nn.functional as F
    from torch.nn.attention import sdpa_kernel, SDPBackend

    mm = load("capture_examples/triton_mm_bf16.py", "mm")
    r4 = load("r4_triton_capture.py", "r4")

    def tmm(rev):
        def f(a, b):
            c = torch.empty(64, 64, dtype=torch.bfloat16, device=a.device)
            mm.mm_bf16_kernel[(1,)](
                a, b, c, M=64, N=64, K=64, BK=32, REV=rev, num_warps=4, num_stages=1
            )
            return c

        return f

    def sdpa(q, k, v):
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            return F.scaled_dot_product_attention(
                q.view(1, 1, 128, 64), k.view(1, 1, 128, 64), v.view(1, 1, 128, 64)
            )

    def tattn(q, k, v):
        o = torch.empty_like(q)
        r4.attn_fwd_kernel[(2, 1)](
            q, k, v, o, 0.125, N_CTX=128, HEAD=64, BM=64, BN=64, num_warps=4
        )
        return o

    pairs = [
        ("c_triton_mm_bf16", ["a", "b"], (64, 64), lambda a, b: torch.mm(a, b), tmm(0)),
        (
            "c_triton_mm_bf16_rev",
            ["a", "b"],
            (64, 64),
            lambda a, b: torch.mm(a, b),
            tmm(1),
        ),
        ("r4_attn_triton", ["q", "k", "v"], (128, 64), sdpa, tattn),
    ]
    for name, bufs, shape, ref, cand in pairs:
        d = os.path.join(root, name)
        os.makedirs(d, exist_ok=True)
        g = torch.Generator().manual_seed(3)
        for t in range(trials):
            ins = {}
            for b in bufs:
                x = (
                    torch.randn(*shape, generator=g)
                    if t % 2 == 0
                    else 1.0 + torch.rand(*shape, generator=g)
                )
                ins[b] = x.bfloat16().cuda()
            r = ref(**ins)
            c = cand(**ins)
            torch.cuda.synchronize()
            for b, v in ins.items():
                v.float().cpu().numpy().tofile(os.path.join(d, f"t{t}.{b}.bin"))
            r.float().cpu().numpy().tofile(os.path.join(d, f"t{t}.ref.bin"))
            c.float().cpu().numpy().tofile(os.path.join(d, f"t{t}.cand.bin"))
        print(f"{name}: {trials} GPU trials written to {d}")


def main():  # noqa: C901 (GPU check script: one sequential driver)
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=1000)
    ap.add_argument("--sum")
    ap.add_argument("--cumsum")
    ap.add_argument("--softmax")
    ap.add_argument(
        "--dump", help="write layer/RMS norm GPU trials for PTXV_EQUIV_HW_TRIALS"
    )
    ap.add_argument(
        "--dump-mma", help="write the bf16 tensor-core pairs' GPU trials (likewise)"
    )
    a = ap.parse_args()
    if a.dump_mma:
        dump_mma(a.dump_mma, a.trials)
        return
    if a.dump:
        dump_norms(a.dump, a.trials)
    r4 = load("r4_triton_capture.py", "r4")
    atomic = load("capture_examples/triton_rowsum_atomic.py", "atomic")
    splitk = load("capture_examples/triton_rowsum_splitk.py", "splitk")
    cums = load("capture_examples/triton_cumsum.py", "cums")
    online = load("capture_examples/triton_softmax_online.py", "online")

    def rowreduce(x):
        s = torch.empty(4, device=x.device)
        m = torch.empty(4, device=x.device)
        r4.rowreduce_kernel[(4,)](x, s, m, 128, 128, BLOCK=128, num_warps=2)
        return s

    def softmax2(x):
        y = torch.empty_like(x)
        r4.softmax_kernel[(4,)](x, y, 128, 128, BLOCK=128, num_warps=2)
        return y

    pairs = []
    if a.sum:
        for name, fn in [
            ("r4_rowreduce_triton", rowreduce),
            ("c_triton_rowsum_splitk", splitk.run),
            ("c_triton_rowsum_atomic", atomic.run),
        ]:
            pairs.append(
                (name, lambda x: x.sum(-1), fn, per_output(a.sum, name), "sum")
            )
    if a.cumsum:
        pairs.append(
            (
                "c_triton_cumsum",
                lambda x: x.cumsum(-1),
                cums.run,
                per_output(a.cumsum, "c_triton_cumsum"),
                "cumsum",
            )
        )
    if a.softmax:
        for name, fn in [
            ("r4_softmax_triton", softmax2),
            ("c_triton_softmax_online", online.run),
        ]:
            pairs.append(
                (
                    name,
                    lambda x: torch.softmax(x, -1),
                    fn,
                    per_output(a.softmax, name),
                    "softmax",
                )
            )
    for name, ref, cand, po, form in pairs:
        g = torch.Generator().manual_seed(1)
        worst, viol, distinct = 0.0, 0, set()
        for t in range(a.trials):
            x = inputs(t, g)
            r = ref(x).double().cpu().flatten()
            c = cand(x).double().cpu().flatten()
            torch.cuda.synchronize()
            xd = x.double().cpu()
            if form == "sum":
                scale = xd.abs().sum(-1)
            elif form == "cumsum":
                scale = xd.abs().cumsum(-1).flatten()
            else:
                y = torch.softmax(xd, -1).flatten()
            for i, o in enumerate(po):
                if form == "softmax":
                    b = (o["rel_u"] * y[i].item() + o["add_u"]) * U + o["abs_extra"]
                else:
                    b = o["bound_u"] * U * scale[i].item() + o["abs_extra"]
                d = abs(c[i].item() - r[i].item())
                if b > 0:
                    worst = max(worst, d / b)
                viol += d > b
            if t == 0 and name == "c_triton_rowsum_atomic":
                # The same input many times: GB300's atomic orders.
                for _ in range(500):
                    distinct.add(tuple(cand(x).cpu().tolist()))
        extra = (
            f", {len(distinct)} distinct outputs over 500 runs of one input"
            if distinct
            else ""
        )
        print(
            f"{name}: {a.trials} trials, {viol} outputs over the bound, "
            f"largest |C - R| / bound = {worst:.4f}{extra}"
        )


if __name__ == "__main__":
    main()
