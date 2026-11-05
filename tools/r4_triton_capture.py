# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Rung 4 (non-GEMM kernel classes): JIT-compile small Triton 3.5 kernels for softmax,
# layernorm, rmsnorm, a row sum/max reduction and a flash-attention forward, check them
# against torch, and write fixtures/r4_<op>_triton/{kernel.ptx,kernel.cubin,case.toml}. The case
# is binary-first: kernel.cubin is the cubin Triton itself built and launched (through CUDA 13
# ptxas); kernel.ptx is kept for reference (the PTX front-end does not parse these kernels yet).
#   TRITON_PTXAS_PATH=/usr/local/cuda-13.0/bin/ptxas CUDA_VISIBLE_DEVICES=0 \
#       $PTXV_PYTHON tools/r4_triton_capture.py [op ...]
import os
import sys

import torch
import triton
import triton.language as tl

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


@triton.jit
def softmax_kernel(x_ptr, y_ptr, stride, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(x_ptr + row * stride + offs, mask=mask, other=-float("inf"))
    x = x - tl.max(x, axis=0)
    e = tl.exp(x)
    y = e / tl.sum(e, axis=0)
    tl.store(y_ptr + row * stride + offs, y, mask=mask)


@triton.jit
def layernorm_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr, stride, n_cols, eps, BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(x_ptr + row * stride + offs, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / n_cols
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + offs, mask=mask).to(tl.float32)
    b = tl.load(b_ptr + offs, mask=mask).to(tl.float32)
    y = d * rstd * w + b
    tl.store(y_ptr + row * stride + offs, y.to(tl.bfloat16), mask=mask)


@triton.jit
def rmsnorm_kernel(x_ptr, w_ptr, y_ptr, stride, n_cols, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(x_ptr + row * stride + offs, mask=mask, other=0.0).to(tl.float32)
    ms = tl.sum(x * x, axis=0) / n_cols
    r = tl.rsqrt(ms + eps)
    w = tl.load(w_ptr + offs, mask=mask).to(tl.float32)
    tl.store(y_ptr + row * stride + offs, (x * r * w).to(tl.bfloat16), mask=mask)


@triton.jit
def rowreduce_kernel(x_ptr, s_ptr, m_ptr, stride, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(x_ptr + row * stride + offs, mask=mask, other=0.0)
    tl.store(s_ptr + row, tl.sum(x, axis=0))
    xm = tl.where(mask, x, -float("inf"))
    tl.store(m_ptr + row, tl.max(xm, axis=0))


@triton.jit
def attn_fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    sm_scale,
    N_CTX: tl.constexpr,
    HEAD: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    # One (batch*head) per program along axis 1, BM query rows per program along axis 0.
    start_m = tl.program_id(0)
    bh = tl.program_id(1)
    base = bh * N_CTX * HEAD
    offs_m = start_m * BM + tl.arange(0, BM)
    offs_n = tl.arange(0, BN)
    offs_d = tl.arange(0, HEAD)
    q = tl.load(q_ptr + base + offs_m[:, None] * HEAD + offs_d[None, :])
    m_i = tl.zeros([BM], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BM], dtype=tl.float32)
    acc = tl.zeros([BM, HEAD], dtype=tl.float32)
    qk_scale = sm_scale * 1.44269504
    for start_n in range(0, N_CTX, BN):
        k = tl.load(k_ptr + base + (start_n + offs_n)[None, :] * HEAD + offs_d[:, None])
        qk = tl.dot(q, k) * qk_scale
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.math.exp2(qk - m_new[:, None])
        alpha = tl.math.exp2(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(v_ptr + base + (start_n + offs_n)[:, None] * HEAD + offs_d[None, :])
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
        m_i = m_new
    acc = acc / l_i[:, None]
    tl.store(
        o_ptr + base + offs_m[:, None] * HEAD + offs_d[None, :], acc.to(tl.bfloat16)
    )


DT = {torch.float32: "f32", torch.bfloat16: "bf16"}


def tensor_bytes(t):
    t = t.detach().contiguous().cpu()
    if t.dtype == torch.bfloat16:
        t = t.view(torch.int16)
    return t.numpy().tobytes()


def write_case(name, h, desc, grid, bufs, params, outputs, expect=("clean",), note=""):
    """bufs: (name, tensor, init, seed); inputs are written as their exact bytes (data/),
    outputs as the bytes the GPU produced (expected/, compared bitwise by `ptxv check`)."""
    d = os.path.join(ROOT, "fixtures", name)
    os.makedirs(os.path.join(d, "data"), exist_ok=True)
    os.makedirs(os.path.join(d, "expected"), exist_ok=True)
    open(os.path.join(d, "kernel.ptx"), "w").write(h.asm["ptx"])
    open(os.path.join(d, "kernel.cubin"), "wb").write(h.asm["cubin"])
    meta = h.metadata
    out = [
        f'''name = "{name}"
description = """
{desc.strip()}
Triton {triton.__version__}, num_warps={meta.num_warps}. Captured by tools/r4_triton_capture.py (inputs
data/, GPU outputs expected/).{note}"""
ptx = "kernel.ptx"
cubin = "kernel.cubin"
entry = "{h.name}"
grid = {list(grid) + [1] * (3 - len(grid))}
block = [{32 * meta.num_warps}, 1, 1]
dynamic_smem = {meta.shared}
outputs = {outputs}
'''
    ]
    for n, t, _init, _seed in bufs:
        b = f'[[buffers]]\nname = "{n}"\ndtype = "{DT[t.dtype]}"\nshape = {list(t.shape)}\n'
        if n in outputs:
            open(os.path.join(d, "expected", f"{n}.bin"), "wb").write(tensor_bytes(t))
            b += f'init = "zeros"\nexpect = "expected/{n}.bin"\n'
        else:
            open(os.path.join(d, "data", f"{n}.bin"), "wb").write(tensor_bytes(t))
            b += f'init = "file"\nfile = "data/{n}.bin"\n'
        out.append(b)
    for kind, v in params:
        if kind == "ptr":
            out.append(f'[[params]]\nkind = "ptr"\nbuffer = "{v}"\n')
        else:
            out.append(f'[[params]]\nkind = "{kind}"\nvalue = {v}\n')
    # Triton appends two scratch pointers (global / profile scratch), unused here.
    for _ in range(2):
        out.append('[[params]]\nkind = "u64"\nvalue = 0\n')
    hw = ""
    if "unmodeled" in expect:
        hw = (
            'hw_note = "lifts completely; execution stops at the first MUFU (transcendental, '
            "agent B's N-MUFU work): compare with expected/ once it lands\"\n"
        )
    out.append(
        f'[[variants]]\nid = "good"\nsummary = "as compiled by Triton"\nexpect = {list(expect)}\n'.replace(
            "'", '"'
        )
        + hw
    )
    open(os.path.join(d, "case.toml"), "w").write("\n".join(out))
    print("wrote", name, h.name, "warps", meta.num_warps, "smem", meta.shared)


def f32bits(x):
    return (
        int(torch.tensor([x], dtype=torch.float32).view(torch.int32).item())
        & 0xFFFFFFFF
    )


def run(ops):
    R, C = 4, 128
    dev = "cuda"
    if "softmax" in ops:
        x = torch.randn(R, C, device=dev)
        y = torch.empty_like(x)
        h = softmax_kernel[(R,)](x, y, C, C, BLOCK=C, num_warps=2)
        torch.cuda.synchronize()
        assert (y - torch.softmax(x, -1)).abs().max() < 1e-6
        write_case(
            "r4_softmax_triton",
            h,
            f"Rung 4. Triton row softmax, y[{R}x{C}] f32 = softmax(x, dim=-1), one CTA per row\n"
            "(max, exp via MUFU.EX2, sum, divide).",
            (R,),
            [("x", x, "normal", 1), ("y", y, "zeros", 0)],
            [("ptr", "x"), ("ptr", "y"), ("u32", C), ("u32", C)],
            ["y"],
            expect=("unmodeled",),
        )
    if "layernorm" in ops:
        x = torch.randn(R, C, device=dev, dtype=torch.bfloat16)
        w = torch.randn(C, device=dev, dtype=torch.bfloat16)
        b = torch.randn(C, device=dev, dtype=torch.bfloat16)
        y = torch.empty_like(x)
        h = layernorm_kernel[(R,)](x, w, b, y, C, C, 1e-5, BLOCK=C, num_warps=2)
        torch.cuda.synchronize()
        ref = torch.nn.functional.layer_norm(
            x.float(), (C,), w.float(), b.float(), 1e-5
        )
        assert (y.float() - ref).abs().max() < 0.05
        write_case(
            "r4_layernorm_triton",
            h,
            f"Rung 4. Triton layernorm, y[{R}x{C}] bf16 = layer_norm(x bf16, w, b, eps=1e-5),\n"
            "one CTA per row, f32 statistics.",
            (R,),
            [
                ("x", x, "normal", 1),
                ("w", w, "normal", 2),
                ("b", b, "normal", 3),
                ("y", y, "zeros", 0),
            ],
            [
                ("ptr", "x"),
                ("ptr", "w"),
                ("ptr", "b"),
                ("ptr", "y"),
                ("u32", C),
                ("u32", C),
                ("u32", f32bits(1e-5)),
            ],
            ["y"],
            expect=("unmodeled",),
        )
    if "rmsnorm" in ops:
        x = torch.randn(R, C, device=dev, dtype=torch.bfloat16)
        w = torch.randn(C, device=dev, dtype=torch.bfloat16)
        y = torch.empty_like(x)
        h = rmsnorm_kernel[(R,)](x, w, y, C, C, 1e-6, BLOCK=C, num_warps=2)
        torch.cuda.synchronize()
        ref = torch.nn.functional.rms_norm(x.float(), (C,), w.float(), 1e-6)
        assert (y.float() - ref).abs().max() < 0.05
        write_case(
            "r4_rmsnorm_triton",
            h,
            f"Rung 4. Triton rmsnorm, y[{R}x{C}] bf16 = rms_norm(x bf16, w, eps=1e-6), one CTA\n"
            "per row (MUFU.RSQ).",
            (R,),
            [("x", x, "normal", 1), ("w", w, "normal", 2), ("y", y, "zeros", 0)],
            [
                ("ptr", "x"),
                ("ptr", "w"),
                ("ptr", "y"),
                ("u32", C),
                ("u32", C),
                ("u32", f32bits(1e-6)),
            ],
            ["y"],
            expect=("unmodeled",),
        )
    if "rowreduce" in ops:
        x = torch.randn(R, C, device=dev)
        s = torch.empty(R, device=dev)
        m = torch.empty(R, device=dev)
        h = rowreduce_kernel[(R,)](x, s, m, C, C, BLOCK=C, num_warps=2)
        torch.cuda.synchronize()
        assert (s - x.sum(-1)).abs().max() < 1e-4 and (m == x.max(-1).values).all()
        write_case(
            "r4_rowreduce_triton",
            h,
            f"Rung 4. Triton row reductions over x[{R}x{C}] f32: s = sum(x, -1), m = max(x, -1),\n"
            "one CTA per row (SHFL.BFLY trees, FMNMX).",
            (R,),
            [("x", x, "normal", 1), ("s", s, "zeros", 0), ("m", m, "zeros", 0)],
            [("ptr", "x"), ("ptr", "s"), ("ptr", "m"), ("u32", C), ("u32", C)],
            ["s", "m"],
        )
    if "attn" in ops:
        B, N, D, BM, BN = 1, 128, 64, 64, 64
        q = torch.randn(B, N, D, device=dev, dtype=torch.bfloat16)
        k = torch.randn(B, N, D, device=dev, dtype=torch.bfloat16)
        v = torch.randn(B, N, D, device=dev, dtype=torch.bfloat16)
        o = torch.empty_like(q)
        sc = D**-0.5
        h = attn_fwd_kernel[(N // BM, B)](
            q, k, v, o, sc, N_CTX=N, HEAD=D, BM=BM, BN=BN, num_warps=4
        )
        torch.cuda.synchronize()
        ref = torch.nn.functional.scaled_dot_product_attention(
            q.float().unsqueeze(1), k.float().unsqueeze(1), v.float().unsqueeze(1)
        ).squeeze(1)
        assert (o.float() - ref).abs().max() < 0.05, (o.float() - ref).abs().max()
        write_case(
            "r4_attn_triton",
            h,
            f"Rung 4. Triton flash-attention forward (online softmax, exp2), one head,\n"
            f"N_CTX={N}, HEAD={D}, BLOCK_M={BM}, BLOCK_N={BN}, bf16 q/k/v/o, f32 accumulators.",
            (N // BM, B),
            [
                ("q", q, "normal", 1),
                ("k", k, "normal", 2),
                ("v", v, "normal", 3),
                ("o", o, "zeros", 0),
            ],
            [
                ("ptr", "q"),
                ("ptr", "k"),
                ("ptr", "v"),
                ("ptr", "o"),
                ("u32", f32bits(sc)),
            ],
            ["o"],
            expect=("unlifted",),
        )


if __name__ == "__main__":
    run(sys.argv[1:] or ["softmax", "layernorm", "rmsnorm", "rowreduce", "attn"])
