# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Capture for rung 2: JIT-compiles Triton 3.5's persistent TMA + tcgen05 matmul (the
# 09-persistent-matmul pattern with host tensor descriptors) and writes kernel.ptx and
# case.toml. Run with the conda python on a free GPU, TRITON_PTXAS_PATH = CUDA 13 ptxas
# (Triton's bundled ptxas does not know sm_103a):
#   TRITON_PTXAS_PATH=/usr/local/cuda-13.0/bin/ptxas CUDA_VISIBLE_DEVICES=0 $PTXV_PYTHON capture.py
import sys

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def matmul_tma_persistent(
    a_desc,
    b_desc,
    c_desc,
    M,
    N,
    K,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    start_pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BM)
    num_pid_n = tl.cdiv(N, BN)
    k_tiles = tl.cdiv(K, BK)
    num_tiles = num_pid_m * num_pid_n
    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True):
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n
        offs_am = pid_m * BM
        offs_bn = pid_n * BN
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BK
            a = a_desc.load([offs_am, offs_k])
            b = b_desc.load([offs_bn, offs_k])
            acc = tl.dot(a, b.T, acc)
        c_desc.store([offs_am, offs_bn], acc)


M, N, K, NUM_SMS = (
    int(x) for x in (sys.argv[1:5] if len(sys.argv) > 4 else (512, 128, 128, 2))
)
BM, BN, BK = 128, 128, 64
a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
b = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
c = torch.empty(M, N, device="cuda", dtype=torch.float32)
triton.set_allocator(
    lambda size, align, stream: torch.empty(size, dtype=torch.int8, device="cuda")
)
descs = [
    TensorDescriptor.from_tensor(a, [BM, BK]),
    TensorDescriptor.from_tensor(b, [BN, BK]),
    TensorDescriptor.from_tensor(c, [BM, BN]),
]
h = matmul_tma_persistent[(NUM_SMS,)](
    *descs, M, N, K, BM=BM, BN=BN, BK=BK, NUM_SMS=NUM_SMS
)
torch.cuda.synchronize()
assert (c - a.float() @ b.float().T).abs().max().item() < 1e-3
meta = h.metadata
open("kernel.ptx", "w").write(h.asm["ptx"])

swz = {0: "none", 1: "32B", 2: "64B", 3: "128B"}
bufs = [
    ("A", "bf16", [M, K], "normal", 1),
    ("B", "bf16", [N, K], "normal", 2),
    ("C", "f32", [M, N], "zeros", 0),
]
out = [
    f'''name = "r2_triton_gemm"
description = """
Rung 2. Triton 3.5 persistent matmul (host TensorDescriptors -> TMA, tl.dot -> tcgen05.mma),
C[{M}x{N}] f32 = A[{M}x{K}] bf16 @ B[{N}x{K}]^T, {NUM_SMS} persistent CTAs. Captured by capture.py."""
ptx = "kernel.ptx"
entry = "{h.name}"
grid = [{NUM_SMS}, 1, 1]
block = [{32 * meta.num_warps}, 1, 1]
dynamic_smem = {meta.shared}
outputs = ["C"]
spec = "gemm_tn"
'''
]
for n, dt, shape, init, seed in bufs:
    out.append(
        f'[[buffers]]\nname = "{n}"\ndtype = "{dt}"\nshape = {shape}\ninit = "{init}"\nseed = {seed}\n'
    )
# Triton expands each host descriptor into (CUtensorMap, shape..., strides...).
for (n, _, shape, _, _), tm in zip(bufs, meta.tensordesc_meta):
    box = list(reversed(tm["block_size"]))
    out.append(
        f'[[params]]\nkind = "tensormap"\nbuffer = "{n}"\nbox = {box}\nswizzle = "{swz[tm["swizzle"]]}"\n'
    )
    for s in shape:
        out.append(f'[[params]]\nkind = "u32"\nvalue = {s}\n')
    for s in (shape[1], 1):
        out.append(f'[[params]]\nkind = "u64"\nvalue = {s}\n')
for v in (M, N, K):
    out.append(f'[[params]]\nkind = "u32"\nvalue = {v}\n')
for _ in range(2):  # global / profile scratch pointers (unused here)
    out.append('[[params]]\nkind = "u64"\nvalue = 0\n')
out.append(
    '[[variants]]\nid = "good"\nsummary = "as compiled by Triton"\nexpect = ["clean"]\n'
)
open("case.toml", "w").write("\n".join(out))
import subprocess

subprocess.run([sys.executable, "variants.py"], check=True)  # markers + [[variants]]
print("captured", h.name, "warps", meta.num_warps, "smem", meta.shared)
