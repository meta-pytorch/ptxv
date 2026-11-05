# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Kernels from three modules in one capture (tools/capture.py --file mixed.py): a cuBLAS GEMM,
# a libtorch_cuda elementwise kernel on a view into the middle of an allocation, and a
# Triton kernel.
import torch
import triton
import triton.language as tl


@triton.jit
def scale_kernel(x_ptr, y_ptr, n, s, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    tl.store(y_ptr + i, tl.load(x_ptr + i, mask=m) * s, mask=m)


def inputs():
    g = torch.Generator().manual_seed(0)
    return {
        "a": torch.randn(64, 64, generator=g).to(torch.bfloat16).cuda(),
        "b": torch.randn(64, 64, generator=g).to(torch.bfloat16).cuda(),
    }


def run(a, b):
    c = a @ b
    d = c[16:48].float()  # reads from the middle of c's allocation
    e = torch.empty_like(d)
    scale_kernel[(triton.cdiv(d.numel(), 1024),)](d, e, d.numel(), 0.5, BLOCK=1024)
    return e
