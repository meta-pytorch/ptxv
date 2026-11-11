# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# GB300 probe for `ptxv check --nonfinite` provenance: fp8 conversions (e4m3 satfinite, e5m2),
# MUFU.EX2 overflow, inf - inf, and min / max with a NaN operand.
import torch
import triton
import triton.language as tl


@triton.jit
def nf_probe_kernel(x_ptr, e4_ptr, e5_ptr, ex_ptr, mx_ptr, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    x = tl.load(x_ptr + i, mask=m)
    tl.store(e4_ptr + i, x.to(tl.float8e4nv), mask=m)
    tl.store(e5_ptr + i, x.to(tl.float8e5), mask=m)
    e = tl.exp2(x * 0.25)
    tl.store(ex_ptr + i, e, mask=m)
    z = e - e
    tl.store(mx_ptr + i, tl.maximum(z, 0.0), mask=m)


def inputs():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(512, generator=g) * 300.0
    return {"x": x.cuda()}


def run(x):
    n = x.numel()
    e4 = torch.empty(n, dtype=torch.float8_e4m3fn, device=x.device)
    e5 = torch.empty(n, dtype=torch.float8_e5m2, device=x.device)
    ex = torch.empty(n, dtype=torch.float32, device=x.device)
    mx = torch.empty(n, dtype=torch.float32, device=x.device)
    nf_probe_kernel[(n // 128,)](x, e4, e5, ex, mx, n, BLOCK=128)
    return e4, e5, ex, mx
