# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# CuTeDSL elementwise add (one thread per element), for
# tools/capture.py --file cute_add.py (PTX from CUTE_DSL_KEEP, so a PTX case).
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack


@cute.kernel
def add_kernel(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()
    i = bidx * bdim + tidx
    m, n = gA.shape
    gC[i // n, i % n] = gA[i // n, i % n] + gB[i // n, i % n]


@cute.jit
def add(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
    m, n = mA.shape
    add_kernel(mA, mB, mC).launch(grid=((m * n) // 128, 1, 1), block=(128, 1, 1))


_compiled = {}


def inputs():
    g = torch.Generator().manual_seed(0)
    return {
        "a": torch.randn(4, 128, generator=g).cuda(),
        "b": torch.randn(4, 128, generator=g).cuda(),
    }


def run(a, b):
    c = torch.empty_like(a)
    t = [from_dlpack(x) for x in (a, b, c)]
    if "f" not in _compiled:
        _compiled["f"] = cute.compile(add, *t)
    _compiled["f"](*t)
    return c
