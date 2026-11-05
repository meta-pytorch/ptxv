# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# torch symmetric-memory one-shot all-reduce, two back-to-back calls that reuse the symmetric
# buffer (tools/capture_dist.py, 3 ranks). one_shot_all_reduce_copy copies the rank's input
# into its symmetric buffer inside the kernel, so nothing but the two kernels runs between the
# calls: call 2's copy overwrites the buffer that peers read in call 1.
import torch
import torch.distributed._symmetric_memory as symm_mem

N = 64
REPLICAS = ["o1", "o2"]  # all-reduce results: every rank must hold the same bits


def setup(rank, world, group):
    buf = symm_mem.empty(N, device="cuda", dtype=torch.float32)
    hdl = symm_mem.rendezvous(buf, group)
    gen = torch.Generator().manual_seed(100 + rank)
    x1 = torch.randn(N, generator=gen).cuda()
    x2 = torch.randn(N, generator=gen).cuda()
    return {"buf": buf, "x1": x1, "x2": x2, "hdl": hdl, "group": group}


def run(buf, x1, x2, hdl, group):
    o1 = torch.ops.symm_mem.one_shot_all_reduce_copy(buf, x1, "sum", group)
    o2 = torch.ops.symm_mem.one_shot_all_reduce_copy(buf, x2, "sum", group)
    return {"o1": o1, "o2": o2}
