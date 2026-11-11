# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# torch symmetric-memory two-shot all-reduce (in place), two back-to-back calls on the same
# symmetric buffer (tools/capture_dist.py, 3 ranks). Each rank reduces its third of the buffer
# from every peer and writes the sum into every peer's buffer; call 2 reduces call 1's result.
import torch
import torch.distributed._symmetric_memory as symm_mem

N = 64
REPLICAS = ["buf"]  # all-reduce results: every rank must hold the same bits


def setup(rank, world, group):
    buf = symm_mem.empty(N, device="cuda", dtype=torch.float32)
    hdl = symm_mem.rendezvous(buf, group)
    gen = torch.Generator().manual_seed(200 + rank)
    x = torch.randn(N, generator=gen).cuda()
    return {"buf": buf, "x": x, "hdl": hdl, "group": group}


def run(buf, x, hdl, group):
    buf.copy_(x)  # a device copy before the first kernel: the buffer's initial contents
    torch.ops.symm_mem.two_shot_all_reduce_(buf, "sum", group)
    torch.ops.symm_mem.two_shot_all_reduce_(buf, "sum", group)
    return {"buf": buf}
