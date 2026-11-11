# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# The fixtures/errdist/mm2.toml pair (tools/capture.py --file ... --binary):
#   PTXV_MM2=fused (default): x @ (w1 + w2): an add launch, then one cuBLASLt sgemm
#                             (fixtures/c_mm2_fused)
#   PTXV_MM2=sum:             x @ w1 + x @ w2: two sgemms, then an add launch (fixtures/c_mm2_sum)
# x [32 x 64], w1, w2 [64 x 32], f32. The checked-in cases were captured in process with
# ptxv.capture(); this script reproduces the same launches for tools/regen_fixture.py.
import os

import torch


def inputs():
    g = torch.Generator().manual_seed(0)
    return {
        "x": torch.randn(32, 64, generator=g).cuda(),
        "w1": torch.randn(64, 32, generator=g).cuda(),
        "w2": torch.randn(64, 32, generator=g).cuda(),
    }


def run(x, w1, w2):
    if os.environ.get("PTXV_MM2", "fused") == "sum":
        return x @ w1 + x @ w2
    return x @ (w1 + w2)
