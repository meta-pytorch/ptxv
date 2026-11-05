# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# MXF8 GEMM through cuBLASLt (tools/capture.py --file): torch._scaled_mm with e4m3 A, B and
# ue8m0 block scales per 32 K elements, f32 out. The inputs are fixtures/r3_cute_mxf8's bytes
# (same names and layouts: torch's blocked scale layout is CUTLASS's 128x4 scale atom), so the
# two cases' leaves are the same tensors.
import os

import numpy as np
import torch

FIX = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../../fixtures/r3_cute_mxf8"
)
M = N = 128
K = 256


def inputs():
    def rd(f):
        return torch.from_numpy(np.fromfile(os.path.join(FIX, f), np.uint8)).cuda()

    return {
        "A": rd("A.bin").view(M, K),
        "B": rd("B.bin").view(N, K),
        "SFA": rd("SFA.bin"),
        "SFB": rd("SFB.bin"),
    }


def run(A, B, SFA, SFB):
    e4, e8 = torch.float8_e4m3fn, torch.float8_e8m0fnu
    return torch._scaled_mm(
        A.view(e4), B.view(e4).t(), SFA.view(e8), SFB.view(e8), out_dtype=torch.float32
    )
