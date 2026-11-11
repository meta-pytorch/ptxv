# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# NVFP4 GEMM through cuBLASLt (tools/capture.py --file): torch._scaled_mm with e2m1 A, B (two
# per byte, low nibble first) and ue4m3 block scales per 16 K elements, f32 out, no global
# scale. The inputs are fixtures/r3_cute_nvf4's bytes (same names and layouts: torch's blocked
# scale layout is CUTLASS's 128x4 scale atom). cuBLASLt picks
# nvjet_sm103_oosss_128x128_256x7_1x2_2cta (cta_group::2) at this shape.
import os

import numpy as np
import torch

FIX = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../../fixtures/r3_cute_nvf4"
)
M = N = 128
K = 256


def inputs():
    def rd(f):
        return torch.from_numpy(np.fromfile(os.path.join(FIX, f), np.uint8)).cuda()

    return {
        "A": rd("A.bin").view(M, K // 2),
        "B": rd("B.bin").view(N, K // 2),
        "SFA": rd("SFA.bin"),
        "SFB": rd("SFB.bin"),
    }


def run(A, B, SFA, SFB):
    f4, e4 = torch.float4_e2m1fn_x2, torch.float8_e4m3fn
    return torch._scaled_mm(
        A.view(f4), B.view(f4).t(), SFA.view(e4), SFB.view(e4), out_dtype=torch.float32
    )
