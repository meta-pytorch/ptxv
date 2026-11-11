# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# How torch applies NVFP4's global f32 scales (docs/equiv-mma.md §5): F.scaled_mm with
# BlockWise1x16 + TensorWise scales (cuBLASLt's nvjet ..._2cta_h_bx kernel, alpha by device
# pointer) against torch._scaled_mm without them (fixtures/c_smm_nvf4's kernel, acc). Counts
# the outputs equal to fl(fl(GA GB) acc) and to fl(fl(acc GA) GB), over random e2m1 bytes,
# nonzero finite ue4m3 scales and GA, GB uniform in [0.01, 10).
#   CUDA_VISIBLE_DEVICES=3 $PTXV_PYTHON tools/nvf4_global_scale_check.py [--trials 200]
import argparse

import numpy as np
import torch
import torch.nn.functional as F

M = N = 128
K = 256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=200)
    a = ap.parse_args()
    f4, e4 = torch.float4_e2m1fn_x2, torch.float8_e4m3fn
    bw, tw = F.ScalingType.BlockWise1x16, F.ScalingType.TensorWise
    sw, ns = F.SwizzleType.SWIZZLE_32_4_4, F.SwizzleType.NO_SWIZZLE
    rng = np.random.default_rng(1)
    tot = alpha_first = assoc = 0
    for _ in range(a.trials):

        def u8(*s, lo=0, hi=256):
            return torch.from_numpy(rng.integers(lo, hi, s, dtype=np.uint8)).cuda()

        A, B = u8(M, K // 2), u8(N, K // 2)
        SA, SB = u8(M * K // 16, lo=1, hi=127), u8(N * K // 16, lo=1, hi=127)
        ga, gb = np.float32(rng.uniform(0.01, 10)), np.float32(rng.uniform(0.01, 10))
        acc = torch._scaled_mm(
            A.view(f4),
            B.view(f4).t(),
            SA.view(e4),
            SB.view(e4),
            out_dtype=torch.float32,
        )
        g = F.scaled_mm(
            A.view(f4),
            B.view(f4).t(),
            [SA.view(e4), torch.tensor(ga).cuda()],
            [bw, tw],
            [SB.view(e4), torch.tensor(gb).cuda()],
            [bw, tw],
            swizzle_a=[sw, ns],
            swizzle_b=[sw, ns],
            output_dtype=torch.float32,
        )
        acc, g = acc.cpu().numpy(), g.cpu().numpy().view(np.uint32)
        tot += g.size
        alpha_first += (g == (acc * np.float32(ga * gb)).view(np.uint32)).sum()
        assoc += (g == ((acc * ga).astype(np.float32) * gb).view(np.uint32)).sum()
    print(
        f"{tot} outputs: {alpha_first} equal fl(fl(GA GB) acc), {assoc} equal fl(fl(acc GA) GB)"
    )


if __name__ == "__main__":
    main()
