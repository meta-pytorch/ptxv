# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# torch._grouped_mm with 2 groups on one 2-CTA cluster (the SM carveout leaves 2 SMs), so one
# CTA pair runs both groups: CUTLASS's async tensormap-updater warps prepare the second
# group's maps (rung 4b, ASSUMPTIONS.md S-TMAP). Single GPU:
#   CUDA_VISIBLE_DEVICES=3 $PTXV_PYTHON tools/capture.py --file tools/capture_examples/grouped_mm_2groups.py --out DIR
import torch


def inputs():
    torch._C._set_sm_carveout_experimental(
        torch.cuda.get_device_properties(0).multi_processor_count - 2
    )
    g = torch.Generator().manual_seed(0)
    return {
        "xs": torch.randn(9, 128, generator=g).to(torch.bfloat16).cuda(),
        "W": (torch.randn(2, 128, 128, generator=g) / 11).to(torch.bfloat16).cuda(),
        "off": torch.tensor([4, 9], dtype=torch.int32).cuda(),
    }


def run(xs, W, off):
    return torch._grouped_mm(xs, W, offs=off)
