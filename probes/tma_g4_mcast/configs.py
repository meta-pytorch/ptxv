# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Configs of probes/tma_g4_mcast (expected/configs.json): (pair, ctaMask, mbarrier rank,
destination rank), each rank expecting exactly the bytes S-TMA-2CTA-MCAST delivers to it (a
256-byte gather per masked rank r, on rank r's own mbarrier, or with .cta_group::2 on the
mbarrier of rank (r & ~1) | (mbar & 1)). Each config ran on GB300 as a case like
probes/tma_2cta_mcast's (tensor map over a 32 x 16 u32 iota tensor, box [16, 1]; the five u32
parameters mask, mbar_rank, dst_rank, exp, pair) with `ptxv hw-worker <case> <ptxas cubin>
--runs 5 --dump expected/<name>.bin`, all runs alike."""

import json

BOX = 256


def delivered(pair, mask, mbar):
    d = [0] * 4
    for r in range(4):
        if mask >> r & 1:
            d[(r & ~1) | (mbar & 1) if pair else r] += BOX
    return d


cfgs = []
for pair in (1, 0):
    for mask in (0xF, 0x1, 0x2, 0x8, 0x3, 0x6, 0x5):
        for mbar in (0, 1):
            for dst in (0, 1):
                dl = delivered(pair, mask, mbar)
                cfgs.append(
                    dict(
                        name=f"p{pair}_m{mask:x}_b{mbar}_d{dst}",
                        pair=pair,
                        mask=mask,
                        mbar_rank=mbar,
                        dst_rank=dst,
                        exp=sum((e // 128) << (8 * r) for r, e in enumerate(dl)),
                        expect=dl,
                        delivered=dl,
                    )
                )
json.dump(cfgs, open("expected/configs.json", "w"), indent=0)
