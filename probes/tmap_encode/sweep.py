# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Random cuTensorMapEncodeTiled sweep for docs/tensormap.md.

    gcc -O2 -o /tmp/tmap_encode tmap_encode.c -I/usr/local/cuda-13.0/include -ldl
    CUDA_VISIBLE_DEVICES=0 python3 sweep.py /tmp/tmap_encode SEED > out.txt

Prints one "args => hex" line per accepted encode (rejected argument sets are dropped).
samples.txt (checked by crates/sim/src/tmap.rs `encoder_sweep`) is every 8th line of the
seed 1 and seed 2 outputs (3097 accepted samples in total, all of which the decoder matched).
"""

import random
import subprocess
import sys

binary = sys.argv[1]
random.seed(int(sys.argv[2]) if len(sys.argv) > 2 else 1)
# CUtensorMapDataType -> element size (sub-byte types 13..15 are probed by hand).
ES = {0: 1, 1: 2, 2: 4, 3: 4, 4: 8, 5: 8, 6: 2, 7: 4, 8: 8, 9: 2, 10: 4, 11: 4, 12: 4}


def enc(lines):
    out = subprocess.run(
        [binary],
        input="\n".join(lines) + "\n",
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [ln.split(" => ")[1] for ln in out.strip().split("\n")]


cfgs = []
for _ in range(3000):
    dt = random.choice(list(ES))
    es = ES[dt]
    rank = random.randint(1, 5)
    il = random.choice([0, 0, 0, 1, 2]) if rank >= 3 else 0
    sw = random.randint(0, 6)
    l2 = random.randint(0, 3)
    oob = random.randint(0, 1)
    if oob and dt in (0, 1, 2, 3, 4, 5):
        oob = 0
    addr = random.randrange(0, 1 << 48) & ~0xFF
    dims = [
        random.choice(
            [
                1,
                2,
                3,
                7,
                8,
                16,
                64,
                100,
                128,
                256,
                1000,
                4096,
                65536,
                (1 << 20) + 3,
                1 << 31,
                1 << 32,
            ]
        )
        for _ in range(rank)
    ]
    strides = []
    s = dims[0] * es
    for i in range(1, rank):
        s = ((s + random.choice([0, 0, 16, 4096])) + 15) // 16 * 16
        if il == 2:
            s = (s + 31) // 32 * 32
        if random.random() < 0.2:
            s = random.randrange(16, 1 << 40, 16) // 32 * 32
        strides.append(s)
        s = s * dims[i]
    box = [random.choice([1, 2, 4, 8, 16, 32, 64, 128, 256]) for _ in range(rank)]
    if il == 0:
        while box[0] * es % 16:
            box[0] *= 2
    estr = [random.randint(1, 8) for _ in range(rank)]
    cfgs.append([dt, rank, il, sw, l2, oob, addr] + dims + strides + box + estr)
lines = [" ".join(map(str, c)) for c in cfgs]
res = enc(lines)
ok = [(c, h) for c, h in zip(cfgs, res) if not h.startswith("ERR")]
print("ok", len(ok), "of", len(cfgs), file=sys.stderr)
for c, h in ok:
    print(" ".join(map(str, c)), "=>", h)
