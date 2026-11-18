# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Probe (N-MMA-BS, MISS 144): block-scaled tensor-core steps whose scaled sum leaves f32.

probes/mma_bs's kernel (../../kernel.ptx: D[128x64] f32 = 4 block-scaled tcgen05.mma steps,
scale factors written to TMEM), two kinds:
  mxf8   mxf8f6f4.block_scale 1X, e4m3 x e4m3, ue8m0 scales (idesc 143654912); A, B random
         finite codes
  mxf4   mxf4.block_scale 2X, e2m1, ue8m0 scales (idesc 143656064); A, B random bytes
SFB is 2^0 everywhere. SFA is 2^0 except rows 32..63, 2^127 for every step (every step leaves
f32), and rows 96..127, 2^127 for step 0 only (column 64 + 0 + 3, word 3). D starts at a
0x5a5a5a5a fill. GB300 (ptxas 13.0, `ptxv hw-worker`, captured in an adversarial review round;
5 plain runs identical): the overflowing rows give +-Inf, never NaN, though later steps follow
(mxf8 rows 32..63: 1025 +Inf, 1023 -Inf); rows 0..31 and 64..95 are finite. The model reports
the first step that leaves f32 as not modeled (`MMA f32 result overflow`); until MISS 144 its
round-toward-zero saturated it to +-FLT_MAX and the run was clean.
Run from this directory: python3 gen.py [--check]."""
import os
import random
import struct
import sys

SENT = 0x5A5A5A5A
# kind: (entry, idesc, A / B codes)
KINDS = {"mxf8": ("mma_mxf8", 143654912, "e4m3"), "mxf4": ("mma_mxf4", 143656064, "bytes")}


def inputs():
    """A, B, SFA, SFB per kind (the review's seeded generator, kinds in this order)."""
    rng = random.Random(77)

    def fin_e4m3():
        while True:
            c = rng.randrange(256)
            if c & 0x7F != 0x7F:
                return c

    out = {}
    for kind, (_, _, fill) in KINDS.items():
        if fill == "e4m3":
            a = [fin_e4m3() for _ in range(128 * 128)]
            b = [fin_e4m3() for _ in range(64 * 128)]
        else:
            a = [rng.randrange(256) for _ in range(128 * 128)]
            b = [rng.randrange(256) for _ in range(64 * 128)]
        sfa = [0x7F7F7F7F] * (128 * 16)
        for t in range(32, 64):
            sfa[t * 16 : (t + 1) * 16] = [0xFEFEFEFE] * 16
        for t in range(96, 128):
            sfa[t * 16 + 3] = 0xFEFEFEFE
        out[kind] = (a, b, sfa, [0x7F7F7F7F] * (128 * 16))
    return out


def case_toml(kind):
    entry, idesc, _ = KINDS[kind]
    what = {
        "mxf8": "mxf8f6f4.block_scale 1X e4m3 x e4m3, ue8m0",
        "mxf4": "mxf4.block_scale 2X e2m1, ue8m0",
    }[kind]
    return f'''name = "probe_mma_bs_ovf_{kind}"
description = """
Probe (N-MMA-BS, MISS 144; probes/mma_bs/ovf/gen.py): {what};
SFA 2^127 in rows 32..63 (every step leaves f32) and 96..127 (step 0 only). expected/ is GB300's
output (+-Inf where a step overflows); the model reports the overflowing step as not modeled."""
ptx = "../../kernel.ptx"
entry = "{entry}"
grid = [1, 1, 1]
block = [128, 1, 1]
outputs = ["D"]

[[buffers]]
name = "A"
dtype = "u8"
shape = [128, 128]
init = "file"
file = "A.bin"

[[buffers]]
name = "B"
dtype = "u8"
shape = [64, 128]
init = "file"
file = "B.bin"

[[buffers]]
name = "SFA"
dtype = "u32"
shape = [128, 16]
init = "file"
file = "SFA.bin"

[[buffers]]
name = "SFB"
dtype = "u32"
shape = [128, 16]
init = "file"
file = "SFB.bin"

[[buffers]]
name = "D"
dtype = "f32"
shape = [128, 64]
init = "file"
file = "D0.bin"
expect = "expected/out.bin"

[[params]]
kind = "tensormap"
buffer = "A"
box = [16, 128, 8]
dims = [16, 128, 8]
strides = [128, 16]

[[params]]
kind = "tensormap"
buffer = "B"
box = [16, 64, 8]
dims = [16, 64, 8]
strides = [128, 16]

[[params]]
kind = "ptr"
buffer = "D"

[[params]]
kind = "u32"
value = {idesc}

[[params]]
kind = "u32"
value = {idesc}

[[params]]
kind = "ptr"
buffer = "SFA"

[[params]]
kind = "ptr"
buffer = "SFB"

[[variants]]
id = "good"
expect = ["unmodeled"]
'''


def main():
    ins = inputs()
    if "--check" in sys.argv:
        for kind in KINDS:
            p = os.path.join(kind, "expected", "out.bin")
            d = list(struct.unpack("<8192I", open(p, "rb").read()))
            groups = ((0, 32, "control"), (32, 64, "every step"), (64, 96, "control"),
                      (96, 128, "step 0"))
            for lo, hi, what in groups:
                ws = d[lo * 64 : hi * 64]
                pinf = sum(w == 0x7F800000 for w in ws)
                ninf = sum(w == 0xFF800000 for w in ws)
                nan = sum((w & 0x7F800000) == 0x7F800000 and (w & 0x7FFFFF) != 0 for w in ws)
                print(f"{kind} rows {lo}..{hi - 1} ({what}): {pinf} +Inf, {ninf} -Inf, {nan} NaN, "
                      f"{len(ws) - pinf - ninf - nan} finite")
        return
    for kind, (a, b, sfa, sfb) in ins.items():
        os.makedirs(os.path.join(kind, "expected"), exist_ok=True)
        with open(os.path.join(kind, "A.bin"), "wb") as f:
            f.write(bytes(a))
        with open(os.path.join(kind, "B.bin"), "wb") as f:
            f.write(bytes(b))
        for name, ws in (("SFA", sfa), ("SFB", sfb), ("D0", [SENT] * (128 * 64))):
            with open(os.path.join(kind, f"{name}.bin"), "wb") as f:
                f.write(struct.pack(f"<{len(ws)}I", *ws))
        with open(os.path.join(kind, "case.toml"), "w") as f:
            f.write(case_toml(kind))


if __name__ == "__main__":
    main()
