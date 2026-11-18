# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Overlap check of probes/i2f8_lat_sat: does every probe window run inside the saturating loops?

Run `good` with the clk buffer as the output (a copy of case.toml with outputs = ["clk"]):
    ptxv probe <copy> --variant good --out DIR; python3 satclk.py DIR/good.probe.bin
clk holds 4 words per warp: (SM id, clock at the loop start, clock at its end, rounds) for the
saturating (odd) warps, (SM id, clock before the window, 0, 0) for the probe (even) warps."""

import struct, sys, collections
b = open(sys.argv[1], 'rb').read()
w = struct.unpack(f'<{len(b)//4}I', b)
sat = collections.defaultdict(list); prb = collections.defaultdict(list)
for i in range(len(w)//4):
    sm, a, c, it = w[4*i:4*i+4]
    (sat if i % 2 else prb)[sm].append((a, c, it) if i % 2 else a)
inside = tot = 0; per_i2f = []; margins = []
for sm in prb:
    s0 = max(a for a, _, _ in sat[sm]); s1 = min(c for _, c, _ in sat[sm])
    for t in prb[sm]:
        tot += 1; inside += s0 < t < s1; margins.append(min(t - s0, s1 - t))
    for a, c, it in sat[sm]:
        per_i2f.append((c - a) / (it * 16))
print(f"SMs {len(prb)}; probe warps {tot}; window clock inside every saturating warp's loop on its SM: {inside}/{tot}")
margins.sort(); per_i2f.sort()
print(f"margin to the nearer loop edge (cycles): min {margins[0]} median {margins[len(margins)//2]}")
print(f"saturating warp cycles per I2F (per warp, 8 such warps per SMSP): min {per_i2f[0]:.1f} median {per_i2f[len(per_i2f)//2]:.1f} max {per_i2f[-1]:.1f}")
print(f"=> SMSP conversion issue rate ~ {8/per_i2f[len(per_i2f)//2]:.3f} warp-I2F per cycle ({32*8/per_i2f[len(per_i2f)//2]:.1f} lanes/cycle)")
