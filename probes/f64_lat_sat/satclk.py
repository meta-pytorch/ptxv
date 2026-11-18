# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Overlap check of probes/f64_lat_sat: does every probe window run inside the saturating loops?

Run a variant with the clk buffer as the output (a copy of case.toml with outputs = ["clk"]):
    ptxv probe <copy> --variant V --out DIR; python3 satclk.py DIR/V.probe.bin MASK
MASK: the pool selector (1: the odd warps saturate; 4: _Q variants). clk holds 8 words per warp:
saturating warps (SM id, warp slot, clock at the loop start, at its end, rounds); probe warps
(SM id, warp slot, clock before the window, the nanosleep delay, clock after the window's
stores). Clocks are SR_CLOCKLO (32 bits, compared modulo 2^32 on each SM)."""

import collections
import struct
import sys

b = open(sys.argv[1], "rb").read()
mask = int(sys.argv[2])
w = struct.unpack(f"<{len(b) // 4}I", b)
sat = collections.defaultdict(list)
prb = collections.defaultdict(list)
part = collections.Counter()
for i in range(len(w) // 8):
    sm, slot, a, x, c = w[8 * i : 8 * i + 5]
    s = (i % 32) & mask != 0
    (sat if s else prb)[sm].append((slot, a, x, c))
    part[(sm, slot % 4, s)] += 1
inside = tot = 0
margins, elapsed, per_op = [], [], []
for sm in prb:
    ref = min(a for _, a, _, _ in sat[sm])
    r = lambda t: (t - ref) & 0xFFFFFFFF
    s0 = max(r(a) for _, a, _, _ in sat[sm])
    s1 = min(r(e) for _, _, e, _ in sat[sm])
    for _, a, _, c in prb[sm]:
        tot += 1
        t0, t1 = r(a), r(c)
        inside += s0 < t0 and t1 < s1
        margins.append(min(t0 - s0, s1 - t1))
        elapsed.append(t1 - t0)
    for _, a, e, it in sat[sm]:
        per_op.append(((e - a) & 0xFFFFFFFF) / (it * 16))
sms = sorted(prb)
both = sum(1 for sm in sms for p in range(4) if part[(sm, p, True)] and part[(sm, p, False)])
print(f"SMs {len(sms)}; probe warps {tot}; window (clock before, clock after) inside every saturating warp's loop on its SM: {inside}/{tot}")
margins.sort(); elapsed.sort(); per_op.sort()
print(f"margin to the nearer loop edge (cycles): min {margins[0]} median {margins[len(margins) // 2]}")
print(f"window elapsed (clock after - before, cycles): min {elapsed[0]} median {elapsed[len(elapsed) // 2]} max {elapsed[-1]}")
print(f"saturating warp cycles per f64 op: min {per_op[0]:.1f} median {per_op[len(per_op) // 2]:.1f} max {per_op[-1]:.1f}")
print(f"(SM, warp slot mod 4) groups holding both pools: {both}/{4 * len(sms)}")
