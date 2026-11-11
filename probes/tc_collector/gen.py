# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Writes kernel.ptx, A.bin, B.bin for probes/tc_collector: tcgen05.mma `.collector::a::*`
(S-TC-COLLECTOR). One CTA, kind::f16, M = 128, N = 64, K = 16 per MMA. A = 4 K chunks
(chunk c: smem A + 4096 c), B = 4 chunks; MMA i writes D_i = TMEM columns 64 i..64 i + 63
(enable-input-d false). The `mode` constant selects the qualifier sequence:

  mode  MMA0          MMA1            MMA2              MMA3
  0     fill A0 B0    use A0 B1       lastuse A0 B2     -  A0 B3   (contract holds)
  1     fill A0 B0    use A0 B1 *     lastuse A0 B2 *   -  A0 B3 * (* A0's smem overwritten
                                                                    with chunk 1 after MMA0)
  2     fill A0 B0    use A1 B1       lastuse A1 B2     -  A0 B3   (other descriptor)
  3     use A0 B0     -  A0 B1        -  A0 B2          -  A0 B3   (no fill)
  4     fill A0 B0    lastuse A0 B1   use A0 B2         -  A0 B3   (use after lastuse)
  5     fill A0 B0    -  A1 B1        use A0 B2         -  A0 B3   (use after a discard)
  6     fill A0 B0    use A0 B1       lastuse A0 B2     (thread 32: use A0 B3, unordered)

`-` = no qualifier (= collector::a::discard). MMA0 is committed and waited on before the rest
(mode 1 rewrites A0 then, with fence.proxy.async)."""

import random
import struct
import sys

d = sys.argv[1] if len(sys.argv) > 1 else "."
r = random.Random(23)


def vals(n):
    out = []
    for _ in range(n):
        x = r.gauss(0.0, 1.0)
        if abs(x) < 2.0**-14:
            x = 2.0**-14 if x >= 0 else -(2.0**-14)
        out.append(x)
    return out


open(f"{d}/A.bin", "wb").write(struct.pack("<8192e", *vals(128 * 64)))
open(f"{d}/B.bin", "wb").write(struct.pack("<4096e", *vals(64 * 64)))

Q = {
    "f": ".collector::a::fill",
    "u": ".collector::a::use",
    "l": ".collector::a::lastuse",
    "-": "",
}
MODES = [
    [("f", 0, 0), ("u", 0, 1), ("l", 0, 2), ("-", 0, 3)],
    [("f", 0, 0), ("u", 0, 1), ("l", 0, 2), ("-", 0, 3)],
    [("f", 0, 0), ("u", 1, 1), ("l", 1, 2), ("-", 0, 3)],
    [("u", 0, 0), ("-", 0, 1), ("-", 0, 2), ("-", 0, 3)],
    [("f", 0, 0), ("l", 0, 1), ("u", 0, 2), ("-", 0, 3)],
    [("f", 0, 0), ("-", 1, 1), ("u", 0, 2), ("-", 0, 3)],
    [("f", 0, 0), ("u", 0, 1), ("l", 0, 2)],
]


def mma(i, q, a, b):
    return (
        f"    add.u32         %r20, %r5, {64 * i};\n"
        f"    add.u64         %rd8, %rd5, {256 * a};\n"
        f"    add.u64         %rd9, %rd7, {128 * b};\n"
        f"    tcgen05.mma.cta_group::1.kind::f16{Q[q]} [%r20], %rd8, %rd9, %r11, %p3;\n"
    )


body = []
for m, seq in enumerate(MODES):
    body.append(
        f"MODE{m}:\n    setp.ne.u32     %p6, %r40, {m};\n    @%p6 bra        MODE{m + 1};\n"
    )
    body.append(mma(0, *seq[0]))
    body.append(
        "    tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%r4];\n"
        f"W{m}:\n    mbarrier.try_wait.parity.shared::cta.b64 %p2, [%r4], 0;\n"
        f"    @!%p2 bra       W{m};\n    tcgen05.fence::after_thread_sync;\n"
    )
    if m == 1:
        body.append(
            "    // A0 <- chunk 1 (16 bytes at a time), then make it visible to the async proxy.\n"
            "    mov.u32         %r21, 0;\n"
            "CPY:\n"
            "    add.u32         %r22, %r6, %r21;\n"
            "    ld.shared.v4.b32 {%r24, %r25, %r26, %r27}, [%r22+4096];\n"
            "    st.shared.v4.b32 [%r22], {%r24, %r25, %r26, %r27};\n"
            "    add.u32         %r21, %r21, 16;\n"
            "    setp.lt.u32     %p7, %r21, 4096;\n"
            "    @%p7 bra        CPY;\n"
            "    fence.proxy.async.shared::cta;\n"
        )
    for i in range(1, len(seq)):
        body.append(mma(i, *seq[i]))
    body.append("    bra.uni         ISSUED;\n")
body.append(f"MODE{len(MODES)}:\nISSUED:\n")

src = open(f"{d}/kernel.tmpl").read().replace("@BODY@\n", "".join(body))
open(f"{d}/kernel.ptx", "w").write(src)
