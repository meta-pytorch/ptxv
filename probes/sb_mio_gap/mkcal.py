# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# kernel_cal.ptx for probes/sb_mio_gap's calibration runs: kernel.ptx with a %clock read before
# and after the placeholder run and their difference (plus the xor word, 0) as the fourth output
# word (patch.py's cal_<G> variants move both reads into the window: the chain's cycles).  python3 mkcal.py kernel.ptx > kernel_cal.ptx
import re
import sys

src = open(sys.argv[1]).read().split("\n")
ph = [i for i, l in enumerate(src) if re.match(r"\s+st\.volatile\.shared\.u32 \[%r4\+12288\], %r5;", l)]
src.insert(ph[-1] + 1, "    @%p1 mov.u32 %r76, %clock;")
# Both reads under a per-thread predicate (always true): ptxas then emits CS2R, not S2UR.
src.insert(ph[0], "    setp.ne.u32 %p1, %r1, 99999;\n    @%p1 mov.u32 %r75, %clock;")
out = "\n".join(src)
out = out.replace(
    "st.global.v4.u32 [%rd6], {%r50, %r51, %r52, %r53};",
    "sub.u32 %r77, %r76, %r75;\n    add.u32 %r77, %r77, %r53;\n    st.global.v4.u32 [%rd6], {%r50, %r51, %r52, %r77};",
)
assert "%r77}" in out
sys.stdout.write(out)
