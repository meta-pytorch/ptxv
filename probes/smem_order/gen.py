#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Single-instruction mapping probes: PTX memory op -> SASS (sm_103a).

Each variant is one PTX kernel with identical surroundings and one varying instruction,
compiled with ptxas -arch=sm_103a and disassembled with nvdisasm -c -hex. Prints every
instruction between the marker-free prologue and the final STG, with its 128-bit hex and the
decoded control word, so encodings can be diffed bit by bit.

usage: python3 gen.py [OUTDIR] [-O0|-O3]
"""

import os
import re
import subprocess
import sys

CUDA = "/usr/local/cuda-13.0/bin"
OUT = sys.argv[1] if len(sys.argv) > 1 and __name__ == "__main__" else "out"
OPT = sys.argv[2] if len(sys.argv) > 2 and __name__ == "__main__" else "-O3"

HDR = """.version 9.0
.target sm_103a
.address_size 64
"""


# {ss}: state space for the address (shared::cta / shared::cluster / global)
def kernel(body, space):
    if space == "global":
        addr = """
    ld.param.u64 %rd2, [q];
    mul.wide.u32 %rd3, %r2, 4;
    add.u64 %rd4, %rd2, %rd3;"""
        a = "%rd4"
    else:
        addr = """
    mov.u32 %r3, smem;
    shl.b32 %r4, %r2, 2;
    add.u32 %r5, %r3, %r4;"""
        a = "%r5"
    return (
        HDR
        + f"""
.shared .align 16 .b32 smem[256];
.visible .entry k(.param .u64 p, .param .u64 q)
{{
    .reg .b32 %r<32>;
    .reg .b64 %rd<16>;
    .reg .pred %p<4>;
    ld.param.u64 %rd1, [p];
    mov.u32 %r2, %tid.x;
    {addr}
    mov.u32 %r10, %r2;
{body.replace("[A]", "[" + a + "]").replace("[A+4]", "[" + a + "+4]")}
    st.global.u32 [%rd1], %r11;
    ret;
}}
"""
    )


V = {}


def ld(name, op, space):
    V[name] = (space, f"    {op} %r11, [A];")


def st(name, op, space):
    V[name] = (space, f"    {op} [A], %r10;\n    mov.u32 %r11, 0;")


for sp, ssn in [("shared::cta", "cta"), ("shared::cluster", "clu"), ("global", "glb")]:
    spc = "global" if sp == "global" else sp
    ld(f"{ssn}.ld.weak", f"ld.{spc}.u32", spc)
    ld(f"{ssn}.ld.weakx", f"ld.weak.{spc}.u32", spc)
    ld(f"{ssn}.ld.volatile", f"ld.volatile.{spc}.u32", spc)
    st(f"{ssn}.st.weak", f"st.{spc}.u32", spc)
    st(f"{ssn}.st.volatile", f"st.volatile.{spc}.u32", spc)
    for sc in ["cta", "cluster", "gpu", "sys"]:
        ld(f"{ssn}.ld.relaxed.{sc}", f"ld.relaxed.{sc}.{spc}.u32", spc)
        ld(f"{ssn}.ld.acquire.{sc}", f"ld.acquire.{sc}.{spc}.u32", spc)
        st(f"{ssn}.st.relaxed.{sc}", f"st.relaxed.{sc}.{spc}.u32", spc)
        st(f"{ssn}.st.release.{sc}", f"st.release.{sc}.{spc}.u32", spc)
        V[f"{ssn}.atom.add.relaxed.{sc}"] = (
            spc,
            f"    atom.relaxed.{sc}.{spc}.add.u32 %r11, [A], %r10;",
        )
        V[f"{ssn}.atom.add.acq_rel.{sc}"] = (
            spc,
            f"    atom.acq_rel.{sc}.{spc}.add.u32 %r11, [A], %r10;",
        )
        V[f"{ssn}.atom.exch.relaxed.{sc}"] = (
            spc,
            f"    atom.relaxed.{sc}.{spc}.exch.b32 %r11, [A], %r10;",
        )
        V[f"{ssn}.red.add.relaxed.{sc}"] = (
            spc,
            f"    red.relaxed.{sc}.{spc}.add.u32 [A], %r10;\n    mov.u32 %r11, 0;",
        )
        V[f"{ssn}.red.add.release.{sc}"] = (
            spc,
            f"    red.release.{sc}.{spc}.add.u32 [A], %r10;\n    mov.u32 %r11, 0;",
        )
    V[f"{ssn}.atom.add.default"] = (spc, f"    atom.{spc}.add.u32 %r11, [A], %r10;")
    V[f"{ssn}.atom.cas.default"] = (spc, f"    atom.{spc}.cas.b32 %r11, [A], %r10, 7;")
    V[f"{ssn}.ld.mmio.sys"] = (spc, f"    ld.mmio.relaxed.sys.{spc}.u32 %r11, [A];")
    V[f"{ssn}.st.mmio.sys"] = (
        spc,
        f"    st.mmio.relaxed.sys.{spc}.u32 [A], %r10;\n    mov.u32 %r11, 0;",
    )

# fences between a weak store and a weak load (shared::cta)
for f in [
    "fence.sc.cta",
    "fence.acq_rel.cta",
    "fence.sc.cluster",
    "fence.acq_rel.cluster",
    "fence.sc.gpu",
    "fence.acq_rel.gpu",
    "fence.sc.sys",
    "fence.acq_rel.sys",
    "fence.proxy.async",
    "fence.proxy.async.shared::cta",
    "membar.cta",
    "membar.gl",
    "fence.acquire.cluster",
    "fence.release.cluster",
    "fence.mbarrier_init.release.cluster",
    "bar.sync 0",
    "bar.warp.sync -1",
]:
    V["fence." + f.replace(" ", "_")] = (
        "shared::cta",
        f"    st.shared::cta.u32 [A], %r10;\n    {f};\n    ld.shared::cta.u32 %r11, [A+4];",
    )

# mbarrier for contrast
V["mbar.arrive"] = (
    "shared::cta",
    "    .reg .b64 %st;\n    mbarrier.arrive.shared::cta.b64 %st, [A];\n    cvt.u32.u64 %r11, %st;",
)
V["mbar.arrive.relaxed"] = (
    "shared::cta",
    "    .reg .b64 %st;\n    mbarrier.arrive.relaxed.cta.shared::cta.b64 %st, [A];\n    cvt.u32.u64 %r11, %st;",
)
V["mbar.try_wait"] = (
    "shared::cta",
    "    mbarrier.try_wait.parity.shared::cta.b64 %p1, [A], %r10;\n    selp.u32 %r11, 1, 0, %p1;",
)
V["mbar.try_wait.relaxed"] = (
    "shared::cta",
    "    mbarrier.try_wait.parity.relaxed.cta.shared::cta.b64 %p1, [A], %r10;\n    selp.u32 %r11, 1, 0, %p1;",
)
V["mbar.test_wait"] = (
    "shared::cta",
    "    mbarrier.test_wait.parity.shared::cta.b64 %p1, [A], %r10;\n    selp.u32 %r11, 1, 0, %p1;",
)


def ctrl(hi):
    c = hi >> 41
    wait = c >> 11 & 63
    return (
        f"stall={c & 15:2d} y={c >> 4 & 1} wb={c >> 5 & 7} rb={c >> 8 & 7} "
        f"wait={wait:06b} reuse={c >> 17 & 15}"
    )


INS = re.compile(r"/\*([0-9a-f]{4})\*/\s+(.*?)\s*;\s*/\*\s*(0x[0-9a-f]+)\s*\*/")
HI = re.compile(r"/\*\s*(0x[0-9a-f]+)\s*\*/")


def disasm(cubin):
    d = subprocess.run(
        [f"{CUDA}/nvdisasm", "-c", "-hex", cubin],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    out = []
    for i, l in enumerate(d):
        m = INS.search(l)
        if m:
            lo = int(m.group(3), 16)
            hm = HI.search(d[i + 1])
            hi = int(hm.group(1), 16)
            out.append((m.group(1), m.group(2), lo, hi))
    return out


SKIP = re.compile(
    r"^(LDC|S2R|NOP|BRA|EXIT|MOV R1|LDCU|S2UR|ULEA|UMOV|IMAD|LEA|ISETP|SEL|MOV|IADD)"
)


def run(V, OUT=OUT, OPT=OPT):
    os.makedirs(OUT, exist_ok=True)
    for name, (space, body) in V.items():
        ptx = os.path.join(OUT, name + ".ptx")
        cub = os.path.join(OUT, name + ".cubin")
        open(ptx, "w").write(kernel(body, space))
        r = subprocess.run(
            [f"{CUDA}/ptxas", OPT, "-arch=sm_103a", ptx, "-o", cub],
            capture_output=True,
            text=True,
        )
        if r.returncode:
            err = r.stderr.strip().splitlines()
            print(f"## {name}: PTXAS ERROR: {err[0] if err else ''}")
            continue
        print(f"## {name}")
        for addr, txt, lo, hi in disasm(cub):
            # keep a NOP only if it waits on a scoreboard (ptxas' acquire idiom)
            if txt.startswith("NOP") and (hi >> 52) & 63 == 0:
                continue
            if txt.startswith(("BRA", "EXIT")) or "LDC" in txt or txt.startswith("S2R"):
                continue
            print(f"  {addr} {txt:<48} {hi:016x}_{lo:016x}  {ctrl(hi)}")


if __name__ == "__main__":
    run(V)
