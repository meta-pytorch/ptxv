#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Ordering and optimization probes: how does ptxas treat weak vs strong shared accesses?

usage: python3 gen_opt.py [OUTDIR] [-O0|-O3]
"""

import sys

import gen

OUT = sys.argv[1] if len(sys.argv) > 1 else "out_opt"
OPT = sys.argv[2] if len(sys.argv) > 2 else "-O3"

LD = {
    "weak": "ld.{s}.u32",
    "volatile": "ld.volatile.{s}.u32",
    "relaxed.cta": "ld.relaxed.cta.{s}.u32",
    "relaxed.gpu": "ld.relaxed.gpu.{s}.u32",
    "acquire.cta": "ld.acquire.cta.{s}.u32",
}
ST = {
    "weak": "st.{s}.u32",
    "volatile": "st.volatile.{s}.u32",
    "relaxed.cta": "st.relaxed.cta.{s}.u32",
    "relaxed.gpu": "st.relaxed.gpu.{s}.u32",
    "release.cta": "st.release.cta.{s}.u32",
}

V = {}
for s, sn in [("shared::cta", "cta"), ("global", "glb")]:
    for k, ln in LD.items():
        ln = ln.format(s=s)
        w = f"ld.{s}.u32"
        # O1: strong/acq load then a weak load to another address
        V[f"{sn}.O1.ld_{k}.then.ld"] = (
            s,
            f"    {ln} %r11, [A];\n    {w} %r12, [A+4];\n    add.u32 %r11, %r11, %r12;",
        )
        # O3: load then a weak store to another address
        V[f"{sn}.O3.ld_{k}.then.st"] = (
            s,
            f"    {ln} %r11, [A];\n    st.{s}.u32 [A+4], %r10;",
        )
        # P1: two identical loads (redundant-load elimination)
        V[f"{sn}.P1.2x_ld_{k}"] = (
            s,
            f"    {ln} %r11, [A];\n    {ln} %r12, [A];\n    add.u32 %r11, %r11, %r12;",
        )
        # P4: spin loop
        V[f"{sn}.P4.spin_{k}"] = (
            s,
            f"SPIN:\n    {ln} %r11, [A];\n    setp.eq.u32 %p1, %r11, 0;\n    @%p1 bra SPIN;",
        )
        # P6: spin loop with weak payload load after it (acquire-pattern)
        V[f"{sn}.P6.spin_{k}.payload"] = (
            s,
            f"SPIN:\n    {ln} %r11, [A];\n    setp.eq.u32 %p1, %r11, 0;\n    @%p1 bra SPIN;\n    {w} %r11, [A+4];",
        )
    for k, st in ST.items():
        st = st.format(s=s)
        w = f"st.{s}.u32"
        # O2: weak store then strong/rel store
        V[f"{sn}.O2.st.then.st_{k}"] = (
            s,
            f"    {w} [A+4], %r10;\n    {st} [A], %r10;\n    mov.u32 %r11, 0;",
        )
        # P2: two stores to the same address (dead-store elimination)
        V[f"{sn}.P2.2x_st_{k}"] = (
            s,
            f"    add.u32 %r12, %r10, 1;\n    {st} [A], %r10;\n    {st} [A], %r12;\n    mov.u32 %r11, 0;",
        )
        # P5: store then load same address (store-to-load forwarding)
        lk = {"release.cta": "acquire.cta"}.get(k, k)
        V[f"{sn}.P5.st_ld_fwd_{k}"] = (
            s,
            f"    {st} [A], %r10;\n    {LD[lk].format(s=s)} %r11, [A];",
        )
# P3: adjacent aligned accesses (vectorization), shared only
for k, ln in LD.items():
    ln = ln.format(s="shared::cta")
    V[f"cta.P3.adjacent_ld_{k}"] = (
        "shared::cta",
        f"    shl.b32 %r20, %r2, 4;\n    add.u32 %r21, %r3, %r20;\n    {ln} %r11, [%r21];\n    {ln} %r12, [%r21+4];\n    add.u32 %r11, %r11, %r12;",
    )
for k, st in ST.items():
    st = st.format(s="shared::cta")
    V[f"cta.P3.adjacent_st_{k}"] = (
        "shared::cta",
        f"    shl.b32 %r20, %r2, 4;\n    add.u32 %r21, %r3, %r20;\n    {st} [%r21], %r10;\n    {st} [%r21+4], %r10;\n    mov.u32 %r11, 0;",
    )

gen.run(V, OUT, OPT)
