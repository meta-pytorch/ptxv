#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Writes probes/cluster_launch_ops/<op>/ (case.toml, kernel.ptx): one cluster-scope operation per
kernel, run with .reqnctapercluster 1, 1, 1 (variant cluster1) and without (variant none, the
directive deleted: a launch without cluster dimensions); and st.async in two more launches:
stasync_attr1 (no directive, case.toml cluster = [1, 1, 1]: the runner's launch attribute) and
stasync_explicit (.explicitcluster, no shape). Each variant's expected classes are GB300's result
(GB300 below: "runs" -> clean, "traps" -> fault).
usage: gen.py PROBE_DIR (probes/cluster_launch_ops)"""
import os
import sys

P = sys.argv[1]
HDR = """// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
"""
WAIT = """WAIT:
    mbarrier.try_wait.parity.shared::cta.b64 %p1, [%r22], 0;
    @!%p1 bra WAIT;"""
EXPECT128 = """    @%p2 mbarrier.arrive.expect_tx.shared::cta.b64 %rd9, [%r22], 128;
    bar.sync 0;"""
OPS = {
    "stasync": ("st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 of the thread's own word (S[t] = t + 7)",
                EXPECT128 + "\n    add.u32 %r10, %r0, 7;\n    st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%r21], %r10, [%r22];\n" + WAIT),
    "stasync_v2": ("st.async ... .v2.b32 by threads 0..15 (S[2t], S[2t + 1] = t + 7, t + 8)",
                   "    @%p2 mbarrier.arrive.expect_tx.shared::cta.b64 %rd9, [%r22], 128;\n    bar.sync 0;\n"
                   "    setp.lt.u32 %p3, %r0, 16;\n    mad.lo.u32 %r12, %r0, 8, %r20;\n    add.u32 %r10, %r0, 7;\n    add.u32 %r11, %r0, 8;\n"
                   "    @%p3 st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.b32 [%r12], {%r10, %r11}, [%r22];\n" + WAIT),
    "redasync": ("red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.add.u32 of 7 into S[t]",
                 EXPECT128 + "\n    red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.add.u32 [%r21], 7, [%r22];\n" + WAIT),
    "mbar_remote_arrive": ("mbarrier.arrive.release.cluster.shared::cluster.b64 on the CTA's own mbarrier through mapa (rank 0)",
                           "    mapa.shared::cluster.u32 %r23, %r22, 0;\n    @%p2 mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%r23];\n" + WAIT),
    "mbar_remote_expect_tx": ("mbarrier.expect_tx.relaxed.cluster.shared::cluster.b64 through mapa (rank 0), then a local arrive",
                              "    mapa.shared::cluster.u32 %r23, %r22, 0;\n    @%p2 mbarrier.expect_tx.relaxed.cluster.shared::cluster.b64 [%r23], 0;\n"
                              "    @%p2 mbarrier.arrive.shared::cta.b64 %rd9, [%r22];\n" + WAIT),
    "ld_cluster": ("ld.shared::cluster.u32 of S[t] through mapa (rank 0)",
                   "    mapa.shared::cluster.u32 %r23, %r21, 0;\n    ld.shared::cluster.u32 %r10, [%r23];\n    add.u32 %r10, %r10, 7;\n    st.shared.u32 [%r21], %r10;"),
    "st_cluster": ("st.shared::cluster.u32 of t + 7 to S[t] through mapa (rank 0)",
                   "    mapa.shared::cluster.u32 %r23, %r21, 0;\n    add.u32 %r10, %r0, 7;\n    st.shared::cluster.u32 [%r23], %r10;\n    bar.sync 0;"),
    "atom_cluster": ("atom.shared::cluster.add.u32 of 7 to S[t] through mapa (rank 0)",
                     "    mapa.shared::cluster.u32 %r23, %r21, 0;\n    atom.shared::cluster.add.u32 %r10, [%r23], 7;\n    bar.sync 0;"),
    "mapa_cta": ("mapa.shared::cluster.u32 (rank 0) of S[t], then st.shared of t + 7 through it",
                 "    mapa.shared::cluster.u32 %r23, %r21, 0;\n    add.u32 %r10, %r0, 7;\n    st.shared.u32 [%r23], %r10;\n    bar.sync 0;"),
    "barrier_cluster": ("barrier.cluster.arrive.release / wait.acquire, then S[t] += 7",
                        "    barrier.cluster.arrive.release.aligned;\n    barrier.cluster.wait.acquire.aligned;\n"
                        "    ld.shared.u32 %r10, [%r21];\n    add.u32 %r10, %r10, 7;\n    st.shared.u32 [%r21], %r10;"),
    "bulk_s2s": ("cp.async.bulk.shared::cluster.shared::cta of S[0..4) to S[32..36) (thread 0), mbarrier complete_tx",
                 "    @%p2 mbarrier.arrive.expect_tx.shared::cta.b64 %rd9, [%r22], 16;\n    bar.sync 0;\n"
                 "    add.u32 %r24, %r20, 128;\n    mapa.shared::cluster.u32 %r25, %r24, 0;\n"
                 "    @%p2 cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%r25], [%r20], 16, [%r22];\n" + WAIT +
                 "\n    setp.lt.u32 %p3, %r0, 4;\n    @%p3 ld.shared.u32 %r10, [%r21 + 128];\n    @%p3 add.u32 %r10, %r10, 7;\n    @%p3 st.shared.u32 [%r21], %r10;"),
    "bulk_g2s": ("cp.async.bulk.shared::cluster.global of 16 bytes of src to S[0..4) (thread 0), mbarrier complete_tx",
                 "    @%p2 mbarrier.arrive.expect_tx.shared::cta.b64 %rd9, [%r22], 16;\n    bar.sync 0;\n"
                 "    @%p2 cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%r20], [%rd5], 16, [%r22];\n" + WAIT),
    "bulk_g2s_mcast": ("cp.async.bulk.shared::cluster.global ... .multicast::cluster, mask 1 (thread 0)",
                       "    @%p2 mbarrier.arrive.expect_tx.shared::cta.b64 %rd9, [%r22], 16;\n    bar.sync 0;\n    mov.b16 %rs1, 1;\n"
                       "    @%p2 cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%r20], [%rd5], 16, [%r22], %rs1;\n" + WAIT),
    "fence_cluster": ("fence.acq_rel.cluster; fence.proxy.async.shared::cluster, then S[t] += 7",
                      "    fence.acq_rel.cluster;\n    fence.proxy.async.shared::cluster;\n"
                      "    ld.shared.u32 %r10, [%r21];\n    add.u32 %r10, %r10, 7;\n    st.shared.u32 [%r21], %r10;"),
    "wait_cluster": ("mbarrier.arrive (local) and try_wait.parity.acquire.cluster",
                     "    @%p2 mbarrier.arrive.shared::cta.b64 %rd9, [%r22];\nWAITC:\n"
                     "    mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 %p1, [%r22], 0;\n    @!%p1 bra WAITC;"),
    "sregs": ("%cluster_ctarank, %cluster_nctarank, %is_explicit_cluster, %clusterid.x in S[t] (t = 0..3)",
              "    mov.u32 %r10, %cluster_ctarank;\n    mov.u32 %r11, %cluster_nctarank;\n    mov.pred %p0, %is_explicit_cluster;\n    selp.u32 %r12, 1, 0, %p0;\n    mov.u32 %r13, %clusterid.x;\n"
              "    setp.eq.u32 %p3, %r0, 0;\n    @%p3 st.shared.u32 [%r21], %r10;\n    setp.eq.u32 %p3, %r0, 1;\n    @%p3 st.shared.u32 [%r21], %r11;\n"
              "    setp.eq.u32 %p3, %r0, 2;\n    @%p3 st.shared.u32 [%r21], %r12;\n    setp.eq.u32 %p3, %r0, 3;\n    @%p3 st.shared.u32 [%r21], %r13;\n    bar.sync 0;"),
    "clc_cancel": ("clusterlaunchcontrol.try_cancel (thread 0) of its 16-byte response into S[0..4), mbarrier complete_tx",
                   "    @%p2 mbarrier.arrive.expect_tx.shared::cta.b64 %rd9, [%r22], 16;\n    bar.sync 0;\n"
                   "    @%p2 clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.b128 [%r20], [%r22];\n" + WAIT),
    "clc_cancel_mc": ("clusterlaunchcontrol.try_cancel ... .multicast::cluster::all (thread 0) into S[0..4)",
                      "    @%p2 mbarrier.arrive.expect_tx.shared::cta.b64 %rd9, [%r22], 16;\n    bar.sync 0;\n"
                      "    @%p2 clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.multicast::cluster::all.b128 [%r20], [%r22];\n" + WAIT),
}
# GB300 (`ptxv probe`, 2026-11-12): (cluster1, none). The four that trap without a cluster
# launch report "an illegal instruction was encountered" at cuCtxSynchronize.
GB300 = {op: ("runs", "runs") for op in OPS}
GB300.update({op: ("runs", "traps") for op in ("stasync", "stasync_v2", "redasync", "bulk_s2s")})
# st.async in the other launches: (directive line, case.toml cluster line, GB300 result).
LAUNCHES = {
    "stasync_attr1": ("", "cluster = [1, 1, 1]\n", "runs",
                      "no directive; case.toml cluster = [1, 1, 1] (the runner passes it as the launch attribute)"),
    "stasync_explicit": (".explicitcluster", "", "runs",
                         ".explicitcluster without a shape; no cluster in case.toml (a 1 x 1 x 1 cluster launch)"),
}


def expect(r, op=None, front_extra=""):
    return '["fault"]' if r == "traps" else '["clean"]'


def said(r):
    return {"runs": "runs", "traps": "traps (illegal instruction)", None: "not measured yet"}[r]
# ptxv's classes where they are more than GB300's trap: on the SASS front lanes 16..31 of
# stasync_v2 wait on the mbarrier the faulting lanes 0..15 never complete.
EXPECT_SASS = {("stasync_v2", "none"): '["deadlock", "fault"]'}


def kernel(op, what, body, directive, note):
    return HDR + f"""
// probes/cluster_launch_ops/{op} (MISS 209): {what}; out[t] = S[t] afterwards. {note}
.version 9.0
.target sm_103a
.address_size 64

.visible .entry cl_{op}(.param .u64 pout, .param .u64 psrc)
{directive}{{
    .reg .pred %p<4>;
    .reg .b16 %rs<4>;
    .reg .b32 %r<32>;
    .reg .b64 %rd<16>;
    .shared .align 16 .b8 S[256];
    .shared .align 8 .b64 MB;
    ld.param.u64 %rd1, [pout];
    ld.param.u64 %rd4, [psrc];
    cvta.to.global.u64 %rd1, %rd1;
    cvta.to.global.u64 %rd5, %rd4;
    mov.u32 %r0, %tid.x;
    mul.wide.u32 %rd2, %r0, 4;
    add.s64 %rd3, %rd1, %rd2;
    mov.u32 %r20, S;
    mad.lo.u32 %r21, %r0, 4, %r20;
    mov.u32 %r22, MB;
    st.shared.u32 [%r21], %r0;
    setp.eq.u32 %p2, %r0, 0;
    @%p2 mbarrier.init.shared::cta.b64 [%r22], 1;
    fence.mbarrier_init.release.cluster;
    fence.proxy.async.shared::cta;
    bar.sync 0;
{body}
    bar.sync 0;
    ld.shared.u32 %r9, [%r21];
    st.global.u32 [%rd3], %r9;
    ret;
}}
"""


def case_toml(op, description, cluster_line, variants):
    return f'''name = "cluster_launch_ops_{op}"
description = """
{description}"""
ptx = "kernel.ptx"
arch = "sm_103a"
entry = "cl_{op}"
grid = [1, 1, 1]
{cluster_line}block = [32, 1, 1]
outputs = ["out"]

[[buffers]]
name = "out"
dtype = "u32"
shape = [32]
init = "zeros"

[[buffers]]
name = "src"
dtype = "u32"
shape = [4]
init = "file"
file = "../src.bin"

[[params]]
kind = "ptr"
buffer = "out"

[[params]]
kind = "ptr"
buffer = "src"
''' + variants


def variant(op, vid, summary, result, extra=""):
    v = f'''
[[variants]]
id = "{vid}"
summary = "{summary}"
{extra}expect = {expect(result)}
'''
    if (op, vid) in EXPECT_SASS:
        v += f"expect_sass = {EXPECT_SASS[(op, vid)]}\n"
    return v


for op, (what, body) in OPS.items():
    d = os.path.join(P, op)
    os.makedirs(d, exist_ok=True)
    note = ("Variant cluster1\n// keeps .reqnctapercluster 1, 1, 1 (a cluster launch); variant none deletes it (a launch without\n"
            "// cluster dimensions).")
    open(os.path.join(d, "kernel.ptx"), "w").write(
        kernel(op, what, body, ".reqnctapercluster 1, 1, 1                                          // @NOCLUSTER\n", note))
    c1, none = GB300[op]
    desc = (f"MISS 209 hardware probe: {what}; out[t] = S[t] afterwards (S[t] starts at t). Variant cluster1:\n"
            f".reqnctapercluster 1, 1, 1; variant none: no cluster dimensions. GB300 (ptxv probe): cluster1\n"
            f"{said(c1)}, none {said(none)} (expected/{op}.gb300.txt).")
    vs = (variant(op, "cluster1", ".reqnctapercluster 1, 1, 1", c1)
          + variant(op, "none", "no cluster dimensions", none, 'marker = "@NOCLUSTER"\nreplace = "<delete>"\n'))
    open(os.path.join(d, "case.toml"), "w").write(case_toml(op, desc, "", vs))
for op, (directive, cluster_line, result, how) in LAUNCHES.items():
    d = os.path.join(P, op)
    os.makedirs(d, exist_ok=True)
    what, body = OPS["stasync"]
    note = f"Launch: {how}."
    open(os.path.join(d, "kernel.ptx"), "w").write(
        kernel(op, what, body, directive + "\n" if directive else "", note))
    desc = (f"MISS 209 hardware probe: {what}; out[t] = S[t] afterwards (S[t] starts at t). Launch: {how}.\n"
            f"GB300 (ptxv probe): {said(result)} (expected/{op}.gb300.txt).")
    open(os.path.join(d, "case.toml"), "w").write(case_toml(op, desc, cluster_line, variant(op, "good", how.split(";")[0], result)))
import struct
open(os.path.join(P, "src.bin"), "wb").write(struct.pack("<4I", 0x1111, 0x2222, 0x3333, 0x4444))
print(len(OPS) + len(LAUNCHES), "cases")
