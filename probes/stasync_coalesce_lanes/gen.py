#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Generates the coalesced-st.async cases of this probe (review repro): gen.py <outdir>
# (default: ./cases next to this file). EXPECTED pins each case's verdict classes per front.
# Each case: rank 0 lanes 0..NL-1 run ONE st.async instruction (lane k writes word k of w[] in
# rank 1, complete_tx on rank 1's M1). Lanes in DR each run their own (separate pc) block
# fence.acq_rel.cluster; mbarrier.arrive.release.cluster on rank 1's M2 (count |DR|). Rank 1
# t0 waits M2, optionally test_waits M1 (TW), reads word R (or not), stores to out, waits M1.
import json, os, sys
out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "cases")
CASES = {
  # name: (NL, drainers, read lane or None, test_wait, expect PTX classes under the PTX model)
  "L3_D0_R2":  (3, [0], 2, False, "race"),
  "L3_D2_R2":  (3, [2], 2, False, "norace"),
  "L3_D12_R0": (3, [1, 2], 0, False, "race"),
  "L3_D01_R1": (3, [0, 1], 1, False, "norace"),
  "L3_D10_R0": (3, [1, 0], 0, False, "norace"),
  "L2_D1_R0":  (2, [1], 0, False, "race"),
  "L2_D1_TW":  (2, [1], None, True, "nondeterministic"),
  "L2_D1_TWR": (2, [1], 0, True, "race"),
  "L2_D0_TW":  (2, [0], None, True, "nondeterministic"),
  "L2_D0_TWR": (2, [0], 1, True, "race"),
  "L2_D01_TW": (2, [0, 1], None, True, "deterministic"),
}
# Verdict classes per case: (PTX, SASS); filled from the per-lane model's runs, each checked
# against the PTX-model category above (run.sh checks both).
EXPECTED = {
  "L3_D0_R2": (["nondeterministic", "race", "violation"], ["nondeterministic", "race", "violation"]),
  "L3_D2_R2": (["clean"], ["clean"]),
  "L3_D12_R0": (["nondeterministic", "race", "violation"], ["nondeterministic", "race", "violation"]),
  "L3_D01_R1": (["clean"], ["clean"]),
  "L3_D10_R0": (["clean"], ["clean"]),
  "L2_D1_R0": (["nondeterministic", "race", "violation"], ["nondeterministic", "race", "violation"]),
  "L2_D1_TW": (["nondeterministic"], ["nondeterministic"]),
  "L2_D1_TWR": (["nondeterministic", "race", "violation"], ["nondeterministic", "race", "violation"]),
  "L2_D0_TW": (["nondeterministic"], ["nondeterministic"]),
  "L2_D0_TWR": (["nondeterministic", "race", "violation"], ["nondeterministic", "race", "violation"]),
  "L2_D01_TW": (["clean"], ["clean"]),
}
for name, (nl, dr, rd, tw, exp) in CASES.items():
    d = os.path.join(out, name); os.makedirs(d, exist_ok=True)
    blocks = []
    br = []
    for k in dr:
        br.append(f"    setp.eq.u32 %p7, %r1, {k};\n    @%p7 bra D{k};")
        blocks.append(f"D{k}:\n    fence.acq_rel.cluster;\n    mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%r10];\n    bra END;")
    rdcode = ""
    if tw:
        rdcode += "    mbarrier.test_wait.parity.acquire.cluster.shared::cta.b64 %p6, [%r4], 0;\n    selp.u32 %r15, 1, 0, %p6;\n"
    else:
        rdcode += "    mov.u32 %r15, 0;\n"
    if rd is not None:
        rdcode += f"    ld.shared.u32 %r16, [%r8+{4*rd}];\n    add.u32 %r15, %r15, %r16;\n"
    ptx = f"""// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// generated case {name}: NL={nl} drainers={dr} read={rd} test_wait={tw}; PTX-model expectation: {exp}
.version 9.0
.target sm_103a
.address_size 64
.visible .entry k(.param .u64 out_p)
.maxntid 32, 1, 1
.reqnctapercluster 2, 1, 1
{{
    .reg .b32 %r<24>;
    .reg .b64 %rd<4>;
    .reg .pred %p<8>;
    .shared .align 8 .b64 m1;
    .shared .align 8 .b64 m2;
    .shared .align 16 .b32 w[4];
    ld.param.u64 %rd1, [out_p];
    cvta.to.global.u64 %rd1, %rd1;
    mov.u32 %r1, %tid.x;
    mov.u32 %r3, %cluster_ctarank;
    mov.u32 %r4, m1;
    mov.u32 %r5, m2;
    mov.u32 %r8, w;
    setp.ne.u32 %p0, %r3, 1;
    setp.ne.u32 %p1, %r1, 0;
    or.pred %p2, %p0, %p1;
    @%p2 bra INIT_DONE;
    mbarrier.init.shared::cta.b64 [%r4], 1;
    mbarrier.init.shared::cta.b64 [%r5], {len(dr)};
    fence.mbarrier_init.release.cluster;
    mbarrier.arrive.expect_tx.relaxed.cluster.shared::cta.b64 _, [%r4], {4*nl};
INIT_DONE:
    barrier.cluster.arrive.release.aligned;
    barrier.cluster.wait.acquire.aligned;
    setp.ge.u32 %p3, %r1, {nl};
    @%p3 bra END;
    @!%p0 bra RANK1;
    mapa.shared::cluster.u32 %r9, %r4, 1;
    mapa.shared::cluster.u32 %r10, %r5, 1;
    shl.b32 %r12, %r1, 2;
    add.u32 %r12, %r8, %r12;
    mapa.shared::cluster.u32 %r13, %r12, 1;
    add.u32 %r14, %r1, 40;
    st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%r13], %r14, [%r9];
{chr(10).join(br)}
    bra END;
{chr(10).join(blocks)}
RANK1:
    @%p1 bra END;
WAIT2:
    mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 %p5, [%r5], 0;
    @!%p5 bra WAIT2;
{rdcode}    st.global.u32 [%rd1], %r15;
WAIT:
    mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 %p4, [%r4], 0;
    @!%p4 bra WAIT;
END:
    barrier.cluster.arrive.release.aligned;
    barrier.cluster.wait.acquire.aligned;
    ret;
}}
"""
    open(os.path.join(d, "kernel.ptx"), "w").write(ptx)
    open(os.path.join(d, "case.toml"), "w").write(f'''name = "probe_coalesce_lanes_{name}"
description = "coalesced st.async case {name}; PTX-model expectation: {exp}"
ptx = "kernel.ptx"
entry = "k"
grid = [2, 1, 1]
cluster = [2, 1, 1]
block = [32, 1, 1]
outputs = ["out"]
[[buffers]]
name = "out"
dtype = "u32"
shape = [1]
init = "zeros"
[[params]]
kind = "ptr"
buffer = "out"
[[variants]]
id = "good"
expect = {json.dumps(EXPECTED.get(name, ([], []))[0])}
expect_sass = {json.dumps(EXPECTED.get(name, ([], []))[1])}
''')
    open(os.path.join(d, "EXPECT"), "w").write(exp + "\n")
