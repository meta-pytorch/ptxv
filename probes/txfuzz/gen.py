#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""mbarrier tx fuzzer (MISS 273, RDP2 finding E): random small mbarrier tx kernels (one phase, one waiter).
Each producer warp's lane 0 stores its slot, fences, then runs 1-2 ops on M:
  AE i  arrive.expect_tx i      A  arrive      E n  expect_tx n      C m  complete_tx m
'pre' ops run on thread 0 before the bar.sync (initial pending / tx state).
The waiter (last warp) waits for phase 0 and sums the slots.
Usage: gen.py OUTDIR SEED N"""
import os, random, sys

def op_text(o):
    k, v = o
    if k == 'AE': return f"mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%r4], {v};"
    if k == 'A':  return "mbarrier.arrive.release.cta.shared::cta.b64 _, [%r4];"
    if k == 'E':  return f"mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%r4], {v};"
    if k == 'C':  return f"mbarrier.complete_tx.relaxed.cta.shared::cta.b64 [%r4], {v};"

def rand_op(r):
    k = r.choice(['AE', 'AE', 'A', 'E', 'C', 'C', 'C'])
    return (k, r.choice([4, 8, 12, 16]) if k != 'A' else 0)

def gen(r):
    # incs: arrivals (AE i / A), pure expects (E n); decs: complete_tx chunks summing to the
    # increments (sometimes off by one chunk: over / under decrement).
    ops = []
    for _ in range(r.choice([1, 2, 2, 3])):
        ops.append(('AE', r.choice([0, 4, 8, 12])) if r.random() < 0.8 else ('A', 0))
    ops = [('A', 0) if o == ('AE', 0) else o for o in ops]
    for _ in range(r.choice([0, 1, 1, 2])):
        ops.append(('E', r.choice([4, 8, 12])))
    tot = sum(v for k, v in ops if k in ('AE', 'E'))
    decs = []
    while tot > 0:
        m = min(tot, r.choice([4, 8, 12]))
        decs.append(('C', m)); tot -= m
    x = r.random()
    if x < 0.15: decs.append(('C', r.choice([4, 8])))
    elif x < 0.3 and decs: decs[r.randrange(len(decs))] = ('C', decs[0][1] + 4)
    allops = ops + decs
    r.shuffle(allops)
    npre = r.choice([0, 0, 0, 1, 2])
    pre, rest = allops[:npre], allops[npre:]
    nw = max(1, min(4, len(rest)))
    progs = [[] for _ in range(nw)]
    for i, o in enumerate(rest):
        progs[i % nw if r.random() < 0.7 else r.randrange(nw)].append(o)
    progs = [p for p in progs if p] or [[('A', 0)]]
    arrivals = sum(1 for o in pre + [x for p in progs for x in p] if o[0] in ('A', 'AE'))
    return pre, progs, max(arrivals, 1)

def kernel(pre, progs, count):
    nw = len(progs)
    L = []
    L.append(".version 8.8\n.target sm_103a\n.address_size 64\n")
    L.append(f".visible .entry txf(.param .u64 out_p)\n.maxntid {32*(nw+1)}, 1, 1\n{{")
    L.append("    .reg .b32 %r<40>;\n    .reg .b64 %rd<4>;\n    .reg .pred %p<8>;")
    L.append("    .shared .align 8 .b64 m1;\n    .shared .align 4 .b32 s[8];")
    L.append("    ld.param.u64 %rd1, [out_p];\n    cvta.to.global.u64 %rd1, %rd1;")
    L.append("    mov.u32 %r1, %tid.x;\n    mov.u32 %r4, m1;\n    mov.u32 %r8, s;\n    shr.u32 %r2, %r1, 5;\n    and.b32 %r3, %r1, 31;")
    L.append("    setp.ne.u32 %p1, %r1, 0;\n    @%p1 bra INIT_DONE;")
    L.append(f"    mbarrier.init.shared::cta.b64 [%r4], {count};")
    for j in range(8):
        L.append(f"    st.shared.u32 [%r8+{4*j}], 0;")
    for o in pre:
        L.append("    " + op_text(o))
    L.append("INIT_DONE:\n    bar.sync 0;\n    setp.ne.u32 %p2, %r3, 0;\n    @%p2 bra END;")
    for w in range(nw):
        L.append(f"    setp.eq.u32 %p3, %r2, {w};\n    @%p3 bra W{w};")
    L.append("    bra WAITER;")
    for w, p in enumerate(progs):
        L.append(f"W{w}:\n    mov.u32 %r9, {w+1};\n    st.shared.u32 [%r8+{4*w}], %r9;\n    fence.acq_rel.cta;")
        for o in p:
            L.append("    " + op_text(o))
        L.append("    bra END;")
    L.append("WAITER:\n    mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 %p5, [%r4], 0;\n    @!%p5 bra WAITER;")
    L.append("    mov.u32 %r20, 0;")
    for w in range(nw):
        L.append(f"    ld.shared.u32 %r21, [%r8+{4*w}];\n    add.u32 %r20, %r20, %r21;")
    L.append("    st.global.u32 [%rd1], %r20;\nEND:\n    ret;\n}")
    return "\n".join(L) + "\n"

TOML = """name = "txf"
description = "{desc}"
ptx = "kernel.ptx"
entry = "txf"
grid = [1, 1, 1]
block = [{block}, 1, 1]
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
expect = ["clean"]
"""

def main():
    out, seed, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    r = random.Random(seed)
    for c in range(n):
        pre, progs, count = gen(r)
        d = os.path.join(out, f"c{seed}_{c:04d}")
        os.makedirs(d, exist_ok=True)
        desc = f"count {count}; pre {pre}; warps {progs}"
        open(os.path.join(d, "kernel.ptx"), "w").write(kernel(pre, progs, count))
        open(os.path.join(d, "case.toml"), "w").write(TOML.format(desc=desc.replace('"', "'"), block=32*(len(progs)+1)))
main()
