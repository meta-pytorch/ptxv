#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MISS 225 repro (run.sh): the simulator's multimem.ld_reduce f32 add lanes (the value ptxv num
and equiv take as the hardware's) next to GB300's recorded lanes.

m4 / m2 (4 / 2 members) reduce the lanes of data/ (mklanes.py, expected/lanes.csv) with each
test's instruction (add.f32 relaxed.sys, add.f32 weak, add.v4.f32); `ptxv check`
(PTXV_DUMP_OUTPUTS=1) writes the simulated output, compared lane by lane with
expected/m<N>.out.bin. A lane is equal, or differs:
- all -0: every member is -0; GB300 gives +0;
- window: GB300's sum keeps fewer member bits than the simulator's (a cancellation: GB300
  gives 0 or another value where the simulator keeps bits of a small member, or the reverse).
With MM=<mm binary built from probes/multimem/mm.cu> and 4 GPUs (PTXV_GPUS / CUDA_VISIBLE_DEVICES)
the lanes are also replayed on GB300 (`mm ldr`, MM_IN) and compared with expected/. Exit 1
while a lane differs, 2 on a tool failure, 3 when GB300 no longer gives the recorded output.
usage: check.py PTXV"""
import collections
import csv
import glob
import os
import struct
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PTXV = sys.argv[1]
TESTS = ["f32", "f32w", "f32x4"]
rows = list(csv.DictReader(open(os.path.join(HERE, "expected", "lanes.csv"))))
count = collections.Counter((int(r["members"]), r["test"]) for r in rows)
kind = {(int(r["members"]), r["test"], int(r["lane"])): (r["kind"], r["source"]) for r in rows}


def u32s(b):
    return struct.unpack("<%dI" % (len(b) // 4), b)


def f32(x):
    return "%.9g" % struct.unpack("<f", struct.pack("<I", x))[0]


tmp = tempfile.mkdtemp(prefix="multimem_f32_window.")
env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PTXV_DUMP_OUTPUTS="1",
           PTXV_THREADS=os.environ.get("PTXV_THREADS", "4"))
rc = 0
print(f"{'members':7} {'test':5} {'source':8} {'front':5} {'lanes':>5} {'equal':>5} {'all -0':>6} {'window':>6}")
examples = {}
for nm in (4, 2):
    ins = [u32s(open(os.path.join(HERE, "data", f"m{nm}.in{d}.bin"), "rb").read()) for d in range(nm)]
    gpu = u32s(open(os.path.join(HERE, "expected", f"m{nm}.out.bin"), "rb").read())
    for front in ("ptx", "sass"):
        out = os.path.join(tmp, f"m{nm}.{front}")
        p = subprocess.run([PTXV, "check", f"m{nm}", "--variant", "good", "--front", front,
                            "--no-input-coverage", "--out", out], cwd=HERE, env=env, capture_output=True, text=True)
        got = glob.glob(os.path.join(out, "**", f"good.{front}.output0.bin"), recursive=True)
        if not got:
            print(f"m{nm} {front}: no simulated output (rc {p.returncode})\n{(p.stdout + p.stderr)[-2000:]}")
            sys.exit(2)
        sim = u32s(open(got[0], "rb").read())
        base = 0
        for test in TESTS:
            c = collections.defaultdict(collections.Counter)
            for lane in range(count[(nm, test)]):
                k = base + lane
                kd, src = kind[(nm, test, lane)]
                if kd == "pad":
                    continue
                g, s = gpu[k], sim[k]
                mem = [ins[d][k] for d in range(nm)]
                if g == s:
                    cls = "equal"
                elif all(h == 0x80000000 for h in mem):
                    cls = "all -0"
                else:
                    cls = "window"
                c[src][cls] += 1
                if cls != "equal":
                    examples.setdefault((test, src, cls), (nm, front, mem, g, s))
            for src in sorted(c):
                n = sum(c[src].values())
                bad = c[src]["all -0"] + c[src]["window"]
                rc |= 1 if bad else 0
                print(f"{nm:7} {test:5} {src:8} {front:5} {n:5} {c[src]['equal']:5} {c[src]['all -0']:6} {c[src]['window']:6}")
            base += count[(nm, test)]
if examples:
    print("\nfirst lane per test, source and class (members: GB300 vs ptxv's simulated value):")
for (test, src, cls), (nm, front, mem, g, s) in sorted(examples.items()):
    eb = max((h >> 23) & 0xFF for h in mem)
    print(f"  {test:5} {src:7} {cls:6} m{nm} {front:4} [{' '.join(f'{h:08x}' for h in mem)}] "
          f"({', '.join(f32(h) for h in mem)}; eB {eb}): GB300 {g:08x} ({f32(g)}), ptxv {s:08x} ({f32(s)})")

mm = os.environ.get("MM")
if mm:
    print("\nGB300 now (mm ldr, MM_IN):")
    for nm in (4, 2):
        ind, outd = os.path.join(tmp, f"in{nm}"), os.path.join(tmp, f"gb{nm}")
        os.makedirs(ind)
        os.makedirs(outd)
        raw = [open(os.path.join(HERE, "data", f"m{nm}.in{d}.bin"), "rb").read() for d in range(nm)]
        exp = open(os.path.join(HERE, "expected", f"m{nm}.out.bin"), "rb").read()
        base, spans = 0, {}
        for test in TESTS:
            n = count[(nm, test)]
            spans[test] = (base, n)
            for d in range(nm):
                with open(os.path.join(ind, f"{test}.in{d}.bin"), "wb") as f:
                    f.write(raw[d][4 * base : 4 * (base + n)].ljust(16 << 16, b"\0"))
            base += n
        q = subprocess.run([mm, "ldr", outd, ",".join(str(d) for d in range(nm))],
                           env=dict(os.environ, MM_IN=ind, MM_TESTS=",".join(TESTS)), capture_output=True, text=True)
        for test in TESTS:
            b, n = spans[test]
            outs = sorted(glob.glob(os.path.join(outd, f"{test}.out*.bin")))
            lanes = [i for i in range(n) if kind[(nm, test, i)][0] != "pad"]
            want = u32s(exp)
            same = bool(outs) and all(all(u32s(open(o, "rb").read()[: 4 * n])[i] == want[b + i] for i in lanes)
                                      for o in outs)
            print(f"  m{nm} {test}: {'equal to expected/' if same else 'DIFFERS from expected/ (rc %d)' % q.returncode}"
                  f" ({len(outs)} issuing GPU / repetition outputs)")
            if not same:
                rc = 3
print("\nexit", rc)
sys.exit(rc)
