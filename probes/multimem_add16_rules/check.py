#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MISS 208 repro (README.md, run.sh): the simulator's multimem.ld_reduce f16 / bf16 add lanes
(the value ptxv num / equiv take as the hardware's) next to GB300's recorded lanes.

m4 / m2 (4 / 2 members) reduce the vectors of data/ (mklanes.py) with each test's instruction;
`ptxv check` (PTXV_DUMP_OUTPUTS=1) writes the simulated output, compared lane by lane with
expected/m<N>.<test>.out.bin. A lane is equal, adjacent (the other rounding of the lane's sum,
one encoding away with the same sign: what MISS 162 accepts on the marked side), or differs:
- all -0: every member is -0; GB300 gives +0;
- flushed: GB300 gives 0 where the simulator gives a nonzero sum (GB300 gives +0 for a sum
  whose leading bit lies below 2^(16 floor(eB / 16) - 23 - bias), eB the largest member's biased
  exponent; no member is dropped);
- other.
With MM=<mm binary built from probes/multimem/mm.cu> and 4 GPUs (PTXV_GPUS / CUDA_VISIBLE_DEVICES)
the vectors are also replayed on GB300 (`mm ldr`, MM_IN) and compared with expected/. Exit 1
while a lane differs, 2 on a tool failure, 3 when GB300 no longer gives the recorded output.
usage: check.py PTXV"""
import csv
import glob
import os
import struct
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PTXV = sys.argv[1]
TESTS = ["bf16x8", "bf16x8a", "f16x8", "f16x8a"]
rows = list(csv.DictReader(open(os.path.join(HERE, "expected", "vectors.csv"))))
counts = {(int(r["members"]), r["test"]): 0 for r in rows}
for r in rows:
    counts[(int(r["members"]), r["test"])] += 1


def u16s(b):
    return struct.unpack("<%dH" % (len(b) // 2), b)


def fmt16(test, h):
    if test.startswith("bf16"):
        return "%.6g" % struct.unpack("<f", struct.pack("<I", h << 16))[0]
    return "%.6g" % struct.unpack("<e", struct.pack("<H", h))[0]


tmp = tempfile.mkdtemp(prefix="multimem_add16.")
env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PTXV_DUMP_OUTPUTS="1",
           PTXV_THREADS=os.environ.get("PTXV_THREADS", "4"))
rc = 0
print(f"{'members':7} {'test':8} {'front':5} {'lanes':>5} {'equal':>5} {'adjacent':>8} {'all -0':>6} {'flushed':>7} {'other':>5}")
examples = {}
for nm in (4, 2):
    ins = [u16s(open(os.path.join(HERE, "data", f"m{nm}.in{d}.bin"), "rb").read()) for d in range(nm)]
    for front in ("ptx", "sass"):
        out = os.path.join(tmp, f"m{nm}.{front}")
        p = subprocess.run([PTXV, "check", f"m{nm}", "--variant", "good", "--front", front,
                            "--no-input-coverage", "--out", out], cwd=HERE, env=env, capture_output=True, text=True)
        got = glob.glob(os.path.join(out, "**", f"good.{front}.output0.bin"), recursive=True)
        if not got:
            print(f"m{nm} {front}: no simulated output (rc {p.returncode})\n{(p.stdout + p.stderr)[-2000:]}")
            sys.exit(2)
        sim = u16s(open(got[0], "rb").read())
        base = 0
        for test in TESTS:
            n = counts[(nm, test)]
            gpu = u16s(open(os.path.join(HERE, "expected", f"m{nm}.{test}.out.bin"), "rb").read())
            c = {"equal": 0, "adjacent": 0, "all -0": 0, "flushed": 0, "other": 0}
            for v in range(n):
                for l in range(8):
                    k = 8 * (base + v) + l
                    g, s = gpu[8 * v + l], sim[k]
                    mem = [ins[d][k] for d in range(nm)]
                    if g == s:
                        cls = "equal"
                    elif g >> 15 == s >> 15 and abs(g - s) == 1 and g & 0x7FFF and s & 0x7FFF:
                        cls = "adjacent"
                    elif all(h == 0x8000 for h in mem):
                        cls = "all -0"
                    elif g & 0x7FFF == 0:
                        cls = "flushed"
                    else:
                        cls = "other"
                    c[cls] += 1
                    if cls not in ("equal", "adjacent"):
                        examples.setdefault((test, cls), (nm, front, mem, g, s))
            bad = c["all -0"] + c["flushed"] + c["other"]
            rc |= 1 if bad else 0
            print(f"{nm:7} {test:8} {front:5} {8 * n:5} {c['equal']:5} {c['adjacent']:8} {c['all -0']:6} {c['flushed']:7} {c['other']:5}")
            base += n
print("\nfirst lane per test and class (members: GB300 vs ptxv's simulated value):")
for (test, cls), (nm, front, mem, g, s) in sorted(examples.items()):
    ms = " ".join(f"{h:04x}" for h in mem)
    print(f"  {test:8} {cls:8} m{nm} {front:4} [{ms}] ({', '.join(fmt16(test, h) for h in mem)}): "
          f"GB300 {g:04x} ({fmt16(test, g)}), ptxv {s:04x} ({fmt16(test, s)})")

mm = os.environ.get("MM")
if mm:
    print("\nGB300 now (mm ldr, MM_IN):")
    for nm in (4, 2):
        ind, outd = os.path.join(tmp, f"in{nm}"), os.path.join(tmp, f"gb{nm}")
        os.makedirs(ind)
        os.makedirs(outd)
        raw = [open(os.path.join(HERE, "data", f"m{nm}.in{d}.bin"), "rb").read() for d in range(nm)]
        base = 0
        for test in TESTS:
            n = counts[(nm, test)]
            for d in range(nm):
                with open(os.path.join(ind, f"{test}.in{d}.bin"), "wb") as f:
                    f.write(raw[d][16 * base : 16 * (base + n)].ljust(16 << 16, b"\0"))
            base += n
        q = subprocess.run([mm, "ldr", outd, ",".join(str(d) for d in range(nm))],
                           env=dict(os.environ, MM_IN=ind, MM_TESTS=",".join(TESTS)), capture_output=True, text=True)
        for test in TESTS:
            n = counts[(nm, test)]
            exp = open(os.path.join(HERE, "expected", f"m{nm}.{test}.out.bin"), "rb").read()
            outs = sorted(glob.glob(os.path.join(outd, f"{test}.out*.bin")))
            same = bool(outs) and all(open(o, "rb").read()[: 16 * n] == exp for o in outs)
            print(f"  m{nm} {test}: {'equal to expected/' if same else 'DIFFERS from expected/ (rc %d)' % q.returncode}"
                  f" ({len(outs)} issuing GPU / repetition outputs)")
            if not same:
                rc = 3
print("\nexit", rc)
sys.exit(rc)
