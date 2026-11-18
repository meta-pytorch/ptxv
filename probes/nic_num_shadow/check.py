#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MISS 224 repro (run.sh): an RDMA write and the numerics shadow.

loop/: one GPU, a loopback IBGDA queue pair. Thread 0 copies the input c into dst (c[i] holds
the f32 value x[i] + y[i] rounds to), writes src[i] = x[i] + y[i], RDMA-writes src into dst,
signals, polls the signal's completion and copies dst to out. Exact: out[i] = x[i] + y[i].
On GB300 out[i] holds the bits of c[i] whatever the NIC delivers: the RDMA write carries
rn(x[i] + y[i]) (arith/: the payload alone, `ptxv hw --runs 20 --scrub`,
expected/arith.out.bin), the bits dst already held. A claimed abs_err or bound of `ptxv num` on
loop/'s out below |GB300 - (x[i] + y[i])| under-claims: until MISS 224 the RDMA write left
dst's numerics nodes in place, so out took the copied input c as its exact value.
With PTXV_GPUS set arith/ is replayed on GB300 (`ptxv hw --runs 5 --scrub`) and compared with
expected/. Exit 1 while a claim is refuted, 2 on a tool failure, 3 when GB300 no longer gives
the recorded output. usage: check.py PTXV"""
import csv
import glob
import os
import struct
import subprocess
import sys
import tempfile
import tomllib
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
PTXV = sys.argv[1]
ENV = dict(os.environ, CUDA_VISIBLE_DEVICES="", PTXV_THREADS=os.environ.get("PTXV_THREADS", "4"))


def f32(v):
    return struct.unpack("<f", struct.pack("<f", v))[0]


def values(case, name):
    """The `values` of buffer `name` in case.toml, as f32 (element order)."""
    with open(os.path.join(HERE, case, "case.toml"), "rb") as f:
        bufs = tomllib.load(f)["buffers"]
    return [f32(v) for _, v in sorted(next(b for b in bufs if b["name"] == name)["values"])]


x, y = values("loop", "x"), values("loop", "y")
exact = [Fraction(a) + Fraction(b) for a, b in zip(x, y)]
b = open(os.path.join(HERE, "expected", "arith.out.bin"), "rb").read()
gpu = struct.unpack("<4f", b)
tmp = tempfile.mkdtemp(prefix="nic_num_shadow.")
p = subprocess.run([PTXV, "num", "loop", "--out", os.path.join(tmp, "num")], cwd=HERE, env=ENV,
                   capture_output=True, text=True)
rc = 0
print("loop/: out[i] = dst[i] after the RDMA write of src[i] = x[i] + y[i] into dst (dst held c[i] = its rounding)")
print(f"{'i':>2} {'x + y (exact)':>22} {'GB300 out':>16} {'|GB300 - exact|':>16} {'front':5} "
      f"{'claimed abs_err':>16} {'bound_cor1a':>12} verdict")
for front in ("ptx", "sass"):
    paths = glob.glob(os.path.join(tmp, "num", "**", f"good.{front}.num.csv"), recursive=True)
    if not paths:
        print(f"{front}: no num csv (rc {p.returncode})\n{(p.stdout + p.stderr)[-2000:]}")
        sys.exit(2)
    for row in csv.DictReader(open(paths[0])):
        i = int(row["index"])
        te = abs(Fraction(gpu[i]) - exact[i])
        if row["untracked"] != "false" or row["nonfinite"] != "false":
            print(f"{i:2} {float(exact[i]):22.12g} {gpu[i]:16.9g} {float(te):16.4e} {front:5} {'(no claim)':>16} {'':>12} ok")
            continue
        ae, b1 = Fraction(float(row["abs_err"])), Fraction(float(row["bound_cor1a"]))
        slack = Fraction(1, 10**9)
        under = ae * (1 + slack) < te or b1 * (1 + slack) < te
        rc |= 1 if under else 0
        print(f"{i:2} {float(exact[i]):22.12g} {gpu[i]:16.9g} {float(te):16.4e} {front:5} {float(ae):16.4e} "
              f"{float(b1):12.4e} {'UNDER-CLAIMED' if under else 'ok'}")

if os.environ.get("PTXV_GPUS"):
    print("\nGB300 now (ptxv hw --runs 5 --scrub, PTXV_GPUS=%s):" % os.environ["PTXV_GPUS"])
    hw_env = dict(os.environ)
    hw_env.pop("CUDA_VISIBLE_DEVICES", None)
    dump = os.path.join(tmp, "hw")
    q = subprocess.run([PTXV, "hw", "arith", "--runs", "5", "--scrub", "--out", os.path.join(tmp, "hwout")],
                       cwd=HERE, env=dict(hw_env, PTXV_HW_DUMP=dump), capture_output=True, text=True)
    got = sorted(glob.glob(os.path.join(dump, "hw_*.bin")))
    same = len(got) == 1 and open(got[0], "rb").read() == b
    print(f"  arith: {'equal to expected/' if same else 'DIFFERS from expected/ (%d distinct outputs, rc %d)' % (len(got), q.returncode)}")
    if not same:
        rc = 3
print("\nexit", rc, "(1: a claim GB300 refutes; 0: none)" if rc in (0, 1) else "")
sys.exit(rc)
