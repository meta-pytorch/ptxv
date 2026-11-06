#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Multicast TMA with and without .cta_group::2 (probes/tma_2cta_mcast, kernel.ptx).

  python3 run.py gen              # write expected/configs.json
  python3 run.py run [cfg...]     # run every config on GB300 (`ptxv probe`), write <cfg>.bin
  python3 run.py report           # per config: which ranks got the box, whose mbarrier completed

A config is (pair, ctaMask, mbarrier rank, destination rank, expected bytes per rank). Rank 0 thread 0 issues one load of the 512-byte box to `buf` of rank `dst_rank` (mapa)
with the ctaMask, naming the mbarrier of rank `mbar_rank`. Every CTA expects its `expect`
bytes on its own mbarrier (a plain arrive for 0) and polls it for ~10 ms; a CTA
whose mbarrier got exactly that many bytes completes, one that got fewer does not (more:
either, depending on how the transactions are split). out[rank] = box (or 0xdeadbeef when
it did not complete) and the flag.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(HERE, "expected")
PTXV = os.environ.get("PTXV", "ptxv")


def delivered(pair, mask, mbar):
    """Bytes each rank's mbarrier receives under S-TMA-2CTA-MCAST: a box per masked rank r,
    signalled on rank r's own mbarrier, or with .cta_group::2 on the mbarrier of rank
    (r & ~1) | (mbar & 1)."""
    d = [0] * 4
    for r in range(4):
        if mask >> r & 1:
            d[(r & ~1) | (mbar & 1) if pair else r] += 512
    return d


def configs():
    """Per (pair, mask, mbarrier rank, destination rank): every rank expecting 512, every rank
    1024, and each rank exactly what the model delivers to it ("x")."""
    out = []
    for pair in (1, 0):
        for mask in (0xF, 0x1, 0x2, 0x4, 0x8, 0x3, 0x6):
            for mbar in (0, 1):
                for dst in (0, 1):
                    dl = delivered(pair, mask, mbar)
                    for tag, exp in (("e512", [512] * 4), ("e1024", [1024] * 4), ("x", dl)):
                        out.append(dict(name=f"p{pair}_m{mask:x}_b{mbar}_d{dst}_{tag}", pair=pair,
                                        mask=mask, mbar_rank=mbar, dst_rank=dst,
                                        exp=sum((e // 128) << (8 * r) for r, e in enumerate(exp)),
                                        expect=exp, delivered=dl))
    return out


CASE = """name = "tma_2cta_mcast_{name}"
description = "probes/tma_2cta_mcast config {name} (run.py)"
ptx = "kernel.ptx"
entry = "tma_2cta_mcast"
grid = [4, 1, 1]
block = [64, 1, 1]
cluster = [4, 1, 1]
outputs = ["out"]

[[buffers]]
name = "src"
dtype = "u32"
shape = [8, 16]
init = "iota"

[[buffers]]
name = "out"
dtype = "u32"
shape = [4, 129]
init = "zeros"

[[params]]
kind = "tensormap"
buffer = "src"
box = [16, 8]

[[params]]
kind = "ptr"
buffer = "out"
{params}
[[variants]]
id = "good"
expect = ["clean"]
"""


def case_toml(c):
    params = "".join(f'\n[[params]]\nkind = "u32"\nvalue = {c[k]}\n'
                     for k in ("mask", "mbar_rank", "dst_rank", "exp", "pair"))
    return CASE.format(name=c["name"], params=params)


def gen():
    os.makedirs(EXP, exist_ok=True)
    with open(os.path.join(EXP, "configs.json"), "w") as f:
        json.dump(configs(), f, indent=0)


def run(names):
    for c in json.load(open(os.path.join(EXP, "configs.json"))):
        if names and c["name"] not in names:
            continue
        d = tempfile.mkdtemp(prefix="tma_2cta_mcast_")
        shutil.copy(os.path.join(HERE, "kernel.ptx"), d)
        with open(os.path.join(d, "case.toml"), "w") as f:
            f.write(case_toml(c))
        r = subprocess.run([PTXV, "probe", d, "--out", os.path.join(d, "out")],
                           capture_output=True, text=True)
        dump = os.path.join(d, "out", "good.probe.bin")
        if r.returncode or not os.path.exists(dump):
            print(c["name"], "FAILED", r.stdout[-600:], r.stderr[-600:])
            continue
        shutil.copy(dump, os.path.join(EXP, c["name"] + ".bin"))
        print(c["name"], "ok")
        shutil.rmtree(d)


def summary(c):
    """(ranks whose box is the source, ranks whose mbarrier completed) of a recorded run."""
    import struct
    raw = open(os.path.join(EXP, c["name"] + ".bin"), "rb").read()
    w = struct.unpack(f"<{len(raw) // 4}I", raw)
    data = [r for r in range(4) if list(w[129 * r:129 * r + 128]) == list(range(128))]
    done = [r for r in range(4) if w[129 * r + 128] != 0]
    return data, done


def report(_):
    for c in json.load(open(os.path.join(EXP, "configs.json"))):
        if os.path.exists(os.path.join(EXP, c["name"] + ".bin")):
            data, done = summary(c)
            print(f"{c['name']:22} box in ranks {data!s:14} mbarrier completed in {done}")


if __name__ == "__main__":
    {"gen": lambda a: gen(), "run": run, "report": report}[sys.argv[1]](sys.argv[2:])
