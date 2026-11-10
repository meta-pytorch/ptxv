#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Certificate-vs-exploration differential on random small synchronization kernels.

Each kernel has one actor per warp (lane 0) in 1-3 CTAs. An actor runs a short random
program over two global counters and four data words: plain / relaxed / release stores,
plain / relaxed / acquire loads, atomic adds with used or unused results (relaxed, release,
acq_rel), polls of a counter (>= k or == k), fences, a relaxed reset of a counter, and stores
or loads guarded by the value an atomic returned. Every observed value goes to the actor's
own output words.

`ptxv check --certify` and plain exploration must agree whenever both finish: a certified
verdict (E-WELLSYNC) with other classes or outputs than the exploration's is a certificate
soundness bug. Cases that disagree are kept under OUT/disagree/.

usage: fuzz_sync.py --ptxv BIN --out DIR [--n N] [--seed S] [--jobs J] [--front ptx|sass]
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import random
import shutil
import subprocess
from pathlib import Path

NCNT, NDATA, NOUT = 2, 4, 8


def gen_actor(rng: random.Random, a: int, nact: int) -> list[tuple]:
    ops = []
    last_atom = False
    for _ in range(rng.randint(1, 6)):
        kind = rng.choices(
            ["std", "ldd", "add", "poll", "fence", "if", "reset"],
            [4, 3, 5, 3, 2, 3 if last_atom else 0, 0.3],
        )[0]
        if kind == "std":
            ops.append(
                (
                    "std",
                    rng.randrange(NDATA),
                    rng.randint(1, 9),
                    rng.choice(["", ".relaxed.gpu", ".release.gpu"]),
                )
            )
        elif kind == "ldd":
            ops.append(
                (
                    "ldd",
                    rng.randrange(NDATA),
                    rng.choice(["", ".relaxed.gpu", ".acquire.gpu"]),
                )
            )
        elif kind == "add":
            used = rng.random() < 0.6
            sem = rng.choice(
                [".relaxed.gpu", ".release.gpu"]
                + ([".acq_rel.gpu", ".acquire.gpu"] if used else [])
            )
            ops.append(
                ("add", rng.randrange(NCNT), rng.choice([1, 1, 1, 2]), used, sem)
            )
            last_atom = last_atom or used
        elif kind == "poll":
            ops.append(
                (
                    "poll",
                    rng.randrange(NCNT),
                    rng.choice(["ge", "ge", "eq"]),
                    rng.randint(1, nact + 1),
                    rng.choice([".acquire.gpu", ".acquire.gpu", ".relaxed.gpu"]),
                )
            )
        elif kind == "fence":
            ops.append(("fence",))
        elif kind == "if":
            sub = rng.choice(
                [
                    (
                        "std",
                        rng.randrange(NDATA),
                        rng.randint(1, 9),
                        rng.choice(["", ".release.gpu"]),
                    ),
                    ("ldd", rng.randrange(NDATA), rng.choice(["", ".acquire.gpu"])),
                ]
            )
            ops.append(("if", rng.randint(0, nact), sub))
        elif kind == "reset":
            ops.append(("reset", rng.randrange(NCNT)))
    return ops


class Emit:
    def __init__(self):
        self.lines: list[str] = []
        self.lbl = 0
        self.slot = 0

    def label(self) -> str:
        self.lbl += 1
        return f"L{self.lbl}"


def emit_op(e: Emit, a: int, op: tuple):
    k = op[0]
    out_addr = lambda: f"[%rdo+{4 * (a * NOUT + e.slot)}]"  # noqa: E731
    if k == "std":
        _, i, v, sem = op
        e.lines.append(f"    mov.u32 %rv, {v};")
        e.lines.append(f"    st{sem}.global.u32 [%rdd+{4 * i}], %rv;")
    elif k == "ldd":
        _, i, sem = op
        e.lines.append(f"    ld{sem}.global.u32 %rv, [%rdd+{4 * i}];")
        if e.slot < NOUT:
            e.lines.append(f"    st.global.u32 {out_addr()}, %rv;")
            e.slot += 1
    elif k == "add":
        _, c, v, used, sem = op
        if used:
            e.lines.append(f"    atom{sem}.global.add.u32 %rt, [%rdc+{4 * c}], {v};")
            if e.slot < NOUT:
                e.lines.append(f"    st.global.u32 {out_addr()}, %rt;")
                e.slot += 1
        else:
            e.lines.append(f"    red{sem}.global.add.u32 [%rdc+{4 * c}], {v};")
    elif k == "poll":
        _, c, cond, v, sem = op
        lp = e.label()
        e.lines.append(f"{lp}:")
        e.lines.append(f"    ld{sem}.global.u32 %rv, [%rdc+{4 * c}];")
        e.lines.append(f"    setp.{'lt' if cond == 'ge' else 'ne'}.u32 %pp, %rv, {v};")
        e.lines.append(f"    @%pp bra {lp};")
    elif k == "fence":
        e.lines.append("    fence.acq_rel.gpu;")
    elif k == "if":
        _, v, sub = op
        sk = e.label()
        e.lines.append(f"    setp.ne.u32 %pp, %rt, {v};")
        e.lines.append(f"    @%pp bra {sk};")
        emit_op(e, a, sub)
        e.lines.append(f"{sk}:")
    elif k == "reset":
        _, c = op
        e.lines.append("    mov.u32 %rv, 0;")
        e.lines.append(f"    st.relaxed.gpu.global.u32 [%rdc+{4 * c}], %rv;")


def gen_case(seed: int, d: Path) -> dict:
    rng = random.Random(seed)
    g = rng.randint(1, 3)
    w = rng.choice([1, 1, 2])
    nact = g * w
    progs = [gen_actor(rng, a, nact) for a in range(nact)]
    if rng.random() < 0.6:
        # Structured skeleton: arrival counter or ticket dispenser, with random noise.
        for a, p in enumerate(progs):
            sk = [("std", a % NDATA, a + 1, rng.choice(["", "", ".relaxed.gpu"]))]
            used = rng.random() < 0.5
            sk.append(
                (
                    "add",
                    0,
                    rng.choice([1, 1, 2]),
                    used,
                    rng.choice(
                        [".release.gpu", ".release.gpu", ".relaxed.gpu", ".acq_rel.gpu"]
                        if used
                        else [".release.gpu", ".relaxed.gpu"]
                    ),
                )
            )
            if used and rng.random() < 0.5:
                sk.append(
                    (
                        "if",
                        rng.randint(0, nact),
                        rng.choice(
                            [
                                (
                                    "ldd",
                                    rng.randrange(NDATA),
                                    rng.choice(["", ".acquire.gpu"]),
                                ),
                                ("std", rng.randrange(NDATA), 7, ""),
                            ]
                        ),
                    )
                )
            if rng.random() < 0.5:
                sk.append(
                    (
                        "poll",
                        0,
                        rng.choice(["ge", "eq"]),
                        0,
                        rng.choice([".acquire.gpu", ".relaxed.gpu"]),
                    )
                )
                sk.append(
                    ("ldd", rng.randrange(NDATA), rng.choice(["", "", ".acquire.gpu"]))
                )
            if rng.random() < 0.3:
                sk.insert(rng.randint(0, len(sk)), ("fence",))
            noise = p[: rng.randint(0, 2)]
            for op in noise:
                sk.insert(rng.randint(0, len(sk)), op)
            progs[a] = sk
    # Poll thresholds within the total each counter can reach (fewer trivial deadlocks).
    tot = [0] * NCNT
    for p in progs:
        for op in p:
            for o in (op, op[2] if op[0] == "if" else ()):
                if o and o[0] == "add":
                    tot[o[1]] += o[2]
    for p in progs:
        for i, op in enumerate(p):
            if op[0] == "poll":
                if tot[op[1]] == 0:
                    p[i] = ("fence",)
                else:
                    p[i] = ("poll", op[1], op[2], rng.randint(1, tot[op[1]]), op[4])
    e = Emit()
    body = []
    for a, p in enumerate(progs):
        e.slot = 0
        e.lines = []
        sk = f"SKIP{a}"
        body.append(f"    setp.ne.u32 %pa, %ra, {a};")
        body.append(f"    @%pa bra {sk};")
        body.append("    mov.u32 %rt, 0;")
        for op in p:
            emit_op(e, a, op)
        body += e.lines
        body.append(f"{sk}:")
    ptx = f"""// fuzz_sync seed {seed}: {g} CTAs x {w} warps, actors = lane 0 of each warp
// programs: {json.dumps(progs)}
.version 9.0
.target sm_103a
.address_size 64

.visible .entry k(.param .u64 out_p, .param .u64 data_p, .param .u64 cnt_p)
{{
    .reg .b32 %r<8>;
    .reg .b32 %ra, %rv, %rt;
    .reg .b64 %rdo, %rdd, %rdc;
    .reg .pred %pa, %pp, %pl;
    ld.param.u64 %rdo, [out_p];
    cvta.to.global.u64 %rdo, %rdo;
    ld.param.u64 %rdd, [data_p];
    cvta.to.global.u64 %rdd, %rdd;
    ld.param.u64 %rdc, [cnt_p];
    cvta.to.global.u64 %rdc, %rdc;
    mov.u32 %r1, %tid.x;
    and.b32 %r2, %r1, 31;
    setp.ne.u32 %pl, %r2, 0;
    @%pl bra DONE;
    shr.u32 %r3, %r1, 5;
    mov.u32 %r4, %ctaid.x;
    mad.lo.u32 %ra, %r4, {w}, %r3;
{chr(10).join(body)}
DONE:
    ret;
}}
"""
    d.mkdir(parents=True, exist_ok=True)
    (d / "kernel.ptx").write_text(ptx)
    (d / "case.toml").write_text(f"""name = "fuzz{seed}"
description = "fuzz_sync seed {seed}"
ptx = "kernel.ptx"
entry = "k"
grid = [{g}, 1, 1]
block = [{32 * w}, 1, 1]
outputs = ["out", "data", "cnt"]

[[buffers]]
name = "out"
dtype = "u32"
shape = [{nact * NOUT}]
init = "zeros"

[[buffers]]
name = "data"
dtype = "u32"
shape = [{NDATA}]
init = "zeros"

[[buffers]]
name = "cnt"
dtype = "u32"
shape = [{NCNT}]
init = "zeros"

[[params]]
kind = "ptr"
buffer = "out"

[[params]]
kind = "ptr"
buffer = "data"

[[params]]
kind = "ptr"
buffer = "cnt"

[[variants]]
id = "good"
summary = "random"
expect = ["clean"]
""")
    return {"seed": seed, "g": g, "w": w, "progs": progs}


def check(
    ptxv: str, d: Path, front: str, certify: bool, max_states: int, timeout: int
) -> dict:
    cmd = [
        "systemd-run",
        "--user",
        "--scope",
        "-q",
        "-p",
        "MemoryMax=8G",
        "-p",
        "MemorySwapMax=0",
        "timeout",
        str(timeout),
        ptxv,
        "check",
        str(d),
        "--variant",
        "good",
        "--front",
        front,
        "--json",
        "--no-input-coverage",
        "--max-states",
        str(max_states),
        "--out",
        str(d / ("out-c" if certify else "out-e")),
    ]
    if certify:
        cmd.append("--certify")
    p = subprocess.run(
        cmd, capture_output=True, text=True, env={**os.environ, "PTXV_THREADS": "2"}
    )
    try:
        doc = json.loads(p.stdout.strip().splitlines()[-1])
        r = doc["rows"][0]
        v = r["verdict"]
        return {
            "got": sorted(r["got"]),
            "complete": v.get("complete"),
            "cert": "E-WELLSYNC" in v.get("assumptions", []),
            "outputs": v.get("distinct_outputs"),
            "sigs": sorted({f.get("kind", "") for f in v.get("findings", [])}),
        }
    except Exception:  # noqa: BLE001
        return {
            "error": f"rc={p.returncode}: {' | '.join(p.stderr.strip().splitlines()[-2:])[:300]}"
        }


def one(a, seed: int) -> dict:
    d = Path(a.out) / "cases" / f"s{seed}"
    info = gen_case(seed, d)
    c = check(a.ptxv, d, a.front, True, a.max_states, a.timeout)
    res = {"seed": seed, "cert": c}
    if c.get("cert") and c.get("complete"):
        x = check(a.ptxv, d, a.front, False, a.max_states, a.timeout)
        res["explore"] = x
        if x.get("complete") and (
            x["got"] != c["got"] or x.get("outputs") != c.get("outputs")
        ):
            res["disagree"] = True
            dd = Path(a.out) / "disagree" / f"s{seed}"
            if dd.exists():
                shutil.rmtree(dd)
            shutil.copytree(d, dd, ignore=shutil.ignore_patterns("out-*"))
            (dd / "result.json").write_text(json.dumps({**res, "info": info}, indent=1))
    shutil.rmtree(d, ignore_errors=True)
    return res


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--ptxv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--front", default="ptx")
    ap.add_argument("--max-states", type=int, default=200000)
    ap.add_argument("--timeout", type=int, default=120)
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    log = (Path(a.out) / "results.jsonl").open("a")
    stats = {"n": 0, "certified": 0, "compared": 0, "disagree": 0, "error": 0}
    with cf.ThreadPoolExecutor(a.jobs) as ex:
        for r in ex.map(lambda s: one(a, s), range(a.seed, a.seed + a.n)):
            log.write(json.dumps(r) + "\n")
            log.flush()
            stats["n"] += 1
            stats["certified"] += bool(r["cert"].get("cert"))
            stats["compared"] += "explore" in r and bool(r["explore"].get("complete"))
            stats["disagree"] += bool(r.get("disagree"))
            stats["error"] += "error" in r["cert"]
            if r.get("disagree"):
                print("DISAGREE", json.dumps(r), flush=True)
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
