#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MISS 258 repro check: ptxv num claims of the case against exact values computed here (Decimal,
60 digits; extended reals for the infinite cases). usage: check.py NUMDIR MANIFEST OUT_CSV
NUMDIR holds <case>.good.<front>.num.csv. A claim: untracked=false and nonfinite=false.
Per element:
  E: the exact value of the chain at the exact operand. 'undef' where a function has no value (sqrt,
     rsqrt, lg2 of a negative number, rcp at 0) and min / max / add / ex2 of 'undef'; -inf / +inf for
     lg2 / rsqrt of 0 and what follows (ex2(-inf) = 0, max(-inf, c) = c, rcp(+inf) = 0).
  nonfin: some MUFU of the chain has a true value that is not finite at a finite operand. Then
     N-NUM-NONFINITE (and MISS 258's rule) say the element is not analyzed.
Verdicts:
  ok, unclaimed;
  CLAIM_NONFINITE  a claim where nonfin holds (the rule says not analyzed); with the error the claim
                   should have had where E is defined: +UNDER (claimed abs_err below |hw - E|, also any
                   finite claim against an infinite error), +OVER (above), +BOUND (bound below |hw - E|);
  WRONG_ERR / WRONG_BOUND  a claim against a finite, defined E with no MUFU going non-finite."""
import csv
import json
import os
import struct
import sys
from decimal import Decimal, getcontext

getcontext().prec = 60
NUMDIR, MAN, OUT = sys.argv[1:4]
man = json.load(open(MAN))
fb = lambda b: struct.unpack("<f", struct.pack("<I", b & 0xFFFFFFFF))[0]
LN2 = Decimal(2).ln()
UNDEF, PINF, NINF = ("undef",), ("+inf",), ("-inf",)


def D(v):
    return Decimal(v) if not isinstance(v, Decimal) else v


class Ev:
    def __init__(self):
        self.nonfin = False

    def mufu(self, f, v):
        if v[0] == "fin":
            x = v[1]
            if f == "lg2":
                r = ("fin", x.ln() / LN2) if x > 0 else (NINF if x == 0 else UNDEF)
            elif f == "rsqrt":
                r = ("fin", 1 / x.sqrt()) if x > 0 else (PINF if x == 0 else UNDEF)
            elif f == "rcp":
                r = ("fin", 1 / x) if x != 0 else UNDEF
            elif f == "sqrt":
                r = ("fin", x.sqrt()) if x >= 0 else UNDEF
            elif f == "ex2":
                r = ("fin", (x * LN2).exp())
            if r[0] != "fin":
                self.nonfin = True
            return r
        if v == UNDEF:
            return UNDEF
        if f == "ex2":
            return ("fin", Decimal(0)) if v == NINF else PINF
        if f in ("rcp", "rsqrt"):
            return ("fin", Decimal(0)) if v == PINF else UNDEF
        if f in ("lg2", "sqrt"):
            return PINF if v == PINF else UNDEF
        return UNDEF


def key(v):
    return {"-inf": -1, "+inf": 1}.get(v[0], 0), (v[1] if v[0] == "fin" else Decimal(0))


def vmax(a, b, mn=False):
    if UNDEF in (a, b):
        return UNDEF
    ka, kb = key(a), key(b)
    return (a if ka <= kb else b) if mn else (a if ka >= kb else b)


def vadd(a, c):
    if a == UNDEF:
        return UNDEF
    return a if a[0] != "fin" else ("fin", a[1] + c)


def chain(name, x):
    e = Ev()
    m = e.mufu
    c = lambda v: ("fin", D(v))
    r = {
        "lg2": lambda: m("lg2", x), "rsqrt": lambda: m("rsqrt", x), "rcp": lambda: m("rcp", x),
        "lg2f": lambda: m("lg2", x), "rsqrtf": lambda: m("rsqrt", x), "rcpf": lambda: m("rcp", x),
        "max(lg2f,-1000)": lambda: vmax(m("lg2", x), c(-1000)),
        "min(rsqrtf,1000)": lambda: vmax(m("rsqrt", x), c(1000), True),
        "min(rcpf,1000)": lambda: vmax(m("rcp", x), c(1000), True),
        "ex2f(lg2f)": lambda: m("ex2", m("lg2", x)),
        "rcpf(rsqrtf)": lambda: m("rcp", m("rsqrt", x)),
        "max(lg2,-1000)": lambda: vmax(m("lg2", x), c(-1000)),
        "add(lg2f,1000)": lambda: vadd(m("lg2", x), D(1000)),
        "min(sqrtf,-200)": lambda: vmax(m("sqrt", x), c(-200), True),
        "max(sqrtf,-200)": lambda: vmax(m("sqrt", x), c(-200)),
        "selp_const(sqrtf)": lambda: c(5),
        "nanguard(sqrt)": lambda: (lambda s: ("fin", Decimal(0)) if s == UNDEF else s)(m("sqrt", x)),
        "sqrt": lambda: m("sqrt", x), "sqrtf": lambda: m("sqrt", x),
        "max(sqrt,-200)": lambda: vmax(m("sqrt", x), c(-200)),
        "ex2f(sqrtf)": lambda: m("ex2", m("sqrt", x)),
        "max(lg2f,-200)": lambda: vmax(m("lg2", x), c(-200)),
    }[name]()
    if name == "selp_const(sqrtf)":
        e.nonfin = False  # the selected value is the constant; the discarded op is not in it
    return r, e.nonfin


def rows(case, front):
    p = os.path.join(NUMDIR, f"{case}.good.{front}.num.csv")
    if not os.path.exists(p):
        return None
    lines = open(p).read().splitlines()
    hdr = lines[0].split(",")
    return {int(r[0]): dict(zip(hdr, r)) for r in (ln.split(",") for ln in lines[1:])}


def fl(s):
    return float("nan") if s in ("unknown", "") else float(s)


def judge(r, E, nonfin):
    if r is None:
        return "unclaimed(no csv)", ""
    if not (r["untracked"] == "false" and r["nonfinite"] == "false"):
        return "unclaimed", ""
    hw = struct.unpack("<f", struct.pack("<f", fl(r["hw"])))[0]  # the f32 value the CSV prints (shortest repr)
    ae, b = fl(r["abs_err"]), fl(r["bound_cor1a"])
    if E == UNDEF:
        true = None
    elif E[0] == "fin":
        true = abs(Decimal(hw) - E[1]) if hw == hw and abs(hw) != float("inf") else Decimal("Infinity")
    else:
        true = Decimal(0) if (hw == float("inf")) == (E == PINF) and abs(hw) == float("inf") else Decimal("Infinity")
    tags = []
    if true is not None:
        tf = float(true)
        if ae < tf * (1 - 1e-6) - 1e-45:
            tags.append("UNDER")
        elif ae > tf * (1 + 1e-6) + 1e-45:
            tags.append("OVER")
        if b == b and b < tf * (1 - 1e-9):
            tags.append("BOUND")
    if nonfin:
        return "CLAIM_NONFINITE" + ("+" + "+".join(tags) if tags else ""), "" if true is None else f"{float(true):.9g}"
    if E == UNDEF:
        return "CLAIM_NONFINITE(undef)", ""
    if "UNDER" in tags or "OVER" in tags:
        return "WRONG_ERR" + ("+BOUND" if "BOUND" in tags else ""), f"{float(true):.9g}"
    if "BOUND" in tags:
        return "WRONG_BOUND", f"{float(true):.9g}"
    return "ok", f"{float(true):.9g}"


def ev_str(E):
    return E[0] if E[0] != "fin" else f"{float(E[1]):.9g}"


out = []
for case, m in man.items():
    for front in ("ptx", "sass"):
        rs = rows(case, front)
        if case == "r8m_mma":
            for i in range(128):
                E, nonfin = (UNDEF, True) if i < 16 else (("fin", Decimal(16)), False)
                r = rs.get(i) if rs else None
                v, t = judge(r, E, nonfin)
                out.append((case, front, i, "D(bf16(sqrtf(x)) x ones)", "row0/8" if i < 16 else "rows1-7/9-15",
                            r["hw"] if r else "", r["abs_err"] if r else "", r["bound_cor1a"] if r else "",
                            ev_str(E), t, v))
            continue
        for k, cname in enumerate(m["chains"]):
            for t in range(32):
                if "kind" not in m or m["kind"][t] == "chain":
                    a, b, c_, e_ = (D(fb(v)) for v in m["abce"][t])
                    x = ("fin", ((a + b) + c_) + e_)
                    hx = fb(struct.unpack("<I", struct.pack("<f", fb(struct.unpack("<I", struct.pack("<f", fb(struct.unpack("<I", struct.pack("<f", float(a) + float(b)))[0]) + float(c_)))[0]) + float(e_)))[0])
                    lane = f"chain hwx={hx:g} x={float(x[1]):g}"
                else:
                    x = ("fin", D(fb(m["x"][t])))
                    lane = f"x={fb(m['x'][t]):.9g}"
                E, nonfin = chain(cname, x)
                r = rs.get(32 * k + t) if rs else None
                v, tr = judge(r, E, nonfin)
                out.append((case, front, 32 * k + t, cname, lane, r["hw"] if r else "", r["abs_err"] if r else "",
                            r["bound_cor1a"] if r else "", ev_str(E), tr, v))
with open(OUT, "w") as f:
    w = csv.writer(f, lineterminator="\n")
    w.writerow(["case", "front", "index", "chain", "lane", "hw", "abs_err", "bound_cor1a", "exact", "true_err", "verdict"])
    w.writerows(out)
agg = {}
for o in out:
    agg.setdefault((o[0], o[1], o[3], o[4]), {}).setdefault(o[10], 0)
    agg[(o[0], o[1], o[3], o[4])][o[10]] += 1
bad = 0
for k_, v in sorted(agg.items()):
    flag = any(x.startswith(("CLAIM_NONFINITE", "WRONG")) for x in v)
    bad |= flag
    if flag:
        print(f"{k_[0]:11} {k_[1]:4} {k_[2]:18} {k_[3]:28} " + " ".join(f"{x}={n}" for x, n in sorted(v.items())))
tot = {}
for o in out:
    tot.setdefault((o[0], o[1]), {}).setdefault(o[10].split("+")[0].split("(")[0], 0)
    tot[(o[0], o[1])][o[10].split("+")[0].split("(")[0]] += 1
for k_, v in sorted(tot.items()):
    print(f"TOTAL {k_[0]:11} {k_[1]:4} " + " ".join(f"{x}={n}" for x, n in sorted(v.items())))
sys.exit(1 if bad else 0)
