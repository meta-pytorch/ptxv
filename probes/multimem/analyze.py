# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Analysis of `mm ldr` / `mm red` dumps (probes/multimem/mm.cu): which reduction-order and
# rounding model reproduces GB300's multimem.ld_reduce / multimem.red results bitwise, and
# whether results depend on the issuing device or the repetition.
#
#   python analyze.py ldr RUNDIR/ldr_0,1,2,3
#   python analyze.py red RUNDIR/red
import itertools
import os
import sys
from fractions import Fraction

import numpy as np

# ------------------------------------------------------------------ formats
FMT = {  # name: (exponent bits, mantissa bits)
    "f32": (8, 23),
    "bf16": (8, 7),
    "f16": (5, 10),
    "e4m3": (4, 3),
    "e5m2": (5, 2),
}


def decode(bits, fmt):
    """bits (int) -> Fraction, or 'nan' / ('inf', sign)."""
    e, m = FMT[fmt]
    w = 1 + e + m
    s = (bits >> (w - 1)) & 1
    ex = (bits >> m) & ((1 << e) - 1)
    man = bits & ((1 << m) - 1)
    bias = (1 << (e - 1)) - 1
    if fmt == "e4m3":
        if ex == 15 and man == 7:
            return "nan"
    elif ex == (1 << e) - 1:
        return "nan" if man else ("inf", s)
    if ex == 0:
        v = Fraction(man, 1 << m) * Fraction(2) ** (1 - bias)
    else:
        v = (1 + Fraction(man, 1 << m)) * Fraction(2) ** (ex - bias)
    return -v if s else (v if v else Fraction(0))  # +0 / -0 kept by zsign below


def zsign(bits, fmt):
    e, m = FMT[fmt]
    return (bits >> (e + m)) & 1


def rnd(v, fmt, ftz=False):
    """Fraction -> (bits) round to nearest even; overflow -> inf (e4m3: nan)."""
    e, m = FMT[fmt]
    w = 1 + e + m
    bias = (1 << (e - 1)) - 1
    s = 1 if v < 0 else 0
    a = -v if v < 0 else v
    if a == 0:
        return s << (w - 1)
    # exponent
    ex = a.numerator.bit_length() - a.denominator.bit_length()
    if Fraction(2) ** ex > a:
        ex -= 1
    while Fraction(2) ** (ex + 1) <= a:
        ex += 1
    emin = 1 - bias
    if ex < emin:
        if ftz:
            return s << (w - 1)
        q = a / (Fraction(2) ** emin) * (1 << m)  # subnormal units
    else:
        q = a / (Fraction(2) ** ex) * (1 << m)  # in [2^m, 2^(m+1))
    fl = q.numerator // q.denominator
    rem = q - fl
    if rem > Fraction(1, 2) or (rem == Fraction(1, 2) and fl & 1):
        fl += 1
    if ex < emin:
        bits = fl  # may carry into the normal range: correct encoding
        if ftz and bits < (1 << m):
            return s << (w - 1)
        return (s << (w - 1)) | bits
    if fl == (1 << (m + 1)):
        fl >>= 1
        ex += 1
    eb = ex + bias
    maxe = (1 << e) - 1 if fmt != "e4m3" else (1 << e)
    if fmt == "e4m3":
        if eb > 15 or (eb == 15 and (fl - (1 << m)) == 7):
            return (s << 7) | 0x7F  # nan (satfinite would be 0x7e)
    elif eb >= maxe:
        return (s << (w - 1)) | (((1 << e) - 1) << m)
    return (s << (w - 1)) | (eb << m) | (fl - (1 << m))


def _round_val(v, z, fmt, ftz):
    """Round exact (v, zero-sign z) to fmt: returns bits."""
    if v == 0:
        e, m = FMT[fmt]
        return (1 << (e + m)) if z else 0
    return rnd(v, fmt, ftz)


def _special(xs):
    if any(x == "nan" for x in xs):
        return "nan"
    infs = [x[1] for x in xs if isinstance(x, tuple)]
    if infs:
        return "nan" if len(set(infs)) > 1 else ("inf", infs[0])
    return None


def add_model(vals, fmt_in, acc, fmt_out, order, ftz=False):
    """Fold vals[order[0]] + vals[order[1]] + ... rounding each step to `acc` (None = exact),
    then to fmt_out. Returns bits, "nan" or ("inf", sign)."""
    xs = [decode(vals[i], fmt_in) for i in order]
    zs = [zsign(vals[i], fmt_in) for i in order]
    sp = _special(xs)
    if sp is not None:
        return sp
    if ftz:
        e, m = FMT[fmt_in]
        tiny = Fraction(2) ** (2 - (1 << (e - 1)))
        xs = [Fraction(0) if abs(x) < tiny else x for x in xs]
    s, sz = xs[0], zs[0] == 1 and xs[0] == 0
    for x, z in zip(xs[1:], zs[1:]):
        r = s + x
        rz = sz and (z == 1 and x == 0) if r == 0 and s == 0 and x == 0 else False
        if acc is not None:
            bits = _round_val(r, rz, acc, ftz)
            d = decode(bits, acc)
            if d == "nan" or isinstance(d, tuple):
                return d if d == "nan" else ("inf", d[1])
            rz = d == 0 and zsign(bits, acc) == 1
            r = d
        s, sz = r, rz
    bits = _round_val(s, sz, fmt_out, ftz)
    d = decode(bits, fmt_out)
    if d == "nan" or isinstance(d, tuple):
        return d if d == "nan" else ("inf", d[1])
    return bits


def canon(bits, fmt):
    d = decode(bits, fmt)
    if d == "nan":
        return "nan"
    if isinstance(d, tuple):
        return d
    return bits


def lanes(arr_u8, lane_bytes):
    if lane_bytes == 4:
        return arr_u8.view(np.uint32).astype(np.int64)
    if lane_bytes == 2:
        return arr_u8.view(np.uint16).astype(np.int64)
    if lane_bytes == 1:
        return arr_u8.astype(np.int64)
    if lane_bytes == 8:
        return arr_u8.view(np.uint64)
    raise ValueError


def orders(n):
    out = {}
    for p in itertools.permutations(range(n)):
        out["fold" + "".join(map(str, p))] = ("fold", p)
    if n == 4:
        for pr in [((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))]:
            out[f"tree{pr[0][0]}{pr[0][1]}|{pr[1][0]}{pr[1][1]}"] = ("tree", pr)
    out["exact"] = ("exact", None)
    return out


def tree_model(vals, fmt_in, acc, fmt_out, pr, ftz=False):
    a = add_model(vals, fmt_in, acc, acc or fmt_out, pr[0], ftz)
    b = add_model(vals, fmt_in, acc, acc or fmt_out, pr[1], ftz)
    if a == "nan" or b == "nan":
        return "nan"
    if isinstance(a, tuple) or isinstance(b, tuple):
        sg = {x[1] for x in (a, b) if isinstance(x, tuple)}
        return "nan" if len(sg) > 1 else ("inf", sg.pop())
    f = acc or fmt_out
    return add_model([a, b], f, acc, fmt_out, (0, 1), ftz)


FLOAT_TESTS = {
    # name: (lane bytes, input fmt, acc fmt (None: same as type), output fmt)
    "f32x4": (4, "f32", "f32", "f32"),
    "f32": (4, "f32", "f32", "f32"),
    "f32w": (4, "f32", "f32", "f32"),
    "bf16x8": (2, "bf16", "bf16", "bf16"),
    "bf16x8a": (2, "bf16", "f32", "bf16"),
    "f16x8": (2, "f16", "f16", "f16"),
    "f16x8a": (2, "f16", "f32", "f16"),
    "e4m3x16a": (1, "e4m3", "f16", "e4m3"),
    "e4m3x16": (1, "e4m3", "e4m3", "e4m3"),
    "e5m2x16a": (1, "e5m2", "f16", "e5m2"),
}


def load(d, name):
    return np.fromfile(os.path.join(d, name), dtype=np.uint8)


def meta(d):
    m = {}
    for line in open(os.path.join(d, "meta.txt")):
        k, *v = line.split()
        m[k] = v
    return m


def analyze_ldr(d, sample=4000):
    m = meta(d)
    nm = len(m["members"])
    reps = int(m["REPS"][0])
    lines = [f"members (add order) {m['members']}"]
    for t in sorted(set(f.split(".")[0] for f in os.listdir(d) if f.endswith(".bin"))):
        ins = [load(d, f"{t}.in{i}.bin") for i in range(nm)]
        outs = {
            (i, r): load(d, f"{t}.out{i}.{r}.bin")
            for i in range(nm)
            for r in range(reps)
            if os.path.exists(os.path.join(d, f"{t}.out{i}.{r}.bin"))
        }
        ref = outs[(0, 0)]
        same = all(np.array_equal(ref, o) for o in outs.values())
        lines.append(
            f"{t}: {len(outs)} results (issuing member x rep) bitwise identical: {same}"
        )
        if t in FLOAT_TESTS:
            lb, fin, acc, fout = FLOAT_TESTS[t]
            X = [lanes(a, lb) for a in ins]
            Y = lanes(ref, lb)
            idx = np.arange(len(Y))
            rng = np.random.default_rng(0)
            pick = rng.choice(idx, size=min(sample, len(Y)), replace=False)
            for ftz in (False, True):
                res = {}
                for name, (kind, arg) in orders(nm).items():
                    bad = 0
                    for j in pick:
                        vals = [int(x[j]) for x in X]
                        if kind == "fold":
                            want = add_model(vals, fin, acc, fout, arg, ftz)
                        elif kind == "tree":
                            want = tree_model(vals, fin, acc, fout, arg, ftz)
                        else:
                            want = add_model(
                                vals, fin, None, fout, tuple(range(nm)), ftz
                            )
                        got = canon(int(Y[j]), fout)
                        w = (
                            want
                            if isinstance(want, (str, tuple))
                            else canon(want, fout)
                        )
                        if w != got:
                            bad += 1
                    res[name] = bad
                best = sorted(res.items(), key=lambda kv: kv[1])
                ok = [k for k, v in best if v == 0]
                lines.append(
                    f"  ftz={int(ftz)} lanes={len(pick)} exact-match models: {ok or 'none'}; "
                    f"best: {best[:4]}"
                )
            # NaN encodings produced
            nanbits = sorted({int(y) for y in Y if canon(int(y), fout) == "nan"})
            lines.append(f"  NaN encodings in results: {[hex(b) for b in nanbits[:4]]}")
        else:
            lb = {"u64add": 8}.get(t, 4)
            X = [lanes(a, lb) for a in ins]
            Y = lanes(ref, lb)
            if t in ("u32add", "u64add"):
                mask = (1 << (8 * lb)) - 1
                want = [sum(int(x[j]) for x in X) & mask for j in range(len(Y))]
            elif t == "xor":
                want = [
                    int(X[0][j])
                    ^ int(X[1][j])
                    ^ (int(X[2][j]) if nm > 2 else 0)
                    ^ (int(X[3][j]) if nm > 3 else 0)
                    for j in range(len(Y))
                ]
            elif t == "s32min":
                sx = [((x + (1 << 31)) % (1 << 32)) - (1 << 31) for x in X]
                want = [min(int(s[j]) for s in sx) & 0xFFFFFFFF for j in range(len(Y))]
            elif t == "u32max":
                want = [max(int(x[j]) for x in X) for j in range(len(Y))]
            elif t in ("bf16x2min", "bf16x2max", "f16x2max"):
                fmt = "bf16" if t.startswith("bf16") else "f16"
                mx = t.endswith("max")
                X16 = [lanes(a, 2) for a in ins]
                Y16 = lanes(ref, 2)
                cats = {}
                bad = 0
                for j in range(len(Y16)):
                    vs = [int(x[j]) for x in X16]
                    ds = [decode(v, fmt) for v in vs]
                    got = int(Y16[j])
                    finite = [
                        (
                            d
                            if not isinstance(d, tuple)
                            else (
                                Fraction(10) ** 40
                                if d[1] == 0
                                else -(Fraction(10) ** 40)
                            ),
                            v,
                        )
                        for d, v in zip(ds, vs)
                        if d != "nan"
                    ]
                    if not finite:
                        cat = "allnan"
                        want = "nan"
                    else:
                        # minNum/maxNum: NaN operands ignored; -0 < +0
                        key = lambda dv: (
                            dv[0],
                            -zsign(dv[1], fmt) if dv[0] == 0 else 0,
                        )
                        pick_ = max(finite, key=key) if mx else min(finite, key=key)
                        want = pick_[1]
                        cat = "nan-in" if len(finite) < nm else "normal"
                    ok_ = (want == "nan" and canon(got, fmt) == "nan") or want == got
                    cats.setdefault(cat, [0, 0])
                    cats[cat][0] += 1
                    cats[cat][1] += 0 if ok_ else 1
                    bad += 0 if ok_ else 1
                lines.append(
                    f"  model minNum/maxNum (NaN ignored, -0 < +0): lanes {len(Y16)} mismatches {bad}; per category [n, bad]: {cats}"
                )
                continue
            else:
                continue
            bad = sum(1 for j in range(len(Y)) if int(Y[j]) != want[j])
            lines.append(f"  integer model mismatches: {bad} of {len(Y)}")
    return lines


def analyze_red(d):
    m = meta(d)
    nm = len(m["members"])
    reps = int(m["REPS"][0])
    lines = [f"members {m['members']}"]
    for kn in ("f32", "bf16x8", "s32"):
        eq_members = 0
        distinct_runs = []
        for r in range(reps):
            fins = [load(d, f"red_{kn}.fin{i}.{r}.bin") for i in range(nm)]
            inits = [load(d, f"red_{kn}.init{i}.{r}.bin") for i in range(nm)]
            # member-wise: fin_i - init_i is the contribution sum; equal across members only if
            # the same values were added in an order-insensitive way
            if kn == "s32":
                f0 = fins[0].view(np.int32)
                want = 8 * sum(range(1, nm + 1))
                lines.append(
                    f"  s32 rep {r}: every member == {want}: {all(np.all(f.view(np.int32) == want) for f in fins)}"
                )
                continue
            if kn == "f32":
                # members started from different values: compare (fin - init) is not meaningful in
                # floating point; report whether each member's result equals some order of adds.
                srcs = [
                    load(d, f"red_{kn}.src{i}.{r}.bin").view(np.uint32)
                    for i in range(nm)
                ]
                fi = [f.view(np.uint32) for f in fins]
                ii = [x.view(np.uint32) for x in inits]
                n = len(fi[0])
                rng = np.random.default_rng(r)
                pick = rng.choice(np.arange(n), size=min(300, n), replace=False)
                okorder = 0
                total = 0
                for j in pick:
                    for mem in range(nm):
                        contrib = [
                            int(srcs[i][k * n + j]) for i in range(nm) for k in range(8)
                        ]
                        got = canon(int(fi[mem][j]), "f32")
                        # arrival orders: device-major interleavings are too many; check the
                        # per-device program order sequences merged in member order and
                        # in reverse, plus a sorted-agnostic check via exact sum.
                        cands = set()
                        for p in itertools.permutations(range(nm)):
                            seq = [int(ii[mem][j])] + [
                                int(srcs[i][k * n + j]) for i in p for k in range(8)
                            ]
                            for ftz in (True, False):
                                w = add_model(
                                    seq,
                                    "f32",
                                    "f32",
                                    "f32",
                                    tuple(range(len(seq))),
                                    ftz,
                                )
                                cands.add(
                                    w
                                    if isinstance(w, (str, tuple))
                                    else canon(w, "f32")
                                )
                        total += 1
                        okorder += got in cands
                lines.append(
                    f"  f32 rep {r}: member results matching a device-sequential arrival order (FTZ or not): {okorder}/{total}"
                )
    return lines


if __name__ == "__main__":
    mode, d = sys.argv[1], sys.argv[2]
    out = analyze_ldr(d) if mode == "ldr" else analyze_red(d)
    print("\n".join(out))
