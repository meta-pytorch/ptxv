# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Check GB300's probes/half_add outputs (expected/out.bin, expected/m.bin) and the controls
(controls/expected/out.bin) against exact rational models (half.py), lane by lane.

Usage: check.py [main dump] [controls dump]   (default: the expected/ files)

The model under test (N-F16X2 / N-BF16X2): one round-to-nearest-even rounding, subnormal
inputs and results kept, every NaN result 0x7fff, an exact zero sum -0 only for (-0) + (-0).
Alternatives, each of which must be refuted by the data (else the probe does not discriminate
it): RZ rounding, flush-to-zero of subnormals, NaN propagation (the first NaN operand quieted,
payload and sign kept), and -0 for any exact zero sum with a -0 operand. The controls must match
their own models (RZ, FTZ) and differ from RN on the classes they target."""

import sys
from collections import Counter

import numpy as np

from half import BF16, F16, add, exact_sum, is_inf, is_nan, is_snan, is_sub, is_tie, val

N = 4096


def lanes(w):
    return [int(w) & 0xFFFF, int(w) >> 16]


def quiet_first(fmt):
    q = 1 << (fmt[1] - 1)
    return lambda a, b: (a if is_nan(fmt, a) else b) | q


def classes(fmt, x, y):
    c = []
    if is_nan(fmt, x) or is_nan(fmt, y):
        c.append("nan")
        if is_snan(fmt, x) or is_snan(fmt, y):
            c.append("nan:signalling")
        if (is_nan(fmt, x) and x & 0x8000) or (is_nan(fmt, y) and y & 0x8000):
            c.append("nan:negative")
        if any(is_nan(fmt, h) and (h & 0x7FFF) != 0x7FFF for h in (x, y)):
            c.append("nan:non-canonical payload")
        return c
    if is_inf(fmt, x) or is_inf(fmt, y):
        c.append("inf operand")
        if is_inf(fmt, x) and is_inf(fmt, y) and x != y:
            c.append("inf - inf")
        return c
    if is_sub(fmt, x) or is_sub(fmt, y):
        c.append("subnormal input")
    if (x & 0x7FFF) == 0 and (y & 0x7FFF) == 0:
        c.append("zero + zero: " + ("-" if x & 0x8000 else "+") + "0 + " + ("-" if y & 0x8000 else "+") + "0")
    s = exact_sum(fmt, x, y)
    if s == 0 and not ((x & 0x7FFF) == 0 and (y & 0x7FFF) == 0):
        c.append("exact cancellation (x + -x)")
    r = add(fmt, x, y)
    if is_tie(fmt, s) and not is_inf(fmt, r):
        c.append("tie, rounded up in magnitude" if abs(val(fmt, r)) > abs(s) else "tie, rounded down in magnitude")
    if is_inf(fmt, r):
        c.append("overflow to inf")
    elif s != 0 and is_sub(fmt, r):
        c.append("subnormal result")
    return c


def pairs(inp):
    """(x, y) lane pairs of the 4096 (a, b) words, in thread-lane order."""
    a, b = inp[:, 0], inp[:, 1]
    out = []
    for t in range(N):
        la, lb = lanes(a[t]), lanes(b[t])
        out += [(la[0], lb[0]), (la[1], lb[1])]
    return out


def sel_pairs(inp, sel):
    """Lane pairs of the selector outputs: sel = ((src a lane, src b lane) for lane 0, lane 1)."""
    a, b = inp[:, 0], inp[:, 1]
    out = []
    for t in range(N):
        la, lb = lanes(a[t]), lanes(b[t])
        out += [(la[sel[0][0]], lb[sel[0][1]]), (la[sel[1][0]], lb[sel[1][1]])]
    return out


def got(words):
    r = []
    for w in words:
        r += lanes(w)
    return r


def main():
    main_dump = sys.argv[1] if len(sys.argv) > 1 else None
    ctl_dump = sys.argv[2] if len(sys.argv) > 2 else None
    inh = np.fromfile("inh.bin", np.uint32).reshape(N, 2)
    inb = np.fromfile("inb.bin", np.uint32).reshape(N, 2)
    if main_dump:
        raw = np.fromfile(main_dump, np.uint32)
        out, m = raw[: 12 * N].reshape(12, N), raw[12 * N :].reshape(8, N)
    else:
        out = np.fromfile("expected/out.bin", np.uint32).reshape(12, N)
        m = np.fromfile("expected/m.bin", np.uint32).reshape(8, N)
    ctl = np.fromfile(ctl_dump or "controls/expected/out.bin", np.uint32).reshape(6, N)
    ph, pb = pairs(inh), pairs(inb)
    a_h, a_b = got(inh[:, 0]), got(inb[:, 0])
    # (name, fmt, lane pairs, GB300 lanes)
    tests = [
        ("add.rn.f16x2 (HFMA2 a, 1, 1, b)", F16, ph, got(out[0])),
        ("HADD2 a, b.H1_H1", F16, sel_pairs(inh, ((0, 1), (1, 1))), got(out[1])),
        ("HADD2 a, b.H0_H0 (lanes swapped)", F16, sel_pairs(inh, ((1, 0), (0, 0))), got(out[2])),
        ("add.rn.bf16x2 (HFMA2.BF16_V2 a, 1, 1, b)", BF16, pb, got(out[3])),
        ("HADD2.BF16_V2 a, b.H1_H1", BF16, sel_pairs(inb, ((0, 1), (1, 1))), got(out[4])),
        ("HADD2.BF16_V2 a, b.H0_H0 (lanes swapped)", BF16, sel_pairs(inb, ((1, 0), (0, 0))), got(out[5])),
        ("ATOMG.E.ADD.F16x2.RN (final word)", F16, ph, got(m[0])),
        ("ATOM.E.ADD.F16x2.RN generic (final word)", F16, ph, got(m[1])),
        ("REDG.E.ADD.F16x2.RN (final word)", F16, ph, got(m[2])),
        ("atom.add.noftz.f16 generic, CAS loop (final word)", F16, ph, got(m[3])),
        ("ATOMG.E.ADD.BF16x2.RN (final word)", BF16, pb, got(m[4])),
        ("ATOM.E.ADD.BF16x2.RN generic (final word)", BF16, pb, got(m[5])),
        ("REDG.E.ADD.BF16x2.RN (final word)", BF16, pb, got(m[6])),
        ("atom.add.noftz.bf16 generic, CAS loop (final word)", BF16, pb, got(m[7])),
    ]
    olds = [
        ("ATOMG.E.ADD.F16x2.RN old", out[6], inh),
        ("ATOM.E.ADD.F16x2.RN old", out[7], inh),
        ("atom.add.noftz.f16 old (two halves)", out[8], inh),
        ("ATOMG.E.ADD.BF16x2.RN old", out[9], inb),
        ("ATOM.E.ADD.BF16x2.RN old", out[10], inb),
        ("atom.add.noftz.bf16 old (two halves)", out[11], inb),
    ]
    alts = {
        "RZ": dict(mode="rz"),
        "FTZ": dict(ftz=True),
        "NaN propagation (first NaN quieted)": None,
    }
    fails = 0
    totals = Counter()
    for name, fmt, ps, hw in tests:
        bad = 0
        cls = Counter()
        refute = Counter()
        for (x, y), h in zip(ps, hw):
            r = add(fmt, x, y)
            if h != r:
                bad += 1
            for c in classes(fmt, x, y):
                cls[c] += 1
            for an, kw in alts.items():
                ra = add(fmt, x, y, nan=quiet_first(fmt)) if kw is None else add(fmt, x, y, **kw)
                if ra != r and h == r:
                    refute[an] += 1
            if (x & 0x7FFF) == 0 and (y & 0x7FFF) == 0 and x != y and h == r:
                refute["-0 for -0 + +0"] += 1
        fails += bad
        totals[name] = len(ps)
        print(f"{name}: {len(ps) - bad} of {len(ps)} lanes = model")
        print("  classes: " + ", ".join(f"{k} {v}" for k, v in sorted(cls.items())))
        print("  alternatives refuted (lanes where the alternative differs and GB300 = model): "
              + ", ".join(f"{k} {refute[k]}" for k in list(alts) + ["-0 for -0 + +0"]))
    for name, o, inp in olds:
        bad = int(np.sum(o != inp[:, 0]))
        fails += bad
        print(f"{name}: {N - bad} of {N} words = the initial word")
    # Controls.
    ctests = [
        ("control add.rn.f16x2 (= main j = 0)", F16, ph, got(ctl[0]), {}),
        ("control HADD2.FTZ", F16, ph, got(ctl[1]), dict(ftz=True)),
        ("control RZ f16 add", F16, ph, got(ctl[2]), dict(mode="rz")),
        ("control add.rn.bf16x2 (= main j = 3)", BF16, pb, got(ctl[3]), {}),
        ("control RZ bf16 add", BF16, pb, got(ctl[4]), dict(mode="rz")),
        ("control bf16 add, subnormals flushed", BF16, pb, got(ctl[5]), dict(ftz=True)),
    ]
    for name, fmt, ps, hw, kw in ctests:
        finite = [(p, h) for p, h in zip(ps, hw) if not (is_nan(fmt, p[0]) or is_nan(fmt, p[1]))]
        bad = sum(1 for (x, y), h in finite if h != add(fmt, x, y, **kw))
        differ = sum(1 for (x, y), h in finite if h != add(fmt, x, y))
        print(f"{name}: {len(finite) - bad} of {len(finite)} non-NaN lanes = its model; "
              f"{differ} differ from the RN no-flush model")
        if kw == {}:
            fails += bad
    nan_ctl = Counter(h for (x, y), h in zip(ph, got(ctl[1])) if is_nan(F16, x) or is_nan(F16, y))
    print(f"control HADD2.FTZ NaN results: {dict((hex(k), v) for k, v in nan_ctl.items())}")
    print("FAIL" if fails else "PASS", fails)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
