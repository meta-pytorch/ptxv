# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Exact rational models of a 16-bit float add for probes/half_add (gen.py, check.py).

A format is (exponent bits, fraction bits): F16 = (5, 10), BF16 = (8, 7). `add(fmt, a, b,
mode)` is the IEEE-754 sum of the two patterns rounded once in `mode` ("rn": nearest even, "rz":
toward zero), with subnormals kept unless `ftz` (subnormal inputs and results become a zero of
the same sign), and every NaN result `nan` (default the canonical 0x7fff). An exact zero sum is
-0 only for (-0) + (-0) (RN and RZ alike)."""

from fractions import Fraction

F16 = (5, 10)
BF16 = (8, 7)


def parts(fmt, h):
    eb, fb = fmt
    s = h >> 15 & 1
    e = h >> fb & ((1 << eb) - 1)
    f = h & ((1 << fb) - 1)
    return s, e, f


def bias(fmt):
    return (1 << (fmt[0] - 1)) - 1


def is_nan(fmt, h):
    s, e, f = parts(fmt, h)
    return e == (1 << fmt[0]) - 1 and f != 0


def is_inf(fmt, h):
    s, e, f = parts(fmt, h)
    return e == (1 << fmt[0]) - 1 and f == 0


def is_sub(fmt, h):
    s, e, f = parts(fmt, h)
    return e == 0 and f != 0


def is_snan(fmt, h):
    return is_nan(fmt, h) and not h >> (fmt[1] - 1) & 1


def val(fmt, h):
    """Exact value of a finite pattern."""
    eb, fb = fmt
    s, e, f = parts(fmt, h)
    sg = -1 if s else 1
    if e == 0:
        return sg * Fraction(f, 1 << fb) * Fraction(2) ** (1 - bias(fmt))
    return sg * (1 + Fraction(f, 1 << fb)) * Fraction(2) ** (e - bias(fmt))


def max_finite(fmt):
    eb, fb = fmt
    return (2 - Fraction(1, 1 << fb)) * Fraction(2) ** ((1 << eb) - 2 - bias(fmt))


def round_val(fmt, x, mode):
    """(pattern, inexact) of the nonzero rational x rounded to fmt in mode."""
    eb, fb = fmt
    s = 1 if x < 0 else 0
    a = abs(x)
    e = 0
    while a >= 2:
        a /= 2
        e += 1
    while a < 1:
        a *= 2
        e -= 1
    emin = 1 - bias(fmt)
    q = max(e, emin) - fb  # quantum exponent
    k = abs(x) / Fraction(2) ** q  # significand in quanta
    fl = k.numerator // k.denominator
    rem = k - fl
    if mode == "rn":
        up = rem > Fraction(1, 2) or (rem == Fraction(1, 2) and fl % 2 == 1)
    elif mode == "rz":
        up = False
    else:
        raise ValueError(mode)
    m = fl + up
    inexact = rem != 0
    if m == 0:
        return s << 15, inexact
    v = m * Fraction(2) ** q
    if v > max_finite(fmt):
        if mode == "rz":
            return s << 15 | ((((1 << eb) - 2) << fb) | ((1 << fb) - 1)), True
        return s << 15 | (((1 << eb) - 1) << fb), True
    if m < 1 << fb:
        return s << 15 | m, inexact  # subnormal
    if m >= 2 << fb:
        m >>= 1
        q += 1
    be = q + fb + bias(fmt)
    return s << 15 | be << fb | (m - (1 << fb)), inexact


def add(fmt, a, b, mode="rn", ftz=False, nan=0x7FFF):
    if is_nan(fmt, a) or is_nan(fmt, b):
        return nan(a, b) if callable(nan) else nan
    if ftz:
        a = a & 0x8000 if is_sub(fmt, a) else a
        b = b & 0x8000 if is_sub(fmt, b) else b
    ia, ib = is_inf(fmt, a), is_inf(fmt, b)
    if ia and ib:
        return a if a == b else (nan(a, b) if callable(nan) else nan)
    if ia:
        return a
    if ib:
        return b
    x = val(fmt, a) + val(fmt, b)
    if x == 0:
        if val(fmt, a) == 0 and val(fmt, b) == 0:
            return 0x8000 if a & 0x8000 and b & 0x8000 else 0
        return 0
    r, _ = round_val(fmt, x, mode)
    if ftz and is_sub(fmt, r):
        r &= 0x8000
    return r


def exact_sum(fmt, a, b):
    """The exact sum of two finite patterns."""
    return val(fmt, a) + val(fmt, b)


def is_tie(fmt, x):
    """x (nonzero) lies exactly halfway between two adjacent values of fmt below overflow."""
    if x == 0:
        return False
    eb, fb = fmt
    a = abs(x)
    e = 0
    while a >= 2:
        a /= 2
        e += 1
    while a < 1:
        a *= 2
        e -= 1
    q = max(e, 1 - bias(fmt)) - fb
    k = abs(x) / Fraction(2) ** q
    return k.denominator == 2
