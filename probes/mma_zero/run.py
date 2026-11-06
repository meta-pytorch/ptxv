#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Signed-zero probe for the MMA models (probes/mma_zero, README.md).

  python3 run.py gen              # write expected/<cfg>.{A,B,C}.bin and expected/configs.json
  python3 run.py run [cfg...]     # run every config on GB300 (`ptxv probe`), write <cfg>.D.bin
  python3 run.py report           # tabulate GB300's signs per case family and config

Every case is one MMA output: products a_k * b_k (k < K) plus C (or C disabled). Cases are
given as exact values and encoded per operand format; a case that a format cannot represent
is skipped for it. The positions are in configs.json (`cases`: name, family, D word index,
and the exact sum's sign / magnitude class), which crates/sim/tests/mma_zero.rs replays.
"""
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(HERE, "expected")
PTXV = os.environ.get("PTXV", "ptxv")

# ------------------------------------------------------------------ operand formats


def f32_bits(v):
    return struct.unpack("<I", struct.pack("<f", v))[0]


def f32_exact(v):
    if v != 0 and (abs(v) < 2.0**-126 or abs(v) > 3.0e38):
        return False  # normal operands only (the models' calibrated domain)
    return struct.unpack("<f", struct.pack("<f", v))[0] == v


def enc_bf16(v):
    if not f32_exact(v):
        return None
    u = f32_bits(v)
    return None if u & 0xFFFF else u >> 16


def enc_tf32(v):
    if not f32_exact(v):
        return None
    u = f32_bits(v)
    return None if u & 0x1FFF else u


def enc_f16(v):
    try:
        h = struct.pack("<e", v)
    except OverflowError:
        return None
    if struct.unpack("<e", h)[0] != v:
        return None
    u = struct.unpack("<H", h)[0]
    if v != 0 and (u & 0x7C00) == 0:
        return None  # subnormal
    return u


def lp_table(ebits, mbits, bias, nan):
    t = {}
    for c in range(1 << (1 + ebits + mbits)):
        if c in nan:
            continue
        s = c >> (ebits + mbits)
        e = (c >> mbits) & ((1 << ebits) - 1)
        m = c & ((1 << mbits) - 1)
        if e == 0:
            if m:
                continue  # normal (or zero) operands only
            v = 0.0
        else:
            v = (1 + m / (1 << mbits)) * 2.0 ** (e - bias)
        v = -v if s else v
        t[(v, math.copysign(1, v))] = c
    return t


E4M3 = lp_table(4, 3, 7, {0x7F, 0xFF})
E2M1 = lp_table(2, 1, 1, set())


def enc_lp(tab):
    return lambda v: tab.get((v, math.copysign(1, v)))


FMT = {  # name: (encoder, K, bytes per element (0.5: packed fp4))
    "bf16": (enc_bf16, 16, 2),
    "f16": (enc_f16, 16, 2),
    "tf32": (enc_tf32, 8, 4),
    "e4m3": (enc_lp(E4M3), 32, 1),
    "e2m1": (enc_lp(E2M1), 32, 0.5),
}

# ------------------------------------------------------------------ cases

P2 = lambda e: 2.0**e  # noqa: E731
WIT_A, WIT_B = 0x16B2, 0x96E5  # the reported witness: one product, exactly -40762 * 2^-178


def bf(u):
    return struct.unpack("<f", struct.pack("<I", u << 16))[0]


def cases():
    """(name, family, [(a, b)...], pad, c): pad fills the remaining K slots with
    p: +0 * +0, n: +0 * -0 (-0 products), m: +0 * (+-1 alternating)."""
    out = []
    pads = ["p", "n", "m"]
    cz = [("c+0", 0.0), ("c-0", -0.0)]

    def add(name, fam, prods, padset, cs=cz):
        for pad in padset:
            for cn, c in cs:
                out.append((f"{name}/{pad}/{cn}", fam, prods, pad, c))

    wa, wb = bf(WIT_A), bf(WIT_B)
    add("witness", "underflow", [(wa, wb)], pads)
    add("witness_neg", "underflow", [(wa, -wb)], pads)
    for pe in [-126, -127, -130, -140, -148, -149, -150, -151, -160, -200, -250]:
        ea = pe // 2
        eb = pe - ea
        for ma, mb in [(1.0, 1.0), (1.5, 1.25)]:
            for sgn in ["+", "-a", "-b"]:
                a = ma * P2(ea) * (-1 if sgn == "-a" else 1)
                b = mb * P2(eb) * (-1 if sgn == "-b" else 1)
                fam = "underflow" if abs(a * b) < P2(-149) else "tiny"
                add(f"single_2^{pe}_{ma}x{mb}_{sgn}", fam, [(a, b)], pads)
    # all products exactly zero
    for az in [0.0, -0.0]:
        for b in [1.0, -1.0]:
            add(f"zeros_{'+' if math.copysign(1, az) > 0 else '-'}0x{b:+}", "zeros", [(az, b)] * 32, ["p"])
    add("zeros_one-0", "zeros", [(-0.0, 1.0)] + [(0.0, 1.0)] * 31, ["p"])
    add("zeros_one+0", "zeros", [(0.0, 1.0)] + [(-0.0, 1.0)] * 31, ["p"])
    # exact cancellation
    for x, y in [(1.0, 1.0), (0.5, 1.5), (P2(-60), P2(-60)), (P2(-100), P2(-20))]:
        add(f"cancel_{x:g}x{y:g}_+-", "cancel", [(x, y), (-x, y)], pads)
        add(f"cancel_{x:g}x{y:g}_-+", "cancel", [(-x, y), (x, y)], pads)
    add("cancel3_1+1-2", "cancel", [(1.0, 1.0), (1.0, 1.0), (-2.0, 1.0)], pads)
    # cancellation leaving a residue below the normal range
    r1 = (1 + P2(-7)) * P2(-75)
    add("resid_+2^-156", "underflow", [(r1, P2(-74)), (-P2(-75), P2(-74))], pads)
    add("resid_-2^-156", "underflow", [(-r1, P2(-74)), (P2(-75), P2(-74))], pads)
    r2 = (1 + P2(-7)) * P2(-65)
    add("resid_+2^-132", "tiny", [(r2, P2(-60)), (-P2(-65), P2(-60))], pads)
    add("resid_-2^-132", "tiny", [(-r2, P2(-60)), (P2(-65), P2(-60))], pads)
    # two underflowing products of opposite signs
    add("two_+2^-161", "underflow", [(P2(-80), P2(-80)), (-P2(-81), P2(-80))], pads)
    add("two_-2^-161", "underflow", [(-P2(-80), P2(-80)), (P2(-81), P2(-80))], pads)
    # an underflowing product next to a nonzero accumulator
    for cv in [P2(-126), -P2(-126), 1.0]:
        for s in [1, -1]:
            add(f"acc_{cv:g}_{'+' if s > 0 else '-'}2^-160", "acc", [(s * P2(-80), P2(-80))], ["p"],
                [(f"c{cv:g}", cv)])
    return out


def exact_class(prods, c, en):
    """Sign / class of the exact result: 'zero' (exact 0), 'uflow+/-' (nonzero, < 2^-149),
    'sub+/-' (subnormal range), 'norm+/-'."""
    from fractions import Fraction

    s = sum(Fraction(a) * Fraction(b) for a, b in prods)
    if en:
        s += Fraction(c)
    if s == 0:
        return "zero"
    sg = "+" if s > 0 else "-"
    if abs(s) < Fraction(2) ** -149:
        return "uflow" + sg
    if abs(s) < Fraction(2) ** -126:
        return "sub" + sg
    return "norm" + sg


# ------------------------------------------------------------------ configs

M128N16, M64N16 = (8 << 24) | (2 << 17), (4 << 24) | (2 << 17)
CONFIGS = []  # (name, entry, fmt, idesc, en, modeled)
for m, md in [("m128", M128N16), ("m64", M64N16)]:
    for en in [0, 1]:
        CONFIGS.append((f"tc_bf16_{m}_en{en}", "tc_f16", "bf16", md | 0x490, en, True))
for en in [0, 1]:
    CONFIGS.append((f"tc_f16_m128_en{en}", "tc_f16", "f16", M128N16 | 0x10, en, False))
    CONFIGS.append((f"tc_tf32_m128_en{en}", "tc_tf32", "tf32", M128N16 | 0x910, en, False))
    CONFIGS.append((f"tc_e4m3_m128_en{en}", "tc_f8", "e4m3", M128N16 | 0x10, en, True))
    CONFIGS.append((f"tc_e4m3f16d_m128_en{en}", "tc_f8", "e4m3", M128N16, en, True))
    CONFIGS.append((f"tc_e2m1_m128_en{en}", "tc_f8", "e2m1", M128N16 | 0x1690, en, True))
CONFIGS += [
    ("hm_bf16_c", "hm", "bf16", 0, 1, True),
    ("hm_bf16_rz", "hm0", "bf16", 0, 0, True),
    ("hm_f16_c", "hmh", "f16", 0, 1, False),
    ("tm_tf32_c", "tm", "tf32", 0, 1, True),
    ("tm_tf32_rz", "tm0", "tf32", 0, 0, True),
]


def pack_row(fmt, codes):
    enc, K, eb = FMT[fmt]
    if eb == 2:
        return struct.pack(f"<{K}H", *codes)
    if eb == 4:
        return struct.pack(f"<{K}I", *codes)
    if eb == 1:
        return bytes(codes)
    # fp4: two 16-element chunks of 8 bytes (low nibble first) + 8 unused bytes each
    out = b""
    for ch in range(2):
        cs = codes[16 * ch : 16 * ch + 16]
        out += bytes(cs[2 * i] | (cs[2 * i + 1] << 4) for i in range(8)) + bytes(8)
    return out


def encode(fmt, prods, pad):
    enc, K, _ = FMT[fmt]
    if len(prods) > K:
        prods = prods[:K]
    pv = []
    for k in range(K):
        if k < len(prods):
            pv.append(prods[k])
        elif pad == "p":
            pv.append((0.0, 0.0))
        elif pad == "n":
            pv.append((0.0, -0.0))
        else:
            pv.append((0.0, 1.0 if k % 2 else -1.0))
    a = [enc(x) for x, _ in pv]
    b = [enc(y) for _, y in pv]
    if None in a or None in b:
        return None
    return a, b, pv[: len(prods)]


def gen():
    os.makedirs(EXP, exist_ok=True)
    allc = cases()
    meta = []
    for name, entry, fmt, idesc, en, modeled in CONFIGS:
        enc, K, eb = FMT[fmt]
        tc = entry.startswith("tc")
        rows = []
        for cn, fam, prods, pad, c in allc:
            r = encode(fmt, prods, pad)
            if r is None or (not en and cn.endswith("c-0") and not tc):
                continue  # C = constant zero: the c-0 copy is the same case
            if fam == "acc" and not en:
                continue
            if not f32_exact(c) and c != 0:
                continue
            if "f16d" in name and enc_f16(c) is None:
                continue
            rows.append((cn, fam, r, c))
        per = 16 if tc else 8
        tiles = (len(rows) + per - 1) // per
        zero = encode(fmt, [], "p")
        if tc:
            A = bytearray(tiles * 128 * 32)
            B = bytearray(tiles * 16 * 32)
            C = bytearray(tiles * 128 * 16 * 4)
            arow = lambda t, s: (t * 128 + s) * 32  # noqa: E731
            brow = lambda t, s: (t * 16 + s) * 32  # noqa: E731
            dword = lambda t, s: (t * 128 + s) * 16 + s  # noqa: E731
            for t in range(tiles):
                for s in range(128):
                    A[arow(t, s) : arow(t, s) + 32] = pack_row(fmt, zero[0])
                for s in range(16):
                    B[brow(t, s) : brow(t, s) + 32] = pack_row(fmt, zero[1])
        else:
            rb = K * 4 if fmt == "tf32" else K * 2
            A = bytearray(tiles * 16 * rb)
            B = bytearray(tiles * 8 * rb)
            C = bytearray(tiles * 128 * 4)
            arow = lambda t, s: (t * 16 + s) * rb  # noqa: E731
            brow = lambda t, s: (t * 8 + s) * rb  # noqa: E731
            dword = lambda t, s: t * 128 + s * 8 + s  # noqa: E731
            for t in range(tiles):
                for s in range(16):
                    A[arow(t, s) : arow(t, s) + rb] = pack_row(fmt, zero[0])
                for s in range(8):
                    B[brow(t, s) : brow(t, s) + rb] = pack_row(fmt, zero[1])
        cmeta = []
        for i, (cn, fam, (a, b, pv), c) in enumerate(rows):
            t, s = divmod(i, per)
            ra, rb_ = pack_row(fmt, a), pack_row(fmt, b)
            A[arow(t, s) : arow(t, s) + len(ra)] = ra
            B[brow(t, s) : brow(t, s) + len(rb_)] = rb_
            w = dword(t, s)
            # an f16 D: one TMEM column per element, in its low half
            cw = enc_f16(c) if "f16d" in name else f32_bits(c)
            C[4 * w : 4 * w + 4] = struct.pack("<I", cw)
            cmeta.append({"name": cn, "family": fam, "word": w,
                          "exact": exact_class(pv, c, en), "c": f32_bits(c)})
        for nm, buf in [("A", A), ("B", B), ("C", C)]:
            with open(os.path.join(EXP, f"{name}.{nm}.bin"), "wb") as f:
                f.write(buf)
        meta.append({"name": name, "entry": entry, "fmt": fmt, "idesc": idesc, "en": en,
                     "modeled": modeled, "grid": tiles, "block": 128 if tc else 32,
                     "d_bytes": len(C), "cases": cmeta})
        print(f"{name}: {len(rows)} cases, {tiles} tiles")
    with open(os.path.join(EXP, "configs.json"), "w") as f:
        json.dump(meta, f, indent=0)


CASE_TOML = """name = "mma_zero_{name}"
description = "probes/mma_zero config {name} (run.py)"
ptx = "kernel.ptx"
entry = "{entry}"
grid = [{grid}, 1, 1]
block = [{block}, 1, 1]
outputs = ["D"]
{bufs}
[[buffers]]
name = "D"
dtype = "u32"
shape = [{dwords}]
init = "nan"
{params}
[[variants]]
id = "good"
expect = ["clean"]
"""


def case_toml(cfg, files_dir):
    bufs = ""
    params = ""
    for nm in ["A", "B", "C"]:
        size = os.path.getsize(os.path.join(files_dir, f"{cfg['name']}.{nm}.bin"))
        bufs += (f'\n[[buffers]]\nname = "{nm}"\ndtype = "u8"\nshape = [{size}]\ninit = "file"\n'
                 f'file = "{os.path.join(files_dir, cfg["name"] + "." + nm + ".bin")}"\n')
    for nm in ["A", "B", "C", "D"]:
        params += f'\n[[params]]\nkind = "ptr"\nbuffer = "{nm}"\n'
    if cfg["entry"].startswith("tc"):
        for v in [cfg["idesc"], cfg["en"]]:
            params += f'\n[[params]]\nkind = "u32"\nvalue = {v}\n'
    return CASE_TOML.format(name=cfg["name"], entry=cfg["entry"], grid=cfg["grid"],
                            block=cfg["block"], bufs=bufs, dwords=cfg["d_bytes"] // 4,
                            params=params)


def run(names):
    meta = json.load(open(os.path.join(EXP, "configs.json")))
    for cfg in meta:
        if names and cfg["name"] not in names:
            continue
        d = tempfile.mkdtemp(prefix="mma_zero_")
        shutil.copy(os.path.join(HERE, "kernel.ptx"), d)
        with open(os.path.join(d, "case.toml"), "w") as f:
            f.write(case_toml(cfg, EXP))
        out = os.path.join(d, "out")
        r = subprocess.run([PTXV, "probe", d, "--out", out], capture_output=True, text=True)
        dump = os.path.join(out, "good.probe.bin")
        if r.returncode or not os.path.exists(dump):
            print(cfg["name"], "FAILED", r.stdout[-2000:], r.stderr[-2000:])
            continue
        shutil.copy(dump, os.path.join(EXP, f"{cfg['name']}.D.bin"))
        print(cfg["name"], "ok")
        shutil.rmtree(d)


def sign_class(u):
    if u == 0:
        return "+0"
    if u == 0x80000000:
        return "-0"
    if (u & 0x7F800000) == 0:
        return "sub" + ("-" if u >> 31 else "+")
    return "norm" + ("-" if u >> 31 else "+")


def report():
    meta = json.load(open(os.path.join(EXP, "configs.json")))
    rows = {}
    for cfg in meta:
        p = os.path.join(EXP, f"{cfg['name']}.D.bin")
        if not os.path.exists(p):
            continue
        D = open(p, "rb").read()
        for c in cfg["cases"]:
            u = struct.unpack("<I", D[4 * c["word"] : 4 * c["word"] + 4])[0]
            if "f16d" in cfg["name"]:  # f16 in the low half: widen for the classes
                u = f32_bits(struct.unpack("<e", struct.pack("<H", u & 0xFFFF))[0])
            key = (cfg["name"], c["family"], c["exact"], c["name"].split("/")[1],
                   c["name"].split("/")[2] if cfg["en"] else "Coff")
            rows.setdefault(key, {}).setdefault(sign_class(u), []).append(c["name"])
    for k in sorted(rows):
        print(" | ".join(k), "->", {s: len(v) for s, v in rows[k].items()})


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "report"
    {"gen": gen, "run": lambda: run(sys.argv[2:]), "report": report}[cmd]()
