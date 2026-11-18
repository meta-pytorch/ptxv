# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Probe (N-ATOMIC, MISS 142): does an f32 atomic add flush subnormals, per state space?

Each thread t adds x[t] to a word holding base[t] with one f32 atomic add and copies the word to
out[t]; the atom forms also store the returned old value to old[t] (else old keeps its 0x5a5a5a5a
fill). 32 (base, x) pairs: 16 flush-sensitive or special ones (subnormal operands and sums, -0,
NaN payloads, Inf - Inf, a signalling NaN) and 16 normal ones. Kernels (PTX, sm_103a):
  shm_red        red.shared.add.f32             grid 1 x 32, the CTA's own shared word
  shm_atom       atom.shared.add.f32            (ptxas 13.0: LDS / FADD / ATOMS.CAST.SPIN)
  gen_red_shm    red.add.f32 on cvta.shared     (generic ATOM.E.ADD.F32.FTZ.RN, then the QSPC
  clu_red_shm    red.shared::cluster.add.f32     fallbacks, every one a non-flushing FADD)
  glb_red        red.global.add.f32             (REDG.E.ADD.F32.FTZ.RN)
  glb_atom       atom.global.add.f32            (ATOMG.E.ADD.F32.FTZ.RN)
  gen_red_glb    red.add.f32 on cvta.global     (ATOM.E.ADD.F32.FTZ.RN, performed: P true)
  clu_red_peer   red.shared::cluster.add.f32    grid 2 x 32 in clusters of 2: each CTA adds
  clu_atom_peer  atom.shared::cluster.add.f32   into the other CTA's word t (mapa), between two
  gen_red_peer   red.add.f32 on mapa.u64 of     cluster barriers; each CTA copies its own words
                 cvta.shared                    (pairs repeated per CTA)
GB300 (ptxas 13.0, `ptxv hw-worker`): every shared form keeps subnormals, every global form
flushes operands and results (PTX ISA, atom / red: ".add.f32 ... flushes subnormal inputs and
results to sign-preserving zero on global memory but supports subnormal inputs and results and
doesn't flush them to zero on shared memory"); NaN results are 0x7fffffff. The seven one-CTA
probes' captures come from an adversarial review round (5 plain runs identical; 20 more plain
runs identical to them), the peer probes' from run 1 (20 plain and 20 scrubbed runs identical
to it). `--check` compares expected/ with the IEEE round-to-nearest model (flush in global
memory only).
Run from this directory: python3 gen.py [--check]."""
import os
import struct
import sys
from fractions import Fraction

SENT = 0x5A5A5A5A
# (base, x): the special pairs, then 16 normal ones (random normals in the review's generator).
PAIRS = [
    (0x00000000, 0x00000001), (0x00000000, 0x80000001), (0x00800000, 0x80000001),
    (0x00000001, 0x00000001), (0x00400000, 0x00400000), (0x3F800000, 0x00000001),
    (0x00800000, 0x80000000), (0x80000000, 0x80000000), (0x00000000, 0x80000000),
    (0x7FC12345, 0x3F800000), (0x3F800000, 0x7FC12345), (0x7F800000, 0xFF800000),
    (0x00000003, 0x80000001), (0x00FFFFFF, 0x80FFFFFE), (0x80800000, 0x00000001),
    (0x7F800001, 0x00000000),
    (0xBF6E4872, 0x3F7534E4), (0x3F3BAFB9, 0x3F5F84BE), (0xBF6EF8C4, 0xBF56FE71),
    (0xBEAA87C3, 0xBE45C8D0), (0xBF1165E4, 0xBF343469), (0x3EE75D9C, 0x3F8DDBD9),
    (0x3EBB455C, 0x3F52A31A), (0x3ECFAECB, 0x401D7CE9), (0xBF10B84D, 0x3A418A35),
    (0x3EB566F3, 0x3D8F87A3), (0xBF2FD9E9, 0x3F237376), (0xBF9A7822, 0x3F9A7865),
    (0x3BDA16B1, 0xBF4D1380), (0xBFD9467A, 0xBF936AF0), (0xBF0C0AE3, 0x3F9E16A6),
    (0xBF739128, 0x3D906E9A),
]
# name: (instruction, space, CTAs)
PROBES = {
    "shm_red": ("red.shared.add.f32 [%r4], %f2;", "shared", 1),
    "shm_atom": ("atom.shared.add.f32 %f4, [%r4], %f2;", "shared", 1),
    "gen_red_shm": ("cvta.shared.u64 %rd9, %rd8; red.add.f32 [%rd9], %f2;", "shared", 1),
    "clu_red_shm": ("red.shared::cluster.add.f32 [%r4], %f2;", "shared", 1),
    "glb_red": ("red.global.add.f32 [%rd7], %f2;", "global", 1),
    "glb_atom": ("atom.global.add.f32 %f4, [%rd7], %f2;", "global", 1),
    "gen_red_glb": ("cvta.global.u64 %rd9, %rd7; red.add.f32 [%rd9], %f2;", "global", 1),
    "clu_red_peer": ("red.shared::cluster.add.f32 [%r6], %f2;", "shared", 2),
    "clu_atom_peer": ("atom.shared::cluster.add.f32 %f4, [%r6], %f2;", "shared", 2),
    "gen_red_peer": ("red.add.f32 [%rd12], %f2;", "shared", 2),
}
PTX_HEADER = """// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

"""


def kernel(name):
    ins, space, ctas = PROBES[name]
    if ctas == 1:
        idx = "    mov.u32 %r1, %tid.x;\n"
        peer = ""
        if space == "shared":
            pre = "    st.shared.f32 [%r4], %f1;\n    bar.sync 0;\n"
            post = "    bar.sync 0;\n    ld.shared.f32 %f3, [%r4];\n    st.global.f32 [%rd7], %f3;\n"
        else:
            pre = "    st.global.f32 [%rd7], %f1;\n"
            post = ""
    else:
        # element 32 rank + tid; the other CTA's copy of word tid: r6 (shared::cluster), rd12
        # (generic, mapa.u64 of the cvta.shared address)
        idx = (
            "    mov.u32 %r10, %tid.x;\n"
            "    mov.u32 %r11, %cluster_ctarank;\n"
            "    mad.lo.u32 %r1, %r11, 32, %r10;\n"
        )
        peer = (
            "    xor.b32 %r5, %r11, 1;\n"
            "    shl.b32 %r12, %r10, 2;\n"
            "    add.u32 %r13, %r2, %r12;\n"
            "    mapa.shared::cluster.u32 %r6, %r13, %r5;\n"
            "    cvt.u64.u32 %rd13, %r13;\n"
            "    cvta.shared.u64 %rd14, %rd13;\n"
            "    mapa.u64 %rd12, %rd14, %r5;\n"
        )
        cbar = "    barrier.cluster.arrive.release.aligned;\n    barrier.cluster.wait.acquire.aligned;\n"
        pre = "    st.shared.f32 [%r4], %f1;\n" + cbar
        post = cbar + "    ld.shared.f32 %f3, [%r4];\n    st.global.f32 [%rd7], %f3;\n"
    if "%f4" in ins:
        post += "    st.global.f32 [%rd10], %f4;\n"
    body = "\n".join("    " + i.strip() + ";" for i in ins.split(";") if i.strip())
    return f"""{PTX_HEADER}.version 8.8
.target sm_103a
.address_size 64

.visible .entry {name}(.param .u64 pbase, .param .u64 px, .param .u64 pout, .param .u64 pold)
{{
    .reg .b32 %r<16>;
    .reg .f32 %f<8>;
    .reg .b64 %rd<16>;
    .shared .align 4 .b32 s[32];
{idx}    ld.param.u64 %rd1, [pbase];
    cvta.to.global.u64 %rd1, %rd1;
    ld.param.u64 %rd2, [px];
    cvta.to.global.u64 %rd2, %rd2;
    ld.param.u64 %rd3, [pout];
    cvta.to.global.u64 %rd3, %rd3;
    ld.param.u64 %rd11, [pold];
    cvta.to.global.u64 %rd11, %rd11;
    mul.wide.u32 %rd4, %r1, 4;
    add.u64 %rd5, %rd1, %rd4;
    ld.global.f32 %f1, [%rd5];
    add.u64 %rd6, %rd2, %rd4;
    ld.global.f32 %f2, [%rd6];
    add.u64 %rd7, %rd3, %rd4;
    add.u64 %rd10, %rd11, %rd4;
    mov.u32 %r2, s;
    shl.b32 %r3, {"%r10" if ctas == 2 else "%r1"}, 2;
    add.u32 %r4, %r2, %r3;
    cvt.u64.u32 %rd8, %r4;
{peer}{pre}{body}
{post}    ret;
}}
"""


def case_toml(name):
    ins, space, ctas = PROBES[name]
    n = 32 * ctas
    what = ins.split(";")[-2].strip() if ins.count(";") > 1 else ins.rstrip(";")
    launch = f"grid = [{ctas}, 1, 1]\nblock = [32, 1, 1]\n" + (
        "cluster = [2, 1, 1]\n" if ctas == 2 else ""
    )
    peer = "; each CTA adds into the other CTA's word" if ctas == 2 else ""
    bufs = ""
    for b, init in (("base", "base.bin"), ("x", "x.bin"), ("out", "out0.bin"), ("old", "old0.bin")):
        bufs += f'''
[[buffers]]
name = "{b}"
dtype = "u32"
shape = [{n}]
init = "file"
file = "{init}"
'''
        if b in ("out", "old"):
            bufs += f'expect = "expected/{b}.bin"\n'
    params = "".join(f'\n[[params]]\nkind = "ptr"\nbuffer = "{b}"\n' for b in ("base", "x", "out", "old"))
    return f'''name = "probe_atom_f32_space_{name}"
description = """
Probe (N-ATOMIC, MISS 142; probes/atom_f32_space/gen.py): {what} ({space} memory{peer}),
base[i] + x[i] into word i. expected/ is GB300's output (ptxas 13.0 -arch=sm_103a,
`ptxv hw-worker`)."""
ptx = "kernel.ptx"
entry = "{name}"
{launch}outputs = ["out", "old"]
{bufs}{params}
[[variants]]
id = "good"
expect = ["clean"]
'''


def val(u):
    e, m, sg = (u >> 23) & 0xFF, u & 0x7FFFFF, -1 if u >> 31 else 1
    if e == 0:
        return sg * Fraction(m) * Fraction(2) ** -149
    return sg * Fraction(m + 2**23) * Fraction(2) ** (e - 150)


def rn(x):
    """A nonzero finite Fraction rounded to the nearest f32 (ties to even), subnormals kept."""
    sg = 0x80000000 if x < 0 else 0
    x = abs(x)
    e = x.numerator.bit_length() - x.denominator.bit_length()
    while Fraction(2) ** e > x:
        e -= 1
    while Fraction(2) ** (e + 1) <= x:
        e += 1
    e = max(e, -126)
    q = x / Fraction(2) ** (e - 23)
    n = q.numerator // q.denominator
    rest = q - n
    if rest > Fraction(1, 2) or (rest == Fraction(1, 2) and n % 2 == 1):
        n += 1
    if n == 2**24:
        n //= 2
        e += 1
    if e > 127:
        return sg | 0x7F800000
    if n < 2**23:
        return sg | n
    return sg | ((e + 127) << 23) | (n - 2**23)


def flush(u):
    return u & 0x80000000 if (u & 0x7F800000) == 0 else u


def add(u, v, ftz):
    """u + v rounded to nearest; ftz: operands and result flushed to zero of their sign; any
    NaN result is 0x7fffffff."""
    if ftz:
        u, v = flush(u), flush(v)
    nan = lambda w: (w & 0x7F800000) == 0x7F800000 and (w & 0x7FFFFF) != 0
    inf = lambda w: (w & 0x7FFFFFFF) == 0x7F800000
    if nan(u) or nan(v) or (inf(u) and inf(v) and u != v):
        return 0x7FFFFFFF
    if inf(u) or inf(v):
        return u if inf(u) else v
    s = val(u) + val(v)
    if s == 0:
        r = 0x80000000 if (u == 0x80000000 and v == 0x80000000) else 0
    else:
        r = rn(s)
    return flush(r) if ftz else r


def model(name):
    ins, space, ctas = PROBES[name]
    out = [add(b, x, space == "global") for b, x in PAIRS] * ctas
    old = [b for b, _ in PAIRS] * ctas if "%f4" in ins else [SENT] * (32 * ctas)
    return out, old


def u32s(ws):
    return struct.pack(f"<{len(ws)}I", *ws)


def main():
    if "--check" in sys.argv:
        bad = 0
        for name in PROBES:
            n = 32 * PROBES[name][2]
            got = [
                list(struct.unpack(f"<{n}I", open(os.path.join(name, "expected", f"{b}.bin"), "rb").read()))
                for b in ("out", "old")
            ]
            want = model(name)
            ok = got[0] == want[0] and got[1] == want[1]
            bad += not ok
            keep = sum(1 for i in range(16) if got[0][i] != add(*PAIRS[i], True))
            print(f"{name}: capture {'=' if ok else 'DIFFERS from'} the model; "
                  f"{keep} of the 16 special pairs differ from a flushing add")
        sys.exit(1 if bad else 0)
    for name in PROBES:
        ctas = PROBES[name][2]
        os.makedirs(os.path.join(name, "expected"), exist_ok=True)
        with open(os.path.join(name, "kernel.ptx"), "w") as f:
            f.write(kernel(name))
        with open(os.path.join(name, "case.toml"), "w") as f:
            f.write(case_toml(name))
        for b, ws in (
            ("base", [p[0] for p in PAIRS] * ctas),
            ("x", [p[1] for p in PAIRS] * ctas),
            ("out0", [SENT] * (32 * ctas)),
            ("old0", [SENT] * (32 * ctas)),
        ):
            with open(os.path.join(name, f"{b}.bin"), "wb") as f:
                f.write(u32s(ws))


if __name__ == "__main__":
    main()
