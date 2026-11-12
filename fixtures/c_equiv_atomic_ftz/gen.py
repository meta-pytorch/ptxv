# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MISS 134 fixture: an fp32 atomic add's flush flag is part of equiv's canonical form.

red/atom.add.f32 (SASS REDG.E.ADD.F32.FTZ.RN) flushes subnormal operands and results to zero
of their sign; the bulk f32 reduction cp.reduce.async.bulk .add.f32 (UBLKRED.G.S.ADD.F32.RN)
keeps them. Kernels (PTX, sm_103a, grid 4 x 256 or 4 x 64; f32 inputs a, b of 1024 elements;
out zero-initialised, so every chain starts at +0):
  ftz1_ref    out[i] = +0 + a[i] (red.add.f32, FTZ)
  blk1_cand   out[i] = +0 + a[i] (a[i] staged in shared memory; thread 0 bulk-adds its CTA's
              256 words, no flush)
  mixA_ref    one thread per 4 elements: red a (FTZ) x4, fence.proxy.async.global, then its
              own 16-byte bulk add of b (no flush)
  mixA_cand   the same thread: the bulk add of b first, then the fence and red a (FTZ) x4
  ftz2_ab / ftz2_ba   red a then red b / red b then red a (one flush mode)
  blk2_ab / blk2_ba   bulk a then bulk b / bulk b then bulk a (one flush mode)
An FTZ add of a subnormal value into +0 gives 0 where the bulk add gives the value, and two adds
of both modes in opposite orders differ where a, b or the sum is subnormal: those pairs (i,
mixA) are not L0; their bound, which covers every order, holds. Adds of one mode commute in
pairs into +0: the controls (cftz, cblk) are L0. Before MISS 134 the canonical form dropped the
flag, and i and mixA were proven L0.
Input sets: n (every a, b normal, |a|, |b| in [2^-20, 2^20], every sum normal) and s (n with 128
special elements at 0 .. 127: subnormals, a cancellation to a subnormal sum, -0). On set s the
GB300 outputs of i differ in 64 words, of mixA in 79; on set n they are equal. Each case dir is
<kernel>_<set>; expected/out.bin is GB300's output (ptxas 13.0 -arch=sm_103a), equal to the
IEEE round-to-nearest model with the PTX flush rules (`--check`), 20 scrubbed and 20 plain runs
identical to it.
Run from this directory: python3 gen.py [--check]."""
import os
import random
import struct
import sys
from fractions import Fraction

N = 1024
# The copyright header of every exported .ptx (fb/oss_check.sh).
PTX_HEADER = """// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

"""
HDR = PTX_HEADER + """.version 8.8
.target sm_103a
.address_size 64
"""


def head(shared=0):
    s = [
        HDR + ".visible .entry k(.param .u64 pa, .param .u64 pb, .param .u64 pout)",
        "{",
        "    .reg .pred %p<4>;",
        "    .reg .b32 %r<16>;",
        "    .reg .f32 %f<16>;",
        "    .reg .b64 %rd<16>;",
    ]
    if shared:
        s.append(f"    .shared .align 128 .b8 sbuf[{shared}];")
    s += [
        "    ld.param.u64 %rd1, [pa];",
        "    ld.param.u64 %rd2, [pb];",
        "    ld.param.u64 %rd3, [pout];",
        "    cvta.to.global.u64 %rd1, %rd1;",
        "    cvta.to.global.u64 %rd2, %rd2;",
        "    cvta.to.global.u64 %rd3, %rd3;",
        "    mov.u32 %r1, %ctaid.x;",
        "    mov.u32 %r2, %tid.x;",
    ]
    return s


def tail(s):
    return "\n".join(s + ["    ret;", "}", ""])


def idx1():
    # i = 256 ctaid + tid; rd5 = &a[i], rd6 = &b[i], rd7 = &out[i]
    return [
        "    mad.lo.u32 %r3, %r1, 256, %r2;",
        "    mul.wide.u32 %rd4, %r3, 4;",
        "    add.s64 %rd5, %rd1, %rd4;",
        "    add.s64 %rd6, %rd2, %rd4;",
        "    add.s64 %rd7, %rd3, %rd4;",
    ]


def idx4():
    # e0 = 256 ctaid + 4 tid; rd5/rd6/rd7 = &a/&b/&out[e0]; r6 = sbuf + 16 tid; r8 = r6 + 1024
    return [
        "    shl.b32 %r7, %r2, 2;",
        "    mad.lo.u32 %r3, %r1, 256, %r7;",
        "    mul.wide.u32 %rd4, %r3, 4;",
        "    add.s64 %rd5, %rd1, %rd4;",
        "    add.s64 %rd6, %rd2, %rd4;",
        "    add.s64 %rd7, %rd3, %rd4;",
        "    mov.u32 %r4, sbuf;",
        "    shl.b32 %r5, %r2, 4;",
        "    add.u32 %r6, %r4, %r5;",
        "    add.u32 %r8, %r6, 1024;",
    ]


def red4(src, f0):
    s = [f"    ld.global.f32 %f{f0 + j}, [{src}+{4 * j}];" for j in range(4)]
    s += [f"    red.global.add.f32 [%rd7+{4 * j}], %f{f0 + j};" for j in range(4)]
    return s


def stage4(src, f0, dst):
    s = [f"    ld.global.f32 %f{f0 + j}, [{src}+{4 * j}];" for j in range(4)]
    s += [f"    st.shared.f32 [{dst}+{4 * j}], %f{f0 + j};" for j in range(4)]
    return s


BULK = "cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32"


def bulk16(sm):
    return [
        f"    {BULK} [%rd7], [{sm}], 16;",
        "    cp.async.bulk.commit_group;",
        "    cp.async.bulk.wait_group 0;",
    ]


def kernels():
    """name: (PTX, block size, what)."""
    k = {}
    s = head() + idx1() + ["    ld.global.f32 %f1, [%rd5];", "    red.global.add.f32 [%rd7], %f1;"]
    k["ftz1_ref"] = (tail(s), 256, "out[i] = +0 + a[i], red.add.f32 (FTZ)")
    s = head(1024) + idx1() + [
        "    ld.global.f32 %f1, [%rd5];",
        "    mov.u32 %r4, sbuf;",
        "    shl.b32 %r5, %r2, 2;",
        "    add.u32 %r6, %r4, %r5;",
        "    st.shared.f32 [%r6], %f1;",
        "    fence.proxy.async.shared::cta;",
        "    bar.sync 0;",
        "    setp.ne.u32 %p1, %r2, 0;",
        "    @%p1 bra DONE;",
        "    mul.wide.u32 %rd8, %r1, 1024;",
        "    add.s64 %rd9, %rd3, %rd8;",
        f"    {BULK} [%rd9], [%r4], 1024;",
        "    cp.async.bulk.commit_group;",
        "    cp.async.bulk.wait_group 0;",
        "DONE:",
    ]
    k["blk1_cand"] = (tail(s), 256, "out[i] = +0 + a[i], a bulk f32 add-reduction (no flush) per CTA")
    s = head(2048) + idx4() + red4("%rd5", 1) + stage4("%rd6", 5, "%r6") + [
        "    fence.proxy.async.shared::cta;",
        "    fence.proxy.async.global;",
    ] + bulk16("%r6")
    k["mixA_ref"] = (tail(s), 64, "out[i] = (+0 + a[i] (FTZ)) + b[i] (bulk, no flush), one thread per 4 elements")
    s = (
        head(2048)
        + idx4()
        + stage4("%rd6", 5, "%r6")
        + ["    fence.proxy.async.shared::cta;"]
        + bulk16("%r6")
        + ["    fence.proxy.async.global;"]
        + red4("%rd5", 1)
    )
    k["mixA_cand"] = (tail(s), 64, "out[i] = (+0 + b[i] (bulk, no flush)) + a[i] (FTZ), one thread per 4 elements")
    k["ftz2_ab"] = (tail(head() + idx4()[:6] + red4("%rd5", 1) + red4("%rd6", 5)), 64, "out[i] = (+0 + a[i]) + b[i], both FTZ")
    k["ftz2_ba"] = (tail(head() + idx4()[:6] + red4("%rd6", 5) + red4("%rd5", 1)), 64, "out[i] = (+0 + b[i]) + a[i], both FTZ")
    s = (
        head(2048)
        + idx4()
        + stage4("%rd5", 1, "%r6")
        + stage4("%rd6", 5, "%r8")
        + ["    fence.proxy.async.shared::cta;"]
        + bulk16("%r6")
        + bulk16("%r8")
    )
    k["blk2_ab"] = (tail(s), 64, "out[i] = (+0 + a[i]) + b[i], both bulk (no flush)")
    s = (
        head(2048)
        + idx4()
        + stage4("%rd5", 1, "%r6")
        + stage4("%rd6", 5, "%r8")
        + ["    fence.proxy.async.shared::cta;"]
        + bulk16("%r8")
        + bulk16("%r6")
    )
    k["blk2_ba"] = (tail(s), 64, "out[i] = (+0 + b[i]) + a[i], both bulk (no flush)")
    return k


# pair: (reference, candidate, input sets, expected level)
PAIRS = {
    "i": ("ftz1_ref", "blk1_cand", "ns", "L2"),
    "mixA": ("mixA_ref", "mixA_cand", "ns", "L2"),
    "cftz": ("ftz2_ab", "ftz2_ba", "s", "L0"),
    "cblk": ("blk2_ab", "blk2_ba", "s", "L0"),
}


def bits(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def inputs():
    """The input sets n and s: (a, b) as f32 encodings."""
    r = random.Random(9804)

    def normal():
        return bits(r.choice([-1, 1]) * 2.0 ** r.uniform(-20, 20))

    def subnormal():
        return r.choice([0, 0x80000000]) | r.randint(1, 0x7FFFFF)

    an = [normal() for _ in range(N)]
    bn = [normal() for _ in range(N)]
    a, b = list(an), list(bn)
    for k in range(16):
        a[k], b[k] = subnormal(), subnormal()  # both subnormal
        a[16 + k] = subnormal()  # a subnormal, b normal
        a[32 + k] = 0x00800000 + k  # a the smallest normals, b subnormal
        b[32 + k] = r.randint(1, 0x3FF)
        a[48 + k] = bits((1 + (k + 1) / 32) * 2.0**-125)  # a + b subnormal (cancellation)
        b[48 + k] = bits(-(2.0**-125))
        a[64 + k], b[64 + k] = 0x80000000, subnormal()  # a = -0, b subnormal
        a[80 + k], b[80 + k] = subnormal(), 0  # a subnormal, b = +0
        a[112 + k] = subnormal()  # b = -a, both subnormal
        b[112 + k] = a[112 + k] ^ 0x80000000
    return {"n": (an, bn), "s": (a, b)}


def val(u):
    e, m, sg = (u >> 23) & 0xFF, u & 0x7FFFFF, -1 if u >> 31 else 1
    assert e != 0xFF
    if e == 0:
        return sg * Fraction(m) * Fraction(2) ** -149
    return sg * Fraction(m + 2**23) * Fraction(2) ** (e - 150)


def rn(x):
    """A nonzero Fraction rounded to the nearest f32 (ties to even), subnormals kept."""
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
    return u & 0x80000000 if (u & 0x7F800000) == 0 and (u & 0x7FFFFF) != 0 else u


def add(u, v, ftz):
    """u + v rounded to nearest; ftz: operands and result flushed to zero of their sign."""
    if ftz:
        u, v = flush(u), flush(v)
    s = val(u) + val(v)
    if s == 0:
        r = 0x80000000 if (u == 0x80000000 and v == 0x80000000) else 0
    else:
        r = rn(s)
    return flush(r) if ftz else r


# kernel: the adds of out[i] in program order, (operand, FTZ)
ADDS = {
    "ftz1_ref": [("a", True)],
    "blk1_cand": [("a", False)],
    "mixA_ref": [("a", True), ("b", False)],
    "mixA_cand": [("b", False), ("a", True)],
    "ftz2_ab": [("a", True), ("b", True)],
    "ftz2_ba": [("b", True), ("a", True)],
    "blk2_ab": [("a", False), ("b", False)],
    "blk2_ba": [("b", False), ("a", False)],
}


def model(kernel, a, b):
    out = []
    for i in range(N):
        r = 0
        for op, ftz in ADDS[kernel]:
            r = add(r, a[i] if op == "a" else b[i], ftz)
        out.append(r)
    return out


def case_toml(name, kernel, block, sname, what):
    bufs = "".join(
        f'''
[[buffers]]
name = "{x}"
dtype = "f32"
shape = [{N}]
init = "file"
file = "../data/{x}_{sname}.bin"
'''
        for x in "ab"
    )
    params = "".join(
        f'''
[[params]]
kind = "ptr"
buffer = "{x}"
'''
        for x in ["a", "b", "out"]
    )
    return f'''name = "c_equiv_atomic_ftz_{name}"
description = """
MISS 134 (fixtures/c_equiv_atomic_ftz/gen.py): {what}; input set {sname}. expected/ is GB300's
output (ptxas 13.0 -arch=sm_103a)."""
ptx = "kernel.ptx"
entry = "k"
grid = [4, 1, 1]
block = [{block}, 1, 1]
outputs = ["out"]
{bufs}
[[buffers]]
name = "out"
dtype = "f32"
shape = [{N}]
init = "zeros"
expect = "expected/out.bin"
{params}
[[variants]]
id = "good"
summary = "as written"
expect = ["clean"]
'''


def eq_toml(tag, sname):
    ref, cand, _, level = PAIRS[tag]
    k = kernels()
    return f'''# MISS 134 (gen.py): {k[ref][2]} against {k[cand][2]}; input set {sname}.
max_states = 200000
reference = {{ case = "{ref}_{sname}", variant = "good", front = "sass", certify = true }}

[[candidates]]
case = "{cand}_{sname}"
variant = "good"
front = "sass"
certify = true
expect = "{level}"
'''


def main():
    sets = inputs()
    if "--check" in sys.argv:
        bad = 0
        for tag, (ref, cand, snames, _) in PAIRS.items():
            for sname in snames:
                got = {}
                for kern in (ref, cand):
                    p = os.path.join(f"{kern}_{sname}", "expected", "out.bin")
                    got[kern] = list(struct.unpack(f"<{N}I", open(p, "rb").read()))
                    same = got[kern] == model(kern, *sets[sname])
                    bad += not same
                    print(f"{kern}_{sname}: capture {'=' if same else 'DIFFERS from'} the model")
                d = sum(1 for x, y in zip(got[ref], got[cand]) if x != y)
                print(f"{tag}_{sname}: {ref} against {cand}: {d} of {N} words differ")
        sys.exit(1 if bad else 0)
    os.makedirs("data", exist_ok=True)
    for sname, (a, b) in sets.items():
        for x, v in (("a", a), ("b", b)):
            with open(os.path.join("data", f"{x}_{sname}.bin"), "wb") as f:
                f.write(struct.pack(f"<{N}I", *v))
    k = kernels()
    for tag, (ref, cand, snames, _) in PAIRS.items():
        for sname in snames:
            for kern in (ref, cand):
                name = f"{kern}_{sname}"
                os.makedirs(name, exist_ok=True)
                with open(os.path.join(name, "kernel.ptx"), "w") as f:
                    f.write(k[kern][0])
                with open(os.path.join(name, "case.toml"), "w") as f:
                    f.write(case_toml(name, kern, k[kern][1], sname, k[kern][2]))
            with open(f"eq_{tag}_{sname}.toml", "w") as f:
                f.write(eq_toml(tag, sname))


if __name__ == "__main__":
    main()
