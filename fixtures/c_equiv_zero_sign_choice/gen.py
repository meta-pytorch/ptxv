# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MISS 135 fixture: a choice that can pick a NaN payload, plus +0, gets no L0~0 claim.

x + (+0) turns a NaN x into the canonical NaN 0x7fffffff (FADD, N-FADD), so L0~0 peels x only
where it holds no other NaN (`Side::nan_canonical`). A choice passes the operand it picks, a NaN
payload included. Kernels (PTX, sm_103a, grid 4 x 256, i = 256 ctaid + tid; f32 inputs a, b of
1024 elements; k = 0x7fc12345, a NaN payload constant):
  gsel_ref     out[i] = a[i] + b[i]; bar.sync; if a[i] > 0: out[i] = b[i]; bar.sync
  gsel_cand    the same, then out[i] = out[i] + (+0) (add.rn.f32, ptxas: FADD with RZ)
  gselk_ref    out[i] = a[i] + b[i]; bar.sync; if a[i] > 0: out[i] = k; bar.sync
  gselk_cand   the same, then out[i] = out[i] + (+0)
ptxas 13.0 keeps the guarded store (@P0 STG). Segment mode makes it the choice node
Sel(P, v, a + b) (`num_gst_post`): v is b's slice in gsel and a constant leaf in gselk. k passes
through a volatile shared word (STS before the first barrier, LDS after it), so ptxas loads it
unconditionally; an immediate becomes a guarded MOV, whose register segment mode cannot slice.
Neither pair reaches the peel. Choices that can pass a payload (a guarded store, a max by
compare and select) exist only in segment mode, and there x + (+0) has no node of its own (one
tracked operand, `num_anchor`): the candidate's output is a slice of the choice, so gsel is
level none (L1 fails). In gselk the NaN constant also gives the polynomial a non-finite
coefficient, which refuses every level (none, unproven). Outside segment mode the only choices
are maxima: a hardware FMNMX passes no payload (peeled on the SASS front), while ptxas folds a
PTX max.f32 of one value into a copy, which does (fixtures/c_equiv_zero_sign_fmnmx, where
srcmut's l0z-nan-choice is caught). l0z-nan-leaf changes no level. Neither mutant changes these
pairs. The pairs pin that outputs which differ on GB300 get no claim.
Input set: -0 at a[8 j] (j = 0 .. 15) with b[8 j] = -0 for odd j and +0 for even j; the NaN
encodings below at b[8 j + 3], with a[8 j + 3] = 1 + j (j = 0 .. 7); -0 at b[8 j + 5] with
a[8 j + 5] = 2 + j (j = 0 .. 7); normal values elsewhere (|x| in [2^-20, 2^20]).
GB300 (ptxas 13.0 -arch=sm_103a): gsel's outputs differ at the 7 non-canonical NaN payloads, at
the 8 elements where a = b = -0 and at the 8 where b = -0 is picked (-0 against +0); gselk's
wherever a > 0 (k against 0x7fffffff) and where a = b = -0. expected/out.bin is GB300's output,
equal to the model below (`--check`), with 20 scrubbed and 20 plain runs identical to it.
Run from this directory: python3 gen.py [--check]."""
import os
import random
import struct
import sys

N = 1024
K = 0x7FC12345
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
NANS = [
    0x7FC12345, 0xFFC00001, 0x7FA00001, 0x7F800001,
    0xFFFFFFFF, 0x7FC00000, 0x7FFFFFFF, 0xFF800001,
]


def kernel(konst, tail):
    s = [
        HDR + ".visible .entry k(.param .u64 pa, .param .u64 pb, .param .u64 pout)",
        "{",
        "    .reg .pred %p<4>;",
        "    .reg .b32 %r<16>;",
        "    .reg .f32 %f<16>;",
        "    .reg .b64 %rd<16>;",
    ]
    if konst:
        s.append("    .shared .align 4 .b32 sk[256];")
    s += [
        "    ld.param.u64 %rd1, [pa];",
        "    ld.param.u64 %rd2, [pb];",
        "    ld.param.u64 %rd3, [pout];",
        "    cvta.to.global.u64 %rd1, %rd1;",
        "    cvta.to.global.u64 %rd2, %rd2;",
        "    cvta.to.global.u64 %rd3, %rd3;",
        "    mov.u32 %r1, %ctaid.x;",
        "    mov.u32 %r2, %tid.x;",
        "    mad.lo.u32 %r3, %r1, 256, %r2;",
        "    mul.wide.u32 %rd4, %r3, 4;",
        "    add.s64 %rd5, %rd1, %rd4;",
        "    add.s64 %rd6, %rd2, %rd4;",
        "    add.s64 %rd7, %rd3, %rd4;",
        "    ld.global.f32 %f1, [%rd5];",
        "    ld.global.f32 %f2, [%rd6];",
    ]
    if konst:
        s += [
            "    mov.u32 %r4, sk;",
            "    shl.b32 %r5, %r2, 2;",
            "    add.u32 %r6, %r4, %r5;",
            f"    mov.b32 %r7, {K:#010x};",
            "    st.volatile.shared.u32 [%r6], %r7;",
        ]
    s += [
        "    add.rn.f32 %f4, %f1, %f2;",
        "    st.global.f32 [%rd7], %f4;",
        "    bar.sync 0;",
    ]
    if konst:
        s.append("    ld.volatile.shared.u32 %r8, [%r6];")
    else:
        s.append("    mov.b32 %r8, %f2;")
    s += [
        "    setp.gt.f32 %p1, %f1, 0f00000000;",
        "    @%p1 st.global.b32 [%rd7], %r8;",
        "    bar.sync 0;",
    ]
    return "\n".join(s + tail + ["    ret;", "}", ""])


PLUS0 = [
    "    ld.global.f32 %f8, [%rd7];",
    "    add.rn.f32 %f9, %f8, 0f00000000;",
    "    st.global.f32 [%rd7], %f9;",
]
KERNELS = {
    "gsel_ref": (kernel(False, []), "a + b, then b where a > 0 (a guarded store)"),
    "gsel_cand": (kernel(False, PLUS0), "(a + b, then b where a > 0) + (+0)"),
    "gselk_ref": (kernel(True, []), "a + b, then k where a > 0 (a guarded store)"),
    "gselk_cand": (kernel(True, PLUS0), "(a + b, then k where a > 0) + (+0)"),
}
# pair: (reference, candidate, expected level, why)
PAIRS = {
    "gsel": (
        "gsel_ref",
        "gsel_cand",
        "none",
        "the choice can be b, whose NaN\n"
        "# payloads x + (+0) turns into 0x7fffffff. Segment mode makes x + (+0) part of a slice\n"
        "# of the choice (one tracked operand), so nothing is peeled and L1 fails: level none.",
    ),
    "gselk": (
        "gselk_ref",
        "gselk_cand",
        "none (unproven)",
        "the choice can be k, a NaN payload,\n"
        "# whose non-finite polynomial coefficient refuses every level: none (unproven).",
    ),
}


def bits(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def inputs():
    r = random.Random(9806)

    def normal():
        return bits(r.choice([-1, 1]) * 2.0 ** r.uniform(-20, 20))

    a = [normal() for _ in range(N)]
    b = [normal() for _ in range(N)]
    for j in range(16):
        a[8 * j] = 0x80000000
        b[8 * j] = 0x80000000 if j % 2 else 0x00000000
    for j, w in enumerate(NANS):
        a[8 * j + 3] = bits(1.0 + j)
        b[8 * j + 3] = w
        a[8 * j + 5] = bits(2.0 + j)
        b[8 * j + 5] = 0x80000000
    return a, b


def isnan(u):
    return (u & 0x7F800000) == 0x7F800000 and (u & 0x7FFFFF) != 0


def val(u):
    return struct.unpack("<f", struct.pack("<I", u))[0]


def add(u, v):
    """u + v, round to nearest (f32 inputs: exact in double, then one f32 rounding); a NaN
    result is 0x7fffffff."""
    if isnan(u) or isnan(v):
        return 0x7FFFFFFF
    s = val(u) + val(v)
    if s == 0 and u == 0x80000000 and v == 0x80000000:
        return 0x80000000
    return bits(struct.unpack("<f", struct.pack("<f", s))[0]) if s != 0 else 0


def gt(u, v):
    return not isnan(u) and not isnan(v) and val(u) > val(v)


def model(kernel, a, b):
    out = []
    for x, y in zip(a, b):
        m = (K if kernel.startswith("gselk") else y) if gt(x, 0) else add(x, y)
        out.append(add(m, 0) if kernel.endswith("cand") else m)
    return out


def case_toml(name, what):
    bufs = "".join(
        f'''
[[buffers]]
name = "{x}"
dtype = "f32"
shape = [{N}]
init = "file"
file = "../data/{x}.bin"
'''
        for x in "ab"
    )
    params = "".join(f'\n[[params]]\nkind = "ptr"\nbuffer = "{x}"\n' for x in ["a", "b", "out"])
    return f'''name = "c_equiv_zero_sign_choice_{name}"
description = """
MISS 135 (fixtures/c_equiv_zero_sign_choice/gen.py):
out[i] = {what}.
expected/ is GB300's output (ptxas 13.0 -arch=sm_103a)."""
ptx = "kernel.ptx"
entry = "k"
grid = [4, 1, 1]
block = [256, 1, 1]
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


def eq_toml(tag):
    ref, cand, level, why = PAIRS[tag]
    return f'''# MISS 135 (gen.py): out[i] = {KERNELS[ref][1]}
# against {KERNELS[cand][1]}.
# The candidate is the reference plus +0, but {why}
max_states = 200000
segments = true
reference = {{ case = "{ref}", variant = "good", front = "sass", certify = true }}

[[candidates]]
case = "{cand}"
variant = "good"
front = "sass"
certify = true
expect = "{level}"
'''


def main():
    a, b = inputs()
    if "--check" in sys.argv:
        bad = 0
        for tag, (ref, cand, _, _) in PAIRS.items():
            got = {}
            for kern in (ref, cand):
                p = os.path.join(kern, "expected", "out.bin")
                got[kern] = list(struct.unpack(f"<{N}I", open(p, "rb").read()))
                same = got[kern] == model(kern, a, b)
                bad += not same
                print(f"{kern}: capture {'=' if same else 'DIFFERS from'} the model")
            d = sum(1 for x, y in zip(got[ref], got[cand]) if x != y)
            print(f"{tag}: {ref} against {cand}: {d} of {N} words differ")
        sys.exit(1 if bad else 0)
    os.makedirs("data", exist_ok=True)
    for x, v in (("a", a), ("b", b)):
        with open(os.path.join("data", f"{x}.bin"), "wb") as f:
            f.write(struct.pack(f"<{N}I", *v))
    for tag, (ref, cand, _, _) in PAIRS.items():
        for kern in (ref, cand):
            os.makedirs(os.path.join(kern, "expected"), exist_ok=True)
            with open(os.path.join(kern, "kernel.ptx"), "w") as f:
                f.write(KERNELS[kern][0])
            with open(os.path.join(kern, "case.toml"), "w") as f:
                f.write(case_toml(kern, KERNELS[kern][1]))
        with open(f"eq_{tag}.toml", "w") as f:
            f.write(eq_toml(tag))


if __name__ == "__main__":
    main()
