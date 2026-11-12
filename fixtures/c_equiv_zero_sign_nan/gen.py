# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MISS 135 fixture: L0~0 (equal bits except -0 -> +0) only where x + (+0) keeps x's NaNs.

x + (+0) also turns a NaN x into the canonical NaN 0x7fffffff (FADD, N-FADD). Kernels (PTX,
sm_103a, grid 4 x 256, i = 256 ctaid + tid; f32 inputs a, b of 1024 elements, b unused):
  zcopy_ref    out[i] = a[i] (a copy)
  zcopy_cand   out[i] = a[i] + (+0) (add.rn.f32, ptxas: FADD with RZ)
The candidate's output is the reference's plus +0, but the reference's is an input, which can
hold any NaN: the pair is L2 (bound 0: equal values), not L0~0 (before MISS 135 it was proven
L0~0).
Input sets (a; b is the same normal values in both): q has -0 at a[8 k] (k = 0 .. 15) and normal
values elsewhere (|a| in [2^-20, 2^20]); p is q with the NaN encodings 0x7fc12345, 0xffc00001,
0x7fa00001, 0x7f800001, 0xffffffff, 0x7fc00000, 0x7fffffff (canonical) and 0xff800001 at
a[8 k + 3] (k = 0 .. 7). GB300's outputs differ at the 16 zero signs on q, and also at the 7
non-canonical NaNs on p, which the candidate turns into 0x7fffffff. expected/out.bin is GB300's
output (ptxas 13.0 -arch=sm_103a), equal to the IEEE model with canonical NaN results
(`--check`), 20 scrubbed and 20 plain runs identical to it.
Run from this directory: python3 gen.py [--check]."""
import os
import random
import struct
import sys

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
NANS = [0x7FC12345, 0xFFC00001, 0x7FA00001, 0x7F800001, 0xFFFFFFFF, 0x7FC00000, 0x7FFFFFFF, 0xFF800001]


def kernel(body):
    s = [
        HDR + ".visible .entry k(.param .u64 pa, .param .u64 pb, .param .u64 pout)",
        "{",
        "    .reg .pred %p<4>;",
        "    .reg .b32 %r<16>;",
        "    .reg .f32 %f<16>;",
        "    .reg .b64 %rd<16>;",
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
    ]
    return "\n".join(s + body + ["    ret;", "}", ""])


# name: (PTX, what)
KERNELS = {
    "zcopy_ref": (
        kernel(["    ld.global.f32 %f1, [%rd5];", "    st.global.f32 [%rd7], %f1;"]),
        "out[i] = a[i] (a copy)",
    ),
    "zcopy_cand": (
        kernel(
            [
                "    ld.global.f32 %f1, [%rd5];",
                "    add.rn.f32 %f2, %f1, 0f00000000;",
                "    st.global.f32 [%rd7], %f2;",
            ]
        ),
        "out[i] = a[i] + (+0) (add.rn.f32, FADD with RZ)",
    ),
}


def bits(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def inputs():
    """The input sets p and q: (a, b) as f32 encodings."""
    r = random.Random(9804)

    def normal():
        return bits(r.choice([-1, 1]) * 2.0 ** r.uniform(-20, 20))

    an = [normal() for _ in range(N)]
    bn = [normal() for _ in range(N)]
    q = list(an)
    for k in range(16):
        q[8 * k] = 0x80000000
    p = list(q)
    for k, v in enumerate(NANS):
        p[8 * k + 3] = v
    return {"p": (p, bn), "q": (q, bn)}


def model(kern, a):
    """The IEEE outputs: a copy keeps every bit; x + (+0) is x, except -0 -> +0 and a NaN ->
    0x7fffffff (no other input of these sets is non-finite or rounds)."""
    if kern == "zcopy_ref":
        return list(a)
    return [0x7FFFFFFF if (u & 0x7F800000) == 0x7F800000 and u & 0x7FFFFF else 0 if u == 0x80000000 else u for u in a]


def case_toml(name, sname, what):
    bufs = "".join(
        f'''
[[buffers]]
name = "{x}"
dtype = "f32"
shape = [{N}]
init = "file"
file = "../data/{f}.bin"
'''
        for x, f in (("a", f"a_{sname}"), ("b", "b"))
    )
    params = "".join(
        f'''
[[params]]
kind = "ptr"
buffer = "{x}"
'''
        for x in ["a", "b", "out"]
    )
    return f'''name = "c_equiv_zero_sign_nan_{name}"
description = """
MISS 135 (fixtures/c_equiv_zero_sign_nan/gen.py): {what}; input set {sname}. expected/ is
GB300's output (ptxas 13.0 -arch=sm_103a)."""
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


def eq_toml(sname):
    return f'''# MISS 135 (gen.py): {KERNELS["zcopy_ref"][1]} against {KERNELS["zcopy_cand"][1]}; input
# set {sname}. The candidate is the reference plus +0, but the reference's output is an input,
# which can hold a NaN payload that x + (+0) turns into 0x7fffffff: L2 (bound 0), not L0~0.
max_states = 200000
reference = {{ case = "zcopy_ref_{sname}", variant = "good", front = "sass", certify = true }}

[[candidates]]
case = "zcopy_cand_{sname}"
variant = "good"
front = "sass"
certify = true
expect = "L2"
'''


def main():
    sets = inputs()
    if "--check" in sys.argv:
        bad = 0
        for sname, (a, _) in sets.items():
            got = {}
            for kern in KERNELS:
                p = os.path.join(f"{kern}_{sname}", "expected", "out.bin")
                got[kern] = list(struct.unpack(f"<{N}I", open(p, "rb").read()))
                same = got[kern] == model(kern, a)
                bad += not same
                print(f"{kern}_{sname}: capture {'=' if same else 'DIFFERS from'} the model")
            d = [i for i in range(N) if got["zcopy_ref"][i] != got["zcopy_cand"][i]]
            z = [i for i in d if {got["zcopy_ref"][i], got["zcopy_cand"][i]} == {0, 0x80000000}]
            print(f"{sname}: {len(d)} of {N} words differ, {len(z)} of them +0 against -0")
        sys.exit(1 if bad else 0)
    os.makedirs("data", exist_ok=True)
    for sname, (a, b) in sets.items():
        with open(os.path.join("data", f"a_{sname}.bin"), "wb") as f:
            f.write(struct.pack(f"<{N}I", *a))
    with open(os.path.join("data", "b.bin"), "wb") as f:
        f.write(struct.pack(f"<{N}I", *sets["p"][1]))
    for sname in sets:
        for kern, (ptx, what) in KERNELS.items():
            name = f"{kern}_{sname}"
            os.makedirs(name, exist_ok=True)
            with open(os.path.join(name, "kernel.ptx"), "w") as f:
                f.write(ptx)
            with open(os.path.join(name, "case.toml"), "w") as f:
                f.write(case_toml(name, sname, what))
        with open(f"eq_{sname}.toml", "w") as f:
            f.write(eq_toml(sname))


if __name__ == "__main__":
    main()
