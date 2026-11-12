# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MISS 109 fixture: L0 up to the sign of zero is a level of its own, L0~0, not L0 (bitwise).

Kernels (PTX, sm_103a, grid 4 x 256, i = 256 ctaid + tid; f32 inputs a, b of 1024 elements, the
same bytes in every case, in data/):
  ref_add     out[i] = a + b
  cand_add0   out[i] = (a + b) + 0: the +0 epilogue of cuBLAS's alpha = 1, beta = 0 GEMMs. Equal
              bits, except that a -0 sum becomes +0: L0~0, and L0 fails at every output.
  cand_ba     out[i] = b + a: the same rounding DAG as ref_add, L0.
The inputs are random in [-4, 4] except at i = 64 k + 5 .. 8 (k = 0 .. 15):
  64 k + 5    a = b = -0           ref_add -0, cand_add0 +0 (the captures differ in the sign)
  64 k + 6    a = -0, b = +0       +0 in both
  64 k + 7    a = x, b = -x        +0 in both
  64 k + 8    a = -2^-149, b = -0  -2^-149 in both
expected/out.bin of these is GB300's output (ptxas 13.0 -arch=sm_103a); `--check` compares the
captures with the IEEE round-to-nearest sums.

The other cases run cand_add0's kernel against its capture with one word changed
(python3 gen.py --tamper):
  cand_add0_t1   out[9] in its last bit
  cand_add0_z    out[69] (a = b = -0) +0 -> -0: equal to ref_add's there, but x + (+0) is never
                 -0 in round to nearest
eq.toml: cand_add0 is L0~0 against ref_add, and its captures, 16 outputs +0 against -0, do not
contradict it; cand_ba is L0; the tampered captures are refused.
Run from this directory: python3 gen.py [--tamper | --check]."""
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


def kernel(body):
    lines = [
        HDR + ".visible .entry k(.param .u64 pa, .param .u64 pb, .param .u64 pout)",
        "{",
        "    .reg .b32 %r<4>;",
        "    .reg .f32 %f<8>;",
        "    .reg .b64 %rd<8>;",
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
    lines += ["    " + l for l in body]
    lines += ["    st.global.f32 [%rd7], %f4;", "    ret;", "}", ""]
    return "\n".join(lines)


# name: (body, what)
BASE = {
    "ref_add": (["add.rn.f32 %f4, %f1, %f2;"], "out = a + b"),
    "cand_add0": (
        ["add.rn.f32 %f3, %f1, %f2;", "add.rn.f32 %f4, %f3, 0f00000000;"],
        "out = (a + b) + 0",
    ),
    "cand_ba": (["add.rn.f32 %f4, %f2, %f1;"], "out = b + a"),
}
# tampered: (base case, captured word, new bits or "flip")
TAMPER = {"cand_add0_t1": ("cand_add0", 9, "flip"), "cand_add0_z": ("cand_add0", 69, 0x80000000)}


def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def inputs():
    r = random.Random(109)
    a, b = [], []
    for i in range(N):
        x, y = f32(r.uniform(-4, 4)), f32(r.uniform(-4, 4))
        k = i % 64
        if k == 5:
            x, y = -0.0, -0.0
        elif k == 6:
            x, y = -0.0, 0.0
        elif k == 7:
            y = -x
        elif k == 8:
            x, y = -struct.unpack("<f", struct.pack("<I", 1))[0], -0.0
        a.append(x)
        b.append(y)
    return a, b


def case_toml(name, base, what):
    return f'''name = "c_equiv_zero_sign_{name}"
description = """
MISS 109 (fixtures/c_equiv_zero_sign/gen.py): {what}. expected/ is GB300's output (ptxas 13.0
-arch=sm_103a){'' if name == base else ", one word changed (gen.py --tamper)"}."""
ptx = "kernel.ptx"
entry = "k"
grid = [4, 1, 1]
block = [256, 1, 1]
outputs = ["out"]

[[buffers]]
name = "a"
dtype = "f32"
shape = [{N}]
init = "file"
file = "../data/a.bin"

[[buffers]]
name = "b"
dtype = "f32"
shape = [{N}]
init = "file"
file = "../data/b.bin"

[[buffers]]
name = "out"
dtype = "f32"
shape = [{N}]
init = "zeros"
expect = "expected/out.bin"

[[params]]
kind = "ptr"
buffer = "a"

[[params]]
kind = "ptr"
buffer = "b"

[[params]]
kind = "ptr"
buffer = "out"

[[variants]]
id = "good"
summary = "as written"
expect = ["clean"]
'''


def write_case(name, base, what):
    os.makedirs(name, exist_ok=True)
    with open(os.path.join(name, "kernel.ptx"), "w") as f:
        f.write(kernel(BASE[base][0]))
    with open(os.path.join(name, "case.toml"), "w") as f:
        f.write(case_toml(name, base, what))


def sums():
    """The IEEE round-to-nearest outputs of the base kernels: the sum of two of these f32
    values is exact in double (their exponents differ by less than 29), then rounded once;
    -0 + -0 = -0, x + (-x) = +0, -0 + +0 = +0."""
    a, b = inputs()
    add = [f32(x + y) for x, y in zip(a, b)]
    return {
        "ref_add": add,
        "cand_add0": [s + 0.0 for s in add],
        "cand_ba": [f32(y + x) for x, y in zip(a, b)],
    }


def main():
    if "--check" in sys.argv:
        bad = 0
        for name, out in sums().items():
            p = os.path.join(name, "expected", "out.bin")
            got = open(p, "rb").read()
            want = struct.pack(f"<{N}f", *out)
            neg0 = sum(1 for v in struct.unpack(f"<{N}I", got) if v == 0x80000000)
            same = got == want
            bad += not same
            print(f"{name}: capture {'=' if same else 'DIFFERS from'} the IEEE sums ({neg0} outputs -0)")
        ref = open("ref_add/expected/out.bin", "rb").read()
        cand = open("cand_add0/expected/out.bin", "rb").read()
        rw, cw = struct.unpack(f"<{N}I", ref), struct.unpack(f"<{N}I", cand)
        diff = [i for i in range(N) if rw[i] != cw[i]]
        zs = [i for i in diff if {rw[i], cw[i]} == {0, 0x80000000}]
        print(f"ref_add vs cand_add0: {len(diff)} outputs differ, {len(zs)} of them +0 against -0")
        sys.exit(1 if bad else 0)
    if "--tamper" in sys.argv:
        for name, (base, word, how) in TAMPER.items():
            write_case(name, base, BASE[base][1])
            src = open(os.path.join(base, "expected", "out.bin"), "rb").read()
            w = list(struct.unpack(f"<{N}I", src))
            w[word] = w[word] ^ 1 if how == "flip" else how
            os.makedirs(os.path.join(name, "expected"), exist_ok=True)
            with open(os.path.join(name, "expected", "out.bin"), "wb") as f:
                f.write(struct.pack(f"<{N}I", *w))
        return
    a, b = inputs()
    os.makedirs("data", exist_ok=True)
    with open("data/a.bin", "wb") as f:
        f.write(struct.pack(f"<{N}f", *a))
    with open("data/b.bin", "wb") as f:
        f.write(struct.pack(f"<{N}f", *b))
    for name, (_, what) in BASE.items():
        write_case(name, name, what)


if __name__ == "__main__":
    main()
