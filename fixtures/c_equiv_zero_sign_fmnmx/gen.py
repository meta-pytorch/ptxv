# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MISS 135 follow-up fixture: a hardware FMNMX passes no NaN payload; a PTX max.f32 can.

x + (+0) turns a NaN x into the canonical NaN 0x7fffffff (FADD), so L0~0 peels x only where it
holds no other NaN (`Side::nan_canonical`). An FMNMX instruction gives 0x7fffffff for two NaN
operands and the other operand for one, so its NaN results are canonical whatever its operands
hold: on the SASS front a max node of an FMNMX / FMNMX3 instruction is peeled outright
(`NumState::fmnmx_hw`). On the PTX front a max.f32 is not an instruction yet: ptxas folds
max.f32 (min.f32) of one value into a copy, which passes the value's payload, so there a max is
peeled only where its operands are. Kernels (PTX, sm_103a, grid 4 x 256, i = 256 ctaid + tid;
f32 inputs a, b, c of 1024 elements):
  fmax      out[i] = max.f32(a[i], b[i])                    (SASS: FMNMX)
  fmax0     the same, + 0 (add.rn.f32 with 0f00000000)      (SASS: FMNMX, FADD with RZ)
  fmaxaa    out[i] = max.f32(a[i], a[i])                    (SASS: a copy, no FMNMX)
  fmaxaa0   the same, + 0                                   (SASS: FADD with RZ)
  fmax3     out[i] = max.f32(max.f32(a[i], b[i]), c[i])     (SASS: FMNMX3)
  fmax30    the same, + 0
Pairs (reference, candidate = reference + 0), each on the PTX and the SASS front, over set f:
  fmax      SASS L0~0, PTX L2 (the max of two inputs is not peeled there)
  fmaxaa    L2 on both fronts (the copy keeps a's payloads, which the + 0 makes 0x7fffffff)
  fmax3     SASS L0~0, PTX L2
The levels are claims for every input; the model's runs use set f, since outside segment mode a
max with a NaN operand in the run is untracked. Set n's GB300 outputs test the claims at NaN
operands. Input sets (copied as measured, with fmax's and fmax0's captures): f has -0 / +0
pairs at 0 .. 63, equal values and x against -x at 64 .. 127, subnormals at 128 .. 191 and
normal values elsewhere; n is f with NaN payloads (0x7fc12345, 0x7fc00001, 0xffc00002,
0x7f800001, 0xff812345, 0x7fffffff, 0x7fc00000) and infinities in a and b at 64 .. 127, and
c = 0x7fc0beef at every third element there. Each case dir is <kernel>_<set>; expected/out.bin
is GB300's output (ptxas 13.0 -arch=sm_103a), equal to the model below (`--check`), with 5
scrubbed and 5 plain runs identical to it. GB300, set n: fmax against fmax0 differs in 16 words,
all zero signs; fmax3 against fmax30 in 8, all zero signs (FMNMX3's NaN results are canonical
too); fmaxaa against fmaxaa0 in 56, 24 of them NaN payloads of a that the copy keeps.
Run from this directory: python3 gen.py [--check]."""
import os
import struct
import sys

N = 1024
CANON = 0x7FFFFFFF
PTX_HEADER = """// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

"""
PLUS0 = "\n    add.rn.f32 %f10, %f11, 0f00000000;"
# kernel -> (body, description)
KERNELS = {
    "fmax": ("max.f32 %f10, %f1, %f2;", "out[i] = max.f32(a[i], b[i])"),
    "fmax0": ("max.f32 %f11, %f1, %f2;" + PLUS0, "out[i] = max.f32(a[i], b[i]) + 0"),
    "fmaxaa": ("max.f32 %f10, %f1, %f1;", "out[i] = max.f32(a[i], a[i])"),
    "fmaxaa0": ("max.f32 %f11, %f1, %f1;" + PLUS0, "out[i] = max.f32(a[i], a[i]) + 0"),
    "fmax3": (
        "max.f32 %f12, %f1, %f2;\n    max.f32 %f10, %f12, %f3;",
        "out[i] = max.f32(max.f32(a[i], b[i]), c[i])",
    ),
    "fmax30": (
        "max.f32 %f12, %f1, %f2;\n    max.f32 %f11, %f12, %f3;" + PLUS0,
        "out[i] = max.f32(max.f32(a[i], b[i]), c[i]) + 0",
    ),
}
# pair -> (reference, candidate, expected level on the PTX front, on the SASS front)
PAIRS = {
    "fmax": ("fmax", "fmax0", "L2", "L0~0"),
    "fmaxaa": ("fmaxaa", "fmaxaa0", "L2", "L2"),
    "fmax3": ("fmax3", "fmax30", "L2", "L0~0"),
}


def kernel(name):
    body, what = KERNELS[name]
    return (
        PTX_HEADER
        + f""".version 8.8
.target sm_103a
.address_size 64
// {name}: {what}.
.visible .entry k(.param .u64 pa, .param .u64 pb, .param .u64 pc, .param .u64 pout)
{{
    .reg .pred %p<8>;
    .reg .b16 %h<8>;
    .reg .b32 %r<16>;
    .reg .f32 %f<24>;
    .reg .b64 %rd<16>;
    ld.param.u64 %rd1, [pa];
    ld.param.u64 %rd2, [pb];
    ld.param.u64 %rd3, [pc];
    ld.param.u64 %rd4, [pout];
    cvta.to.global.u64 %rd1, %rd1;
    cvta.to.global.u64 %rd2, %rd2;
    cvta.to.global.u64 %rd3, %rd3;
    cvta.to.global.u64 %rd4, %rd4;
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %tid.x;
    mad.lo.u32 %r3, %r1, 256, %r2;
    mul.wide.u32 %rd5, %r3, 4;
    add.s64 %rd6, %rd1, %rd5;
    add.s64 %rd7, %rd2, %rd5;
    add.s64 %rd8, %rd3, %rd5;
    add.s64 %rd9, %rd4, %rd5;
    ld.global.f32 %f1, [%rd6];
    ld.global.f32 %f2, [%rd7];
    ld.global.f32 %f3, [%rd8];
    {body}
    st.global.f32 [%rd9], %f10;
    ret;
}}
"""
    )


SETS = ("f", "n")


def case_toml(name, sname):
    buffers = "".join(
        f"""
[[buffers]]
name = "{x}"
dtype = "f32"
shape = [1024]
init = "file"
file = "../data/{x}_{sname}.bin"
"""
        for x in "abc"
    )
    params = "".join(f'\n[[params]]\nkind = "ptr"\nbuffer = "{x}"\n' for x in ("a", "b", "c", "out"))
    return f'''name = "c_equiv_zero_sign_fmnmx_{name}_{sname}"
description = """
MISS 135 (fixtures/c_equiv_zero_sign_fmnmx/gen.py): {KERNELS[name][1]}; input set {sname}. expected/
is GB300's output (ptxas 13.0 -arch=sm_103a)."""
ptx = "kernel.ptx"
entry = "k"
grid = [4, 1, 1]
block = [256, 1, 1]
outputs = ["out"]
{buffers}
[[buffers]]
name = "out"
dtype = "f32"
shape = [1024]
init = "zeros"
expect = "expected/out.bin"
{params}
[[variants]]
id = "good"
summary = "as written"
expect = ["clean"]
'''


def eq_toml(pair, front):
    ref, cand, lp, ls = PAIRS[pair]
    level = lp if front == "ptx" else ls
    return f'''# MISS 135 (gen.py): {KERNELS[ref][1]} against the same + 0; input set f, {front} front.
max_states = 200000
reference = {{ case = "{ref}_f", variant = "good", front = "{front}", certify = true }}

[[candidates]]
case = "{cand}_f"
variant = "good"
front = "{front}"
certify = true
expect = "{level}"
'''


def isnan(w):
    return (w & 0x7F800000) == 0x7F800000 and (w & 0x7FFFFF) != 0


def fval(w):
    return struct.unpack("<f", struct.pack("<I", w))[0]


def fmnmx(x, y):
    """GB300's FMNMX (max): two NaNs give 0x7fffffff, one NaN the other operand; -0 < +0."""
    if isnan(x) and isnan(y):
        return CANON
    if isnan(x):
        return y
    if isnan(y):
        return x
    fx, fy = fval(x), fval(y)
    if fx == fy:
        return y if x == 0x80000000 else x
    return x if fx > fy else y


def plus0(x):
    """FADD x, RZ: the canonical NaN for a NaN, +0 for -0."""
    if isnan(x):
        return CANON
    return 0 if x == 0x80000000 else x


def model(name, a, b, c):
    base = name[:-1] if name.endswith("0") else name
    out = []
    for x, y, z in zip(a, b, c):
        if base == "fmax":
            v = fmnmx(x, y)
        elif base == "fmaxaa":
            v = x  # ptxas folds max.f32(a, a) into a copy
        else:
            v = fmnmx(fmnmx(x, y), z)
        out.append(plus0(v) if name.endswith("0") else v)
    return out


def read(p):
    return list(struct.unpack(f"<{N}I", open(p, "rb").read()))


def main():
    if "--check" in sys.argv:
        bad = 0
        for sname in SETS:
            a, b, c = (read(os.path.join("data", f"{x}_{sname}.bin")) for x in "abc")
            got = {}
            for k in KERNELS:
                got[k] = read(os.path.join(f"{k}_{sname}", "expected", "out.bin"))
                same = got[k] == model(k, a, b, c)
                bad += not same
                print(f"{k}_{sname}: capture {'=' if same else 'DIFFERS from'} the model")
            for p, (ref, cand, _, _) in PAIRS.items():
                d = [(x, y) for x, y in zip(got[ref], got[cand]) if x != y]
                nan = sum(1 for x, y in d if isnan(x) or isnan(y))
                print(
                    f"{p}_{sname}: {ref} against {cand}: {len(d)} of {N} words differ ({nan} at NaNs)"
                )
        sys.exit(1 if bad else 0)
    for k in KERNELS:
        for sname in SETS:
            d = f"{k}_{sname}"
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, "kernel.ptx"), "w") as f:
                f.write(kernel(k))
            with open(os.path.join(d, "case.toml"), "w") as f:
                f.write(case_toml(k, sname))
    for p in PAIRS:
        for front in ("ptx", "sass"):
            with open(f"eq_{p}_{front}.toml", "w") as f:
                f.write(eq_toml(p, front))


if __name__ == "__main__":
    main()
