# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""probes/fmad_ftz_neg (MISS 62): `mul.ftz.f32` read only by `neg.f32`, whose result an
`add.ftz` / `sub.ftz` [.sat] reads. ptxas 13.0 and 13.3 contract it into one FFMA.FTZ (-a or a);
the PTX front kept the mul apart (it followed a neg only when the mul had no .ftz). Forms 4, 5 are
controls both fronts contract. Each form reads its own a, b, c planes (no CSE across forms); in
half the lanes c cancels the product to a few ulps. Writes kernel.ptx and in.bin;
out[k * N + i] = form k, input i."""

import random
import struct
import sys

N = 16384
FORMS = [
    ("mul.ftz.f32 P, A, B;", "neg.f32 Q, P;", "add.ftz.f32 D, Q, C;", "c - ab"),
    ("mul.ftz.f32 P, A, B;", "neg.f32 Q, P;", "sub.ftz.f32 D, C, Q;", "c + ab"),
    ("mul.ftz.f32 P, A, B;", "neg.f32 Q, P;", "add.ftz.sat.f32 D, Q, C;", "c - ab"),
    ("mul.ftz.f32 P, A, B;", "neg.f32 Q, P;", "sub.ftz.sat.f32 D, C, Q;", "c + ab"),
    (
        "mul.f32 P, A, B;",
        "neg.f32 Q, P;",
        "add.f32 D, Q, C;",
        "c - ab",
    ),  # control: both fronts contract
    (
        "mul.ftz.f32 P, A, B;",
        "mov.f32 Q, P;",
        "add.ftz.f32 D, Q, C;",
        "c + ab",
    ),  # control: both contract
]
f32 = lambda x: struct.unpack("<f", struct.pack("<f", x))[0]


def rnd_f32(rng):
    r = rng.random()
    if r < 0.03:
        return rng.choice(
            [0.0, -0.0, float("inf"), float("-inf"), float("nan"), 1.0, -1.0]
        )
    e = rng.randint(-40, 40) if r < 0.9 else rng.randint(-149, -110)
    return f32(rng.choice([1, -1]) * (1 + rng.random()) * 2.0**e)


def main():
    rng = random.Random(20261110)
    planes = []
    for k, (_, _, _, kind) in enumerate(FORMS):
        A, B, C = [], [], []
        for i in range(N):
            a, b = rnd_f32(rng), rnd_f32(rng)
            if i % 2 == 0 and all(
                map(lambda v: v == v and abs(v) != float("inf"), (a, b))
            ):
                ab = f32(a * b) if abs(a * b) < 3e38 else 0.0
                ulp = 2**-23 * abs(ab) if ab else 2**-149
                c = f32((ab if kind == "c - ab" else -ab) + rng.randint(-4, 4) * ulp)
            else:
                c = rnd_f32(rng)
            A.append(a)
            B.append(b)
            C.append(c)
        planes += [A, B, C]
    with open("in.bin", "wb") as f:
        for p in planes:
            f.write(struct.pack(f"<{N}f", *p))
    body = []
    for k, (m, n, a, kind) in enumerate(FORMS):
        R = {
            "A": f"%f{10 * k + 1}",
            "B": f"%f{10 * k + 2}",
            "C": f"%f{10 * k + 3}",
            "P": f"%f{10 * k + 5}",
            "Q": f"%f{10 * k + 6}",
            "D": f"%f{10 * k + 9}",
        }
        sub = lambda s: "".join(R.get(ch, ch) for ch in s)
        body.append(f"    // form {k}: {m} {n} {a}")
        for j, reg in enumerate("ABC"):
            body.append(f"    ld.global.f32   {R[reg]}, [%rd4+{4 * N * (3 * k + j)}];")
        for s in (m, n, a):
            body.append("    " + sub(s))
        body.append(f"    st.global.f32   [%rd5+{4 * N * k}], {R['D']};")
    names = "\n".join(f"//   {k}: {m} {n} {a}" for k, (m, n, a, _) in enumerate(FORMS))
    open(
        "kernel.ptx", "w"
    ).write(f"""// probes/fmad_ftz_neg (gen.py). 256 threads x 64 iterations, i = 256 j + t.
{names}
.version 9.0
.target sm_103a
.address_size 64

.visible .entry fmad_ftz_neg(
    .param .u64 pin,
    .param .u64 pout
)
.maxntid 256, 1, 1
{{
    .reg .pred  %p<2>;
    .reg .f32   %f<{10 * len(FORMS) + 10}>;
    .reg .b32   %r<8>;
    .reg .b64   %rd<8>;

    ld.param.u64    %rd1, [pin];
    ld.param.u64    %rd2, [pout];
    cvta.to.global.u64 %rd1, %rd1;
    cvta.to.global.u64 %rd2, %rd2;
    mov.u32         %r4, %tid.x;
LOOP:
    mul.wide.u32    %rd3, %r4, 4;
    add.s64         %rd4, %rd1, %rd3;
    add.s64         %rd5, %rd2, %rd3;
{chr(10).join(body)}
    add.u32         %r4, %r4, 256;
    setp.lt.u32     %p1, %r4, {N};
    @%p1 bra        LOOP;
    ret;
}}
""")
    print(len(FORMS), file=sys.stderr)


main()
