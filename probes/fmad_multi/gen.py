# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# probes/fmad_multi: kernel.ptx and in.bin. Each shape reads a, b, c, d, e, f = in[8 t ..] and
# writes its results to out[128 t + k] (SHAPES: name, body, outputs; %fA.. are the inputs,
# %fO<k> the k-th output). Inputs: in 3 of 4 threads c and d are -(a b) +- 1 ulp, e a random
# value and f = -c (fused and unfused results differ); the 4th thread is random words.
import random
import struct
import sys

SHAPES = [
    # every reader a plain add / sub: contracted into each (FFMA, FFMA -a)
    (
        "two_adds",
        "mul.f32 %p, %fA, %fB; add.f32 %fO0, %p, %fC; add.f32 %fO1, %fD, %p;",
        2,
    ),
    (
        "add_sub",
        "mul.f32 %p, %fA, %fB; add.f32 %fO0, %p, %fC; sub.f32 %fO1, %fD, %p;",
        2,
    ),
    (
        "sub_both",
        "mul.f32 %p, %fA, %fB; sub.f32 %fO0, %fC, %p; sub.f32 %fO1, %p, %fD;",
        2,
    ),
    (
        "sat_and_plain",
        "mul.f32 %p, %fA, %fB; add.sat.f32 %fO0, %p, %fC; add.f32 %fO1, %p, %fD;",
        2,
    ),
    ("neg_add", "mul.f32 %p, %fA, %fB; neg.f32 %q, %p; add.f32 %fO0, %q, %fC;", 1),
    (
        "neg_and_add",
        "mul.f32 %p, %fA, %fB; neg.f32 %q, %p; add.f32 %fO0, %q, %fC; add.f32 %fO1, %p, %fD;",
        2,
    ),
    (
        "neg_sub",
        "mul.f32 %p, %fA, %fB; neg.f32 %q, %p; sub.f32 %fO0, %q, %fC; sub.f32 %fO1, %fD, %q;",
        2,
    ),
    (
        "copy_adds",
        "mul.f32 %p, %fA, %fB; mov.f32 %q, %p; add.f32 %fO0, %p, %fC; add.f32 %fO1, %q, %fD;",
        2,
    ),
    (
        "dead_readers",
        "mul.f32 %p, %fA, %fB; add.f32 %fO0, %p, %fC; setp.gt.f32 %x, %p, %fD; mul.f32 %q, %p, %fD;",
        1,
    ),
    (
        "ftz_adds",
        "mul.ftz.f32 %p, %fA, %fB; add.ftz.f32 %fO0, %p, %fC; sub.ftz.f32 %fO1, %fD, %p;",
        2,
    ),
    # a mul.ftz through neg.f32 (no .ftz) or a copy into add.ftz / sub.ftz [.sat]: contracted
    # (ptxas 13.0 / 13.3: FFMA.FTZ [-a]; MISS 62)
    (
        "ftz_neg_add",
        "mul.ftz.f32 %p, %fA, %fB; neg.f32 %q, %p; add.ftz.f32 %fO0, %q, %fF;",
        1,
    ),
    (
        "ftz_neg_add_sat",
        "mul.ftz.f32 %p, %fA, %fB; neg.f32 %q, %p; add.ftz.sat.f32 %fO0, %q, %fF;",
        1,
    ),
    (
        "ftz_neg_subcq",
        "mul.ftz.f32 %p, %fA, %fB; neg.f32 %q, %p; sub.ftz.f32 %fO0, %fC, %q;",
        1,
    ),
    (
        "ftz_neg_subcq_sat",
        "mul.ftz.f32 %p, %fA, %fB; neg.f32 %q, %p; sub.ftz.sat.f32 %fO0, %fC, %q;",
        1,
    ),
    (
        "ftz_neg_sub_qc",
        "mul.ftz.f32 %p, %fA, %fB; neg.f32 %q, %p; sub.ftz.f32 %fO0, %q, %fC;",
        1,
    ),
    (
        "ftz_neg_sub_qc_sat",
        "mul.ftz.f32 %p, %fA, %fB; neg.f32 %q, %p; sub.ftz.sat.f32 %fO0, %q, %fC;",
        1,
    ),
    (
        "ftz_neg_add_imm",
        "mul.ftz.f32 %p, %fA, %fB; neg.f32 %q, %p; add.ftz.f32 %fO0, %q, 0f3F800000;",
        1,
    ),
    (
        "ftz_neg_add_imm_sat",
        "mul.ftz.f32 %p, %fA, %fB; neg.f32 %q, %p; add.ftz.sat.f32 %fO0, %q, 0f3F800000;",
        1,
    ),
    (
        "ftz_neg_neg_add",
        "mul.ftz.f32 %p, %fA, %fB; neg.f32 %q, %p; neg.f32 %s, %q; add.ftz.f32 %fO0, %s, %fC;",
        1,
    ),
    (
        "ftz_neg_neg_add_sat",
        "mul.ftz.f32 %p, %fA, %fB; neg.f32 %q, %p; neg.f32 %s, %q; add.ftz.sat.f32 %fO0, %s, %fC;",
        1,
    ),
    (
        "ftz_neg_mov_add",
        "mul.ftz.f32 %p, %fA, %fB; neg.f32 %q, %p; mov.f32 %s, %q; add.ftz.f32 %fO0, %s, %fF;",
        1,
    ),
    (
        "ftz_neg_mov_add_sat",
        "mul.ftz.f32 %p, %fA, %fB; neg.f32 %q, %p; mov.f32 %s, %q; add.ftz.sat.f32 %fO0, %s, %fF;",
        1,
    ),
    (
        "ftz_mov_neg_add",
        "mul.ftz.f32 %p, %fA, %fB; mov.f32 %q, %p; neg.f32 %s, %q; add.ftz.f32 %fO0, %s, %fF;",
        1,
    ),
    (
        "ftz_mov_neg_add_sat",
        "mul.ftz.f32 %p, %fA, %fB; mov.f32 %q, %p; neg.f32 %s, %q; add.ftz.sat.f32 %fO0, %s, %fF;",
        1,
    ),
    # another live reader: no contraction (FMUL + FADD)
    ("add_store", "mul.f32 %p, %fA, %fB; add.f32 %fO0, %p, %fC; mov.f32 %fO1, %p;", 2),
    (
        "add_mul",
        "mul.f32 %p, %fA, %fB; add.f32 %fO0, %p, %fC; mul.f32 %fO1, %p, %fE;",
        2,
    ),
    (
        "add_setp",
        "mul.f32 %p, %fA, %fB; add.f32 %fO0, %p, %fC; setp.gt.f32 %x, %p, %fD; selp.f32 %fO1, 0f3F800000, 0f00000000, %x;",
        2,
    ),
    (
        "add_rn",
        "mul.f32 %p, %fA, %fB; add.f32 %fO0, %p, %fC; add.rn.f32 %fO1, %p, %fD;",
        2,
    ),
    (
        "add_fma_addend",
        "mul.f32 %p, %fA, %fB; add.f32 %fO0, %p, %fC; fma.rn.f32 %fO1, %fD, %fE, %p;",
        2,
    ),
    ("add_pp", "mul.f32 %p, %fA, %fB; add.f32 %fO0, %p, %fC; add.f32 %fO1, %p, %p;", 2),
    (
        "ftz_mismatch",
        "mul.ftz.f32 %p, %fA, %fB; add.ftz.f32 %fO0, %p, %fC; add.f32 %fO1, %p, %fD;",
        2,
    ),
    (
        "add_neg_store",
        "mul.f32 %p, %fA, %fB; add.f32 %fO0, %p, %fC; neg.f32 %fO1, %p;",
        2,
    ),
    (
        "neg_add_store",
        "mul.f32 %p, %fA, %fB; neg.f32 %q, %p; add.f32 %fO0, %q, %fC; mov.f32 %fO1, %q;",
        2,
    ),
    (
        "add_max",
        "mul.f32 %p, %fA, %fB; add.f32 %fO0, %p, %fC; max.f32 %fO1, %p, %fD;",
        2,
    ),
    (
        "add_cvt",
        "mul.f32 %p, %fA, %fB; add.f32 %fO0, %p, %fC; cvt.rzi.f32.f32 %fO1, %p;",
        2,
    ),
    (
        "copy_store",
        "mul.f32 %p, %fA, %fB; add.f32 %fO0, %p, %fC; mov.f32 %q, %p; mov.b32 %iO1, %q;",
        2,
    ),
    # an add of two products: the one with no other reader folds (both: the first operand's)
    (
        "two_muls",
        "mul.f32 %p, %fA, %fB; mul.f32 %q, %fC, %fE; add.f32 %fO0, %p, %q;",
        1,
    ),
    (
        "two_muls_late",
        "mul.f32 %q, %fC, %fE; mul.f32 %p, %fA, %fB; sub.f32 %fO0, %p, %q;",
        1,
    ),
    (
        "two_muls_q_stored",
        "mul.f32 %p, %fA, %fB; mul.f32 %q, %fC, %fE; add.f32 %fO0, %p, %q; mov.f32 %fO1, %q;",
        2,
    ),
    (
        "two_muls_p_stored",
        "mul.f32 %p, %fA, %fB; mul.f32 %q, %fD, %fE; add.f32 %fO0, %p, %q; mov.f32 %fO1, %p;",
        2,
    ),
    (
        "two_muls_p_added",
        "mul.f32 %p, %fA, %fB; mul.f32 %q, %fD, %fE; add.f32 %fO0, %p, %q; add.f32 %fO1, %p, %fC;",
        2,
    ),
    (
        "two_muls_q_added",
        "mul.f32 %p, %fA, %fB; mul.f32 %q, %fD, %fE; add.f32 %fO0, %p, %q; add.f32 %fO1, %q, %fC;",
        2,
    ),
]


def kernel():
    L = [
        "// probes/fmad_multi (generated by gen.py): PTX-FMAD for a mul with several readers.",
        "// out[128 t + k], shapes in gen.py order.",
        ".version 8.8",
        ".target sm_103a",
        ".address_size 64",
        "",
        ".visible .entry fmad_multi(",
        "    .param .u64 pin,",
        "    .param .u64 pout",
        ")",
        ".maxntid 256, 1, 1",
        "{",
        "    .reg .b32 %r<4>;",
        "    .reg .b64 %rd<8>;",
        "    .reg .f32 %fA, %fB, %fC, %fD, %fE, %fF;",
    ]
    body, k, regs = [], 0, []
    for si, (name, src, nout) in enumerate(SHAPES):
        s = (
            src.replace("%p", f"%p{si}")
            .replace("%q", f"%q{si}")
            .replace("%s", f"%s{si}")
            .replace("%x", f"%x{si}")
        )
        regs += [f"%p{si}", f"%q{si}", f"%s{si}"]
        outs = []
        for o in range(nout):
            s = s.replace(f"%fO{o}", f"%o{k}").replace(f"%iO{o}", f"%i{k}")
            outs.append(k)
            k += 1
        body.append(f"    // {name}")
        body += ["    " + x.strip() + ";" for x in s.split(";") if x.strip()]
        for o in outs:
            if f"%i{o}" in s:
                body.append(f"    st.global.b32 [%rd5+{4 * o}], %i{o};")
            else:
                body.append(f"    st.global.f32 [%rd5+{4 * o}], %o{o};")
    assert k <= 128
    L.append("    .reg .f32 " + ", ".join(regs + [f"%o{i}" for i in range(k)]) + ";")
    L.append("    .reg .b32 " + ", ".join(f"%i{i}" for i in range(k)) + ";")
    L.append("    .reg .pred " + ", ".join(f"%x{i}" for i in range(len(SHAPES))) + ";")
    L += [
        "    ld.param.u64 %rd1, [pin];",
        "    ld.param.u64 %rd2, [pout];",
        "    cvta.to.global.u64 %rd1, %rd1;",
        "    cvta.to.global.u64 %rd2, %rd2;",
        "    mov.u32 %r1, %tid.x;",
        "    mul.wide.u32 %rd3, %r1, 32;",
        "    add.s64 %rd4, %rd1, %rd3;",
        "    ld.global.v4.f32 {%fA, %fB, %fC, %fD}, [%rd4];",
        "    ld.global.v2.f32 {%fE, %fF}, [%rd4+16];",
        "    mul.wide.u32 %rd3, %r1, 512;",
        "    add.s64 %rd5, %rd2, %rd3;",
    ]
    L += body + ["    ret;", "}", ""]
    return "\n".join(L), k


def inputs():
    r = random.Random(29)

    def f(v):
        return struct.unpack("<I", struct.pack("<f", v))[0]

    def g(w):
        return struct.unpack("<f", struct.pack("<I", w))[0]

    out = []
    for t in range(256):
        if t % 4 == 3:
            out += [r.getrandbits(32) for _ in range(8)]
            continue
        a, b = (r.choice([1, -1]) * (1 + r.random()) for _ in range(2))
        a, b = g(f(a)), g(f(b))
        p = a * b
        c = f(-p) ^ r.choice([0, 1])
        d = f(-p) ^ r.choice([0, 1, 2])
        e = f(r.choice([1, -1]) * (1 + r.random()))
        out += [f(a), f(b), c, d, e, c ^ 0x80000000, 0, 0]
    return struct.pack("<2048I", *out)


if __name__ == "__main__":
    src, k = kernel()
    open("kernel.ptx", "w").write(src)
    open("in.bin", "wb").write(inputs())
    print(k, "outputs per thread")
