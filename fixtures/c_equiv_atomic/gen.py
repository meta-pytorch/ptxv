# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MISS 98 fixture (from its review): float atomic adds exempt from the simulated-vs-captured
check only the outputs whose value depends on their order; every other output is compared
bitwise, and the captured pair is held to the level (L0: bitwise, NaN payloads included; L1:
the bound, and no non-finite value against a different one). An unproven level is held to
the captures too.

Kernels (PTX, sm_103a, grid 4 x 256, i = 256 ctaid + tid; f32 inputs a, b, c of 1024
elements, the same bytes in every case, in data/):
  ref_sum        out[i] = (a + b) + c
  cand_sum       out[i] = a + (b + c)
  cand_sum_scr   cand_sum; threads i < 4 also red.global.add.f32 [scr], a[i] (scr is no output)
  ref_B          ref_sum; thread 0: out[1024] = ((a0 + a1) + a2) + a3           (1025 outputs)
  cand_B         cand_sum; threads i < 4: red.global.add.f32 [out + 4096], a[i]
  cand_Bn        cand_sum; thread 0: out[1024] = ((a3 + a2) + a1) + a0          (no atomic)
  ref_B2         ref_sum; thread 0: out[1024] = a0 + a1
  cand_B2        cand_sum; threads i < 2: red.global.add.f32 [out + 4096], a[i] (two adds
                 into +0: every order gives the same bits, so out[1024] is compared too)
  ref_copy       out[i] = a[i]  (a: anan.bin, with NaN payloads, -0 and +inf)
  cand_copy_scr  ref_copy and cand_sum_scr's scratch atomic
  cand_copy_g    ref_copy; threads with a[i] < 0 also store a[i] into scr (a guard that
                 depends on input data, and the stores race: the level is unproven)
expected/out.bin of these is GB300's output (ptxas 13.0 -arch=sm_103a).

The other cases are a base case with one captured word changed (python3 gen.py --tamper,
from the base cases' captures):
  cand_sum_t1, cand_sum_scr_t1, cand_B_t1, cand_Bn_t1   out[225] in its last bit (1 ulp; the
      proven bound there is about 2,700 ulp, so only the simulated-vs-captured check sees it)
  cand_sum_scr_inf / cand_sum_scr_nan                   out[225] = +inf / 0x7fc00000
  cand_copy_nan (ref_copy's capture), cand_copy_scr_nan out[7] 0x7fc12345 -> 0x7fc00000
  cand_B_ulp                                            out[1024] (the atomic sum) in its last
      bit: an order of the adds the GPU may run, inside the bound
  cand_B_inf / cand_B_far                               out[1024] = +inf / + 1.0
  cand_B2_t1                                            out[1024] in its last bit
  cand_copy_g_bit                                       out[5] in its last bit (an unproven
      level against a capture the model does not reproduce)
and two cases whose capture covers part of the output (expect_offset = 2048: out[512..]):
  ref_sum_off, cand_sum_off (also expect_ignore of out[600])   ref_sum's / cand_sum's capture
Run from this directory: python3 gen.py [--tamper]."""
import os
import random
import struct
import sys
import textwrap

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


def kernel(body, tail=(), scr=False):
    params = [".param .u64 pa", ".param .u64 pb", ".param .u64 pc", ".param .u64 pout"]
    if scr:
        params.append(".param .u64 pscr")
    lines = [
        HDR + ".visible .entry k(" + ", ".join(params) + ")",
        "{",
        "    .reg .pred %p<4>;",
        "    .reg .b32 %r<8>;",
        "    .reg .f32 %f<16>;",
        "    .reg .b64 %rd<16>;",
        "    ld.param.u64 %rd1, [pa];",
        "    ld.param.u64 %rd2, [pb];",
        "    ld.param.u64 %rd3, [pc];",
        "    ld.param.u64 %rd4, [pout];",
        "    cvta.to.global.u64 %rd1, %rd1;",
        "    cvta.to.global.u64 %rd2, %rd2;",
        "    cvta.to.global.u64 %rd3, %rd3;",
        "    cvta.to.global.u64 %rd4, %rd4;",
    ]
    if scr:
        lines += ["    ld.param.u64 %rd10, [pscr];", "    cvta.to.global.u64 %rd10, %rd10;"]
    lines += [
        "    mov.u32 %r1, %ctaid.x;",
        "    mov.u32 %r2, %tid.x;",
        "    mad.lo.u32 %r3, %r1, 256, %r2;",
        "    mul.wide.u32 %rd5, %r3, 4;",
        "    add.s64 %rd6, %rd1, %rd5;",
        "    add.s64 %rd7, %rd2, %rd5;",
        "    add.s64 %rd8, %rd3, %rd5;",
        "    add.s64 %rd9, %rd4, %rd5;",
        "    ld.global.f32 %f1, [%rd6];",
        "    ld.global.f32 %f2, [%rd7];",
        "    ld.global.f32 %f3, [%rd8];",
    ]
    lines += ["    " + l for l in body]
    lines += ["    st.global.f32 [%rd9], %f4;"]
    lines += [l if l.endswith(":") else "    " + l for l in tail]
    lines += ["    ret;", "}", ""]
    return "\n".join(lines)


# Rounding-DAG mode tracks sums of inputs, so the sum pairs are a reassociated three-term sum
# (L1 with a bound).
SUML = ["add.rn.f32 %f5, %f1, %f2;", "add.rn.f32 %f4, %f5, %f3;"]
SUMR = ["add.rn.f32 %f5, %f2, %f3;", "add.rn.f32 %f4, %f1, %f5;"]
COPY = ["mov.f32 %f4, %f1;"]
SCR = ["setp.lt.u32 %p1, %r3, 4;", "@%p1 red.global.add.f32 [%rd10], %f1;"]
ATOM_B = ["setp.lt.u32 %p1, %r3, 4;", "@%p1 red.global.add.f32 [%rd4+4096], %f1;"]
ATOM_B2 = ["setp.lt.u32 %p1, %r3, 2;", "@%p1 red.global.add.f32 [%rd4+4096], %f1;"]
GUARD = ["setp.lt.f32 %p2, %f1, 0f00000000;", "@%p2 st.global.f32 [%rd10], %f1;"]


def seq(order):
    """Thread 0: out[1024] = the sequential sum of a[order[0]], a[order[1]], ..."""
    t = ["setp.ne.u32 %p2, %r3, 0;", "@%p2 bra DONE;"]
    for k, j in enumerate(order):
        t.append(f"ld.global.f32 %f{10 + k}, [%rd1+{4 * j}];")
    t.append("add.rn.f32 %f6, %f10, %f11;")
    for k in range(2, len(order)):
        t.append(f"add.rn.f32 %f6, %f6, %f{10 + k};")
    t.append("st.global.f32 [%rd4+4096], %f6;")
    t.append("DONE:")
    return t


# name: (kernel, outputs, scratch buffer, input a, what)
BASE = {
    "ref_sum": (kernel(SUML), N, False, "a", "out = (a + b) + c"),
    "cand_sum": (kernel(SUMR), N, False, "a", "out = a + (b + c)"),
    "cand_sum_scr": (
        kernel(SUMR, SCR, scr=True),
        N,
        True,
        "a",
        "cand_sum and a float atomic add into a scratch buffer (no output)",
    ),
    "ref_B": (kernel(SUML, seq([0, 1, 2, 3])), N + 1, False, "a",
              "ref_sum; out[1024] = ((a0 + a1) + a2) + a3"),
    "cand_B": (kernel(SUMR, ATOM_B), N + 1, False, "a",
               "cand_sum; out[1024] = float atomic adds of a0 .. a3"),
    "cand_Bn": (kernel(SUMR, seq([3, 2, 1, 0])), N + 1, False, "a",
                "cand_sum; out[1024] = ((a3 + a2) + a1) + a0"),
    "ref_B2": (kernel(SUML, seq([0, 1])), N + 1, False, "a", "ref_sum; out[1024] = a0 + a1"),
    "cand_B2": (kernel(SUMR, ATOM_B2), N + 1, False, "a",
                "cand_sum; out[1024] = two float atomic adds (a0, a1) into +0"),
    "ref_copy": (kernel(COPY), N, False, "anan", "out = a (a with NaN payloads)"),
    "cand_copy_scr": (
        kernel(COPY, SCR, scr=True),
        N,
        True,
        "anan",
        "out = a and a float atomic add into a scratch buffer (no output)",
    ),
    "cand_copy_g": (
        kernel(COPY, GUARD, scr=True),
        N,
        True,
        "anan",
        "out = a; a[i] < 0 also stored into a scratch buffer (an input-dependent guard)",
    ),
}
J, JB, JN = 225, 1024, 7
# tampered: (base case whose kernel it runs, base case whose capture it changes, word, how)
TAMPER = {
    "cand_sum_t1": ("cand_sum", "cand_sum", J, "flip"),
    "cand_sum_scr_t1": ("cand_sum_scr", "cand_sum_scr", J, "flip"),
    "cand_sum_scr_inf": ("cand_sum_scr", "cand_sum_scr", J, 0x7F800000),
    "cand_sum_scr_nan": ("cand_sum_scr", "cand_sum_scr", J, 0x7FC00000),
    "cand_B_t1": ("cand_B", "cand_B", J, "flip"),
    "cand_Bn_t1": ("cand_Bn", "cand_Bn", J, "flip"),
    "cand_B_ulp": ("cand_B", "cand_B", JB, "flip"),
    "cand_B_inf": ("cand_B", "cand_B", JB, 0x7F800000),
    "cand_B_far": ("cand_B", "cand_B", JB, "far"),
    "cand_B2_t1": ("cand_B2", "cand_B2", JB, "flip"),
    "cand_copy_nan": ("ref_copy", "ref_copy", JN, 0x7FC00000),
    "cand_copy_scr_nan": ("cand_copy_scr", "cand_copy_scr", JN, 0x7FC00000),
    "cand_copy_g_bit": ("cand_copy_g", "cand_copy_g", 5, "flip"),
}


# Partial captures: (base case, expect_offset, expect_ignore).
PART = {"ref_sum_off": ("ref_sum", 2048, []), "cand_sum_off": ("cand_sum", 2048, [[2400, 4]])}


def case_toml(name, base, what, part=None):
    _, nout, scr, a, _ = BASE[base]
    expect = 'expect = "expected/out.bin"'
    if part:
        expect += f"\nexpect_offset = {part[1]}"
        if part[2]:
            expect += f"\nexpect_ignore = {part[2]}"
    bufs = ""
    for k, f in [("a", a), ("b", "b"), ("c", "c")]:
        bufs += f'''
[[buffers]]
name = "{k}"
dtype = "f32"
shape = [{N}]
init = "file"
file = "../data/{f}.bin"
'''
    bufs += f'''
[[buffers]]
name = "out"
dtype = "f32"
shape = [{nout}]
init = "zeros"
{expect}
'''
    if scr:
        bufs += '''
[[buffers]]
name = "scr"
dtype = "f32"
shape = [1]
init = "zeros"
'''
    params = "".join(
        f'''
[[params]]
kind = "ptr"
buffer = "{k}"
'''
        for k in ["a", "b", "c", "out"] + (["scr"] if scr else [])
    )
    desc = "\n".join(textwrap.wrap(f"MISS 98 (fixtures/c_equiv_atomic/gen.py): {what}.", 92))
    return f'''name = "c_equiv_atomic_{name}"
description = """
{desc}"""
ptx = "kernel.ptx"
entry = "k"
grid = [4, 1, 1]
block = [256, 1, 1]
outputs = ["out"]
{bufs}{params}
[[variants]]
id = "good"
summary = "as written"
expect = ["clean"]
'''


def words(path):
    b = open(path, "rb").read()
    return list(struct.unpack(f"<{len(b) // 4}I", b))


def f32(w):
    return struct.unpack("<f", struct.pack("<I", w))[0]


if "--tamper" in sys.argv:
    # The pairs' captures agree at out[225], so a last-bit change there is a 1-ulp difference.
    for r, c in [("ref_sum", "cand_sum"), ("ref_sum", "cand_sum_scr"), ("ref_B", "cand_B"),
                 ("ref_B", "cand_Bn")]:
        wr, wc = words(f"{r}/expected/out.bin"), words(f"{c}/expected/out.bin")
        assert wr[J] == wc[J], (r, c, hex(wr[J]), hex(wc[J]))
    assert words("ref_copy/expected/out.bin")[JN] == 0x7FC12345
    for name, (kern, src, j, how) in TAMPER.items():
        ws = words(f"{src}/expected/out.bin")
        old = ws[j]
        if how == "flip":
            ws[j] = old ^ 1
        elif how == "far":
            ws[j] = struct.unpack("<I", struct.pack("<f", f32(old) + 1.0))[0]
        else:
            ws[j] = how
        os.makedirs(f"{name}/expected", exist_ok=True)
        open(f"{name}/expected/out.bin", "wb").write(struct.pack(f"<{len(ws)}I", *ws))
        what = (
            f"{BASE[kern][4]}. expected/ is {src}'s GB300 output with out[{j}] "
            f"{old:#010x} -> {ws[j]:#010x} (gen.py --tamper)"
        )
        open(f"{name}/kernel.ptx", "w").write(BASE[kern][0])
        open(f"{name}/case.toml", "w").write(case_toml(name, kern, what))
    for name, (src, off, ign) in PART.items():
        os.makedirs(f"{name}/expected", exist_ok=True)
        open(f"{name}/expected/out.bin", "wb").write(open(f"{src}/expected/out.bin", "rb").read()[off:])
        what = (
            f"{BASE[src][4]}. expected/ is bytes {off}.. of {src}'s GB300 output (expect_offset)"
            + (f", bytes {ign[0][0]}..{ign[0][0] + ign[0][1]} ignored (expect_ignore)" if ign else "")
            + " (gen.py --tamper)"
        )
        open(f"{name}/kernel.ptx", "w").write(BASE[src][0])
        open(f"{name}/case.toml", "w").write(case_toml(name, src, what, (src, off, ign)))
    print("ok")
    sys.exit(0)

# The inputs of the review's probes (the same seed and draws).
random.seed(9803)
f32s = lambda xs: struct.pack(f"<{len(xs)}f", *xs)  # noqa: E731
DATA = {k: f32s([random.uniform(-4, 4) for _ in range(N)]) for k in ["a", "b", "c"]}
nan = [struct.unpack("<I", struct.pack("<f", random.uniform(-4, 4)))[0] for _ in range(N)]
for k, v in {7: 0x7FC12345, 9: 0xFFC00001, 11: 0x7FA00001, 13: 0x80000000, 15: 0x7F800000}.items():
    nan[k] = v
DATA["anan"] = struct.pack(f"<{N}I", *nan)
os.makedirs("data", exist_ok=True)
for k, v in DATA.items():
    open(f"data/{k}.bin", "wb").write(v)
for name, (ptx, _, _, _, what) in BASE.items():
    os.makedirs(f"{name}/expected", exist_ok=True)
    open(f"{name}/kernel.ptx", "w").write(ptx)
    open(f"{name}/case.toml", "w").write(
        case_toml(name, name, f"{what}. expected/ is GB300's output (ptxas 13.0 -arch=sm_103a)")
    )
L1 = "L1"
NO = "none (contradicted by the captured GPU outputs)"
TOMLS = {
    "eq_l0.toml": (
        "L0 with a float atomic add into a scratch buffer: a NaN payload changed in the "
        "candidate's capture is refused (before, L0 was kept: the exemption covered the whole "
        "kernel and L0 took NaN as equal to NaN); the control without the atomic is refused too",
        "ref_copy",
        [("cand_copy_scr", "L0"), ("cand_copy_scr_nan", NO), ("cand_copy_nan", NO)],
    ),
    "eq_l1_out.toml": (
        "L1 with float atomic adds into one output (out[1024]): a 1-ulp change at another "
        "output is refused (before, L1 was kept); out[1024] itself is held to the bound only "
        "(the GPU may add in another order): its last bit passes, +inf (before, L1: only "
        "finite pairs were checked) and + 1.0 do not",
        "ref_B",
        [("cand_B", L1), ("cand_B_t1", NO), ("cand_Bn_t1", NO), ("cand_B_ulp", L1),
         ("cand_B_inf", NO), ("cand_B_far", NO)],
    ),
    "eq_l1_two.toml": (
        "two float atomic adds into one output (out[1024]) from +0, which every order adds to "
        "the same bits: out[1024] is held to its capture as the other outputs are, and its "
        "last bit changed is refused",
        "ref_B2",
        [("cand_B2", L1), ("cand_B2_t1", NO)],
    ),
    "eq_l1_scr.toml": (
        "L1 with a float atomic add into a scratch buffer: a 1-ulp change, +inf and a NaN at "
        "an output are refused (before, L1 was kept); the control without the atomic is "
        "refused too",
        "ref_sum",
        [("cand_sum_scr", L1), ("cand_sum_scr_t1", NO), ("cand_sum_scr_inf", NO),
         ("cand_sum_scr_nan", NO), ("cand_sum_t1", NO)],
    ),
    "eq_offset.toml": (
        "captures of part of the output (expect_offset, expect_ignore): the captured pair is "
        "compared where both cases captured it, each output against its own bound (before, "
        "the captures were read from byte 0 and met the bounds of other outputs)",
        "ref_sum_off",
        [("cand_sum_off", L1)],
    ),
    "eq_unproven.toml": (
        "an unproven level (an input-dependent guard; racing stores into a scratch buffer) "
        "against a capture with one word changed in its last bit is refused as contradicted "
        "(before, L0 (unproven) was kept, \"holding on the captured inputs\")",
        "ref_copy",
        [("cand_copy_g", "L0 (unproven)"), ("cand_copy_g_bit", NO)],
    ),
}
for f, (what, ref, cands) in TOMLS.items():
    head = textwrap.wrap(
        f"MISS 98 (gen.py), rounding-DAG mode: {what}.", 92, initial_indent="# ",
        subsequent_indent="# ",
    )
    s = "\n".join(head) + "\nmax_states = 200000\n"
    s += f'reference = {{ case = "{ref}", variant = "good", front = "sass", certify = true }}\n'
    for c, e in cands:
        s += (
            f'\n[[candidates]]\ncase = "{c}"\nvariant = "good"\nfront = "sass"\n'
            f'certify = true\nexpect = "{e}"\n'
        )
    open(f, "w").write(s)
print("ok")
