# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# probes/fmad_sat kernel.ptx: each form reads its own triple, (a, b, c) = in[3 j ..] with
# j = (i + 1024 k) mod 8192 for form k, so ptxas cannot merge the forms' products (with one
# triple for all, it emitted one FMUL.SAT for forms 0-2, read by an add, a sub and a store).
FORMS = [
    ("mul.sat(a, b) + c", ["mul.sat.f32 P, A, B;", "add.f32 D, P, C;"]),
    ("c - mul.sat(a, b)", ["mul.sat.f32 P, A, B;", "sub.f32 D, C, P;"]),
    ("mul.sat(a, b)", ["mul.sat.f32 D, A, B;"]),
    ("add.sat(mul(a, c), b)", ["mul.f32 P, A, C;", "add.sat.f32 D, P, B;"]),
    ("sub.sat(mul(a, c), b)", ["mul.f32 P, A, C;", "sub.sat.f32 D, P, B;"]),
    ("sub.sat(b, mul(a, c))", ["mul.f32 P, A, C;", "sub.sat.f32 D, B, P;"]),
    (
        "mul.ftz.sat(a, b) + c (.ftz)",
        ["mul.ftz.sat.f32 P, A, B;", "add.ftz.f32 D, P, C;"],
    ),
    (
        "add.ftz.sat(mul.ftz(a, c), b)",
        ["mul.ftz.f32 P, A, C;", "add.ftz.sat.f32 D, P, B;"],
    ),
]
body = []
for k, (_, ops) in enumerate(FORMS):
    R = {
        "A": f"%r{10 * k + 12}",
        "B": f"%r{10 * k + 13}",
        "C": f"%r{10 * k + 14}",
        "P": f"%r{10 * k + 15}",
        "D": f"%r{10 * k + 16}",
    }
    body.append(f"    add.u32         %r{10 * k + 10}, %r1, {1024 * k};")
    body.append(f"    and.b32         %r{10 * k + 10}, %r{10 * k + 10}, 8191;")
    body.append(f"    mul.wide.u32    %rd{2 * k + 10}, %r{10 * k + 10}, 12;")
    body.append(f"    add.s64         %rd{2 * k + 11}, %rd1, %rd{2 * k + 10};")
    for j, x in enumerate("ABC"):
        body.append(f"    ld.global.f32   {R[x]}, [%rd{2 * k + 11}+{4 * j}];")
    for o in ops:
        body.append("    " + "".join(R.get(ch, ch) for ch in o))
outs = [f"%r{10 * k + 16}" for k in range(len(FORMS))]
names = "\n".join(f"//   {k}: {n}" for k, (n, _) in enumerate(FORMS))
print(f"""// probes/fmad_sat (genptx.py): a mul.sat.f32 read by plain add / sub. ptxas 13.0 and 13.3 keep
// the FMUL.SAT and emit FADD (no contraction: the clamp sits between the product and the sum),
// while a plain mul read by an add.sat / sub.sat is one FFMA.SAT. 32 CTAs x 256 threads;
// i = 256 ctaid + tid; form k reads (a, b, c) = in[3 j ..], j = (i + 1024 k) mod 8192 (gen.py);
// out[8 i + k]:
{names}
.version 8.8
.target sm_103a
.address_size 64

.visible .entry fmad_sat(
    .param .u64 pin,
    .param .u64 pout
)
.maxntid 256, 1, 1
{{
    .reg .b32   %r<{10 * len(FORMS) + 10}>;
    .reg .b64   %rd<{2 * len(FORMS) + 10}>;

    ld.param.u64    %rd1, [pin];
    ld.param.u64    %rd2, [pout];
    cvta.to.global.u64 %rd1, %rd1;
    cvta.to.global.u64 %rd2, %rd2;
    mov.u32         %r2, %ctaid.x;
    mov.u32         %r3, %tid.x;
    mad.lo.u32      %r1, %r2, 256, %r3;
    mul.wide.u32    %rd5, %r1, 32;
    add.s64         %rd6, %rd2, %rd5;
{chr(10).join(body)}
    st.global.v4.f32 [%rd6], {{{", ".join(outs[:4])}}};
    st.global.v4.f32 [%rd6+16], {{{", ".join(outs[4:])}}};
    ret;
}}""")
