# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Writes kernel.ptx for probes/cvt_f16 (see case.toml): `python3 genptx.py > kernel.ptx`.
PLANES order is the output order: out[k * 32768 + i] = plane k for element i. x = in[i],
y = in[i ^ 0x4000] (the f32 -> f16 / bf16 forms; 16-bit results zero-extended); h = i, i + 32768
(the widening forms over every 16-bit code). Tag: the front-ends that model the form."""

N = 32768
PLANES = [
    ("cvt.rn.f16x2.f32", "pack", "both"),
    ("cvt.rn.bf16x2.f32", "pack", "both"),
    ("cvt.rz.f16x2.f32", "pack", "both"),
    ("cvt.rn.satfinite.f16x2.f32", "pack", "both"),
    ("cvt.rn.relu.satfinite.f16x2.f32", "pack", "both"),
    ("cvt.rn.satfinite.bf16x2.f32", "pack", "both"),
    ("cvt.rn.relu.satfinite.bf16x2.f32", "pack", "both"),
    ("cvt.rn.f16.f32", "narrow", "both"),
    ("cvt.rn.bf16.f32", "narrow", "both"),
    ("cvt.rn.sat.f16.f32", "narrow", "ptx"),
    ("cvt.rn.ftz.f16.f32", "narrow", "sass"),
]
for op, tag in [
    ("cvt.f32.f16", "both"),
    ("cvt.f32.bf16", "both"),
    ("cvt.ftz.f32.f16", "sass"),
    ("cvt.sat.f32.f16", "sass"),
    ("cvt.ftz.sat.f32.f16", "sass"),
    ("cvt.ftz.f32.bf16", "none"),
]:
    for half in (0, 1):
        PLANES.append((op, f"widen{half}", tag))
MARK = {"both": "", "ptx": "  // @SASSNO", "sass": "  // @PTXNO", "none": "  // @NONE"}

body = []
for k, (op, kind, tag) in enumerate(PLANES):
    m = MARK[tag] + (f" k{k}" if MARK[tag] else "")
    if kind == "pack":
        body.append(f"    {op} %r{20 + k}, %f1, %f2;{m}")
    elif kind == "narrow":
        body.append(f"    {op} %h{k}, %f1;{m}")
        body.append(f"    cvt.u32.u16 %r{20 + k}, %h{k};{m}")
    else:
        body.append(f"    {op} %g{k}, %w{kind[-1]};{m}")
        body.append(f"    mov.b32 %r{20 + k}, %g{k};{m}")
    body.append(f"    st.global.b32 [%rd5+{4 * N * k}], %r{20 + k};{m}")
names = "\n".join(
    f"//   {k:2}: {op}{' (h = i + 32768)' if kind == 'widen1' else ''} [{tag}]"
    for k, (op, kind, tag) in enumerate(PLANES)
)
print(f"""// probes/cvt_f16: cvt between f32 and f16 / bf16 (MISS 22; N-CVT-F16). 256 threads, 128
// iterations, i = 256 j + t: x = in[i], y = in[i ^ 0x4000] (gen.py: every sign and exponent,
// mantissas at the f16 and bf16 rounding edges), h = the 16-bit codes i and i + 32768;
// out[k * {N} + i] = plane k (16-bit results zero-extended). [both] / [ptx] / [sass]: the
// front-ends that model the form; [none]: refused by both. Variant `ptx` deletes the @PTXNO
// and @NONE lines, `sass` the @SASSNO and @NONE ones.
{names}
.version 8.8
.target sm_103a
.address_size 64

.visible .entry cvt_f16(
    .param .u64 pin,
    .param .u64 pout
)
.maxntid 256, 1, 1
{{
    .reg .pred  %p<2>;
    .reg .b16   %h<{len(PLANES)}>, %w<2>;
    .reg .f32   %f<3>, %g<{len(PLANES)}>;
    .reg .b32   %r<{20 + len(PLANES)}>;
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
    ld.global.f32   %f1, [%rd4];
    xor.b32         %r5, %r4, 16384;
    mul.wide.u32    %rd6, %r5, 4;
    add.s64         %rd7, %rd1, %rd6;
    ld.global.f32   %f2, [%rd7];
    cvt.u16.u32     %w0, %r4;
    or.b32          %r6, %r4, 32768;
    cvt.u16.u32     %w1, %r6;
{chr(10).join(body)}
    add.u32         %r4, %r4, 256;
    setp.lt.u32     %p1, %r4, {N};
    @%p1 bra        LOOP;
    ret;
}}""")
