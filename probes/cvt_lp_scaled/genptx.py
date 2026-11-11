# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""kernel.ptx for probes/cvt_lp_scaled: cvt.rn[.relu].scaled::n2::ue8m0.bf16x2.<f8/f6/f4>x2 d, a, s
(ptxas 13.3, PTX 9.3; SASS F2FP[.RELU].BF16.<fmt>.UNPACK_B d, a, s.H0). in[2i] = a (codes),
in[2i + 1] = s (two ue8m0 scales). out[k * 65536 + i] = plane k."""

N = 65536
P = []
for t in ["e4m3", "e5m2", "e2m3", "e3m2", "e2m1"]:
    for m in ["", ".relu"]:
        P.append((f"cvt.rn{m}.scaled::n2::ue8m0.bf16x2.{t}x2", t))
body = []
for k, (op, t) in enumerate(P):
    src = "%c1" if t == "e2m1" else "%h1"
    body.append(f"    {op} %r{20 + k}, {src}, %h2;")
    body.append(f"    st.global.b32 [%rd5+{4 * N * k}], %r{20 + k};")
names = "\n".join(f"//   {k:2}: {op}" for k, (op, t) in enumerate(P))
print(f"""// probes/cvt_lp_scaled (genptx.py). 256 threads x 256 iterations, i = 256 j + t.
{names}
.version 9.3
.target sm_103a
.address_size 64

.visible .entry cvt_lp_scaled(
    .param .u64 pin,
    .param .u64 pout
)
.maxntid 256, 1, 1
{{
    .reg .pred  %p<2>;
    .reg .b8    %c<2>;
    .reg .b16   %h<4>;
    .reg .b32   %r<{20 + len(P)}>;
    .reg .b64   %rd<8>;

    ld.param.u64    %rd1, [pin];
    ld.param.u64    %rd2, [pout];
    cvta.to.global.u64 %rd1, %rd1;
    cvta.to.global.u64 %rd2, %rd2;
    mov.u32         %r4, %tid.x;
LOOP:
    mul.wide.u32    %rd3, %r4, 8;
    add.s64         %rd4, %rd1, %rd3;
    mul.wide.u32    %rd6, %r4, 4;
    add.s64         %rd5, %rd2, %rd6;
    ld.global.v2.u32 {{%r2, %r3}}, [%rd4];
    cvt.u16.u32     %h1, %r2;
    cvt.u8.u32      %c1, %r2;
    cvt.u16.u32     %h2, %r3;
{chr(10).join(body)}
    add.u32         %r4, %r4, 256;
    setp.lt.u32     %p1, %r4, {N};
    @%p1 bra        LOOP;
    ret;
}}""")
