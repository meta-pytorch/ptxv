# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Variants of the LDS write-back-order litmus (probes/lds_waw). Each lane's region is 128
bytes at base + lane * s1 (first, 128-bit load) / base + lane * s2 (second, 32-bit load);
s = 0 broadcasts, s = 128 puts all 32 lanes on the same banks (32-way conflict). usage:
gen.py NAME S1 S2 OUT.ptx"""

import sys

name, s1, s2, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
xs = "\n".join(
    f"    ld.volatile.shared.u32 %x{i}, [%r10+{32 + 4 * i}];" for i in range(17)
)
adds = "\n".join(f"    add.u32 %r20, %r20, %x{i};" for i in range(17))
ptx = f"""// probes/lds_waw variant {name}: `LDS.128 Rn, [a]` then `LDS Rn+2|3, [b]` with no wait
// between them (ptxas's own schedule under .maxnreg 24). First load lane stride {s1} bytes,
// second {s2} (128: 32-way bank conflict, 0: broadcast). w[k] of lane l's region = 0x100000 *
// warp + 0x1000 * l + k. out[3 * gtid + {{0, 1, 2}}] = {{first.x, first.y + sum, second}}.
.version 9.0
.target sm_103a
.address_size 64
.visible .entry lds_waw(.param .u64 out)
.maxntid 128, 1, 1
.maxnreg 24
{{
    .reg .pred  %p<4>;
    .reg .b32   %r<32>;
    .reg .b32   %x<17>;
    .reg .b64   %rd<8>;
    .shared .align 16 .b8 w[16384];
    mov.u32         %r0, %tid.x;
    shr.u32         %r1, %r0, 5;
    and.b32         %r2, %r0, 31;
    mov.u32         %r3, w;
    shl.b32         %r9, %r1, 12;
    add.u32         %r10, %r3, %r9;
    // fill this warp's 4 KB: word k of lane l's 128-byte region
    mov.u32         %r21, 0;
FILL:
    shl.b32         %r22, %r21, 5;
    add.u32         %r22, %r22, %r2;
    shl.b32         %r23, %r22, 2;
    add.u32         %r23, %r10, %r23;
    shr.u32         %r24, %r22, 5;
    and.b32         %r25, %r24, 31;
    shr.u32         %r26, %r22, 10;
    shl.b32         %r27, %r1, 20;
    shr.u32         %r28, %r22, 5;
    shl.b32         %r28, %r28, 0;
    // word index g = 32 * r21 + lane; region l = g / 32, k = g % 32
    and.b32         %r29, %r22, 31;
    shl.b32         %r30, %r28, 12;
    add.u32         %r30, %r30, %r27;
    add.u32         %r30, %r30, %r29;
    st.shared.u32   [%r23], %r30;
    add.u32         %r21, %r21, 1;
    setp.lt.u32     %p1, %r21, 32;
    @%p1 bra        FILL;
    bar.warp.sync   0xffffffff;
{xs}
    mul.lo.u32      %r12, %r2, {s1};
    add.u32         %r12, %r10, %r12;
    mul.lo.u32      %r13, %r2, {s2};
    add.u32         %r13, %r10, %r13;
    ld.volatile.shared.v4.u32 {{%r4, %r5, %r6, %r7}}, [%r12];
    ld.volatile.shared.u32 %r8, [%r13+16];
    mov.u32 %r20, %r5;
{adds}
    ld.param.u64    %rd0, [out];
    cvta.to.global.u64 %rd1, %rd0;
    mov.u32         %r15, %ctaid.x;
    mad.lo.u32      %r16, %r15, 128, %r0;
    mul.lo.u32      %r16, %r16, 3;
    mul.wide.u32    %rd2, %r16, 4;
    add.u64         %rd3, %rd1, %rd2;
    st.global.u32   [%rd3], %r4;
    st.global.u32   [%rd3+4], %r20;
    st.global.u32   [%rd3+8], %r8;
    ret;
}}
"""
open(out, "w").write(ptx)
