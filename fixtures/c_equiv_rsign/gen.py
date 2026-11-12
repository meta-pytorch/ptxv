# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MISS 97 fixture (rA's repro, bf16 inputs): writes cand/ and ref/ (kernel.ptx, data/x.bin,
case.toml) and equiv.toml. Run from this directory: python3 gen.py
  cand: y = bits(add.sat.s32(bits(f32(x)), 0x40000000))  (ptxas: VIADD + PLOP3 on R/UR.SIGN + SEL)
  ref:  y = bits(f32(x)) + 0x40000000 wrapping, except bits 0xC0000000 -> 0x80000000: what a
        slice reading the `.SIGN` predicates as `register != 0` computes for cand.
They differ where the add overflows (x >= 2: cand 0x7fffffff) and at x = -2 (cand +0, ref -0).
expected/out.bin of each case is GB300's output (the SASS of ptxas 13.0 -arch=sm_103a)."""
import os
import random
import struct

random.seed(97)
# The copyright header of every exported .ptx (fb/oss_check.sh).
PTX_HEADER = """// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

"""
HDR = PTX_HEADER + """.version 8.8
.target sm_103a
.address_size 64
.visible .entry k(.param .u64 px, .param .u64 py)
{{
    .reg .b16 %h<2>;
    .reg .b32 %r<12>;
    .reg .f32 %f<4>;
    .reg .b64 %rd<8>;
    .reg .pred %p<2>;
    ld.param.u64 %rd1, [px];
    ld.param.u64 %rd2, [py];
    cvta.to.global.u64 %rd1, %rd1;
    cvta.to.global.u64 %rd2, %rd2;
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %tid.x;
    mad.lo.u32 %r3, %r1, 256, %r2;
    mul.wide.u32 %rd3, %r3, 2;
    add.s64 %rd4, %rd1, %rd3;
    mul.wide.u32 %rd5, %r3, 4;
    add.s64 %rd6, %rd2, %rd5;
    ld.global.b16 %h1, [%rd4];
    cvt.f32.bf16 %f1, %h1;
    mov.b32 %r4, %f1;
{body}
    mov.b32 %f2, %r6;
    st.global.f32 [%rd6], %f2;
    ret;
}}
"""
BODY = {
    "cand": "    add.sat.s32 %r6, %r4, 1073741824;",
    "ref": "    add.s32 %r5, %r4, 1073741824;\n"
    "    setp.eq.s32 %p1, %r4, -1073741824;\n"
    "    selp.b32 %r6, -2147483648, %r5, %p1;",
}
N = 1024
# bf16 inputs: 2, 3, 1e30-ish, -2, 1, +-0, just below 2, NaNs, +-Inf, just below -2, then random.
edge = [0x4000, 0x4040, 0x7149, 0xC000, 0x3F80, 0x0000, 0x8000, 0x3FFF, 0x7FC0, 0x7FFF, 0x7F80,
        0xFF80, 0xBFFF, 0x4001, 0xC001, 0x7F7F]
words = edge + [random.getrandbits(16) for _ in range(N - len(edge))]
for name, body in BODY.items():
    os.makedirs(f"{name}/data", exist_ok=True)
    os.makedirs(f"{name}/expected", exist_ok=True)
    open(f"{name}/kernel.ptx", "w").write(HDR.format(body=body))
    open(f"{name}/data/x.bin", "wb").write(struct.pack(f"<{N}H", *words))
    open(f"{name}/case.toml", "w").write(f'''name = "c_equiv_rsign_{name}"
description = """
MISS 97 (fixtures/c_equiv_rsign/gen.py): {name}. y[i] = f(x[i]) on 1024 bf16 inputs, f32 bits out.
expected/ is GB300's output (ptxas 13.0 -arch=sm_103a)."""
ptx = "kernel.ptx"
entry = "k"
grid = [4, 1, 1]
block = [256, 1, 1]
outputs = ["out"]

[[buffers]]
name = "x"
dtype = "bf16"
shape = [1024]
init = "file"
file = "data/x.bin"

[[buffers]]
name = "out"
dtype = "f32"
shape = [1024]
init = "zeros"
expect = "expected/out.bin"

[[params]]
kind = "ptr"
buffer = "x"

[[params]]
kind = "ptr"
buffer = "out"

[[variants]]
id = "good"
summary = "as written"
expect = ["clean"]
''')
EQUIV = '''# MISS 97 (gen.py): add.sat.s32 on the bits (cand) against the function that a
# slice reading the `.SIGN` predicates as nonzero computes for it (ref). Elementwise mode
# sweeps every bf16 input (no sweep cache: the sweep is the check). The level is EW: the two
# differ on 16,385 of 65,536 inputs. Before the fix the slices read the `.SIGN` predicates as
# nonzero, and the level was L0.
max_states = 200000
segments = true
elementwise = true
reference = { case = "ref", variant = "good", front = "sass", certify = true }

[[candidates]]
case = "cand"
variant = "good"
front = "sass"
certify = true
expect = "EW"
'''
open("equiv.toml", "w").write(EQUIV)
print("ok")
