# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MISS 98 fixture: two cases with the same kernel (y = x, f32) and the same inputs.
a/expected/out.bin is GB300's output; a_bad/expected/out.bin is that output with out[5]
changed in its last bit: a captured GPU output the model does not reproduce (as a model miss
would show). The pair is L0 in both modes, which the capture contradicts: equiv.toml
(rounding-DAG mode) and equiv_ew.toml (elementwise mode) must not claim equivalence. Run from
this directory: python3 gen.py [--bad] (--bad writes a_bad/expected/out.bin from
a/expected/out.bin)."""
import os
import random
import struct
import sys

random.seed(98)
# The copyright header of every exported .ptx (fb/oss_check.sh).
PTX_HEADER = """// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

"""
PTX = PTX_HEADER + """.version 8.8
.target sm_103a
.address_size 64
.visible .entry k(.param .u64 px, .param .u64 py)
{
    .reg .b32 %r<4>;
    .reg .f32 %f<2>;
    .reg .b64 %rd<8>;
    ld.param.u64 %rd1, [px];
    ld.param.u64 %rd2, [py];
    cvta.to.global.u64 %rd1, %rd1;
    cvta.to.global.u64 %rd2, %rd2;
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %tid.x;
    mad.lo.u32 %r3, %r1, 256, %r2;
    mul.wide.u32 %rd3, %r3, 4;
    add.s64 %rd4, %rd1, %rd3;
    add.s64 %rd5, %rd2, %rd3;
    ld.global.f32 %f1, [%rd4];
    st.global.f32 [%rd5], %f1;
    ret;
}
"""
N = 1024
BAD = 5
if "--bad" in sys.argv:
    out = bytearray(open("a/expected/out.bin", "rb").read())
    out[4 * BAD] ^= 1
    os.makedirs("a_bad/expected", exist_ok=True)
    open("a_bad/expected/out.bin", "wb").write(out)
    print("ok")
    sys.exit(0)
xs = struct.pack(f"<{N}f", *[random.uniform(-4, 4) for _ in range(N)])
for name in ["a", "a_bad"]:
    os.makedirs(f"{name}/data", exist_ok=True)
    os.makedirs(f"{name}/expected", exist_ok=True)
    open(f"{name}/kernel.ptx", "w").write(PTX)
    open(f"{name}/data/x.bin", "wb").write(xs)
    what = (
        "expected/ is GB300's output (ptxas 13.0 -arch=sm_103a)"
        if name == "a"
        else f"expected/ is a/'s GB300 output, out[{BAD}] changed in its last bit (gen.py --bad)"
    )
    open(f"{name}/case.toml", "w").write(f'''name = "c_equiv_captured_{name}"
description = """
MISS 98 (fixtures/c_equiv_captured/gen.py): y[i] = x[i] on 1024 finite f32 inputs.
{what}."""
ptx = "kernel.ptx"
entry = "k"
grid = [4, 1, 1]
block = [256, 1, 1]
outputs = ["out"]

[[buffers]]
name = "x"
dtype = "f32"
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
for f, ew in [("equiv.toml", False), ("equiv_ew.toml", True)]:
    mode = "elementwise mode" if ew else "rounding-DAG mode (no segments)"
    extra = "segments = true\nelementwise = true\n" if ew else ""
    check = (
        "the candidate's element function differs from its capture"
        if ew
        else "the candidate's simulated output differs from its capture"
    )
    open(f, "w").write(f'''# MISS 98 (gen.py), {mode}.
# The same kernel twice; the candidate's captured GPU output differs from the model's in
# one word. The pair is L0, which the capture contradicts:
# {check}.
# equiv refuses the level; before, it reported L0.
max_states = 200000
{extra}reference = {{ case = "a", variant = "good", front = "sass", certify = true }}

[[candidates]]
case = "a_bad"
variant = "good"
front = "sass"
certify = true
expect = "none (contradicted by the captured GPU outputs)"
''')
print("ok")
