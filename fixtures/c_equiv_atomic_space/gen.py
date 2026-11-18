# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MISS 142 fixture: an f32 atomic add flushes subnormals in global memory only.

Kernels (PTX, sm_103a, one CTA of 256 threads; out zero-initialised): word i = +0, then one f32
atomic add of x[i]:
  ref_shm    in shared memory, red.shared.add.f32, the word copied out (ptxas 13.0: LDS / FADD /
             ATOMS.CAST.SPIN, no flush)
  ref_gen    in shared memory through a generic address, cvta.shared then red.add.f32 (ptxas: a
             generic ATOM.E.ADD.F32.FTZ.RN, not performed there, then the QSPC.E.S fallback's
             LDS / FADD / ATOMS.CAST.SPIN, no flush)
  cand_glb   in global memory, red.global.add.f32 (REDG.E.ADD.F32.FTZ.RN, flushing)
x: 128 subnormals of both signs, 64 small normals, 32 random normals, +-0 and +-2^-126. GB300
(ptxas 13.0 -arch=sm_103a; an adversarial review round's capture, 5 plain runs identical; 20
plain and 20 scrubbed runs identical to it): the shared forms give x[i], the global one +-0 at
the 128 subnormals, so the pairs differ in 128 of 256 words. The PTX front's pairs are not L0, nor L2 (MISS 143: the flushing candidate's absolute
term exceeds the reference's, which is 0); their bound, with the flush term, holds. The SASS
front's shared forms are unproven (the CAS loop's guard depends on the data). Until MISS 142 the
PTX front flushed every f32 atomic add: both PTX pairs were proven L0.
Each eq_<ref>_<front>.toml compares ref_<ref> with cand_glb on that front.
Run from this directory: python3 gen.py [--check]."""
import os
import struct
import sys

N = 256
X_HEX = (
    "000d2ecc8024764b0001ab90804aedc40038875c805bc9cc0078b3d3805a78fd0078114980263be50066ff78"
    "8065fde000737b5e8001808d005303c48052bdfa0006ceaa806e65df004349a0800e92700004dae2801a3607"
    "002e92d48019860f00631d9e8060edee00028454805a231a007f59d080144d9500102abb804531b9003a3913"
    "805eaa9f00635bdb804087c800196e9c806af6c6003586c680477779002de864805661f4001186de8007506c"
    "006573a28017462f005d447f803034c70031dcd48053a0fd000aac7b8064b7780074c08c800b15f2002319c4"
    "802280ea0007ace180224dfd006cbc2f8022f192002a6dac80643ae4004de575803d620a007710f48067b985"
    "002f6784806a503b00696439801148fc004cabc5803a36c9006adc75807e867600213d1c8071edc0005b06a5"
    "803af3ac0005bc42805c3f9b00784c50802644a7002217f6805b0b8f000f87ec802f5c62003b14c480024b3f"
    "007bf38d801b38760064e2028069c94700784041803e2e42002d6b3c801d35be007f23f68012eb4c00369468"
    "801db6b800414baf80214240007cfca7805bfd23005b7a96806339bb0036e2e1803c7ac00027faca800bd1f9"
    "001659578034cac40042a56a800f6e44000f24fe807b7ca4005aaaf980608aa9000de342805469d300652098"
    "8069718d00192abb805c41b00041010c80119a710062ae068059e87a008ce972808cd2ec0089801c808f5cb1"
    "0080519580820fa200877d2a808c1199008deb4580888d5c008705588088980100830ce6808efb66008dccf4"
    "80890c14008afcda80846524008209678084863b00811c1780882865008bdab88082a12d0088e7ff80835b59"
    "00896593808c731b008660908081a6e0008074ec808fe42100851ba48088ff1e008cc24580878da50081df43"
    "808641cf008625b6808688d70082083080832931008ff860808f53780089d45e8084ed57008de68c8089e29e"
    "008e4b63808b8cb90089900c808887d20084268d8084fd82008e54db808e9591008e1732808d3bff008c583b"
    "80859aee008f8dc380834ba7008d7e5680853e823e68d51bbfe63594bf41aa83c0092c103de2605abe5f5850"
    "bece3f253e1baeff3fcf29d83d93909abdc7974cc0001c63c021cbae3fa1ff5bbf9e4b4bbfb20c8c3f27420f"
    "bf8334b13e8f8fd9c004416abeb82bbc3f8526e4bf9831153f45f797bf5eb6e23ead02733f9371853e1bbdc6"
    "bf8d4d8cbe9fe519bef9463abfca47ba00000000800000000000000080000000000000008000000000000000"
    "8000000000000000800000000000000080000000000000008000000000000000800000000080000080800000"
    "0080000080800000008000008080000000800000808000000080000080800000008000008080000000800000"
    "808000000080000080800000"
)
PTX_HEADER = """// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

"""
HEAD = """.version 8.8
.target sm_103a
.address_size 64

.visible .entry {name}(.param .u64 px, .param .u64 pout)
{{
    .reg .b32 %r<8>;
    .reg .f32 %f<4>;
    .reg .b64 %rd<8>;
"""
LOAD = """    mov.u32 %r1, %tid.x;
    ld.param.u64 %rd1, [px];
    cvta.to.global.u64 %rd1, %rd1;
    ld.param.u64 %rd2, [pout];
    cvta.to.global.u64 %rd2, %rd2;
    mul.wide.u32 %rd3, %r1, 4;
    add.u64 %rd4, %rd1, %rd3;
    ld.global.f32 %f1, [%rd4];
"""
SHARED = """    mov.u32 %r2, s;
    shl.b32 %r3, %r1, 2;
    add.u32 %r4, %r2, %r3;
    mov.f32 %f2, 0f00000000;
    st.shared.f32 [%r4], %f2;
    bar.sync 0;
{red}    bar.sync 0;
    ld.shared.f32 %f3, [%r4];
    add.u64 %rd5, %rd2, %rd3;
    st.global.f32 [%rd5], %f3;
    ret;
}}
"""
KERNELS = {
    "ref_shm": (
        "red.shared.add.f32",
        "    .shared .align 4 .b32 s[256];\n" + LOAD
        + SHARED.format(red="    red.shared.add.f32 [%r4], %f1;\n"),
    ),
    "ref_gen": (
        "cvta.shared, red.add.f32",
        "    .shared .align 4 .b32 s[256];\n" + LOAD
        + SHARED.format(
            red="    cvt.u64.u32 %rd6, %r4;\n    cvta.shared.u64 %rd7, %rd6;\n    red.add.f32 [%rd7], %f1;\n"
        ),
    ),
    "cand_glb": (
        "red.global.add.f32",
        LOAD
        + """    add.u64 %rd5, %rd2, %rd3;
    mov.f32 %f2, 0f00000000;
    st.global.f32 [%rd5], %f2;
    red.global.add.f32 [%rd5], %f1;
    ret;
}
""",
    ),
}
# eq file: (reference, front, expected level)
PAIRS = {
    "eq_shm_ptx": ("ref_shm", "ptx", "L1"),
    "eq_gen_ptx": ("ref_gen", "ptx", "L1"),
    "eq_shm_sass": ("ref_shm", "sass", "none (unproven)"),
    "eq_gen_sass": ("ref_gen", "sass", "none (unproven)"),
}


def xs():
    s = "".join(X_HEX)
    return [int(s[8 * i : 8 * i + 8], 16) for i in range(N)]


def flush(u):
    return u & 0x80000000 if (u & 0x7F800000) == 0 else u


def model(kernel):
    """+0 + x: x itself (an add to +0 is exact; +0 + -0 = +0), flushed in global memory."""
    out = []
    for u in xs():
        if kernel == "cand_glb":
            u = flush(u)
        out.append(0 if u == 0x80000000 else u)
    return out


def case_toml(name):
    what = KERNELS[name][0]
    space = "global memory" if name == "cand_glb" else "shared memory"
    return f'''name = "c_equiv_atomic_space_{name}"
description = """
MISS 142 (fixtures/c_equiv_atomic_space/gen.py): word i = +0, then one f32 atomic add of x[i]
in {space} ({what}), the word copied out. expected/ is GB300's output (ptxas 13.0
-arch=sm_103a)."""
ptx = "kernel.ptx"
entry = "{name}"
grid = [1, 1, 1]
block = [256, 1, 1]
outputs = ["out"]

[[buffers]]
name = "x"
dtype = "f32"
shape = [{N}]
init = "file"
file = "../data/x.bin"

[[buffers]]
name = "out"
dtype = "f32"
shape = [{N}]
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
'''


def eq_toml(tag):
    ref, front, level = PAIRS[tag]
    return f'''# MISS 142 (gen.py): {KERNELS[ref][0]} (shared memory) against red.global.add.f32 of the same value into +0, {front} front.
max_states = 200000
reference = {{ case = "{ref}", variant = "good", front = "{front}", certify = true }}

[[candidates]]
case = "cand_glb"
variant = "good"
front = "{front}"
certify = true
expect = "{level}"
'''


def main():
    if "--check" in sys.argv:
        bad = 0
        got = {}
        for k in KERNELS:
            p = os.path.join(k, "expected", "out.bin")
            got[k] = list(struct.unpack(f"<{N}I", open(p, "rb").read()))
            same = got[k] == model(k)
            bad += not same
            print(f"{k}: capture {'=' if same else 'DIFFERS from'} the model")
        for r in ("ref_shm", "ref_gen"):
            d = sum(1 for a, b in zip(got[r], got["cand_glb"]) if a != b)
            print(f"{r} against cand_glb: {d} of {N} words differ")
        sys.exit(1 if bad else 0)
    os.makedirs("data", exist_ok=True)
    with open(os.path.join("data", "x.bin"), "wb") as f:
        f.write(struct.pack(f"<{N}I", *xs()))
    for k, (_, ptx) in KERNELS.items():
        os.makedirs(os.path.join(k, "expected"), exist_ok=True)
        with open(os.path.join(k, "kernel.ptx"), "w") as f:
            f.write(PTX_HEADER + HEAD.format(name=k) + ptx)
        with open(os.path.join(k, "case.toml"), "w") as f:
            f.write(case_toml(k))
    for tag in PAIRS:
        with open(f"{tag}.toml", "w") as f:
            f.write(eq_toml(tag))


if __name__ == "__main__":
    main()
