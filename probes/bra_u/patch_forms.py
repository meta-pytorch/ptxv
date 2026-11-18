# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# probes/bra_u: the thread-predicate operand forms of ptxas's guarded `@!P0 BRA` (opcode 0x947,
# guard !P0, operand PT), argv[1] -> argv[2]_{pp,pp_u,upt_u}.cubin; argv[3]: which guarded
# BRA, in code order (default: the only one; split: 1, the inner branch, as for patch.py).
#   pp:    BRA !P0, `(label)          no guard, the predicate as the operand (per lane)
#   pp_u:  BRA.U !P0, `(label)
#   upt_u: BRA.U !P0, UPT, `(label)   (opcode 0x547, uniform operand UPT: cuBLAS's form)
# Fields, from ptxas output: low word bits 0-11 opcode, 12-15 guard (bit 15 negates, 7 = PT),
# 24-27 the uniform operand (0x547 only), bit 32 .U; high word bits 23-25 the predicate
# operand, bit 26 its negation, bit 27 set in every 0x547 seen. The *_{pp,pp_u,upt_u}.cubin
# here are full / exit / split_bra.cubin through this script (cuobjdump prints the three
# forms). Each case dir also holds GB300's outputs of the three binaries (gb300_pp.bin,
# gb300_pp_u.bin, gb300_upt_u.bin; `ptxv hw-worker <case> <cubin> --runs 20 --dump`, 20
# identical runs each, 2026-11-11).
import struct, sys
src, out = sys.argv[1], sys.argv[2]
b0 = bytearray(open(src, "rb").read())
hits = []
for i in range(0, len(b0) - 15, 16):
    lo, hi = struct.unpack_from("<QQ", b0, i)
    if lo & 0xFFF == 0x947 and (lo >> 12) & 0xF != 0x7 and hi & 0xFFFFFFFF == 0x03800000:
        hits.append(i)
if len(sys.argv) > 3:
    i = hits[int(sys.argv[3])]
else:
    assert len(hits) == 1, hits
    i = hits[0]
lo0, hi0 = struct.unpack_from("<QQ", b0, i)
assert (lo0 >> 12) & 0xF == 0x8, hex(lo0)  # @!P0
assert (lo0 >> 32) == 0
off = lo0 & 0xFFFF0000


def form(op, upp, u, bit27):
    lo = op | 0x7 << 12 | off | upp << 24 | u << 32
    hi = (hi0 & ~(0x1F << 23)) | 0 << 23 | 1 << 26 | bit27 << 27
    return lo, hi


forms = {
    "pp": form(0x947, 0, 0, 0),
    "pp_u": form(0x947, 0, 1, 0),
    "upt_u": form(0x547, 7, 1, 1),
}
for name, (lo, hi) in forms.items():
    b = bytearray(b0)
    struct.pack_into("<QQ", b, i, lo, hi)
    open(f"{out}_{name}.cubin", "wb").write(b)
