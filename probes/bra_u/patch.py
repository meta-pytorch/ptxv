# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# probes/bra_u: turn a guarded `BRA` (opcode 0x947, low-word bit 32 clear) into `BRA.U`
# (low-word bit 32 set, as ptxas encodes `@!P2 BRA.U`), argv[1] -> argv[2]. argv[3]: which of
# the guarded BRAs, in code order (default: the only one). The cubins here: full / exit /
# split.ptx built with ptxas 13.1 -arch=sm_103a (*_bra.cubin) and patched (*_bra_u.cubin; for
# split, argv[3] = 1, the inner branch). Each case dir holds one parameter set and GB300's
# outputs of both binaries (`ptxv hw-worker <case> <cubin> --runs 20 --dump`, 20 identical
# runs each, 2026-11-09).
import struct, sys
b = bytearray(open(sys.argv[1], "rb").read())
hits = []
for i in range(0, len(b) - 15, 16):
    lo, hi = struct.unpack_from("<QQ", b, i)
    if lo & 0xFFF == 0x947 and (lo >> 12) & 0xF != 0x7 and hi & 0xFFFFFFFF == 0x03800000:
        hits.append(i)
if len(sys.argv) > 3:
    i = hits[int(sys.argv[3])]
else:
    assert len(hits) == 1, hits
    i = hits[0]
lo, = struct.unpack_from("<Q", b, i)
struct.pack_into("<Q", b, i, lo | (1 << 32))
open(sys.argv[2], "wb").write(b)
