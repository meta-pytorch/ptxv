# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""m.bin for probes/atom_bf16x4: m[k][g] (k = 0, 1; 2048 8-byte slots each) = the a lanes of
probes/bf16x2's triples 2g and 2g + 1 (the case reads ../bf16x2/in.bin)."""

import struct

w = struct.unpack("<12288I", open("../bf16x2/in.bin", "rb").read())
a = [w[3 * i] for i in range(4096)]
open("m.bin", "wb").write(struct.pack("<8192I", *(a + a)))
