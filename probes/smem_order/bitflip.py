#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Flip each non-control bit of one instruction in a cubin and ask nvdisasm what it decodes to.

usage: python3 bitflip.py CUBIN LO_HEX HI_HEX [bit ...]   (bits are 0..104; default: 64..104)
Reveals modifier fields nvdisasm knows for an opcode (e.g. is there a .STRONG field on LDS?).
"""

import os
import struct
import subprocess
import sys
import tempfile

cub, lo, hi = sys.argv[1], int(sys.argv[2], 16), int(sys.argv[3], 16)
bits = [int(b) for b in sys.argv[4:]] or list(range(64, 105))
data = open(cub, "rb").read()
pat = struct.pack("<QQ", lo, hi)
off = data.find(pat)
assert off >= 0 and data.find(pat, off + 1) < 0, "pattern not unique"


def dis(buf):
    with tempfile.NamedTemporaryFile(suffix=".cubin", delete=False) as f:
        f.write(buf)
    r = subprocess.run(
        ["/usr/local/cuda-13.0/bin/nvdisasm", "-c", f.name],
        capture_output=True,
        text=True,
    )
    os.unlink(f.name)
    return r


base = dis(data).stdout
for b in bits:
    v = (hi | lo << 64) ^ (1 << (b + 64) if b < 64 else 1 << (b - 64))
    nlo, nhi = (
        (lo ^ (1 << b)) if b < 64 else lo,
        (hi ^ (1 << (b - 64))) if b >= 64 else hi,
    )
    buf = data[:off] + struct.pack("<QQ", nlo, nhi) + data[off + 16 :]
    r = dis(buf)
    out = r.stdout
    # find the changed line
    diff = [ln.strip() for ln, m in zip(out.splitlines(), base.splitlines()) if ln != m]
    print(
        f"bit {b:3d}: {diff[0] if diff else ('ERR ' + r.stderr.strip()[:80] if r.returncode else 'same')}"
    )
