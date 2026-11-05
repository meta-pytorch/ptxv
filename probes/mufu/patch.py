# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""mufu_raw.cubin: replace the 1/(2 pi) FMUL.RZ immediate of the sin/cos kernels by 1.0.
python patch.py mufu.cubin mufu_raw.cubin"""

import sys

b = bytearray(open(sys.argv[1], "rb").read())
old, new = (0x3E22F983).to_bytes(4, "little"), (0x3F800000).to_bytes(4, "little")
n = b.count(old)
assert n == 2, f"expected the constant in exactly the sin and cos kernels, found {n}"
open(sys.argv[2], "wb").write(b.replace(old, new))
