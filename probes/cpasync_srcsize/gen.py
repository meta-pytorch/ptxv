# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# in.bin: 128 x 16 bytes, byte i = (i * 37 + 11) & 0xff.
open("in.bin", "wb").write(bytes((i * 37 + 11) & 0xFF for i in range(128 * 16)))
