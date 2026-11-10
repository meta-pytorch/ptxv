# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Case writer: tensor maps whose base lies outside every allocation (negative byte_offset)."""

import struct

from ptxv._casewriter import _ragged_shift


def _tmap(addr: int, dims: list[int], strides: list[int], box: list[int]) -> bytes:
    """A driver-style 128-byte bf16 tiled CUtensorMap (128B swizzle) at global address addr."""
    rank = len(dims)
    raw = bytearray(128)
    struct.pack_into("<Q", raw, 0, addr)
    w2 = ((rank - 1) << 4) | (10 << 7) | (3 << 13)  # bf16, 128B swizzle
    struct.pack_into("<I", raw, 8, w2)
    hi = 0
    for i, s in enumerate(strides, start=1):
        struct.pack_into("<I", raw, 8 + 4 * i, (s >> 4) & 0xFFFFFFFF)
        hi |= ((s >> 36) & 15) << (4 * (i - 1))
    struct.pack_into("<I", raw, 0x1C, hi)
    for i, d in enumerate(dims):
        struct.pack_into("<I", raw, 0x20 + 4 * i, d - 1)
    for i, b in enumerate(box):
        raw[0x37 + i] = b - 1
    bb = 2
    for b in box:
        bb *= b
    struct.pack_into("<Q", raw, 0x40, bb)
    struct.pack_into("<Q", raw, 0x48, 8 * 128)
    return bytes(raw)


def _comp_of(lo: int, hi: int):
    def comp_of(v):
        return (0, v - lo) if lo <= v < hi else None

    return comp_of


def test_ragged_ptr_shift():
    # quack create_ragged_tensor_for_tma(ptr_shift=True): base = O - 2^30 rows of 128 B.
    o = 0xE43A47C15E00
    raw = _tmap(
        o - (1 << 37), [64, 1 << 30, 1, 2**31 - 1], [128, 128, 128], [64, 128, 1, 1]
    )
    r = _ragged_shift(raw, 0, o - (1 << 37), _comp_of(o, o + 89600))
    assert r is not None
    g, ci, off, shift = r
    assert (ci, off, shift) == (0, -(1 << 37), 1 << 37)
    assert g["dims"][1] == 1 << 30


def test_base_just_before_allocation():
    # A box offset folded into the base: 1024 B before the buffer.
    buf = 0x7F0000100000
    raw = _tmap(buf - 1024, [64, 256], [128], [64, 64])
    g, ci, off, gap = _ragged_shift(raw, 0, buf - 1024, _comp_of(buf, buf + 32768))
    assert (ci, off) == (0, -1024)


def test_unrelated_address_unresolved():
    raw = _tmap(0x10000, [64, 256], [128], [64, 64])
    assert (
        _ragged_shift(raw, 0, 0x10000, _comp_of(0x7F0000100000, 0x7F0000108000)) is None
    )
