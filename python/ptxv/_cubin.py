# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Cubin and tensor-map helpers for the case writer.

Copied from tools/capture.py (decode_tmap, prune, ESIZE) so the package works without the
tools directory; keep the two in sync.
"""

import struct

ESIZE = {
    "bf16": 2,
    "f16": 2,
    "u16": 2,
    "i16": 2,
    "f32": 4,
    "u32": 4,
    "i32": 4,
    "f64": 8,
    "u64": 8,
    "i64": 8,
    "u8": 1,
    "i8": 1,
}


# CUtensorMap element type codes (docs/tensormap.md) -> (dtype, bytes).
TMAP_DT = {
    0: "u8",
    11: "u8",  # 16U4_ALIGN8B, decoded in bytes
    1: "u16",
    2: "u32",
    3: "i32",
    4: "u64",
    5: "i64",
    6: "f16",
    7: "f32",
    8: "f32",
    9: "f64",
    10: "bf16",
}


def decode_tmap(raw):  # noqa: C901 (flat field-by-field decoder of the CUtensorMap layout)
    """Geometry of a 128-byte tiled CUtensorMap, or None if it does not look like one."""
    if len(raw) < 128:
        return None
    w2 = struct.unpack_from("<I", raw, 8)[0]
    if (
        w2 & 0xD or (w2 >> 11) & 3 or w2 >> 22
    ):  # im2col, unused bits 2-3, interleave, 22-31
        return None
    rank = ((w2 >> 4) & 7) + 1
    code = (w2 >> 7) & 15
    if rank > 5 or code not in TMAP_DT:
        return None
    dt = TMAP_DT[code]
    es = ESIZE[dt]
    dims = [struct.unpack_from("<I", raw, 0x20 + 4 * i)[0] + 1 for i in range(rank)]
    box = [raw[0x37 + i] + 1 for i in range(rank)]
    elem4 = code == 11  # 16U4_ALIGN8B: the case gives it in bytes (`elem4`, S-TMA)
    if elem4:
        if dims[0] % 2 or box[0] % 2:
            return None
        dims[0] //= 2
        box[0] //= 2
    hi = struct.unpack_from("<I", raw, 0x1C)[0]
    strides = []
    for i in range(1, rank):
        lo = struct.unpack_from("<I", raw, 8 + 4 * i)[0]
        s = (lo << 4) | (((hi >> (4 * (i - 1))) & 15) << 36)
        # A dimension of extent 1 is never stepped: keep its stride consistent.
        strides.append(
            s if dims[i] > 1 or s else dims[i - 1] * (strides[-1] if strides else es)
        )
    if any(b > 256 for b in box) or (strides and strides[0] < es):
        return None
    # Slots past the rank are 0 in every descriptor seen (docs/tensormap.md: the dims and
    # boxes; strides may hold junk), and a row fits its stride.
    if any(struct.unpack_from("<I", raw, 0x20 + 4 * i)[0] for i in range(rank, 5)):
        return None
    if any(raw[0x37 + i] for i in range(rank, 5)) or any(raw[0x3C:0x40]):
        return None
    if strides and dims[0] * es > strides[0]:
        return None
    if not w2 & 2:
        # Driver-built: the derived words (box bytes, swizzle repeat) are consistent.
        span = [0, 32, 64, 128][(w2 >> 13) & 3]
        bb = es
        for b in box:
            bb *= b
        if u64(raw, 0x40) != bb or u64(raw, 0x48) != (8 * span if span else 16):
            return None
    estr = struct.unpack_from("<I", raw, 0x34)[0]
    notes = []
    if estr & 0x7FFF:
        notes.append("element strides != 1 (not modeled)")
    if (w2 >> 19) & 3:
        notes.append("swizzle atom != 16B (not modeled)")
    swz = ["none", "32B", "64B", "128B"][(w2 >> 13) & 3]
    return {
        "dtype": dt,
        "dims": dims,
        "box": box,
        "strides": strides,
        "swizzle": swz,
        "notes": notes,
        "elem4": elem4,
    }


def u64(raw, off):
    return struct.unpack_from("<Q", raw, off)[0]


def elf_defines(data: bytes, name: str) -> bool:
    """Whether an ELF image defines kernel `name` (has its .text section)."""
    return data[:4] == b"\x7fELF" and (b".text." + name.encode() + b"\x00") in data


def prune(data, keep):
    """A module cubin (libtorch_cuda's hold hundreds of kernels) with the code of every entry
    kernel not in `keep` overwritten by NOPs: same layout, symbols and relocations, so it
    loads and disassembles as before, but compresses to a fraction (fixtures stay small).
    Device functions are kept (a kept kernel may call them)."""
    d = bytearray(data)
    shoff = u64(d, 0x28)
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", d, 0x3A)
    secs = [
        struct.unpack_from("<IIQQQQIIQQ", d, shoff + i * shentsize)
        for i in range(shnum)
    ]

    def cstr(off):
        return bytes(d[off : d.index(b"\0", off)]).decode()

    names = [cstr(secs[shstrndx][4] + sec[0]) for sec in secs]
    entries = set()
    for st in (sec for sec in secs if sec[1] == 2):  # SHT_SYMTAB
        strtab = secs[st[6]][4]
        for o in range(st[4], st[4] + st[5], 24):
            n, info, other = struct.unpack_from("<IBB", d, o)
            if info & 15 == 2 and other & 0x10:  # STT_FUNC, CUDA entry
                entries.add(cstr(strtab + n))
    text = {n[6:]: sec for n, sec in zip(names, secs) if n.startswith(".text.")}
    k = text[next(iter(keep))]
    nop = bytes(d[k[4] + k[5] - 16 : k[4] + k[5]])  # ptxas pads every kernel with NOPs
    for n, sec in text.items():
        if n in entries and n not in keep and sec[1] != 8:  # not SHT_NOBITS
            d[sec[4] : sec[4] + sec[5]] = nop * (sec[5] // 16)
    return bytes(d)
