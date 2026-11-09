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
    "e4m3": 1,
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


def _sections(d: bytes) -> dict[str, tuple[int, int]]:
    """ELF section name -> (file offset, size)."""
    shoff = u64(d, 0x28)
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", d, 0x3A)
    secs = [
        struct.unpack_from("<IIQQQQIIQQ", d, shoff + i * shentsize)
        for i in range(shnum)
    ]
    stro = secs[shstrndx][4]

    def cstr(off):
        return bytes(d[off : d.index(b"\0", off)]).decode(errors="replace")

    return {cstr(stro + sec[0]): (sec[4], sec[5]) for sec in secs}


def reqntid(data: bytes, name: str) -> tuple[int, int, int] | None:
    """The block shape kernel `name` was compiled for (EIATTR_REQNTID, 0x10, from
    `.maxntid`/`.reqntid` or a CuTeDSL launch config), or None."""
    try:
        sec = _sections(data).get(".nv.info." + name)
    except Exception:
        return None
    if not sec:
        return None
    off, size = sec
    i, end = off, off + size
    while i + 2 <= end:
        fmt, attr = data[i], data[i + 1]
        i += 2
        if fmt == 0x04:
            n = struct.unpack_from("<H", data, i)[0]
            i += 2
            if attr == 0x10 and n == 12:
                return struct.unpack_from("<III", data, i)
            i += n
        else:
            i += 2
    return None


def kparam_sizes(data: bytes, name: str) -> list[int] | None:
    """Parameter sizes of kernel `name`, by ordinal, from EIATTR_KPARAM_INFO (0x17) in its
    `.nv.info.<name>` section; None when the section or the attribute is absent."""
    try:
        sec = _sections(data).get(".nv.info." + name)
    except Exception:
        return None
    if not sec:
        return None
    off, size = sec
    d, i, end = data, off, off + size
    sizes: dict[int, int] = {}
    while i + 2 <= end:
        fmt, attr = d[i], d[i + 1]
        i += 2
        if fmt == 0x04:  # EIFMT_SVAL: u16 length, payload
            n = struct.unpack_from("<H", d, i)[0]
            i += 2
            if attr == 0x17 and n >= 12:
                _, ordinal, _poff, tmp = struct.unpack_from("<IHHI", d, i)
                sizes[ordinal] = (tmp >> 18) & 0x3FFF
            i += n
        elif fmt == 0x03:  # EIFMT_HVAL: u16 value
            i += 2
        elif fmt == 0x02:  # EIFMT_BVAL: u8 value (padded to 2)
            i += 2
        elif fmt == 0x01:  # EIFMT_NVAL: no value
            pass
        else:
            break
    if not sizes:
        return None
    return [sizes.get(k, 0) for k in range(max(sizes) + 1)]


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


def data_globals(data: bytes, kernel: str) -> list[tuple[str, int, bytes | None]]:
    """Module data globals (`__device__` variables) whose addresses `kernel`'s constant banks
    hold (R_CUDA_64 relocations of `.nv.constant<N>.<kernel>` or the module's `.nv.constant<N>`
    against a `.nv.global` / `.nv.global.init` symbol): [(symbol, size, ELF initial bytes or
    None for zeros)], in relocation order (crates/sass/src/elf.rs `Elf::kernel_globals`)."""
    d = data
    if d[:4] != b"\x7fELF":
        return []
    shoff = u64(d, 0x28)
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", d, 0x3A)
    secs = [
        struct.unpack_from("<IIQQQQIIQQ", d, shoff + i * shentsize)
        for i in range(shnum)
    ]

    def cstr(off):
        return bytes(d[off : d.index(b"\0", off)]).decode(errors="replace")

    names = [cstr(secs[shstrndx][4] + s[0]) for s in secs]
    symtab = next((s for s in secs if s[1] == 2), None)
    if symtab is None:
        return []
    strtab = secs[symtab[6]][4]

    def symbol(i):
        n, _info, _other, shndx, value, size = struct.unpack_from(
            "<IBBHQQ", d, symtab[4] + 24 * i
        )
        nm = cstr(strtab + n) or (names[shndx] if shndx < len(names) else "")
        return nm, shndx, value, size

    out, seen = [], set()
    for bank in range(1, 18):
        own = f".nv.constant{bank}.{kernel}"
        si = names.index(own) if own in names else None
        if si is None and f".nv.constant{bank}" in names:
            si = names.index(f".nv.constant{bank}")
        if si is None:
            continue
        for rs in secs:
            if rs[1] != 4 or rs[7] != si:
                continue
            for o in range(rs[4], rs[4] + rs[5], 24):
                _off, info, _add = struct.unpack_from("<QQq", d, o)
                if info & 0xFFFFFFFF != 2:
                    continue
                nm, shndx, value, size = symbol(info >> 32)
                if shndx >= len(secs) or names[shndx] not in (
                    ".nv.global",
                    ".nv.global.init",
                ):
                    continue
                sec = secs[shndx]
                if nm == names[shndx] and size == 0:
                    value, size = 0, sec[5]
                if nm in seen:
                    continue
                seen.add(nm)
                init = None
                if sec[1] != 8:  # not NOBITS
                    b = d[sec[4] + value : sec[4] + value + size]
                    init = bytes(b) + bytes(size - len(b))
                out.append((nm, size, init))
    return out
