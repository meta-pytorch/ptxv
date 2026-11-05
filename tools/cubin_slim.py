# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Shrink a library cubin (libtorch_cuda modules hold hundreds of kernels, 2-11 MB) to the
# kernels a case launches: every per-function section (.text.F, .nv.constant0.F,
# .nv.shared.F, .rel*.text.F, ...) of any other function F is emptied (size 0); section headers,
# the symbol table and the global .nv.info stay as they are, so every index remains valid.
# Debug frames (unwind info) are emptied as well. Section data is re-laid out and the program headers are recomputed. Functions the kept ones
# call (relocations of their code against another function's symbol) are kept too.
#   python3 tools/cubin_slim.py in.cubin out.cubin KERNEL [KERNEL ...]
import struct
import sys


def main(src, dst, keep_fns):  # noqa: C901 (single ELF rewrite pass over shared section tables)
    b = bytearray(open(src, "rb").read())
    assert b[:6] == b"\x7fELF\x02\x01"
    (shoff,) = struct.unpack_from("<Q", b, 0x28)
    (phoff,) = struct.unpack_from("<Q", b, 0x20)
    phentsize, phnum, shentsize, shnum, shstrndx = struct.unpack_from("<HHHHH", b, 0x36)
    sh = []
    for i in range(shnum):
        o = shoff + i * shentsize
        name, typ, flags, addr, off, size, link, info, align, entsize = (
            struct.unpack_from("<IIQQQQIIQQ", b, o)
        )
        sh.append(
            {
                "name": name,
                "typ": typ,
                "flags": flags,
                "addr": addr,
                "off": off,
                "size": size,
                "link": link,
                "info": info,
                "align": align,
                "entsize": entsize,
            }
        )
    stroff = sh[shstrndx]["off"]

    def cstr(o):
        return bytes(b[o : b.index(0, o)]).decode()

    names = [cstr(stroff + s["name"]) for s in sh]
    fns = {n[len(".text.") :] for n in names if n.startswith(".text.")}
    # Symbol table: names by index, and the section each symbol lives in.
    symtab = next(i for i, s in enumerate(sh) if s["typ"] == 2)
    st, ss = sh[symtab], sh[sh[symtab]["link"]]
    syms = []
    for k in range(st["size"] // 24):
        nm, info, other, shndx, val, size = struct.unpack_from(
            "<IBBHQQ", b, st["off"] + 24 * k
        )
        syms.append((cstr(ss["off"] + nm), shndx))
    # Closure over relocations: a kept function's relocations naming another function's symbol.
    keep = set(keep_fns)
    while True:
        add = set()
        for i, _n in enumerate(names):
            if sh[i]["typ"] not in (4, 9):  # RELA / REL
                continue
            tgt = names[sh[i]["info"]]
            if not (tgt.startswith(".text.") and tgt[len(".text.") :] in keep):
                continue
            ent = 24 if sh[i]["typ"] == 4 else 16
            for k in range(sh[i]["size"] // ent):
                (info,) = struct.unpack_from("<Q", b, sh[i]["off"] + ent * k + 8)
                sname = syms[info >> 32][0]
                if sname in fns and sname not in keep:
                    add.add(sname)
        if not add:
            break
        keep |= add

    def owner(n):
        # Longest function-name suffix of a section name.
        best = None
        for f in fns:
            if n.endswith("." + f) and (best is None or len(f) > len(best)):
                best = f
        return best

    # .nv.info.F stays (cuobjdump rejects a cubin whose function lacks it; it is small).
    drop = [
        i
        for i, n in enumerate(names)
        if (o := owner(n)) is not None
        and o not in keep
        and not n.startswith(".nv.info.")
        or n.endswith("debug_frame")
    ]
    # Symbols in emptied sections: value and size 0 (nvdisasm checks them against the section).
    # The driver rejects a module whose functions have no code: an emptied .text.F keeps its
    # first instructions (a stub that is never run).
    dset = set(drop)
    size = {
        i: (min(sh[i]["size"], 128) if names[i].startswith(".text.") else 0)
        if i in dset
        else sh[i]["size"]
        for i in range(shnum)
    }
    for k, (nm, shndx) in enumerate(syms):
        if shndx in dset:
            z = size[shndx] if nm == names[shndx][len(".text.") :] else 0
            struct.pack_into("<QQ", b, st["off"] + 24 * k + 8, 0, z)
    # Lay out kept section data in original order of file offset.
    out = bytearray(b[:0x40])
    order = sorted(range(1, shnum), key=lambda i: sh[i]["off"])
    newoff = {}
    for i in order:
        s = sh[i]
        if size[i] == 0 or s["typ"] == 8:  # emptied, or NOBITS (no file data)
            newoff[i] = len(out)
            continue
        a = max(s["align"], 1)
        while len(out) % a:
            out.append(0)
        newoff[i] = len(out)
        out += b[s["off"] : s["off"] + size[i]]
    # Program headers: each segment spans the sections it held.
    phs = []
    for k in range(phnum):
        o = phoff + k * phentsize
        typ, flags, off, va, pa, filesz, memsz, align = struct.unpack_from(
            "<IIQQQQQQ", b, o
        )
        phs.append([typ, flags, off, va, pa, filesz, memsz, align])
    new_phoff = None
    while len(out) % 8:
        out.append(0)
    new_phoff = len(out)
    out += b"\0" * (phnum * phentsize)
    for p in phs:
        typ, flags, off, va, pa, filesz, memsz, align = p
        if off == phoff:  # PHDR / the LOAD of the header table
            p[2] = new_phoff
            continue
        inside = [
            i
            for i in range(1, shnum)
            if sh[i]["typ"] != 8
            and off <= sh[i]["off"]
            and sh[i]["off"] + sh[i]["size"] <= off + filesz
            and sh[i]["size"]
        ]
        kept = [i for i in inside if size[i]]
        if not kept:
            p[2], p[5] = new_phoff, 0
            p[6] = 0 if memsz == filesz else memsz
            continue
        lo = min(newoff[i] for i in kept)
        hi = max(newoff[i] + size[i] for i in kept)
        p[2], p[5] = lo, hi - lo
        p[6] = hi - lo + (memsz - filesz)
    for k, p in enumerate(phs):
        struct.pack_into("<IIQQQQQQ", out, new_phoff + k * phentsize, *p)
    while len(out) % 8:
        out.append(0)
    new_shoff = len(out)
    for i, s in enumerate(sh):
        off = newoff.get(i, 0)
        out += struct.pack(
            "<IIQQQQIIQQ",
            s["name"],
            s["typ"],
            s["flags"],
            s["addr"],
            off,
            size[i],
            s["link"],
            s["info"],
            s["align"],
            s["entsize"],
        )
    struct.pack_into("<Q", out, 0x20, new_phoff)
    struct.pack_into("<Q", out, 0x28, new_shoff)
    open(dst, "wb").write(out)
    print(
        f"{src}: {len(b)} -> {len(out)} bytes, kept {sorted(keep)[:4]}..., "
        f"dropped {len(drop)} sections"
    )


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
