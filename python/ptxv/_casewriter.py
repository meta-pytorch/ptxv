# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Writing captured launches as a ptxv case (case.toml + data/ + expected/ + cubins).

The format is tools/capture.py's (docs/capture.md). The difference is where bytes come from:
here every launch's referenced allocations were copied right before and after it, so a
buffer's initial bytes are those before the first launch of the group that reaches it, and
its expected bytes those after the last one.

Buffers: intervals are named tensors' storages, torch caching-allocator blocks (live before
the region, or allocated in it), and whole driver allocations no block covers; overlapping
intervals merge. A component whose references all fall inside one named tensor is that tensor
(dtype, shape); otherwise it is the whole component as u8.

Outputs: buffers the group changed, plus buffers allocated inside the region (a kernel may
write the bytes they already held). Outputs allocated in the region, and `write_only` ones,
start as zeros; outputs that existed before keep their bytes (in-place updates).
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import re
import shutil
import struct
import tempfile
from pathlib import Path
from typing import Any, Sequence

from . import _cupti
from ._cubin import decode_tmap, elf_defines, ESIZE, prune as prune_cubin, u64
from ._util import cache_dir, tree_hash


class CaseError(RuntimeError):
    pass


def _read(path: Path) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def write_case(
    launches: Sequence[Any],
    cap: Any,
    *,
    outputs: Sequence[str] | None = None,
    write_only: Sequence[str] = (),
    ptx: bool = False,
    prune: bool = True,
    rename: dict[str, str] | None = None,
    name: str = "captured",
) -> Path:
    """Write the case; returns its content-addressed directory under $PTXV_CACHE/cases."""
    tmp = Path(tempfile.mkdtemp(prefix="case-", dir=cache_dir("cases")))
    try:
        _write(launches, cap, tmp, outputs, write_only, ptx, prune, rename or {}, name)
        h = tree_hash(tmp)[:24]
        dst = cache_dir("cases") / h
        if dst.exists():
            shutil.rmtree(tmp)
        else:
            tmp.rename(dst)
        return dst
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def _write(  # noqa: C901 (case writer: one linear pass over shared capture state)
    launches, cap, out: Path, outputs, write_only, ptx, prune, rename, name
) -> None:
    warnings: list[str] = []
    snapdir = cap.dir / "snap"

    # ---------------------------------------------------------------- snapshots
    allocs: dict[int, int] = {}  # base -> size
    first_pre: dict[int, Path] = {}
    last_post: dict[int, Path] = {}
    seen_post: dict[int, tuple[int, Path]] = {}  # base -> (launch index, post file)
    for _i, la in enumerate(launches):
        if la.snap is None:
            warnings.append(
                f"launch {la.index} ({la.short_name}) has no snapshot: buffers only it reaches start as zeros"
            )
            continue
        for f in glob.glob(str(snapdir / f"s{la.snap}_pre_*.bin")):
            base = int(f.rsplit("_", 1)[1][:-4], 16)
            allocs[base] = os.path.getsize(f)
            if base not in first_pre:
                first_pre[base] = Path(f)
            elif base in seen_post and _read(Path(f)) != _read(seen_post[base][1]):
                warnings.append(
                    f"allocation {base:#x} changed between launch {seen_post[base][0]} and launch "
                    f"{la.index} outside this group (memcpy, another stream, or a launch not in it)"
                )
            post = snapdir / f"s{la.snap}_post_{base:x}.bin"
            if post.exists():
                last_post[base] = post
                seen_post[base] = (la.index, post)

    def alloc_of(a: int) -> int | None:
        for b, z in allocs.items():
            if b <= a < b + z:
                return b
        return None

    img_cache: dict[Path, bytes] = {}

    def image(p: Path) -> bytes:
        if p not in img_cache:
            img_cache[p] = _read(p)
        return img_cache[p]

    def bytes_at(start: int, size: int, which: dict[int, Path]) -> bytes | None:
        b = alloc_of(start)
        if b is None or b not in which:
            return None
        d = image(which[b])
        o = start - b
        chunk = d[o : o + size]
        if len(chunk) < size:
            return None
        return chunk

    # ---------------------------------------------------------------- intervals
    ivs = []  # (start, end, kind, name)
    for n, t in cap.names.items():
        ivs.append((t.ptr, t.ptr + t.nbytes, "named", n))
    for a, z, kind in cap.blocks:
        ivs.append((a, a + z, kind, None))
    for b, z in allocs.items():
        if not any(b <= s < b + z for s, _, _, _ in ivs):
            ivs.append((b, b + z, "alloc", None))
    ivs.sort()
    comps: list[list] = []
    for iv in ivs:
        if comps and iv[0] < comps[-1][1]:
            comps[-1][1] = max(comps[-1][1], iv[1])
            comps[-1][2].append(iv)
        else:
            comps.append([iv[0], iv[1], [iv]])

    def comp_of(v: int):
        for i, (s, e, _) in enumerate(comps):
            if s <= v < e:
                return i, v - s
        for i, (s, e, _) in enumerate(comps):
            if v == e:
                return i, v - s
        return None

    def near_alloc(v: int) -> bool:
        return any(s - (1 << 30) <= v < e + (1 << 30) for s, e, _ in comps)

    # ---------------------------------------------------------------- parameters
    refs: dict[int, list] = {}
    tmap_refs: dict[int, set] = {}
    unresolved: list[dict] = []

    def note_ref(ci, off, n=1):
        refs.setdefault(ci, []).append((off, n))

    def scan_blob(raw: bytes, li: int, pi: int):
        fields = []
        off = 0
        while off + 8 <= len(raw):
            v = u64(raw, off)
            loc = comp_of(v) if v else None
            if loc and off % 64 == 0 and off + 128 <= len(raw):
                g = decode_tmap(raw[off : off + 128])
                if g and raw[off + 8 : off + 16] != bytes(8):
                    ci, o = loc
                    tbytes = max(
                        g["dims"][-1]
                        * (g["strides"][-1] if g["strides"] else ESIZE[g["dtype"]]),
                        1,
                    )
                    note_ref(ci, o, tbytes)
                    tmap_refs.setdefault(ci, set()).add(g["dtype"])
                    n = 64 if raw[off + 8] & 2 else 128
                    fields.append((off, ("tmap", ci, o, g, raw[off : off + n])))
                    off += n
                    continue
            if loc:
                note_ref(*loc)
                fields.append((off, ("ptr",) + loc))
            elif v and near_alloc(v):
                unresolved.append(
                    {"launch": li, "param": pi, "offset": off, "value": hex(v)}
                )
            off += 8
        return fields

    plaunches = []
    first_ref: dict[int, int] = {}
    for li, la in enumerate(launches):
        params = []
        before = set(refs)
        for pi, p in enumerate(la.record["params"]):
            raw = bytes.fromhex(p["hex"])
            if len(raw) == 8 and u64(raw, 0) and comp_of(u64(raw, 0)):
                loc = comp_of(u64(raw, 0))
                note_ref(*loc)
                params.append(("ptr",) + loc)
                continue
            if len(raw) == 128 and not p.get("packed"):
                f = scan_blob(raw, li, pi)
                if len(f) == 1 and f[0][0] == 0 and f[0][1][0] == "tmap":
                    params.append(f[0][1])
                    continue
            if len(raw) == 4:
                params.append(("u32", struct.unpack("<I", raw)[0]))
                continue
            params.append(("bytes", raw, scan_blob(raw, li, pi)))
        for ci in set(refs) - before:
            first_ref.setdefault(ci, li)
        plaunches.append((la, params))

    # ---------------------------------------------------------------- buffers
    buffers: list[dict] = []
    comp_buf: dict[int, tuple[int, int]] = {}
    nws = 0
    for ci in sorted(refs):
        s, e, members = comps[ci]
        nm = [m for m in members if m[2] == "named"]
        rs = refs[ci]
        solo = None
        for m in nm:
            if (
                all(m[0] <= s + o and s + o + max(k, 1) - 1 <= m[1] for o, k in rs)
                and cap.names[m[3]].whole
            ):
                solo = m
                break
        if solo:
            ti = cap.names[solo[3]]
            start, size, bname = ti.ptr, ti.nbytes, solo[3]
            dt, shape = ti.dtype, ti.shape
            if size % ESIZE[dt]:
                dt, shape = "u8", [size]
        else:
            start, size = s, e - s
            bname = nm[0][3] if nm else None
            dt, shape = "u8", [size]
        if bname is None:
            bname = f"W{nws}"
            nws += 1
        tds = tmap_refs.get(ci, set())
        if tds:
            td = sorted(tds)[0]
            if len(tds) > 1:
                warnings.append(
                    f"{bname}: tensor maps of several element types {sorted(tds)}"
                )
            if ESIZE[dt] != ESIZE[td]:
                if dt == "u8" and size % ESIZE[td] == 0:
                    dt, shape = td, [size // ESIZE[td]]
                else:
                    warnings.append(f"{bname}: dtype {dt} but a tensor map reads {td}")
        kinds = {m[2] for m in members}
        named_before = any(cap.names[m[3]].before for m in nm)
        fresh = "during" in kinds and "pre" not in kinds and not named_before
        init = bytes_at(start, size, first_pre)
        final = bytes_at(start, size, last_post)
        if init is None:
            warnings.append(f"{bname}: no snapshot covers it; starts as zeros")
        buffers.append(
            {
                "name": bname,
                "dtype": dt,
                "shape": shape,
                "start": start,
                "size": size,
                "init": init,
                "final": final,
                "fresh": fresh,
                "named": bool(nm),
                "changed": init is not None and final is not None and init != final,
                "first_ref": first_ref.get(ci, 0),
            }
        )
        comp_buf[ci] = (len(buffers) - 1, start - s)

    # Unique names, then renames.
    taken: set[str] = set()
    for b in buffers:
        base, k = b["name"], 1
        while b["name"] in taken:
            b["name"] = f"{base}_{k}"
            k += 1
        taken.add(b["name"])
    for b in buffers:
        b["name"] = rename.get(b["name"], b["name"])
    if len({b["name"] for b in buffers}) != len(buffers):
        raise CaseError(
            f"rename makes buffer names collide: {[b['name'] for b in buffers]}"
        )

    # Outputs.
    if outputs is not None:
        outs = [b for n in outputs for b in buffers if b["name"] == n]
        missing = set(outputs) - {b["name"] for b in outs}
        if missing:
            raise CaseError(
                f"outputs {sorted(missing)} are not buffers of this case: {[b['name'] for b in buffers]}"
            )
    else:
        outs = [
            b
            for b in buffers
            if b["changed"] or (b["fresh"] and b["final"] is not None)
        ]
        outs.sort(key=lambda b: (not b["changed"], not b["named"], -b["first_ref"]))
    if not outs:
        raise CaseError(
            "no output: no buffer changed and none was allocated in the capture region"
        )
    for b in outs:
        if b["final"] is None:
            raise CaseError(f"output {b['name']} has no post-launch snapshot")
        b["is_out"] = True
        if b["fresh"] or b["name"] in write_only:
            b["init"] = None

    def buf_ref(ci: int, off: int):
        bi, delta = comp_buf[ci]
        return buffers[bi]["name"], off - delta

    # ---------------------------------------------------------------- cubins
    mods = [_cupti.MODULE_DIR / f"{h}.cubin" for h in dict.fromkeys(cap.modules)]
    mods = [m for m in mods if m.exists()]
    names = {la.name for la, _ in plaunches}
    cubin_of: dict[str, Path] = {}
    for n in names:
        c = [m for m in mods if elf_defines(image(m), n)]
        if not c:
            raise CaseError(
                f"no captured module defines {n}: its module was loaded before ptxv's hook. "
                "Import ptxv before the program's first CUDA work, or run under `python -m ptxv run`."
            )
        cubin_of[n] = min(c, key=os.path.getsize)
    ptx_of = _triton_ptx(names, {n: image(p) for n, p in cubin_of.items()})
    use_ptx = ptx and len(names) == 1 and next(iter(names)) in ptx_of
    by_hash: dict[str, str] = {}
    cubin_file: dict[str, str] = {}
    for la, _ in plaunches:
        src = cubin_of[la.name]
        data = image(src)
        h = hashlib.sha1(data).hexdigest()
        if h not in by_hash:
            by_hash[h] = "jit.cubin" if use_ptx else f"k{len(by_hash)}.cubin"
            keep = {m for m in names if cubin_of[m] == src}
            (out / by_hash[h]).write_bytes(
                data if not prune else prune_cubin(data, keep)
            )
        cubin_file[la.name] = by_hash[h]
    if len(names) == 1 and next(iter(names)) in ptx_of:
        src = ptx_of[next(iter(names))]
        m = re.search(r"^\.version (\d+)\.(\d+)", src, re.M)
        if m and (int(m.group(1)), int(m.group(2))) > (9, 0):
            src = src.replace(m.group(0), ".version 9.0", 1)
        (out / ("kernel.ptx" if use_ptx else "jit.ptx")).write_text(src)

    # ---------------------------------------------------------------- case.toml
    q = json.dumps

    def toml_tmap(prefix, ci, o, g, raw, extra=""):
        b, bo = buf_ref(ci, o)
        return (
            f'{prefix}{extra}kind = "tensormap"\nbuffer = {q(b)}\nbyte_offset = {bo}\n'
            f"box = {g['box']}\ndims = {g['dims']}\nstrides = {g['strides']}\n"
            f'swizzle = "{g["swizzle"]}"\nraw = "{raw.hex()}"\n'
            + ("elem4 = true\n" if g.get("elem4") else "")
        )

    def toml_params(params, key):
        s = []
        for p in params:
            if p[0] == "ptr":
                b, bo = buf_ref(p[1], p[2])
                s.append(
                    f'[[{key}]]\nkind = "ptr"\nbuffer = {q(b)}\nbyte_offset = {bo}\n'
                )
            elif p[0] == "tmap":
                s.append(toml_tmap(f"[[{key}]]\n", *p[1:]))
            elif p[0] == "u32":
                s.append(f'[[{key}]]\nkind = "u32"\nvalue = {p[1]}\n')
            else:
                _, raw, fields = p
                s.append(f'[[{key}]]\nkind = "bytes"\nhex = "{raw.hex()}"\n')
                for off, f in fields:
                    if f[0] == "tmap":
                        s.append(
                            toml_tmap(
                                f"[[{key}.fields]]\n", *f[1:], extra=f"offset = {off}\n"
                            )
                        )
                    else:
                        b, bo = buf_ref(f[1], f[2])
                        s.append(
                            f'[[{key}.fields]]\noffset = {off}\nkind = "ptr"\nbuffer = {q(b)}\nbyte_offset = {bo}\n'
                        )
        return s

    l0 = plaunches[0][0]
    kern_lines = "\n".join(
        f"  {i}: {la.name[:110]} grid {list(la.grid)} block {list(la.block)}"
        + (f" cluster {list(la.cluster)}" if la.cluster != (1, 1, 1) else "")
        for i, (la, _) in enumerate(plaunches)
    )
    desc = (
        f"\nCaptured in process by ptxv.capture (python/ptxv). {len(plaunches)} launch(es), "
        + ("PTX from Triton's cache" if use_ptx else "binary only (cubins from CUPTI)")
        + f":\n{kern_lines}"
    )
    desc = desc.replace("\\", "\\\\").replace('"', '\\"')
    out_names = [b["name"] for b in outs]
    toml = [
        f'name = {q(name)}\ndescription = """{desc}"""\n'
        + ('ptx = "kernel.ptx"\n' if use_ptx else f'cubin = "{cubin_file[l0.name]}"\n')
        + f"entry = {q(l0.name)}\ngrid = {list(l0.grid)}\nblock = {list(l0.block)}\n"
        + (f"cluster = {list(l0.cluster)}\n" if l0.cluster != (1, 1, 1) else "")
        # cudaLaunchAttributePreferredClusterDimension (S-CGA-SHAPE).
        + (
            f"preferred_cluster = {l0.record['preferred_cluster']}\n"
            if l0.record.get("preferred_cluster")
            else ""
        )
        + f"dynamic_smem = {l0.smem}\noutputs = {json.dumps(out_names)}\n"
    ]
    (out / "data").mkdir()
    (out / "expected").mkdir()
    for b in buffers:
        s = f'[[buffers]]\nname = {q(b["name"])}\ndtype = "{b["dtype"]}"\nshape = {b["shape"]}\n'
        if b["init"] is not None and any(b["init"]):
            (out / "data" / f"{b['name']}.bin").write_bytes(b["init"])
            s += f'init = "file"\nfile = "data/{b["name"]}.bin"\n'
        else:
            s += 'init = "zeros"\n'
        if b.get("is_out"):
            (out / "expected" / f"{b['name']}.bin").write_bytes(b["final"])
            s += f'expect = "expected/{b["name"]}.bin"\n'
        toml.append(s)
    toml += toml_params(plaunches[0][1], "params")
    for la, params in plaunches[1:]:
        s = f"[[launches]]\nentry = {q(la.name)}\ngrid = {list(la.grid)}\nblock = {list(la.block)}\n"
        if la.cluster != (1, 1, 1):
            s += f"cluster = {list(la.cluster)}\n"
        if la.record.get("preferred_cluster"):
            s += f"preferred_cluster = {la.record['preferred_cluster']}\n"
        s += f"dynamic_smem = {la.smem}\n"
        if la.pdl:
            s += "pdl = true\n"
        if not use_ptx and cubin_file[la.name] != cubin_file[l0.name]:
            s += f'cubin = "{cubin_file[la.name]}"\n'
        toml.append(s)
        toml += toml_params(params, "launches.params")
    toml.append(
        '[[variants]]\nid = "good"\nsummary = "as captured"\nexpect = ["clean"]\n'
    )
    (out / "case.toml").write_text("\n".join(toml))
    meta = {
        "what": "ptxv.capture",
        "launches": [
            {
                "index": la.index,
                "name": la.name,
                "grid": list(la.grid),
                "block": list(la.block),
                "cluster": list(la.cluster),
                "smem": la.smem,
                "pdl": int(la.pdl),
                "op": la.tag,
                "cubin": cubin_file[la.name],
            }
            for la, _ in plaunches
        ],
        "ptx": use_ptx,
        "buffers": [
            {
                "name": b["name"],
                "dtype": b["dtype"],
                "shape": b["shape"],
                "size": b["size"],
                "fresh": b["fresh"],
                "output": bool(b.get("is_out")),
            }
            for b in buffers
        ],
        "unresolved": unresolved,
        "warnings": warnings + list(cap.warnings),
    }
    (out / "capture.json").write_text(json.dumps(meta, indent=1))


def _triton_ptx(names: set[str], cubins: dict[str, bytes]) -> dict[str, str]:
    """PTX of Triton kernels whose compiled cubin in Triton's cache equals the captured one."""
    root = os.environ.get("TRITON_CACHE_DIR") or os.path.expanduser("~/.triton/cache")
    out = {}
    if not os.path.isdir(root):
        return out
    for n in names:
        want = hashlib.sha1(cubins[n]).hexdigest()
        for c in glob.glob(os.path.join(root, "*", f"{n}.cubin")):
            try:
                if hashlib.sha1(_read(Path(c))).hexdigest() == want:
                    p = c[: -len(".cubin")] + ".ptx"
                    if os.path.exists(p):
                        out[n] = Path(p).read_text()
                        break
            except OSError:
                continue
    return out
