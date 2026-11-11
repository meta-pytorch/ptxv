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
write the bytes they already held). Every output starts from its pre-launch bytes, as on the
GPU (bytes no kernel writes then compare equal); `write_only` ones start as zeros.
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

from . import _cupti, _ptxsrc
from ._cubin import (
    decode_tmap,
    elf_defines,
    ESIZE,
    kparam_sizes,
    prune as prune_cubin,
    reqntid,
    u64,
)
from ._util import cache_dir, tree_hash

PTR_SCAN_MAX = 1 << 20  # as ptxvcap.c: larger buffers are not scanned for pointer words
FLOAT_DT = ("bf16", "f16", "f32", "f64", "e4m3")


class CaseError(RuntimeError):
    pass


RAGGED_BIG = 1 << 30  # quack copy_utils BIG_INT


def _ragged_shift(raw: bytes, off: int, v: int, comp_of):
    """(geometry, comp, offset, shift) when the 128 bytes at `off` are a tensor map whose base
    `v` lies outside every allocation but `v + 2^30 * stride[d]` (a dim of extent 2^30) lies in
    one: the ragged `ptr_shift` layout. `offset` (negative) is relative to that component."""
    if not v or off % 64 or off + 128 > len(raw):
        return None
    g = decode_tmap(raw[off : off + 128])
    if not g or raw[off + 8 : off + 16] == bytes(8):
        return None
    for d in range(1, len(g["dims"])):
        if g["dims"][d] != RAGGED_BIG:
            continue
        shift = RAGGED_BIG * g["strides"][d - 1]
        loc = comp_of((v + shift) & ((1 << 64) - 1))
        if loc:
            return g, loc[0], loc[1] - shift, shift
    # A map whose base sits a little before an allocation (a box offset folded into the
    # base, e.g. the cascade WGRAD scale-tail boxes): name the next allocation with a negative
    # byte_offset; the simulator then reports any element the kernel reads before it.
    for gap in (1 << k for k in range(4, 21)):
        loc = comp_of(v + gap)
        if loc and loc[1] < gap:
            return g, loc[0], loc[1] - gap, gap
    return None


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
    scratch: Sequence[str] = (),
    suggest_scratch: bool = True,
) -> Path:
    """Write the case; returns its content-addressed directory under $PTXV_CACHE/cases.

    scratch: buffer names (as in the case) that are dead workspace: `init = "undef"`,
    `scratch = true`, never an output (S-INIT scratch buffers). suggest_scratch: run the ptxv
    binary's first-access pass (when it is available) and list, in capture.json
    `scratch_suggestions` and a warning, the buffers whose first access is a write or a
    discard; a suggestion is never applied (PTXV_SUGGEST_SCRATCH=0: off)."""
    tmp = Path(tempfile.mkdtemp(prefix="case-", dir=cache_dir("cases")))
    try:
        _write(
            launches,
            cap,
            tmp,
            outputs,
            write_only,
            ptx,
            prune,
            rename or {},
            name,
            tuple(scratch),
            suggest_scratch,
        )
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
    launches,
    cap,
    out: Path,
    outputs,
    write_only,
    ptx,
    prune,
    rename,
    name,
    scratch=(),
    suggest_scratch=False,
) -> None:
    warnings: list[str] = []
    snapdir = cap.dir / "snap"

    # ---------------------------------------------------------------- snapshots
    allocs: dict[int, int] = {}  # base -> size
    first_pre: dict[int, Path] = {}
    first_pre_idx: dict[
        int, int
    ] = {}  # base -> region index of the launch it was taken at
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
                first_pre_idx[base] = la.index
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
    # Allocator blocks of the region overlap when one was freed and its memory reused; merging
    # them would join unrelated tensors into one buffer. Keep the blocks live at some launch of
    # this group (allocator trace time_us vs. the launch record's t_us, both wall clock).
    t_l = [la.record.get("t_us") for la in launches]
    life = getattr(cap, "block_life", {})
    # A named tensor whose storage is an allocator block allocated after the group's last
    # launch is not what the launches saw: its memory was freed and reused (a scalar result of
    # a later op at the address of a kernel's freed workspace / LSE output). Its name, dtype
    # and extent would be stale; the launch-time block stands instead.
    late = []
    if all(t_l) and t_l:
        tmax = max(t_l)
        for bi, (a, z, kind) in enumerate(cap.blocks):
            lt = life.get(bi)
            if kind == "during" and lt and lt[0] is not None and lt[0] > tmax:
                late.append((a, a + z))
    for n, t in cap.names.items():
        if any(a <= t.ptr and t.ptr + t.nbytes <= e for a, e in late):
            warnings.append(
                f"{n}: allocated after the launch (memory reused); name not used"
            )
            continue
        ivs.append((t.ptr, t.ptr + t.nbytes, "named", n))
    for bi, (a, z, kind) in enumerate(cap.blocks):
        lt = life.get(bi)
        if lt and lt[0] is not None and all(t_l):
            t0, t1 = lt
            if not any(t0 <= t and (t1 is None or t < t1) for t in t_l):
                continue
        ivs.append((a, a + z, kind, None))
    for b, z in allocs.items():
        # The parts of a snapshotted driver allocation that no named tensor or allocator
        # block covers are their own intervals: a pointer just past a tensor's end then
        # resolves to the allocation that follows it (a scalar tensor the allocator placed
        # right after another), not to "one past the end" of the first.
        cov = sorted(
            (max(s, b), min(e, b + z)) for s, e, _, _ in ivs if s < b + z and e > b
        )
        pos = b
        for s, e in cov:
            if s > pos:
                ivs.append((pos, s, "alloc", None))
            pos = max(pos, e)
        if pos < b + z:
            ivs.append((pos, b + z, "alloc", None))
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
            elif (r := _ragged_shift(raw, off, v, comp_of)) is not None:
                # A ragged `ptr_shift` tensor map (CuTe create_ragged_tensor_for_tma): its base
                # sits 2^30 rows before its tensor, so the map names the tensor's buffer with a
                # negative byte_offset. ASSUME(S-TMA)
                g, ci, o, shift = r
                note_ref(ci, o + shift, 1)
                tmap_refs.setdefault(ci, set()).add(g["dtype"])
                n = 64 if raw[off + 8] & 2 else 128
                fields.append((off, ("tmap", ci, o, g, raw[off : off + n])))
                off += n
                continue
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

    # ------------------------------------------------- pointer tables (one level)
    # A small, non-float buffer a parameter points to may hold addresses of other allocations
    # (a multi-tensor kernel's pointer array). Such words become `ptrs` (filled with the
    # target buffer's address in the simulator and on hardware), and their targets buffers.
    ptr_words: list[
        tuple[int, int, int, int]
    ] = []  # (ci, offset in comp, cj, offset in cj)
    for ci in sorted(refs):
        s0, e0, members = comps[ci]
        if e0 - s0 > PTR_SCAN_MAX or any(
            m[2] == "named" and cap.names[m[3]].dtype in FLOAT_DT for m in members
        ):
            continue
        data = bytes_at(s0, e0 - s0, first_pre)
        if data is None:
            continue
        for o in range(0, len(data) - 7, 8):
            v = u64(data, o)
            if v < 0x10000 or v >> 52:
                continue
            loc = comp_of(v)
            if loc is None or loc[0] == ci:
                continue
            ptr_words.append((ci, o, loc[0], loc[1]))
            note_ref(*loc)
    if ptr_words:
        warnings.append(
            f"{len(ptr_words)} pointer words inside buffers resolved to captured allocations (`ptrs`)"
        )

    # Components an earlier launch of the region (not in this group) points into: an
    # allocation made in the region but written before the group starts is an input of the
    # group, not a fresh output (a combine kernel reading a decode kernel's partials).
    first_idx = min((la.index for la in launches), default=0)
    earlier: set[int] = set()
    for la in getattr(cap, "launches", []):
        if la.index >= first_idx:
            continue
        for p in la.record.get("params", []):
            raw = bytes.fromhex(p["hex"])
            for off in range(0, len(raw) - 7, 8):
                v = u64(raw, off)
                loc = comp_of(v) if v else None
                if loc:
                    earlier.add(loc[0])

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
                # A pointer equal to the tensor's end is inside it only when nothing follows
                # in the component (otherwise it is the start of the next tensor).
                all(
                    m[0] <= s + o and (s + o + max(k, 1) <= m[1] or s + o == m[1] == e)
                    for o, k in rs
                )
                and cap.names[m[3]].whole
            ):
                # A named tensor smaller than the allocator block at its address (beyond the
                # allocator's 512-byte rounding) may be a stale name for memory the launch
                # uses as something larger (a workspace reusing a freed tensor's block): take
                # the component instead.
                big = [
                    x
                    for x in members
                    if x[2] in ("pre", "during")
                    and x[0] <= m[0] < x[1]
                    and x[1] > m[1]
                    and x[1] - x[0] > -(-(m[1] - m[0]) // 512) * 512
                ]
                if big:
                    warnings.append(
                        f"{m[3]}: the named tensor ({m[1] - m[0]} B) is smaller than the allocator "
                        f"block at its address ({max(x[1] - x[0] for x in big)} B); the buffer "
                        "takes the block's extent"
                    )
                    continue
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
            if ESIZE[dt] != ESIZE[td] or (dt == "u8" and td != "u8"):
                if size % ESIZE[td] == 0 and (dt == "u8" or len(tds) == 1):
                    if dt != "u8":
                        # The kernel reads it as the tensor map's type (a stale or reused name).
                        warnings.append(
                            f"{bname}: named as {dt}, a tensor map reads {td}; typed {td}"
                        )
                    dt, shape = td, [size // ESIZE[td]]
                else:
                    warnings.append(f"{bname}: dtype {dt} but a tensor map reads {td}")
        kinds = {m[2] for m in members}
        named_before = any(cap.names[m[3]].before for m in nm)
        fresh = (
            "during" in kinds
            and "pre" not in kinds
            and not named_before
            and ci not in earlier
        )
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

    for ci, o, cj, oj in ptr_words:
        bi, delta = comp_buf[ci]
        bo = o - delta
        if 0 <= bo <= buffers[bi]["size"] - 8:
            tb, tdelta = comp_buf[cj]
            buffers[bi].setdefault("ptrs", []).append(
                (bo, buffers[tb]["name"], oj - tdelta)
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
    # Declared scratch (dead workspace): undefined until written, never compared.
    unknown = set(scratch) - {b["name"] for b in buffers}
    if unknown:
        raise CaseError(
            f"scratch {sorted(unknown)} are not buffers of this case: {[b['name'] for b in buffers]}"
        )
    if outputs is not None and set(scratch) & set(outputs):
        raise CaseError(
            f"buffers {sorted(set(scratch) & set(outputs))} are both outputs and scratch"
        )
    outs = [b for b in outs if b["name"] not in scratch]
    if not outs and getattr(cap, "unchanged_outputs", False):
        # A launch that changed nothing on this input (a bounds check that found every index
        # in range, a backward pass with no long segments): every snapshotted buffer is an
        # output expected unchanged, so the check still covers its races, faults and writes.
        outs = [b for b in buffers if b["final"] is not None]
        if outs:
            warnings.append(
                "no buffer changed: every buffer is an output expected to stay unchanged"
            )
    if not outs:
        raise CaseError(
            "no output: no buffer changed and none was allocated in the capture region"
        )
    for b in outs:
        if b["final"] is None:
            raise CaseError(f"output {b['name']} has no post-launch snapshot")
        b["is_out"] = True
        # A fresh output (allocated in the region: torch.empty, a reused allocator block)
        # keeps its pre-launch bytes as init, as on the GPU: bytes no kernel writes then
        # compare equal with the captured output. Declared write_only outputs start as zeros.
        if b["name"] in write_only or (b["fresh"] and b["init"] is None):
            b["init"] = None

    # Vacuous-input guards. (1) An input whose bytes are all one value (all zeros: a buffer a
    # collective or another stream fills after the snapshot) makes a check on it say little.
    # (2) A launch of the region that is not in this case and points into an input between
    # that input's snapshot and the case's first launch reading it may have written bytes
    # the case's initial contents miss.
    group_idx = {la.index for la in launches}
    for bi, b in enumerate(buffers):
        if b.get("is_out") or not b["size"]:
            continue
        if b["init"] is not None and b["init"].count(b["init"][:1]) == len(b["init"]):
            warnings.append(
                f"input {b['name']}: all {b['size']} bytes are {b['init'][0]:#04x} "
                "(uniform input: the check may be vacuous; was it filled after the snapshot?)"
            )
        ci = next((c for c, (j, _) in comp_buf.items() if j == bi), None)
        if ci is None:
            continue
        reader = max(
            group_idx
        )  # a later launch of the case may be the one that reads it
        a = alloc_of(b["start"])
        snap_at = first_pre_idx.get(a) if a is not None else None
        lo, hi = b["start"], b["start"] + b["size"]
        for la in getattr(cap, "launches", []):
            if la.index in group_idx or la.index >= reader:
                continue
            if snap_at is not None and la.index < snap_at:
                continue  # ran before the snapshot: its writes are in the initial contents
            hit = False
            for p in la.record.get("params", []):
                raw = bytes.fromhex(p["hex"])
                if any(lo <= u64(raw, o) < hi for o in range(0, len(raw) - 7, 8)):
                    hit = True
                    break
            if hit:
                warnings.append(
                    f"input {b['name']}: launch {la.index} ({la.short_name[:60]}), not in this case, "
                    f"points into it after its snapshot and before launch {reader} of the case: "
                    "bytes it writes are missing from the initial contents"
                )

    # Device addresses a kernel wrote into an output (a CUtensorMap built on the device in a
    # global scratch buffer, a pointer table): the GPU's bytes are capture-space addresses, a
    # simulator run writes its own. Such words (8-byte aligned, pointing into a captured
    # allocation, changed by the launches) are left out of the comparison (`expect_ignore`).
    for b in outs:
        fin, ini = b["final"], b["init"]
        if fin is None or b["dtype"] in FLOAT_DT:
            continue
        rng = []
        for o in range(0, len(fin) - 7, 8):
            v = u64(fin, o)
            if v < 0x10000 or v >> 52 or comp_of(v) is None:
                continue
            if ini is not None and ini[o : o + 8] == fin[o : o + 8]:
                continue
            if rng and rng[-1][0] + rng[-1][1] == o:
                rng[-1][1] += 8
            else:
                rng.append([o, 8])
        if rng:
            b["expect_ignore"] = rng
            warnings.append(
                f"{b['name']}: {sum(n for _, n in rng)} bytes of device addresses the launches wrote "
                "are excluded from the output comparison (expect_ignore)"
            )

    def buf_ref(ci: int, off: int):
        bi, delta = comp_buf[ci]
        return buffers[bi]["name"], off - delta

    # ---------------------------------------------------------------- cubins
    mods = [_cupti.MODULE_DIR / f"{h}.cubin" for h in dict.fromkeys(cap.modules)]
    mods = [m for m in mods if m.exists()]
    names = {la.name for la, _ in plaunches}
    cubin_of: dict[str, Path] = {}
    defining: dict[str, list[Path]] = {}
    psizes = {}
    for la, _ in plaunches:
        ps = la.record.get("params") or []
        if ps and not any(p.get("packed") for p in ps):
            psizes.setdefault(la.name, [len(p["hex"]) // 2 for p in ps])
    for n in names:
        c = [m for m in mods if elf_defines(image(m), n)]
        defining[n] = c
        if not c:
            raise CaseError(
                f"no captured module defines {n}: its module was loaded before ptxv's hook. "
                "Import ptxv before the program's first CUDA work, or run under `python -m ptxv run`."
            )
        # A module compiled for another block shape (EIATTR_REQNTID) is not the launched one:
        # same-name CuTeDSL specializations differ in their launch config.
        blocks = {tuple(la.block) for la, _ in plaunches if la.name == n}
        fits = [m for m in c if (r := reqntid(image(m), n)) is None or {r} >= blocks]
        if fits and len(fits) < len(c):
            c = fits
        elif not fits:
            warnings.append(
                f"{n}: no captured module's REQNTID matches the launch block {sorted(blocks)}"
            )
        rec_mod = {la.record.get("module") for la, _ in plaunches if la.name == n} - {
            None
        }
        exact = [m for m in c if m.stem in rec_mod]
        if len(exact) == 1:
            c = exact  # the launch recorded its module image (ptxvcap's handle map)
        elif len(c) > 1 and n in psizes:
            # Several modules define this name (CuTeDSL truncates specialized kernel names):
            # keep those whose parameter layout (EIATTR_KPARAM_INFO) is the launch's.
            same = [m for m in c if kparam_sizes(image(m), n) == psizes[n]]
            if same:
                c = same
            if len({hashlib.sha1(image(m)).hexdigest() for m in c}) > 1:
                warnings.append(
                    f"{n}: {len(c)} captured modules define it with the launch's parameter "
                    "layout; the smallest is used"
                )
        cubin_of[n] = min(c, key=os.path.getsize)
    # Each launch's image: its recorded module when that is one of several different images
    # defining the name (specializations of one CuTeDSL / Triton kernel launched in one
    # region), else the name's choice above.
    lcubin: list[Path] = []
    for la, _ in plaunches:
        own = [m for m in defining[la.name] if m.stem == la.record.get("module")]
        lcubin.append(own[0] if own else cubin_of[la.name])
    split = {
        la.name
        for (la, _), src in zip(plaunches, lcubin)
        if hashlib.sha1(image(src)).digest()
        != hashlib.sha1(image(cubin_of[la.name])).digest()
    }
    if split:
        warnings.append(
            f"launches of {sorted(split)} run different modules under one name: each keeps "
            "its own cubin"
        )
    # PTX kept by the toolchain (Triton cache, CuTeDSL dumps, PTXV_PTX_DIRS), matched to the
    # captured machine code (_ptxsrc). Every kernel of the case needs its PTX; kernels from
    # several modules become one merged PTX text. Launches of one name running different
    # modules cannot share one PTX text (one definition per entry): those stay binary-only.
    ptx_origin: dict[str, str] = {}
    ptx_of = _ptxsrc.find_ptx({n: image(p) for n, p in cubin_of.items()}, ptx_origin)
    ptx_src = None
    if names and all(n in ptx_of for n in names):
        try:
            ptx_src = _ptxsrc.merge_ptx([ptx_of[n] for n in sorted(names)])
        except ValueError as e:
            warnings.append(f"PTX of the case's kernels not merged: {e}")
    elif ptx_of:
        warnings.append(
            "no PTX kept for " + ", ".join(sorted(n[:60] for n in names - set(ptx_of)))
        )
    use_ptx = ptx and not split and ptx_src is not None
    by_hash: dict[str, str] = {}
    cubin_files: list[str] = []
    for (la, _), src in zip(plaunches, lcubin):
        data = image(src)
        h = hashlib.sha1(data).hexdigest()
        if h not in by_hash:
            by_hash[h] = "jit.cubin" if use_ptx else f"k{len(by_hash)}.cubin"
            keep = {
                m.name
                for (m, _), q in zip(plaunches, lcubin)
                if hashlib.sha1(image(q)).hexdigest() == h
            }
            (out / by_hash[h]).write_bytes(
                data if not prune else prune_cubin(data, keep)
            )
        cubin_files.append(by_hash[h])
    if ptx_src is not None:
        # The PTX as the toolchain produced it (its `.version` included): ptxv compiles it with
        # the ptxas of the toolkit that built the executed cubin (`ptxas_toolkit`, PTXV_TOOLKITS)
        # and compares the result with `executed_cubin`.
        (out / ("kernel.ptx" if use_ptx else "jit.ptx")).write_text(ptx_src)
    # The CUDA toolkit whose ptxas built the executed cubin (its .note.nv.tkinfo).
    tk = re.search(rb"release (\d+)\.(\d+),", (out / cubin_files[0]).read_bytes())
    ptxas_toolkit = f"{int(tk.group(1))}.{int(tk.group(2))}" if tk else None
    # Options Triton gave ptxas that the toolkit note omits (`ptxas_args`), verified by
    # rebuilding each kernel with Triton's own ptxas; one set for the whole case, else none
    # (ptxv then refuses the PTX front of a Triton-built case).
    ptxas_args = None
    if use_ptx and names and all(ptx_origin.get(n) == "triton" for n in names):
        sets = [
            _ptxsrc.triton_ptxas_args(image(cubin_of[n]), n, ptx_of[n])
            for n in sorted(names)
        ]
        if all(x is not None for x in sets) and all(x == sets[0] for x in sets):
            ptxas_args = sets[0]
        else:
            warnings.append(
                "no ptxas option set rebuilds the Triton kernels' executed code: the case "
                "records no ptxas_args"
            )

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
        + (
            "PTX kept by the toolchain"
            if use_ptx
            else "binary only (cubins from CUPTI)"
            + (", PTX in jit.ptx" if ptx_src is not None else "")
        )
        + f":\n{kern_lines}"
    )
    desc = desc.replace("\\", "\\\\").replace('"', '\\"')
    out_names = [b["name"] for b in outs]
    toml = [
        f'name = {q(name)}\ndescription = """{desc}"""\n'
        + (
            f'ptx = "kernel.ptx"\nexecuted_cubin = "{cubin_files[0]}"\n'
            if use_ptx
            else f'cubin = "{cubin_files[0]}"\n'
        )
        + (f"ptxas_toolkit = {q(ptxas_toolkit)}\n" if ptxas_toolkit else "")
        + (f"ptxas_args = {q(ptxas_args)}\n" if ptxas_args is not None else "")
        + f"entry = {q(l0.name)}\ngrid = {list(l0.grid)}\nblock = {list(l0.block)}\n"
        # S-CGA-SHAPE: a preferred cluster needs the explicit regular one, even 1 x 1 x 1.
        + (
            f"cluster = {list(l0.cluster)}\n"
            if l0.cluster != (1, 1, 1) or l0.record.get("preferred_cluster")
            else ""
        )
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
        if b["name"] in scratch:
            # Declared dead workspace (capture's `scratch`): every byte undefined until written.
            s += 'init = "undef"\nscratch = true\n'
        elif b["init"] is not None and any(b["init"]):
            (out / "data" / f"{b['name']}.bin").write_bytes(b["init"])
            s += f'init = "file"\nfile = "data/{b["name"]}.bin"\n'
        else:
            s += 'init = "zeros"\n'
        if b["fresh"] and b["name"] not in scratch:
            # Allocated inside the capture region: when the case's launches write it before
            # reading it, its captured bytes are allocator garbage (S-INIT scratch buffers).
            s += "scratch = true\n"
        if b.get("is_out"):
            (out / "expected" / f"{b['name']}.bin").write_bytes(b["final"])
            s += f'expect = "expected/{b["name"]}.bin"\n'
            if b.get("expect_ignore"):
                s += f"expect_ignore = {b['expect_ignore']}\n"
        if b.get("ptrs"):
            s += (
                "ptrs = ["
                + ", ".join(
                    f"{{offset = {o}, buffer = {q(t)}, byte_offset = {to}}}"
                    for o, t, to in b["ptrs"]
                )
                + "]\n"
            )
        toml.append(s)
    toml += toml_params(plaunches[0][1], "params")
    for i, (la, params) in enumerate(plaunches[1:], 1):
        s = f"[[launches]]\nentry = {q(la.name)}\ngrid = {list(la.grid)}\nblock = {list(la.block)}\n"
        if la.cluster != (1, 1, 1) or la.record.get("preferred_cluster"):
            s += f"cluster = {list(la.cluster)}\n"
        if la.record.get("preferred_cluster"):
            s += f"preferred_cluster = {la.record['preferred_cluster']}\n"
        s += f"dynamic_smem = {la.smem}\n"
        if la.pdl:
            s += "pdl = true\n"
        if not use_ptx and cubin_files[i] != cubin_files[0]:
            s += f'cubin = "{cubin_files[i]}"\n'
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
                "cubin": cubin_files[i],
            }
            for i, (la, _) in enumerate(plaunches)
        ],
        "ptx": use_ptx,
        "jit_ptx": ptx_src is not None and not use_ptx,
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
    if scratch:
        meta["scratch"] = sorted(scratch)
    if suggest_scratch and os.environ.get("PTXV_SUGGEST_SCRATCH", "1") != "0":
        sug = scratch_suggestions(out, exclude=set(scratch))
        if sug:
            meta["scratch_suggestions"] = sug
            meta["warnings"].append(
                "possible scratch (dead workspace) buffers, not applied: "
                + ", ".join(f"{x['buffer']} ({x['why']})" for x in sug)
                + '; pass scratch=(...) to mark them init = "undef", scratch = true'
            )
    (out / "capture.json").write_text(json.dumps(meta, indent=1))


def scratch_suggestions(
    case_dir: Path, exclude=frozenset(), timeout: float = 60
) -> list:
    """Buffers of a written case whose first access, in one deterministic run (`ptxv
    first-access`), is a write or a discard: candidates for `init = "undef"`, `scratch = true`.
    Buffers already undefined or scratch with captured inputs resolved by ptxv itself (fresh
    allocations) are skipped. [] when the ptxv binary is unavailable or the run fails.
    """
    import subprocess
    import tomllib

    from . import _binary

    try:
        exe = _binary.find(build=False)
    except FileNotFoundError:
        return []
    man = tomllib.loads((case_dir / "case.toml").read_text())
    skip = set(exclude) | {
        b["name"]
        for b in man.get("buffers", [])
        if b.get("scratch") or b.get("init") == "undef"
    }
    first: dict = {}
    # The SASS front first; a PTX case falls back to its PTX front (an instruction the SASS
    # front refuses, e.g. CCTL.E.RML2).
    for front in ["sass"] + (["ptx"] if man.get("ptx") else []):
        try:
            r = subprocess.run(
                [
                    str(exe),
                    "first-access",
                    str(case_dir),
                    "--front",
                    front,
                    "--out",
                    str(case_dir / ".first_access"),
                ],
                capture_output=True,
                text=True,
                timeout=timeout,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
            )
            if r.returncode == 0:
                first = json.loads(r.stdout)
                break
        except (subprocess.TimeoutExpired, ValueError, OSError):
            pass
        finally:
            shutil.rmtree(case_dir / ".first_access", ignore_errors=True)
    # Written first and read back (a workspace), or written first and not an output of the
    # case (a discarded one). A buffer only written is an output, not a suggestion.
    outs = set(man.get("outputs", []))
    return [
        {
            "buffer": n,
            "why": f"first access is a write or discard by {f.get('by')}"
            + ("; read back after it" if f.get("read_back") else "; not an output"),
        }
        for n, f in sorted(first.items())
        if f.get("first") == "write"
        and n not in skip
        and (f.get("read_back") or n not in outs)
    ]
