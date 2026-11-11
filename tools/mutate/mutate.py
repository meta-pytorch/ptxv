#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Synchronization mutants of ptxv cases (docs/soundness-harness.md).

Each mutant deletes or weakens ONE synchronization point of a case's base kernel and is
written as a sibling case directory:

    <out>/<case>/<mutant-id>/case.toml     the case, variants replaced by `good` + `mut`
    <out>/<case>/<mutant-id>/kernel.ptx    the base PTX, the mutated line(s) tagged `// @MUTk`
    <out>/<case>/<mutant-id>/mutant.json   operator, line(s), before / after text
    (every other file of the case is symlinked)

`good` is the unmodified base kernel (so `ptxv hw --variant mut` compares the mutant against
the good kernel's output); `mut` carries the edit as ordinary marker edits
(crates/case/src/lib.rs `apply_edit`), so `ptxv check --variant mut` checks it through both
front-ends (the SASS front-end compiles the mutated PTX with ptxas).

Binary-only cases (`cubin = ...`) get SASS mutants instead: `mut` NOPs one synchronization
instruction of the cubin (a `patch = [{at, nop = true}]` variant, crates/sass/src/scoreboard.rs).

Operators (PTX): see OPERATORS below; (SASS): SASS_OPS.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------------------------
# PTX line model


@dataclass
class Inst:
    """One PTX instruction line: optional label, optional guard, opcode, operands."""

    idx: int  # 0-based line index
    label: str | None
    guard: str | None  # e.g. "@!%p3"
    opcode: str  # e.g. "mbarrier.try_wait.parity.shared::cta.b64"
    operands: str  # text after the opcode up to ';'
    text: str  # the instruction text as written (guard .. ';'), the `find` string


_LABEL = r"(?:(?P<label>[A-Za-z_$][\w$]*)\s*:\s*)?"
_GUARD = r"(?:(?P<guard>@!?%[\w$]+)\s+)?"
_INST_RE = re.compile(
    r"^\s*" + _LABEL + r"(?P<text>" + _GUARD + r"(?P<op>[a-z][\w.:]*)(?P<ops>[^;]*);)"
)


def strip_comment(line: str) -> str:
    i = line.find("//")
    return line if i < 0 else line[:i]


def parse_insts(lines: list[str]) -> list[Inst]:
    out = []
    in_block_comment = False
    for i, line in enumerate(lines):
        code = strip_comment(line)
        if in_block_comment:
            if "*/" in code:
                in_block_comment = False
            continue
        if "/*" in code:
            if "*/" not in code:
                in_block_comment = True
            continue
        m = _INST_RE.match(code)
        if not m:
            continue
        op = m.group("op")
        if op.startswith(".") or op in ("{", "}"):
            continue
        # One instruction per line only (generated PTX occasionally packs several).
        if code.count(";") != 1:
            continue
        out.append(
            Inst(
                idx=i,
                label=m.group("label"),
                guard=m.group("guard"),
                opcode=op,
                operands=m.group("ops").strip(),
                text=m.group("text"),
            )
        )
    return out


def operand_list(ops: str) -> list[str]:
    """Split operands at top-level commas (brackets / braces kept together)."""
    out, depth, cur = [], 0, ""
    for ch in ops:
        if ch in "[{":
            depth += 1
        elif ch in "]}":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


# --------------------------------------------------------------------------------------------
# Mutants


@dataclass
class Edit:
    line: int  # 0-based
    find: str
    replace: str


@dataclass
class Mutant:
    id: str
    op: str  # operator name
    summary: str
    edits: list[Edit] = field(default_factory=list)
    # SASS mutants: control-bit / NOP patches instead of PTX edits.
    patch: list[dict] = field(default_factory=list)
    lines: list[int] = field(
        default_factory=list
    )  # 1-based lines of the base PTX (sync sites)


def _scope_tokens(opcode: str) -> list[str]:
    """Opcode qualifiers; `.shared::cluster` stays one token, so `cluster` is only a scope."""
    return opcode.split(".")


def _is_barrier_cluster(op: str) -> bool:
    return op.startswith("barrier.cluster")


DELETED = "// ptxv-mutate: deleted"


def ptx_mutants(src: str) -> list[Mutant]:
    lines = src.split("\n")
    insts = parse_insts(lines)
    muts: list[Mutant] = []
    seen_ids: dict[str, int] = {}

    def add(op: str, inst_or_insts, summary: str, replaces: list[str]):
        group = inst_or_insts if isinstance(inst_or_insts, list) else [inst_or_insts]
        base = f"{op}-L{group[0].idx + 1}"
        n = seen_ids.get(base, 0)
        seen_ids[base] = n + 1
        mid = base if n == 0 else f"{base}.{n}"
        m = Mutant(id=mid, op=op, summary=summary, lines=[g.idx + 1 for g in group])
        for g, r in zip(group, replaces):
            m.edits.append(Edit(line=g.idx, find=g.text, replace=r))
        muts.append(m)

    for k, ins in enumerate(insts):
        op = ins.opcode
        base = op.split(".")[0]
        toks = _scope_tokens(op)
        L = ins.idx + 1
        # --- deletions of whole synchronization instructions -------------------------------
        if op.startswith(
            (
                "bar.sync",
                "bar.arrive",
                "barrier.sync",
                "barrier.arrive",
                "bar.cta.sync",
                "bar.cta.arrive",
                "barrier.cta.sync",
                "barrier.cta.arrive",
            )
        ):
            add("del-bar", ins, f"delete `{op}` (line {L})", [DELETED])
        elif op.startswith("bar.warp.sync"):
            add("del-bar", ins, f"delete `{op}` (line {L})", [DELETED])
        elif _is_barrier_cluster(op):
            add("del-bar", ins, f"delete `{op}` (line {L})", [DELETED])
            # .release arrive -> .relaxed
            if ".arrive" in op and ".release" in op:
                add(
                    "relax",
                    ins,
                    f"`{op}`: .release -> .relaxed (line {L})",
                    [ins.text.replace(".release", ".relaxed", 1)],
                )
        elif op.startswith("griddepcontrol.wait"):
            add("del-bar", ins, f"delete `{op}` (line {L})", [DELETED])
        elif op.startswith(("mbarrier.try_wait", "mbarrier.test_wait")):
            ops = operand_list(ins.operands)
            pred = ops[0] if ops else None
            if pred and pred.startswith("%"):
                g = (ins.guard + " ") if ins.guard else ""
                add(
                    "del-wait",
                    ins,
                    f"`{op}` succeeds at once (line {L}): the wait is gone",
                    [f"{g}mov.pred {pred}, 1;"],
                )
            if ".parity" in op and len(ops) >= 3:
                par = ops[2]
                if par in ("0", "1"):
                    new_ops = ops[:2] + [str(1 - int(par))] + ops[3:]
                    g = (ins.guard + " ") if ins.guard else ""
                    add(
                        "parity",
                        ins,
                        f"`{op}`: parity {par} -> {1 - int(par)} (line {L})",
                        [f"{g}{op} {', '.join(new_ops)};"],
                    )
                elif par.startswith("%"):
                    g = (ins.guard + " ") if ins.guard else ""
                    new_ops = ops[:2] + ["%ptxv_mut_par"] + ops[3:]
                    add(
                        "parity",
                        ins,
                        f"`{op}`: parity operand {par} flipped (line {L})",
                        [
                            f"{{ .reg .b32 %ptxv_mut_par; xor.b32 %ptxv_mut_par, {par}, 1; "
                            f"{g}{op} {', '.join(new_ops)}; }}"
                        ],
                    )
            if ".acquire" in op:
                add(
                    "relax",
                    ins,
                    f"`{op}`: .acquire -> .relaxed (line {L})",
                    [ins.text.replace(".acquire", ".relaxed", 1)],
                )
            if "cluster" in toks:
                add(
                    "scope",
                    ins,
                    f"`{op}`: .cluster -> .cta (line {L})",
                    [_rescope(ins.text, op, "cluster", "cta")],
                )
        elif op.startswith(
            ("mbarrier.arrive", "mbarrier.arrive_drop")
        ) or op.startswith("cp.async.mbarrier.arrive"):
            if ".expect_tx" in op:
                # Weaken: arrive without announcing the transaction bytes.
                ops = operand_list(ins.operands)
                new_op = op.replace(".expect_tx", "", 1)
                g = (ins.guard + " ") if ins.guard else ""
                add(
                    "drop-expect-tx",
                    ins,
                    f"`{op}`: arrive without expect_tx (line {L}): the phase can complete "
                    "before the async bytes land",
                    [f"{g}{new_op} {', '.join(ops[:2])};"],
                )
            add("del-arrive", ins, f"delete `{op}` (line {L})", [DELETED])
            if ".release" in op:
                add(
                    "relax",
                    ins,
                    f"`{op}`: .release -> .relaxed (line {L})",
                    [ins.text.replace(".release", ".relaxed", 1)],
                )
            if "cluster" in toks:
                add(
                    "scope",
                    ins,
                    f"`{op}`: .cluster -> .cta (line {L})",
                    [_rescope(ins.text, op, "cluster", "cta")],
                )
        elif op.startswith("mbarrier.expect_tx"):
            add("drop-expect-tx", ins, f"delete `{op}` (line {L})", [DELETED])
        elif base == "fence" or base == "membar":
            add("del-fence", ins, f"delete `{op}` (line {L})", [DELETED])
            if (
                base == "fence"
                and not op.startswith("fence.proxy")
                and "mbarrier_init" not in op
            ):
                for s, t in (("sys", "gpu"), ("gpu", "cta"), ("cluster", "cta")):
                    if s in toks:
                        add(
                            "scope",
                            ins,
                            f"`{op}`: .{s} -> .{t} (line {L})",
                            [_rescope(ins.text, op, s, t)],
                        )
            if base == "membar":
                for s, t in (("sys", "gl"), ("gl", "cta")):
                    if op.endswith("." + s):
                        add(
                            "scope",
                            ins,
                            f"`{op}`: .{s} -> .{t} (line {L})",
                            [_rescope(ins.text, op, s, t)],
                        )
        elif op.startswith("tcgen05.fence"):
            add("del-fence", ins, f"delete `{op}` (line {L})", [DELETED])
        elif op.startswith("tcgen05.wait"):
            add("del-wait", ins, f"delete `{op}` (line {L})", [DELETED])
        elif op.startswith(
            ("cp.async.bulk.wait_group", "cp.async.wait_group", "cp.async.wait_all")
        ):
            add("del-wait", ins, f"delete `{op}` (line {L})", [DELETED])
            ops = operand_list(ins.operands)
            if ops and ops[0].isdigit():
                n = int(ops[0])
                add(
                    "weaken-wait",
                    ins,
                    f"`{op} {n}` -> `{op} {n + 1}` (line {L})",
                    [re.sub(r"\b" + str(n) + r"\s*;$", f"{n + 1};", ins.text)],
                )
            if op.startswith("cp.async.bulk.wait_group") and ".read" not in op:
                # Only wait for the reads of the source, not completion of the writes.
                add(
                    "weaken-wait",
                    ins,
                    f"`{op}` -> `.read` (line {L})",
                    [
                        ins.text.replace(
                            "cp.async.bulk.wait_group",
                            "cp.async.bulk.wait_group.read",
                            1,
                        )
                    ],
                )
        elif op.startswith(("cp.async.bulk.commit_group", "cp.async.commit_group")):
            add("del-commit", ins, f"delete `{op}` (line {L})", [DELETED])
        elif op.startswith("tcgen05.commit"):
            add("del-commit", ins, f"delete `{op}` (line {L})", [DELETED])
        elif base in ("ld", "st", "atom", "red"):
            sem = next(
                (s for s in ("acq_rel", "acquire", "release") if f".{s}" in op), None
            )
            if sem:
                add(
                    "relax",
                    ins,
                    f"`{op}`: .{sem} -> .relaxed (line {L})",
                    [ins.text.replace(f".{sem}", ".relaxed", 1)],
                )
            if sem or ".relaxed" in op or ".volatile" in op:
                for s, t in (("sys", "gpu"), ("gpu", "cta"), ("cluster", "cta")):
                    if s in toks:
                        add(
                            "scope",
                            ins,
                            f"`{op}`: .{s} -> .{t} (line {L})",
                            [_rescope(ins.text, op, s, t)],
                        )

    # --- order and atomicity ---------------------------------------------------------------
    def no_label_between(a: Inst, b: Inst) -> bool:
        if a.label or b.label:
            return False
        return not any(
            re.match(r"^\s*[A-Za-z_$][\w$]*\s*:", strip_comment(lines[i]))
            for i in range(a.idx + 1, b.idx)
        )

    def is_mem(i: Inst) -> bool:
        return i.opcode.split(".")[0] in ("ld", "st", "atom", "red")

    def is_order_fence(i: Inst) -> bool:
        b = i.opcode.split(".")[0]
        return (
            b == "fence"
            and not i.opcode.startswith("fence.proxy")
            and "mbarrier_init" not in i.opcode
        ) or b == "membar"

    for k, ins in enumerate(insts):
        L = ins.idx + 1
        # A fence moved past its neighbouring memory access (after the next one, before the
        # previous one): the access leaves the fence's protection.
        if is_order_fence(ins) and not ins.guard:
            if (
                k + 1 < len(insts)
                and is_mem(insts[k + 1])
                and no_label_between(ins, insts[k + 1])
            ):
                nxt = insts[k + 1]
                add(
                    "move-fence",
                    [ins, nxt],
                    f"`{ins.opcode}` moved after `{nxt.opcode}` (line {L})",
                    [nxt.text, ins.text],
                )
            if k > 0 and is_mem(insts[k - 1]) and no_label_between(insts[k - 1], ins):
                prv = insts[k - 1]
                add(
                    "move-fence",
                    [prv, ins],
                    f"`{ins.opcode}` moved before `{prv.opcode}` (line {L})",
                    [ins.text, prv.text],
                )
        base = ins.opcode.split(".")[0]
        op = ins.opcode
        acquiring = any(f".{s}" in op for s in ("acquire", "acq_rel", "sc"))
        # An acquire fence inserted after a relaxed read or read-modify-write: the access
        # synchronizes with whatever release it observed (fence-acquired), which the
        # exemptions for non-acquiring accesses must not keep.
        if (
            base in ("ld", "atom", "red")
            and not acquiring
            and (".relaxed" in op or ".volatile" in op or base in ("atom", "red"))
            and ".shared" not in op
            and ".param" not in op
            and ".const" not in op
        ):
            add(
                "acq-fence",
                ins,
                f"`fence.acq_rel.gpu` inserted after `{op}` (line {L})",
                [ins.text + " fence.acq_rel.gpu;"],
            )
        # An atomic add whose result is used, split into a load, an add and a store: the
        # read-modify-write is no longer atomic (two claimants can get the same ticket).
        m = re.match(r"^atom((?:\.\w+)*)\.add\.(u32|s32)$", op)
        if m and not ins.guard:
            ops = operand_list(ins.operands)
            mods = [x for x in m.group(1).split(".") if x]
            space = next((x for x in mods if x in ("global", "shared")), None)
            if len(ops) == 3 and ops[0].startswith("%") and space:
                d, a, v = ops
                ty = m.group(2)
                add(
                    "atom-plain",
                    ins,
                    f"`{op}` split into ld / add / st (line {L}): not atomic",
                    [
                        f"{{ .reg .b32 %ptxv_mut_t; ld.relaxed.gpu.{space}.{ty} {d}, {a}; "
                        f"add.{ty} %ptxv_mut_t, {d}, {v}; st.relaxed.gpu.{space}.{ty} {a}, %ptxv_mut_t; }}"
                    ],
                )
    return muts


def _rescope(text: str, op: str, s: str, t: str) -> str:
    """Replace the scope qualifier .s of the opcode with .t (state spaces like ::cluster untouched)."""
    toks = op.split(".")
    new = ".".join(t if tok == s else tok for tok in toks)
    return text.replace(op, new, 1)


# --------------------------------------------------------------------------------------------
# SASS mutants (binary-only cases): NOP one synchronization instruction.

SASS_OPS = [
    # (regex on the opcode, operator name)
    (r"^BAR\.(SYNC|ARV|RED)", "del-bar"),
    (r"^BAR\b", "del-bar"),
    (r"^WARPSYNC", "del-bar"),
    (r"^UCGABAR_(ARV|WAIT)", "del-bar"),
    (r"^SYNCS\.ARRIVE", "del-arrive"),
    (r"^SYNCS\.PHASECHK", "del-wait"),
    (r"^MEMBAR", "del-fence"),
    (r"^FENCE", "del-fence"),
    (r"^DEPBAR", "del-wait"),
    (r"^LDGDEPBAR", "del-commit"),
    (r"^UTC\w*BAR", "del-commit"),
    (r"^ACQBULK", "del-wait"),
    (r"^CCTL", "del-fence"),
    # The drain of the thread's remote shared::cluster operations that ptxas emits before a
    # cluster-scope release (`MEMBAR.ALL.GPU; ERRBAR; CGAERRBAR`): without it a weak remote
    # store can land after the release.
    (r"^(CGA)?ERRBAR", "del-fence"),
]

_SASS_LINE = re.compile(
    r"/\*(?P<addr>[0-9a-f]{4,})\*/\s+(?P<guard>@!?U?P\w+\s+)?(?P<op>[A-Z][A-Z0-9_.]*)(?P<rest>[^;]*);"
)


def sass_mutants(disasm: str, kernel: str | None = None) -> list[Mutant]:
    muts = []
    for line in disasm.splitlines():
        m = _SASS_LINE.search(line)
        if not m:
            continue
        op = m.group("op")
        for rx, name in SASS_OPS:
            if re.match(rx, op):
                addr = int(m.group("addr"), 16)
                # A deletion also drops the instruction's scoreboards (a NOP that still sets
                # one is an encoding ptxas never emits); waits on them then pass at once.
                p = {
                    "at": f"@0x{addr:04x}",
                    "nop": True,
                    "wbar": 7,
                    "rbar": 7,
                    "expect": "NOP",
                }
                if kernel:
                    p["kernel"] = kernel
                text = (m.group("guard") or "") + op + m.group("rest")
                muts.append(
                    Mutant(
                        id=f"sass-{name}-0x{addr:04x}",
                        op="sass-" + name,
                        summary=f"NOP `{text.strip()}` at 0x{addr:04x}",
                        patch=[p],
                    )
                )
                break
    return muts


# BPT.TRAP 0x1 on sm_100+/sm_103a (ptxas 13.0, `trap;`): lo 0x000000040000795c, hi
# 0x000fea0000300000, control bits included (the original's control bits can make the trap an
# illegal encoding, e.g. a write scoreboard). The guard (bits 12..15) is kept.
_TRAP_BITS = [[0, 12, 0x95C], [16, 48, 0x40000], [64, 64, 0x000FEA0000300000]]


def reach_probe(m: Mutant) -> Mutant:
    """The mutant's site replaced by a trap: does any schedule execute it? A mutant whose site
    never executes at this launch is equivalent to the base kernel (dead code here)."""
    p = Mutant(
        id=m.id + ".reach",
        op="reach",
        summary=f"trap at the site of {m.id}",
        lines=m.lines,
    )
    if m.edits:
        e = m.edits[0]
        ins = parse_insts([e.find])
        guard = (ins[0].guard + " ") if ins and ins[0].guard else ""
        p.edits = [Edit(line=e.line, find=e.find, replace=f"{guard}trap;")]
    else:
        q = {
            k: v
            for k, v in m.patch[0].items()
            if k not in ("nop", "expect", "wbar", "rbar")
        }
        q["bits"] = _TRAP_BITS
        g = re.match(r"NOP `(@!?U?P\w+) ", m.summary)
        q["expect"] = (g.group(1) + " " if g else "") + "BPT.TRAP 0x1"
        p.patch = [q]
    return p


def cubin_kernels(disasm: str) -> dict[str, str]:
    """nvdisasm text split by `.text.<kernel>` section."""
    out, cur, buf = {}, None, []
    for line in disasm.splitlines():
        m = re.match(r"\s*\.section\s+\.text\.(\S+?),", line)
        if m:
            if cur:
                out[cur] = "\n".join(buf)
            cur, buf = m.group(1), []
        elif cur:
            buf.append(line)
    if cur:
        out[cur] = "\n".join(buf)
    return out


# --------------------------------------------------------------------------------------------
# case.toml handling


def strip_variants(toml_text: str) -> str:
    """The manifest text without its [[variants]] (and [[variants.edits]]) tables."""
    out, skipping, in_ml = [], False, None
    for line in toml_text.split("\n"):
        stripped = line.strip()
        if in_ml is None:
            header = re.match(r"^\[\[?\s*([\w.]+)\s*\]\]?\s*(#.*)?$", stripped)
            if header:
                skipping = header.group(1).split(".")[0] == "variants"
        if not skipping:
            out.append(line)
        # Track multi-line strings so a header-looking line inside one is not a header.
        for q in ('"""', "'''"):
            n = line.count(q)
            if in_ml is None and n % 2 == 1:
                in_ml = q
                break
            if in_ml == q and n % 2 == 1:
                in_ml = None
                break
    while out and out[-1].strip() == "":
        out.pop()
    return "\n".join(out) + "\n"


def toml_str(s: str) -> str:
    return json.dumps(s, ensure_ascii=False)


def apply_variant_ptx(src: str, variant: dict) -> str:
    """crates/case/src/lib.rs `variant_ptx`, for materializing a non-good base variant."""

    def one(src, marker, find, replace):
        tag = f"// {marker}"
        out, hits = [], 0
        for line in src.split("\n"):
            if tag in line:
                hits += 1
                if replace == "<delete>":
                    continue
                if find not in line:
                    raise ValueError(f"`{find}` not on marked line {marker}")
                out.append(line.replace(find, replace, 1))
            else:
                out.append(line)
        if hits == 0 or (replace != "<delete>" and hits != 1):
            raise ValueError(f"marker {marker} found {hits} times")
        return "\n".join(out)

    if variant.get("marker"):
        src = one(src, variant["marker"], variant.get("find"), variant.get("replace"))
    for e in variant.get("edits", []):
        src = one(src, e["marker"], e.get("find"), e["replace"])
    return src


def write_mutant(
    case_dir: Path,
    out_dir: Path,
    m: Mutant,
    base_ptx: str | None,
    base_variant: dict | None,
    arch: str | None = None,
) -> Path:
    """Write mutant `m` of the case at `case_dir` to `out_dir` (created / replaced)."""
    case_dir = case_dir.resolve()
    out_dir = out_dir.resolve()
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    text = (case_dir / "case.toml").read_text()
    man = tomllib.loads(text)
    ptx_name = man.get("ptx") or ""
    for f in case_dir.iterdir():
        if f.name in ("case.toml", ptx_name, "mutant.json") or f.name.startswith("out"):
            continue
        # Relative, so that a copy of the runs directory next to a copy of the case tree
        # (a batch runner's mirror) still resolves.
        os.symlink(os.path.relpath(f, out_dir), out_dir / f.name)
    body = strip_variants(text)
    if arch:
        if re.search(r"(?m)^arch\s*=", body):
            body = re.sub(r'(?m)^arch\s*=\s*"[^"]*"', f'arch = "{arch}"', body)
    variants = []
    good = [
        "[[variants]]",
        'id = "good"',
        'summary = "the base kernel (ptxv-mutate)"',
        'expect = ["clean"]',
    ]
    if base_variant is not None and base_variant.get("patch") and not m.patch:
        good.append("patch = " + _toml_patch(base_variant["patch"]))
    variants.append("\n".join(good))
    mut = [
        "[[variants]]",
        'id = "mut"',
        f"summary = {toml_str(m.summary)}",
        'expect = ["clean"]',
    ]
    if m.edits:
        lines = base_ptx.split("\n")
        for k, e in enumerate(m.edits):
            marker = f"@MUT{k}"
            if f"// {marker}" in lines[e.line]:
                raise ValueError("marker clash")
            lines[e.line] = lines[e.line] + f"  // {marker}"
            if k == 0:
                mut += [
                    f'marker = "{marker}"',
                    f"find = {toml_str(e.find)}",
                    f"replace = {toml_str(e.replace)}",
                ]
            else:
                mut.append(
                    f'[[variants.edits]]\nmarker = "{marker}"\nfind = {toml_str(e.find)}\n'
                    f"replace = {toml_str(e.replace)}"
                )
        ptx = "\n".join(lines)
        if arch:
            ptx = re.sub(r"(?m)^(\s*\.target\s+)sm_\w+", r"\g<1>" + arch, ptx)
        (out_dir / ptx_name).write_text(ptx)
    else:
        if ptx_name and base_ptx is not None:
            (out_dir / ptx_name).write_text(base_ptx)
        patches = list(m.patch)
        if base_variant is not None and base_variant.get("patch"):
            patches = list(base_variant["patch"]) + patches
        mut.insert(4, "patch = " + _toml_patch(patches))
    # Keep the [[variants.edits]] sub-tables after the variant's own keys.
    variants.append("\n".join(mut))
    (out_dir / "case.toml").write_text(body + "\n" + "\n\n".join(variants) + "\n")
    (out_dir / "mutant.json").write_text(
        json.dumps(
            {
                "case": str(case_dir),
                "id": m.id,
                "op": m.op,
                "summary": m.summary,
                "lines": m.lines,
                "edits": [
                    {"line": e.line + 1, "find": e.find, "replace": e.replace}
                    for e in m.edits
                ],
                "patch": m.patch,
                "base_variant": base_variant["id"] if base_variant else "good",
            },
            indent=1,
        )
        + "\n"
    )
    return out_dir


def _toml_patch(ps: list[dict]) -> str:
    def val(v):
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, (int, float)):
            return str(v)
        if isinstance(v, list):
            return "[" + ", ".join(val(x) for x in v) + "]"
        return toml_str(v)

    return (
        "["
        + ", ".join(
            "{ " + ", ".join(f"{k} = {val(v)}" for k, v in p.items()) + " }" for p in ps
        )
        + "]"
    )


def mutant_from_dir(d: Path) -> tuple[Mutant, Path, str | None]:
    """(mutant, base case dir, base PTX) of a written mutant directory."""
    mj = json.loads((d / "mutant.json").read_text())
    m = Mutant(
        id=mj["id"],
        op=mj["op"],
        summary=mj["summary"],
        lines=mj.get("lines", []),
        edits=[
            Edit(line=e["line"] - 1, find=e["find"], replace=e["replace"])
            for e in mj.get("edits", [])
        ],
        patch=mj.get("patch", []),
    )
    man = tomllib.loads((d / "case.toml").read_text())
    src = None
    if man.get("ptx") and not man.get("cubin"):
        src = re.sub(r"  // @MUT\d+$", "", (d / man["ptx"]).read_text(), flags=re.M)
    return m, Path(mj["case"]), src


def write_probe(d: Path, out: Path) -> Path:
    """The reach probe (reach_probe) of the mutant written at `d`."""
    m, case, src = mutant_from_dir(d)
    man = tomllib.loads((case / "case.toml").read_text())
    bv = next((v for v in man.get("variants", []) if v["id"] == "good"), None)
    return write_mutant(case, out, reach_probe(m), src, bv)


def nvdisasm(cubin: Path) -> str:
    return subprocess.run(
        ["nvdisasm", "-c", str(cubin)], check=True, capture_output=True, text=True
    ).stdout


def generate(
    case_dir: Path,
    out_root: Path,
    base: str = "good",
    arch: str | None = None,
    ops: set[str] | None = None,
) -> list[Path]:
    """All mutants of one case (its variant `base` as the base kernel)."""
    case_dir = Path(case_dir)
    man = tomllib.loads((case_dir / "case.toml").read_text())
    variants = {v["id"]: v for v in man.get("variants", [])}
    bv = variants.get(base)
    if bv is None and base != "good":
        raise ValueError(f"{case_dir}: no variant {base}")
    out = []
    if man.get("cubin"):
        dis = nvdisasm(case_dir / man["cubin"])
        entries = {man["entry"]} | {
            l["entry"] for l in man.get("launches", []) if not l.get("cubin")
        }
        ks = cubin_kernels(dis)
        muts = []
        for k in sorted(entries):
            if k not in ks:
                continue
            muts += sass_mutants(ks[k], kernel=None if k == man["entry"] else k)
        base_ptx = None
    else:
        base_ptx = (case_dir / man["ptx"]).read_text()
        if bv is not None and (bv.get("marker") or bv.get("edits")):
            base_ptx = apply_variant_ptx(base_ptx, bv)
        muts = ptx_mutants(base_ptx)
    for m in muts:
        if ops and m.op not in ops:
            continue
        out.append(write_mutant(case_dir, out_root / m.id, m, base_ptx, bv, arch))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("case", type=Path)
    ap.add_argument(
        "--out", type=Path, required=True, help="mutants go to OUT/<mutant-id>/"
    )
    ap.add_argument("--base", default="good", help="base variant (default: good)")
    ap.add_argument(
        "--arch", default=None, help="rewrite arch / .target (e.g. sm_103a)"
    )
    ap.add_argument("--ops", default=None, help="comma list of operators to keep")
    ap.add_argument("--list", action="store_true", help="only list the mutants")
    a = ap.parse_args(argv)
    if a.list:
        man = tomllib.loads((a.case / "case.toml").read_text())
        if man.get("cubin"):
            ms = sass_mutants(nvdisasm(a.case / man["cubin"]))
        else:
            src = (a.case / man["ptx"]).read_text()
            bv = next((v for v in man.get("variants", []) if v["id"] == a.base), None)
            if bv:
                src = (
                    apply_variant_ptx(src, bv)
                    if (bv.get("marker") or bv.get("edits"))
                    else src
                )
            ms = ptx_mutants(src)
        for m in ms:
            print(f"{m.id:32} {m.summary}")
        return
    ds = generate(
        a.case, a.out, a.base, a.arch, set(a.ops.split(",")) if a.ops else None
    )
    print(f"{len(ds)} mutants -> {a.out}")


if __name__ == "__main__":
    main()
