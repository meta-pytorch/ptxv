# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Draw the numerics DAG errdist analyzes for one output element, for the reference and the
candidate of an equiv.toml pair, side by side (graphviz).

The DAG is ptxv's own: `ptxv errdist <equiv.toml> --candidate N --compile FILE` writes the
programs (`errdist::chain::Prog`, docs/errdist.md §4) that this script reads. Each tensor-core
step is drawn with its 16 products as one node, fed by the input slices (leaves) it reads;
the products are coloured by the K range they cover, so a reordered accumulation shows as a
different colour order down the chain.

    tools/figures/dag.py fixtures/c_equiv_mm_bf16/equiv.toml --candidate 1 \\
        --titles "cuBLAS (K in order)" "Triton (K blocks reversed)" \\
        -o docs/img/errdist_dag            # writes .svg and .png

ptxv is found as $PTXV_BIN, else $CARGO_TARGET_DIR/release/ptxv, else target/release/ptxv.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Reference palette (categorical slots 1-4) for the K ranges; text and rules stay neutral.
K_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK = "#0b0b0b"
INK2 = "#52514e"
RULE = "#9a9994"
SURFACE = "#fcfcfb"
FONT = "DejaVu Sans"
MONO = FONT


def ptxv_bin():
    if os.environ.get("PTXV_BIN"):
        return os.environ["PTXV_BIN"]
    t = os.environ.get("CARGO_TARGET_DIR")
    for p in ([Path(t) / "release/ptxv"] if t else []) + [ROOT / "target/release/ptxv"]:
        if p.exists():
            return str(p)
    sys.exit("ptxv binary not found (set PTXV_BIN or build with cargo build --release)")


def compile_model(equiv, cand, path):
    subprocess.run(
        [ptxv_bin(), "errdist", str(equiv), "--candidate", str(cand), "--compile", str(path)],
        check=True,
        stdout=subprocess.DEVNULL,
    )


def unravel(idx, shape):
    out = []
    for d in reversed(shape):
        out.append(idx % d)
        idx //= d
    return list(reversed(out))


def slice_label(name, idxs, shape):
    """'a[0, 0:16]' for a set of flat indices that forms a box."""
    pts = [unravel(i, shape) for i in idxs]
    parts = []
    for d in range(len(shape)):
        vals = sorted({p[d] for p in pts})
        if len(vals) == 1:
            parts.append(str(vals[0]))
        elif vals == list(range(vals[0], vals[-1] + 1)):
            parts.append(f"{vals[0]}:{vals[-1] + 1}")
        else:
            parts.append("{" + ",".join(map(str, vals)) + "}")
    return f"{name}[{', '.join(parts)}]"


def tint(hexcol, w=0.22):
    """`hexcol` mixed with white (weight w of the colour)."""
    c = [int(hexcol[i : i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(255 - (255 - x) * w):02x}" for x in c)


def krange(label):
    """'a[0, 16:32]' -> '16…31' (the last dimension's range, inclusive)."""
    last = label.rstrip("]").split(",")[-1].strip()
    if ":" in last:
        lo, hi = last.split(":")
        return f"{lo}\u2026{int(hi) - 1}"
    return last


def kind(op):
    return next(iter(op))


def chain(prog, out):
    """The ops of output `out`, children first (reachable set, in program order)."""
    ops = prog["ops"]
    seen = set()
    stack = [prog["outputs"][out]]
    while stack:
        i = stack.pop()
        if i in seen or i is None:
            continue
        seen.add(i)
        k = kind(ops[i])
        v = ops[i][k]
        if k == "Mma":
            if v["acc"] is not None:
                stack.append(v["acc"])
            for a, b in v["terms"]:
                stack += [a, b]
        elif k in ("Round", "Tf32"):
            stack.append(v["arg"])
        elif k in ("Add", "Fma"):
            stack += [v[x] for x in ("a", "b", "c") if isinstance(v.get(x), int)]
    return sorted(seen)


def html(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def kernel_dot(prog, out, shapes, tag, title, krange_color):
    ops = prog["ops"]
    bufs = prog["bufs"]
    lines = [
        f'subgraph cluster_{tag} {{',
        f'label=<<b>{html(title)}</b>>; labeljust=l; fontsize=15; fontname="{FONT}"; '
        f'fontcolor="{INK}"; color="{RULE}"; style="rounded"; margin=14;',
    ]
    name = {}
    step = 0
    out_fmt = prog["out_fmt"].lower() if isinstance(prog["out_fmt"], str) else "out"
    for i in chain(prog, out):
        k = kind(ops[i])
        v = ops[i][k]
        n = f"{tag}{i}"
        name[i] = n
        if k == "Leaf" or k == "Const":
            continue  # drawn with their consumer
        if k == "Mma":
            step += 1
            # Leaves, normalized to (A, B) order by buffer name.
            pairs = [tuple(sorted((a, b), key=lambda x: bufs[ops[x]["Leaf"]["buf"]])) for a, b in v["terms"]]
            labels = []
            for side in range(2):
                lv = [ops[p[side]]["Leaf"] for p in pairs]
                bname = bufs[lv[0]["buf"]]
                labels.append(slice_label(bname, [x["idx"] for x in lv], shapes[bname]))
            key = labels[0]
            col = krange_color(key)
            lines.append(
                f'{n}l [shape=box, style="rounded,filled", fillcolor="white", color="{RULE}", '
                f'fontname="{FONT}", fontsize=12, fontcolor="{INK2}", label="{labels[0]}\\n{labels[1]}"];'
            )
            lines.append(
                f'{n}p [shape=box, style="rounded,filled", fillcolor="{tint(col)}", color="{col}", penwidth=1.6, '
                f'fontcolor="{INK}", fontname="{FONT}", fontsize=12, '
                f'label=<{len(pairs)} exact products<br/>k = {krange(labels[0])}>];'
            )
            lines.append(f'{n}l -> {n}p [color="{RULE}", arrowsize=0.5];')
            first = v["acc"] is None
            detail = (
                "truncate terms to 2<sup>emax-25</sup>,<br/>add, round toward zero to f32"
                if step == 1
                else "+ accumulator, same rounding"
            )
            lines.append(
                f'{n} [shape=box, style="rounded,filled", fillcolor="white", color="{INK}", penwidth=1.4, '
                f'fontname="{FONT}", fontsize=12, fontcolor="{INK}", '
                f'label=<<b>tensor-core step {step}</b><br/><font point-size="11" color="{INK2}">{detail}</font>>];'
            )
            lines.append(f'{n}p -> {n} [color="{col}", penwidth=1.6, arrowsize=0.6];')
            if not first:
                lines.append(f'{name[v["acc"]]} -> {n} [color="{INK}", penwidth=1.4, arrowsize=0.6, weight=10];')
        elif k == "Add":
            a, b = v["a"], v["b"]
            consts = [x for x in (a, b) if kind(ops[x]) == "Const"]
            mode = "toward zero" if v["rz"] else "to nearest even"
            what = (
                f"add {ops[consts[0]]['Const']:g} (epilogue)" if consts else "add"
            )
            lines.append(
                f'{n} [shape=box, style="rounded,filled", fillcolor="white", color="{INK}", penwidth=1.0, '
                f'fontname="{FONT}", fontsize=12, fontcolor="{INK}", '
                f'label=<{what}<br/><font point-size="11" color="{INK2}">round {mode} to f32</font>>];'
            )
            for x in (a, b):
                if kind(ops[x]) != "Const":
                    lines.append(f'{name[x]} -> {n} [color="{INK}", penwidth=1.4, arrowsize=0.6, weight=10];')
        elif k == "Round":
            fmt = "f16" if v["f16"] else "bf16"
            lines.append(
                f'{n} [shape=box, style="rounded,filled", fillcolor="white", color="{INK}", penwidth=2.2, '
                f'fontname="{FONT}", fontsize=12, fontcolor="{INK}", '
                f'label=<<b>round to nearest even to {fmt}</b>>];'
            )
            lines.append(f'{name[v["arg"]]} -> {n} [color="{INK}", penwidth=1.4, arrowsize=0.6, weight=10];')
        else:
            lines.append(f'{n} [shape=box, label="{k}"];')
            for x in ("arg", "a", "b", "c"):
                if isinstance(v, dict) and isinstance(v.get(x), int):
                    lines.append(f"{name[v[x]]} -> {n};")
    root = name[prog["outputs"][out]]
    oname = f"{tag}out"
    lines.append(
        f'{oname} [shape=plaintext, fontname="{MONO}", fontsize=12, fontcolor="{INK}", label="{OUT_LABEL}"];'
    )
    lines.append(f'{root} -> {oname} [color="{INK}", penwidth=1.4, arrowsize=0.6, weight=10];')
    lines.append("}")
    return lines


OUT_LABEL = "out"


def main():
    global OUT_LABEL
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("equiv", help="equiv.toml (reference + candidates)")
    ap.add_argument("--candidate", type=int, default=0)
    ap.add_argument("--model", help="an already compiled model file (skips running ptxv)")
    ap.add_argument("--output", type=int, default=0, help="flat output element index")
    ap.add_argument("--titles", nargs=2, default=None, help="titles for reference, candidate")
    ap.add_argument("-o", "--out", required=True, help="output path without extension")
    a = ap.parse_args()

    if a.model:
        mpath = Path(a.model)
    else:
        mpath = Path(tempfile.mkdtemp()) / "model.json"
        compile_model(a.equiv, a.candidate, mpath)
    m = json.load(open(mpath))["programs"]
    shapes = {l["name"]: l["shape"] for l in m["layouts"]}
    oname, oshape = m["out_name"]
    OUT_LABEL = slice_label(oname, [a.output], oshape)

    colors = {}

    def krange_color(key):
        if key not in colors:
            colors[key] = K_COLORS[len(colors) % len(K_COLORS)]
        return colors[key]

    titles = a.titles or m["labels"]
    g = [
        "digraph errdist_dag {",
        f'graph [rankdir=TB, bgcolor="{SURFACE}", fontname="{FONT}", nodesep=0.25, ranksep=0.32, '
        f'pad=0.25, newrank=true];',
        f'node [fontname="{FONT}"]; edge [color="{RULE}"];',
    ]
    g += kernel_dot(m["r"], a.output, shapes, "r", titles[0], krange_color)
    g += kernel_dot(m["c"], a.output, shapes, "c", titles[1], krange_color)
    g.append("}")
    dot = "\n".join(g) + "\n"
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["dot", "-Tsvg", "-o", str(out.with_suffix(".svg"))], input=dot.encode(), check=True)
    subprocess.run(["dot", "-Tpng", "-Gdpi=144", "-o", str(out.with_suffix(".png"))], input=dot.encode(), check=True)
    print(out.with_suffix(".svg"), out.with_suffix(".png"))


if __name__ == "__main__":
    main()
