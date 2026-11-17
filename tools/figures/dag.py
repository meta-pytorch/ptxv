# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Draw the numerics DAG errdist analyzes for one output element, for the reference and the
candidate of an equiv.toml pair, side by side (a hand-laid-out SVG, light and dark).

The DAG is ptxv's own: `ptxv errdist <equiv.toml> --candidate N --compile FILE` writes the
programs (`errdist::chain::Prog`, docs/errdist.md §4) that this script reads. Each tensor-core
step is drawn with its 16 products as one node, fed by the input slices (leaves) it reads;
the products are coloured by the K range they cover, so a reordered accumulation shows as a
different colour order down the chain. Rows are aligned across the two kernels: tensor-core
steps first, then the epilogue, with an empty slot where one kernel has an operation the other
lacks.

    tools/figures/dag.py fixtures/c_equiv_mm_bf16/equiv.toml --candidate 1 \\
        --titles "cuBLAS (K in order)" "Triton (K blocks reversed)" \\
        -o docs/img/errdist_dag            # writes errdist_dag.svg and errdist_dag-dark.svg

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

# Reference palette (categorical slots 1-4) for the K ranges; everything else stays neutral.
K_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
THEMES = {
    "light": dict(
        bg="#ffffff",
        panel="#f6f8fa",
        border="#d0d7de",
        card="#ffffff",
        ink="#1f2328",
        ink2="#59636e",
        spine="#8c959f",
        ghost="#d0d7de",
        strong="#1f2328",
        on_strong="#ffffff",
        tint=0.14,
        shadow=0.10,
    ),
    "dark": dict(
        bg="#0d1117",
        panel="#151b23",
        border="#3d444d",
        card="#212830",
        ink="#f0f6fc",
        ink2="#9198a1",
        spine="#656c76",
        ghost="#3d444d",
        strong="#f0f6fc",
        on_strong="#0d1117",
        tint=0.22,
        shadow=0.0,
    ),
}
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, 'SF Mono', Menlo, Consolas, 'Liberation Mono', monospace"


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


def mix(hexcol, base, w):
    """`hexcol` mixed into `base` (weight w of the colour)."""
    c = [int(hexcol[i : i + 2], 16) for i in (1, 3, 5)]
    d = [int(base[i : i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(y + (x - y) * w):02x}" for x, y in zip(c, d))


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


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def rows(prog, out, shapes, krange_color):
    """The output's DAG as rows, inputs first: ("mma", ...) per tensor-core step, then
    ("op", title, detail) per epilogue operation, ending with the final rounding."""
    ops = prog["ops"]
    bufs = prog["bufs"]
    mma, post = [], []
    for i in chain(prog, out):
        k = kind(ops[i])
        v = ops[i][k]
        if k in ("Leaf", "Const"):
            continue  # drawn with their consumer
        if k == "Mma":
            # Leaves, normalized to (A, B) order by buffer name.
            pairs = [tuple(sorted((a, b), key=lambda x: bufs[ops[x]["Leaf"]["buf"]])) for a, b in v["terms"]]
            labels = []
            for side in range(2):
                lv = [ops[p[side]]["Leaf"] for p in pairs]
                bname = bufs[lv[0]["buf"]]
                labels.append(slice_label(bname, [x["idx"] for x in lv], shapes[bname]))
            first = v["acc"] is None
            mma.append(
                dict(
                    color=krange_color(labels[0]),
                    k=krange(labels[0]),
                    n=len(pairs),
                    leaves=" · ".join(labels),
                    detail=(
                        ["truncate terms to 2^(emax−25),", "sum, round toward zero → f32"]
                        if first
                        else ["+ accumulator, same rounding"]
                    ),
                )
            )
        elif k == "Add":
            consts = [x for x in (v["a"], v["b"]) if kind(ops[x]) == "Const"]
            mode = "toward zero" if v["rz"] else "to nearest even"
            what = f"add {ops[consts[0]]['Const']:g} (epilogue)" if consts else "add"
            post.append(("op", what, f"round {mode} → f32", False))
        elif k == "Round":
            fmt = "f16" if v["f16"] else "bf16"
            post.append(("op", f"round to nearest even → {fmt}", None, True))
        else:
            post.append(("op", k, None, False))
    return mma, post


class Svg:
    def __init__(self):
        self.parts = []

    def add(self, s):
        self.parts.append(s)

    def text(self, x, y, s, size, fill, weight=400, anchor="start", family=FONT, extra=""):
        # "2^(e)" is drawn as a superscript.
        body = esc(s)
        if "^(" in s:
            pre, rest = s.split("^(", 1)
            sup, post = rest.split(")", 1)
            body = (
                f'{esc(pre)}<tspan dy="-0.45em" font-size="{size * 0.72:.1f}">{esc(sup)}</tspan>'
                f'<tspan dy="0.45em">{esc(post)}</tspan>'
            )
        self.add(
            f'<text x="{x:.1f}" y="{y:.1f}" font-family="{family}" font-size="{size}" font-weight="{weight}" '
            f'fill="{fill}" text-anchor="{anchor}"{extra}>{body}</text>'
        )


# Geometry (px at 1x).
W_PANEL, GAP, MARGIN = 480, 24, 24
HEAD, ROW, CARD_H = 76, 72, 54
CHIP_X, CHIP_W = 20, 186
CARD_X, CARD_W = 236, 224


def render(theme, titles, kernels, out_label):
    t = THEMES[theme]
    n_mma = max(len(m) for m, _ in kernels)
    n_post = max(len(p) for _, p in kernels)
    height_panel = HEAD + (n_mma + n_post) * ROW + 70
    W = 2 * MARGIN + 2 * W_PANEL + GAP
    H = MARGIN + height_panel + 56
    g = Svg()
    g.add(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
        f'role="img" aria-label="The numerics DAG of one output for two kernels">'
    )
    colors = sorted({r["color"] for m, _ in kernels for r in m}) + [t["spine"]]
    defs = ["<defs>"]
    if t["shadow"]:
        defs.append(
            f'<filter id="sh" x="-10%" y="-20%" width="120%" height="150%">'
            f'<feDropShadow dx="0" dy="1" stdDeviation="1.6" flood-color="#1f2328" flood-opacity="{t["shadow"]}"/></filter>'
        )
    for i, c in enumerate(colors):
        defs.append(
            f'<marker id="a{i}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
            f'orient="auto-start-reverse"><path d="M1,1.5 L9,5 L1,8.5 z" fill="{c}"/></marker>'
        )
    defs.append("</defs>")
    g.add("".join(defs))
    arrow = {c: f"url(#a{i})" for i, c in enumerate(colors)}
    shadow = ' filter="url(#sh)"' if t["shadow"] else ""
    g.add(f'<rect width="{W}" height="{H}" fill="{t["bg"]}"/>')

    for p, ((mma, post), title) in enumerate(zip(kernels, titles)):
        x0 = MARGIN + p * (W_PANEL + GAP)
        y0 = MARGIN
        g.add(
            f'<rect x="{x0}" y="{y0}" width="{W_PANEL}" height="{height_panel}" rx="14" '
            f'fill="{t["panel"]}" stroke="{t["border"]}"/>'
        )
        name, _, sub = title.partition("|")
        g.text(x0 + 22, y0 + 34, name.strip(), 17, t["ink"], 600)
        if sub:
            g.text(x0 + 22, y0 + 56, sub.strip(), 13, t["ink2"])
        cx = x0 + CARD_X + CARD_W / 2

        def card_y(row):
            return y0 + HEAD + row * ROW

        # Spine first, so cards sit on top of it.
        slots = list(range(n_mma, n_mma + n_post))
        g.add(
            f'<line x1="{cx}" y1="{card_y(0) + CARD_H}" x2="{cx}" y2="{card_y(n_mma + n_post) + 6}" '
            f'stroke="{t["spine"]}" stroke-width="1.5"/>'
        )
        for r in range(1, n_mma + n_post):
            yy = card_y(r)
            g.add(
                f'<line x1="{cx}" y1="{yy - 12}" x2="{cx}" y2="{yy - 1}" stroke="{t["spine"]}" '
                f'stroke-width="1.5" marker-end="{arrow[t["spine"]]}"/>'
            )

        for r in range(len(mma)):
            s = mma[r]
            y = card_y(r)
            col = s["color"]
            # Products chip.
            g.add(
                f'<rect x="{x0 + CHIP_X}" y="{y}" width="{CHIP_W}" height="{CARD_H}" rx="10" '
                f'fill="{mix(col, t["panel"], t["tint"])}" stroke="{mix(col, t["panel"], 0.55)}"/>'
            )
            g.add(f'<rect x="{x0 + CHIP_X}" y="{y + 10}" width="4" height="{CARD_H - 20}" rx="2" fill="{col}"/>')
            g.text(x0 + CHIP_X + 16, y + 23, f"k = {s['k']}", 13, t["ink"], 600)
            g.text(x0 + CHIP_X + CHIP_W - 12, y + 23, f"{s['n']} products", 11, t["ink2"], anchor="end")
            g.text(x0 + CHIP_X + 16, y + 42, s["leaves"], 10.5, t["ink2"], family=MONO)
            # Feed into the step.
            ym = y + CARD_H / 2
            g.add(
                f'<line x1="{x0 + CHIP_X + CHIP_W}" y1="{ym}" x2="{x0 + CARD_X - 2}" y2="{ym}" '
                f'stroke="{col}" stroke-width="2" marker-end="{arrow[col]}"/>'
            )
            # Step card.
            g.add(
                f'<rect x="{x0 + CARD_X}" y="{y}" width="{CARD_W}" height="{CARD_H}" rx="10" '
                f'fill="{t["card"]}" stroke="{t["border"]}"{shadow}/>'
            )
            g.add(f'<circle cx="{x0 + CARD_X + 22}" cy="{ym}" r="11" fill="{mix(col, t["card"], 0.18)}"/>')
            g.text(x0 + CARD_X + 22, ym + 4.5, str(r + 1), 12, col, 700, "middle")
            lines = s["detail"]
            ty = ym - 4 - 7 * (len(lines) - 1)
            g.text(x0 + CARD_X + 42, ty, "tensor-core step", 13, t["ink"], 600)
            for j, ln in enumerate(lines):
                g.text(x0 + CARD_X + 42, ty + 16 + 14 * j, ln, 11, t["ink2"])

        # Epilogue, bottom-aligned so the final rounding lines up; empty slots where the
        # other kernel has an operation this one lacks.
        pad = n_post - len(post)
        for j, r in enumerate(slots):
            y = card_y(r)
            ym = y + CARD_H / 2
            if j < pad:
                g.add(
                    f'<rect x="{x0 + CARD_X}" y="{y}" width="{CARD_W}" height="{CARD_H}" rx="10" '
                    f'fill="{t["panel"]}" stroke="{t["ghost"]}" stroke-dasharray="4 4"/>'
                )
                g.text(cx, ym + 4, "no epilogue op", 11.5, t["ink2"], 400, "middle", extra=' font-style="italic"')
                continue
            _, what, detail, strong = post[j - pad]
            fill, ink, ink2 = (
                (t["strong"], t["on_strong"], t["on_strong"]) if strong else (t["card"], t["ink"], t["ink2"])
            )
            g.add(
                f'<rect x="{x0 + CARD_X}" y="{y}" width="{CARD_W}" height="{CARD_H}" rx="10" '
                f'fill="{fill}" stroke="{t["border"] if not strong else fill}"{shadow}/>'
            )
            if detail:
                g.text(cx, ym - 3, what, 13, ink, 600, "middle")
                g.text(cx, ym + 14, detail, 11, ink2, 400, "middle")
            else:
                g.text(cx, ym + 4.5, what, 13, ink, 600, "middle")

        # Output.
        y = card_y(n_mma + n_post) + 8
        w = 9 * len(out_label) + 28
        g.add(
            f'<rect x="{cx - w / 2}" y="{y}" width="{w}" height="30" rx="15" fill="{t["card"]}" '
            f'stroke="{t["strong"]}" stroke-width="1.5"/>'
        )
        g.text(cx, y + 20, out_label, 13, t["ink"], 600, "middle", family=MONO)

    # Legend: one swatch per K block, in K order.
    ks = {}
    for m, _ in kernels:
        for r in m:
            ks.setdefault(r["k"], r["color"])
    ly = MARGIN + height_panel + 34
    x = MARGIN + 2
    g.text(x, ly, "Colour = K block:", 12, t["ink2"])
    x += 112
    for k, c in sorted(ks.items(), key=lambda kv: int(kv[0].split("…")[0])):
        g.add(f'<rect x="{x}" y="{ly - 10}" width="12" height="12" rx="3" fill="{c}"/>')
        g.text(x + 18, ly, f"k = {k}", 12, t["ink"])
        x += 96
    g.text(
        W - MARGIN - 2,
        ly,
        "Same 64 exact products in both kernels; only the order of the f32 roundings differs.",
        12,
        t["ink2"],
        anchor="end",
    )
    g.add("</svg>")
    return "\n".join(g.parts) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("equiv", help="equiv.toml (reference + candidates)")
    ap.add_argument("--candidate", type=int, default=0)
    ap.add_argument("--model", help="an already compiled model file (skips running ptxv)")
    ap.add_argument("--output", type=int, default=0, help="flat output element index")
    ap.add_argument("--titles", nargs=2, default=None, help='titles for reference, candidate; "name | subtitle"')
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
    out_label = slice_label(oname, [a.output], oshape)

    colors = {}

    def krange_color(key):
        if key not in colors:
            colors[key] = K_COLORS[len(colors) % len(K_COLORS)]
        return colors[key]

    kernels = [rows(m[side], a.output, shapes, krange_color) for side in ("r", "c")]
    titles = a.titles or m["labels"]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for theme, suffix in (("light", ""), ("dark", "-dark")):
        path = out.with_name(out.name + suffix + ".svg")
        path.write_text(render(theme, titles, kernels, out_label))
        print(path)


if __name__ == "__main__":
    main()
