# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Animate a witness schedule of `ptxv check` on SASS: warps as columns with their SASS and
pc, the mbarrier state, the shared-memory accesses, and the finding when it is reported.

The frames come from ptxv's own replay of the finding's trace (`PTXV_TRACE_DUMP=<dir>`,
crates/cli/src/trace_dump.rs). The last frame states the verdict, the witness schedule, and
the verdict of the variant given with --good.

    tools/figures/sim_gif.py fixtures/e_stasync --variant SA2 --good good \\
        --good-note "CTA 0 waits for the mbarrier phase (SYNCS.PHASECHK.TRANS64.TRYWAIT loop) before the LDS." \\
        -o docs/img/sim.gif

ptxv is found as $PTXV_BIN, else $CARGO_TARGET_DIR/release/ptxv, else target/release/ptxv.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]

W, H = 1000, 600
SURFACE = (252, 252, 251)
INK = (11, 11, 11)
INK2 = (82, 81, 78)
MUTED = (154, 153, 148)
RULE = (222, 221, 216)
# Reference palette: categorical slots 1-2 for the two warps; red only for the finding.
WARP_COL = [(42, 120, 214), (235, 104, 52), (27, 175, 122), (237, 161, 0)]
ASYNC_COL = (120, 119, 114)
RED = (227, 73, 72)


def font(mono=False, bold=False, size=13):
    fam = "DejaVu Sans Mono" if mono else "DejaVu Sans"
    p = font_manager.findfont(font_manager.FontProperties(family=fam, weight="bold" if bold else "normal"))
    return ImageFont.truetype(p, size)


F = {
    "title": font(True, True, 15),
    "sub": font(False, False, 13),
    "head": font(False, True, 14),
    "code": font(True, False, 12),
    "codeb": font(True, True, 12),
    "small": font(False, False, 12),
    "smallb": font(False, True, 12),
    "mono11": font(True, False, 11),
    "big": font(False, True, 18),
}


def tint(c, w):
    return tuple(round(255 - (255 - x) * w) for x in c)


def ptxv_bin():
    if os.environ.get("PTXV_BIN"):
        return os.environ["PTXV_BIN"]
    t = os.environ.get("CARGO_TARGET_DIR")
    for p in ([Path(t) / "release/ptxv"] if t else []) + [ROOT / "target/release/ptxv"]:
        if p.exists():
            return str(p)
    sys.exit("ptxv binary not found (set PTXV_BIN or build with cargo build --release)")


def verdict_line(line):
    """'PASS SA2 sass [race] states=29 ... 0.01s' -> 'SA2 sass [race] states=29' (no PASS/FAIL
    against the case's expectation, no timing)."""
    t = line.split()
    keep = [x for x in t[1:] if not re.fullmatch(r"[0-9.]+s|terminal=\d+|outputs=\d+", x)]
    return " ".join(keep)


def run_check(case, variant, extra=()):
    d = Path(tempfile.mkdtemp())
    env = dict(os.environ, PTXV_TRACE_DUMP=str(d))
    r = subprocess.run(
        [ptxv_bin(), "check", str(case), "--variant", variant, "--front", "sass", "--out", str(d / "out"), *extra],
        env=env,
        capture_output=True,
        text=True,
    )
    first = next((l for l in r.stdout.splitlines() if l.startswith(("PASS", "FAIL"))), "")
    return json.load(open(d / f"{variant}.sass.trace.json")), r.stdout, verdict_line(first)


def hexloc(code, pc):
    return f"{code[pc]['loc']:#06x}"


def lanes_str(ls):
    ls = sorted(ls)
    if len(ls) == 1:
        return f"t{ls[0]}"
    if ls == list(range(ls[0], ls[-1] + 1)):
        return f"t{ls[0]}–{ls[-1]}"
    return f"{len(ls)} lanes"


STATE = {"Run": "ready", "Exited": "exited", "ClusterWait": "waits at cluster barrier", "Bar": "waits at barrier"}


class Anim:
    def __init__(self, dump, case_name, variant, desc_lines):
        self.d = dump
        self.code = dump["code"]
        # Drop the padding after the last EXIT (self-branch and NOPs).
        last = max(i for i, c in enumerate(self.code) if c["text"].startswith("EXIT"))
        self.ncode = last + 1
        self.case_name = case_name
        self.variant = variant
        self.desc_lines = desc_lines
        self.run = next(r for r in dump["runs"] if r.get("finding") and r["finding"]["kind"] == "race" and "steps" in r)
        self.warps = [w["warp"] for w in self.run["initial"]["warps"]]
        self.wcta = {w["warp"]: w["cta"] for w in self.run["initial"]["warps"]}
        f = self.run["finding"]
        self.race_pcs = f["pcs"]

    def actor(self, step):
        a = step["action"]
        if a["kind"] == "warp":
            return a["warp"]
        m = re.search(r"by cta(\d+)\.t(\d+)", step["desc"])
        if m:  # an async op: draw it in the issuing warp's column
            cta = int(m.group(1))
            return next(w for w in self.warps if self.wcta[w] == cta)
        return None

    def executed(self, step, wstate):
        """pcs the step ran in its warp (from the start pc to the next stop or branch)."""
        a = step["action"]
        if a["kind"] != "warp":
            m = re.search(r"sass:(0x[0-9a-f]+)", step["desc"])
            loc = int(m.group(1), 16) if m else None
            return [i for i, c in enumerate(self.code) if c["loc"] == loc]
        start = a["pc"]
        nxt = [g["pc"] for g in wstate["groups"] if g["state"] != "Exited"] or [g["pc"] for g in wstate["groups"]]
        end = max(nxt)
        out = [start]
        pc = start
        while pc + 1 < end and not self.code[pc]["text"].startswith(("BRA", "EXIT")) and len(out) < 12:
            pc += 1
            out.append(pc)
        return out

    # ---------------------------------------------------------------- drawing

    def frame(self, k, final=False, race=False):
        """Frame after step k (k = -1: initial state)."""
        steps = self.run["steps"]
        st = steps[k] if k >= 0 else None
        wsts = (st or self.run["initial"])["warps"]
        bars = (st or self.run["initial"])["barriers"]
        im = Image.new("RGB", (W, H), SURFACE)
        d = ImageDraw.Draw(im)
        # Header.
        d.text((20, 12), f"ptxv check fixtures/{self.case_name} --variant {self.variant} --front sass", font=F["title"], fill=INK)
        y = 36
        for line in self.desc_lines:
            d.text((20, y), line, font=F["sub"], fill=INK2)
            y += 17
        top = y + 8
        # Warp columns.
        act = self.actor(st) if st else None
        colw = 330
        for ci, w in enumerate(self.warps):
            x0 = 20 + ci * (colw + 12)
            col = WARP_COL[ci % len(WARP_COL)]
            ws = next(x for x in wsts if x["warp"] == w)
            d.rectangle([x0, top, x0 + 6, top + 18], fill=col)
            d.text((x0 + 12, top), f"warp {w}  (CTA {ws['cta']})", font=F["head"], fill=INK)
            parts = []
            for g in sorted(ws["groups"], key=lambda g: g["lanes"][0]):
                s = STATE.get(g["state"], g["state"])
                parts.append(f"{lanes_str(g['lanes'])} {s}")
            d.text((x0 + 12, top + 21), ", ".join(parts), font=F["small"], fill=INK2)
            pcs = [g["pc"] for g in ws["groups"] if g["state"] != "Exited"]
            focus = pcs[0] if pcs else max(g["pc"] for g in ws["groups"])
            ran = self.executed(st, ws) if (st and act == w) else []
            nl = 19
            lo = max(0, min(focus - 7, self.ncode - nl))
            ly = top + 44
            d.rectangle([x0, ly - 3, x0 + colw, ly + nl * 18 + 1], outline=RULE)
            for pc in range(lo, min(lo + nl, self.ncode)):
                yy = ly + (pc - lo) * 18
                hot_race = race and pc in self.race_pcs and self.race_owner(pc) == w
                if hot_race:
                    d.rectangle([x0 + 1, yy - 1, x0 + colw - 1, yy + 16], fill=tint(RED, 0.25))
                elif pc in ran:
                    d.rectangle([x0 + 1, yy - 1, x0 + colw - 1, yy + 16], fill=tint(col, 0.18))
                is_pc = pc in pcs
                if is_pc:
                    d.polygon([(x0 + 4, yy + 2), (x0 + 4, yy + 12), (x0 + 11, yy + 7)], fill=col)
                txt = self.code[pc]["text"]
                if len(txt) > 34:
                    txt = txt[:33] + "…"
                fnt = F["codeb"] if (is_pc or hot_race) else F["code"]
                d.text((x0 + 15, yy), hexloc(self.code, pc)[2:], font=F["code"], fill=MUTED)
                d.text((x0 + 58, yy), txt, font=fnt, fill=INK if (is_pc or pc in ran or hot_race) else INK2)
                if hot_race:
                    d.text((x0 + colw - 44, yy), "RACE", font=F["smallb"], fill=RED)
        # Right panel: barrier state and shared-memory accesses.
        px = 20 + 2 * (colw + 12)
        pw = W - 20 - px
        y = top
        d.text((px, y), "mbarrier", font=F["head"], fill=INK)
        y += 22
        for b in bars["mbarriers"]:
            d.text((px, y), f"CTA {b['cta']}, smem {b['addr']:#x}", font=F["small"], fill=INK2)
            y += 17
            d.text((px + 10, y), f"phase {b['phase']}", font=F["code"], fill=INK)
            y += 16
            d.text((px + 10, y), f"pending arrivals {b['pending']}/{b['expected']}", font=F["code"], fill=INK)
            y += 16
            d.text((px + 10, y), f"tx-count {b['tx']:+d} bytes", font=F["code"], fill=INK)
            y += 22
        if not bars["mbarriers"]:
            d.text((px + 10, y), "not initialized", font=F["small"], fill=MUTED)
            y += 22
        for b in bars["bars"]:
            d.text((px, y), f"barrier {b['id']} (CTA {b['cta']}): {b['waiting']} waiting", font=F["small"], fill=INK2)
            y += 18
        y += 6
        d.line([px, y, px + pw, y], fill=RULE)
        y += 8
        d.text((px, y), "shared-memory accesses", font=F["head"], fill=INK)
        y += 22
        log = []
        for i, s in enumerate(steps[: k + 1]):
            for a in s["smem"]:
                log.append((i, a))
        if not log:
            d.text((px + 10, y), "none yet", font=F["small"], fill=MUTED)
        for i, a in log[-6:]:
            by = a["by"]
            w = by.get("warp")
            ci = self.warps.index(w) if w in self.warps else None
            c = WARP_COL[ci] if ci is not None else ASYNC_COL
            hot = race and by.get("pc") in self.race_pcs
            d.rectangle([px, y + 2, px + 6, y + 14], fill=RED if hot else c)
            rw = "write" if a["rw"] == "write" else "read"
            rng = f"{a['off']:#x}" + (f"+{a['bytes']}" if a["bytes"] > 1 else "")
            d.text((px + 12, y), f"step {i + 1}: {rw} {rng}", font=F["small"], fill=INK)
            y += 16
            inst = self.code[by["pc"]]["text"].split()[0] if "pc" in by else "?"
            d.text((px + 12, y), f"CTA {a['cta']} smem, warp {w} {inst}", font=F["small"], fill=INK2)
            y += 20
        # Bottom: schedule strip and the current trace line.
        by0 = H - 92
        d.line([20, by0 - 8, W - 20, by0 - 8], fill=RULE)
        n = len(steps)
        d.text((20, by0), f"schedule  step {k + 1}/{n}" if k >= 0 else f"schedule  ({n} steps)", font=F["smallb"], fill=INK)
        sx = 180
        bw = min(28, (W - 20 - sx) // n)
        for i, s in enumerate(steps):
            a = self.actor(s)
            ci = self.warps.index(a) if a in self.warps else None
            c = WARP_COL[ci] if ci is not None else ASYNC_COL
            if s["action"]["kind"] != "warp":
                c = tint(c, 0.45)
            fill = c if i <= k or final else tint(c, 0.18)
            xx = sx + i * bw
            d.rectangle([xx, by0 + 1, xx + bw - 3, by0 + 15], fill=fill)
            if i == k and not final:
                d.rectangle([xx - 1, by0, xx + bw - 2, by0 + 16], outline=INK, width=2)
            if s["findings"] and any(f["sig"] == self.run["finding"]["sig"] for f in s["findings"]) and (i <= k or final):
                d.text((xx + bw // 2 - 4, by0 + 16), "▲", font=F["small"], fill=RED)
        lx = sx + n * bw + 14
        for ci, w in enumerate(self.warps):
            d.rectangle([lx, by0 + 3, lx + 10, by0 + 13], fill=WARP_COL[ci])
            d.text((lx + 14, by0 + 1), f"warp {w}", font=F["small"], fill=INK2)
            lx += 70
        d.rectangle([lx, by0 + 3, lx + 10, by0 + 13], fill=tint(WARP_COL[1], 0.45))
        d.text((lx + 14, by0 + 1), "async landing", font=F["small"], fill=INK2)
        if st:
            line = st["desc"]
            if len(line) > 118:
                line = line[:117] + "…"
            d.text((20, by0 + 34), line, font=F["mono11"], fill=INK2)
        if race:
            msg = self.run["finding"]["msg"].splitlines()[0]
            msg = re.sub(r"\s+", " ", msg)
            for j, l in enumerate(textwrap.wrap("RACE: " + msg, 120)[:2]):
                d.text((20, by0 + 52 + j * 15), l, font=F["mono11"], fill=RED)
        return im

    def race_owner(self, pc):
        """The warp whose access at pc is in the race (from the finding message)."""
        m = self.run["finding"]["msg"]
        loc = self.code[pc]["loc"]
        for cta, ploc in re.findall(r"cta(\d+)\.t\d+ at sass:(0x[0-9a-f]+)", m):
            if int(ploc, 16) == loc:
                return next(w for w in self.warps if self.wcta[w] == int(cta))
        return None

    def verdict(self, first_line, good_line, good_note, gb300_note, hw_note):
        im = self.frame(len(self.run["steps"]) - 1, final=True, race=True)
        ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        od = ImageDraw.Draw(ov)
        x0, y0, x1, y1 = 60, 120, W - 60, H - 112
        od.rounded_rectangle([x0, y0, x1, y1], 10, fill=SURFACE + (246,), outline=INK2 + (255,), width=1)
        im = Image.alpha_composite(im.convert("RGBA"), ov).convert("RGB")
        d = ImageDraw.Draw(im)
        y = y0 + 16
        d.text((x0 + 20, y), "verdict", font=F["big"], fill=INK)
        y += 30
        d.text((x0 + 20, y), re.sub(r"\s+", " ", first_line.strip()), font=F["codeb"], fill=RED)
        y += 22
        msg = re.sub(r"\s+", " ", self.run["finding"]["msg"].splitlines()[0])
        for l in textwrap.wrap("RACE: " + msg, 108):
            d.text((x0 + 20, y), l, font=F["code"], fill=INK)
            y += 16
        for note in (gb300_note, hw_note):
            if note:
                for l in textwrap.wrap(note, 120):
                    d.text((x0 + 20, y), l, font=F["small"], fill=INK2)
                    y += 16
        y += 6
        sched = []
        for s in self.run["steps"]:
            a = self.actor(s)
            sched.append(("L" if s["action"]["kind"] != "warp" else "") + f"w{a}")
        d.text((x0 + 20, y), f"witness schedule ({len(sched)} steps; wN: warp N runs, LwN: an async operation of warp N lands):", font=F["small"], fill=INK2)
        y += 17
        for l in textwrap.wrap(" ".join(sched), 100):
            d.text((x0 + 20, y), l, font=F["code"], fill=INK)
            y += 16
        y += 10
        d.line([x0 + 20, y, x1 - 20, y], fill=RULE)
        y += 10
        d.text((x0 + 20, y), re.sub(r"\s+", " ", good_line.strip()), font=F["codeb"], fill=INK)
        y += 20
        for l in textwrap.wrap(good_note, 116):
            d.text((x0 + 20, y), l, font=F["small"], fill=INK2)
            y += 16
        return im


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("case")
    ap.add_argument("--variant", required=True, help="the variant with the race")
    ap.add_argument("--good", help="a variant to contrast on the last frame")
    ap.add_argument("--good-note", default="", help="one sentence on what the --good variant does differently")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--step-ms", type=int, default=450)
    ap.add_argument("--event-ms", type=int, default=1100)
    a = ap.parse_args()

    case = Path(a.case)
    dump, out, first = run_check(case, a.variant)
    toml = (case / "case.toml").read_text()
    # One-line description of the case and the variant, from case.toml.
    desc = re.search(r'description = """\n?(.*?)"""', toml, re.S).group(1)
    desc = re.sub(r"\s+", " ", desc).strip()
    vblock = toml.split(f'id = "{a.variant}"', 1)[1]
    summary = re.search(r'summary = "(.*?)"', vblock).group(1)
    hw = re.search(r'hw_note = "(.*?)"', vblock.split("[[variants]]")[0])
    lines = textwrap.wrap(desc, 140)[:2] + [f"{a.variant}: {summary}"]
    an = Anim(dump, case.name, a.variant, lines)

    gb = an.run["finding"].get("gb300")
    gb_note = f"Second level (GB300 hardware model): {gb['class']}." if gb else ""
    hw_note = f"GB300 run of this variant (case.toml): {hw.group(1)}." if hw else ""
    good_line, good_note = "", ""
    if a.good:
        gdump, gout, gfirst = run_check(case, a.good, ["--certify"])
        good_line = gfirst
        notes = [l.strip()[6:] for l in gout.splitlines() if l.strip().startswith("note: well-synchronized")]
        good_note = " ".join(
            x for x in (f"{a.good}: {a.good_note}" if a.good_note else "", notes[0][0].upper() + notes[0][1:] + "." if notes else "") if x
        )

    frames, durs = [], []
    frames.append(an.frame(-1))
    durs.append(1000)
    steps = an.run["steps"]
    for k, s in enumerate(steps):
        changed = s["smem"] or s["findings"] or (k and s["barriers"] != steps[k - 1]["barriers"])
        is_race = any(f["sig"] == an.run["finding"]["sig"] for f in s["findings"])
        frames.append(an.frame(k, race=is_race))
        durs.append(3000 if is_race else (a.event_ms if changed else a.step_ms))
    frames.append(an.verdict(first, good_line, good_note, gb_note, hw_note))
    durs.append(5500)

    pal = [f.quantize(colors=64, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for f in frames]
    out_p = Path(a.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    pal[0].save(out_p, save_all=True, append_images=pal[1:], duration=durs, loop=0, optimize=True, disposal=1)
    print(f"{out_p}: {len(frames)} frames, {sum(durs) / 1000:.1f} s, {out_p.stat().st_size / 1e3:.0f} kB")


if __name__ == "__main__":
    main()
