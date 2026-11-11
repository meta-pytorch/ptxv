/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! `ptxv errdist --html FILE`: one self-contained HTML page (inline SVG and CSS, no
//! scripts or external assets) with P(differ) per tier and observed, the PMFs over output
//! ulps overlaid, a table view of the PMFs, and the text summary.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use ptxv_sim::errdist::report::Report;

/// Categorical slots in fixed order (the reference palette of the dataviz method; light
/// surface). Series beyond eight share the last slot and are told apart by label only.
const SERIES: [&str; 8] = [
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948",
];
const INK: &str = "#0b0b0b";
const INK2: &str = "#52514e";
const GRID: &str = "#e4e3df";
const SURFACE: &str = "#fcfcfb";

fn esc(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

/// One plotted series: a label, P(differ) with an interval, and a PMF.
struct Series {
    label: String,
    detail: String,
    p: Option<(f64, f64, f64)>,
    pmf: Vec<(i64, f64)>,
}

fn series(rep: &Report) -> Vec<Series> {
    let mut v = vec![];
    for (i, e) in rep.estimates.iter().enumerate() {
        let short: String = e
            .method
            .split([';', ':', ','])
            .next()
            .unwrap_or("")
            .to_string();
        let mut label = format!("tier {}: {}", e.tier, short.trim());
        if label.chars().count() > 48 {
            label = label.chars().take(45).collect::<String>() + "...";
        }
        if Some(i) == rep.best {
            label.push_str(" (best)");
        }
        let app = match e.applicable {
            Some(true) => "applicable",
            Some(false) => "not applicable",
            None => "",
        };
        v.push(Series {
            label,
            detail: format!("{} {app}", e.method),
            p: e.p_differ,
            pmf: e.pmf_ulp.clone(),
        });
    }
    // Observations: one per source (the observed side is the same in every entry).
    let mut seen: Vec<String> = vec![];
    for x in &rep.validation {
        let key = format!(
            "{}{}",
            x.source,
            x.notes
                .iter()
                .find(|n| n.starts_with("GPU "))
                .map_or("", |s| s.as_str())
        );
        if seen.contains(&key) {
            continue;
        }
        seen.push(key);
        v.push(Series {
            label: format!("observed ({})", x.source),
            detail: format!("{} inputs, {} outputs", x.inputs, x.outputs),
            p: x.p_differ_obs,
            pmf: x.pmf_ulp.clone(),
        });
    }
    v
}

fn color(i: usize) -> &'static str {
    SERIES[i.min(SERIES.len() - 1)]
}

/// Dot-and-interval chart of P(differ), one row per series.
fn p_chart(ss: &[Series]) -> String {
    let rows: Vec<(usize, &Series, (f64, f64, f64))> = ss
        .iter()
        .enumerate()
        .filter_map(|(i, s)| s.p.map(|p| (i, s, p)))
        .collect();
    if rows.is_empty() {
        return "<p>No tier reports P(differ).</p>".into();
    }
    let (lw, pw, rh, top) = (380.0, 460.0, 28.0, 24.0);
    let w = lw + pw + 110.0;
    let h = top + rh * rows.len() as f64 + 30.0;
    let hi = rows.iter().map(|r| r.2.2.max(r.2.0)).fold(0.0, f64::max);
    let lo_pos = rows
        .iter()
        .map(|r| r.2.1.min(r.2.0))
        .filter(|x| *x > 0.0)
        .fold(f64::INFINITY, f64::min);
    // Log axis when the values span more than two decades.
    let log = hi > 0.0 && lo_pos.is_finite() && hi / lo_pos > 100.0;
    let (a, b) = if log {
        (lo_pos.log10().floor(), hi.log10().ceil())
    } else {
        (0.0, if hi > 0.0 { hi * 1.15 } else { 1.0 })
    };
    let x = |p: f64| -> f64 {
        let t = if log {
            (p.max(10f64.powf(a)).log10() - a) / (b - a)
        } else {
            (p - a) / (b - a)
        };
        lw + pw * t.clamp(0.0, 1.0)
    };
    let mut o = String::new();
    let _ = write!(
        o,
        r#"<svg viewBox="0 0 {w} {h}" width="{w}" role="img" aria-label="P(bits differ) per tier with 95% intervals">"#
    );
    // Ticks.
    let ticks: Vec<f64> = if log {
        (a as i32..=b as i32).map(|e| 10f64.powi(e)).collect()
    } else {
        (0..=4).map(|i| a + (b - a) * i as f64 / 4.0).collect()
    };
    let yb = top + rh * rows.len() as f64;
    for t in ticks {
        let xt = x(t);
        let _ = write!(
            o,
            r#"<line x1="{xt:.1}" x2="{xt:.1}" y1="{top}" y2="{yb}" stroke="{GRID}" stroke-width="1"/><text x="{xt:.1}" y="{:.1}" font-size="11" fill="{INK2}" text-anchor="middle">{}</text>"#,
            yb + 16.0,
            if log {
                format!("1e{}", t.log10().round())
            } else {
                format!("{t:.3}")
            }
        );
    }
    let _ = write!(
        o,
        r#"<text x="{lw}" y="14" font-size="11" fill="{INK2}">P(bits differ), 95% interval{}</text>"#,
        if log { " (log scale)" } else { "" }
    );
    for (r, (i, s, (p, l, u))) in rows.iter().enumerate() {
        let y = top + rh * (r as f64 + 0.5);
        let c = color(*i);
        let _ = write!(
            o,
            r#"<g><title>{}: {p:.4e} [{l:.4e}, {u:.4e}]
{}</title><rect x="0" y="{:.1}" width="{w}" height="{rh}" fill="transparent"/><text x="{:.1}" y="{:.1}" font-size="12" fill="{INK}" text-anchor="end">{}</text>"#,
            esc(&s.label),
            esc(&s.detail),
            y - rh / 2.0,
            lw - 10.0,
            y + 4.0,
            esc(&s.label)
        );
        if u > l {
            let _ = write!(
                o,
                r#"<line x1="{:.1}" x2="{:.1}" y1="{y:.1}" y2="{y:.1}" stroke="{c}" stroke-width="2" stroke-linecap="round"/>"#,
                x(*l),
                x(*u)
            );
        }
        let _ = write!(
            o,
            r#"<circle cx="{:.1}" cy="{y:.1}" r="5" fill="{c}" stroke="{SURFACE}" stroke-width="2"/><text x="{:.1}" y="{:.1}" font-size="11" fill="{INK2}">{p:.4e}</text></g>"#,
            x(*p),
            lw + pw + 10.0,
            y + 4.0
        );
    }
    o.push_str("</svg>");
    o
}

/// PMF overlay on a log probability axis.
fn pmf_chart(ss: &[Series]) -> String {
    let with: Vec<(usize, &Series)> = ss
        .iter()
        .enumerate()
        .filter(|(_, s)| !s.pmf.is_empty())
        .collect();
    if with.is_empty() {
        return "<p>No PMF over output ulps (outputs off the ulp lattice).</p>".into();
    }
    // k range: where any series has mass >= 1e-7, at most +-24 around 0.
    let ks: Vec<i64> = with
        .iter()
        .flat_map(|(_, s)| s.pmf.iter().filter(|x| x.1 >= 1e-7).map(|x| x.0))
        .collect();
    let k0 = ks.iter().copied().min().unwrap_or(-1).clamp(-24, -1);
    let k1 = ks.iter().copied().max().unwrap_or(1).clamp(1, 24);
    let pmin = with
        .iter()
        .flat_map(|(_, s)| s.pmf.iter().map(|x| x.1))
        .filter(|x| *x > 0.0)
        .fold(1.0f64, f64::min)
        .max(1e-9);
    let (ea, eb) = (pmin.log10().floor(), 0.0);
    let (ml, mt, pw, ph) = (60.0, 20.0, 640.0, 300.0);
    let w = ml + pw + 330.0;
    let h = mt + ph + 40.0;
    let x = |k: f64| ml + pw * (k - k0 as f64 + 0.5) / ((k1 - k0) as f64 + 1.0);
    let y = |p: f64| mt + ph * (1.0 - (p.max(10f64.powf(ea)).log10() - ea) / (eb - ea));
    let mut o = String::new();
    let _ = write!(
        o,
        r#"<svg viewBox="0 0 {w} {h}" width="{w}" role="img" aria-label="PMF over output ulps per tier and observed">"#
    );
    for e in ea as i32..=0 {
        let yy = y(10f64.powi(e));
        let _ = write!(
            o,
            r#"<line x1="{ml}" x2="{:.1}" y1="{yy:.1}" y2="{yy:.1}" stroke="{GRID}" stroke-width="1"/><text x="{:.1}" y="{:.1}" font-size="11" fill="{INK2}" text-anchor="end">1e{e}</text>"#,
            ml + pw,
            ml - 6.0,
            yy + 4.0
        );
    }
    let step = ((k1 - k0) / 12).max(1);
    let mut k = k0;
    while k <= k1 {
        let _ = write!(
            o,
            r#"<text x="{:.1}" y="{:.1}" font-size="11" fill="{INK2}" text-anchor="middle">{k}</text>"#,
            x(k as f64),
            mt + ph + 16.0
        );
        k += step;
    }
    let _ = write!(
        o,
        r#"<text x="{:.1}" y="{:.1}" font-size="11" fill="{INK2}" text-anchor="middle">k = (output - reference) / ulp_out(y)</text>"#,
        ml + pw / 2.0,
        mt + ph + 34.0
    );
    for (li, (i, s)) in with.iter().enumerate() {
        let c = color(*i);
        let pts: Vec<(i64, f64)> = s
            .pmf
            .iter()
            .copied()
            .filter(|x| x.0 >= k0 && x.0 <= k1 && x.1 > 0.0)
            .collect();
        let path: Vec<String> = pts
            .iter()
            .map(|(k, p)| format!("{:.1},{:.1}", x(*k as f64), y(*p)))
            .collect();
        let dash = if s.label.starts_with("observed") {
            ""
        } else {
            r#" stroke-dasharray="6 3""#
        };
        let _ = write!(
            o,
            r#"<polyline points="{}" fill="none" stroke="{c}" stroke-width="2"{dash}/>"#,
            path.join(" ")
        );
        for (k, p) in &pts {
            let _ = write!(
                o,
                r#"<g><title>{}: k = {k}, p = {p:.4e}</title><circle cx="{:.1}" cy="{:.1}" r="10" fill="transparent"/><circle cx="{:.1}" cy="{:.1}" r="4" fill="{c}" stroke="{SURFACE}" stroke-width="2"/></g>"#,
                esc(&s.label),
                x(*k as f64),
                y(*p),
                x(*k as f64),
                y(*p)
            );
        }
        // Legend.
        let ly = mt + 10.0 + 20.0 * li as f64;
        let lx = ml + pw + 20.0;
        let _ = write!(
            o,
            r#"<line x1="{lx}" x2="{:.1}" y1="{ly}" y2="{ly}" stroke="{c}" stroke-width="2"{dash}/><circle cx="{:.1}" cy="{ly}" r="4" fill="{c}"/><text x="{:.1}" y="{:.1}" font-size="11" fill="{INK}">{}</text>"#,
            lx + 24.0,
            lx + 12.0,
            lx + 30.0,
            ly + 4.0,
            esc(&s.label)
        );
    }
    o.push_str("</svg>");
    o
}

/// Table view of the PMFs (k by series).
fn pmf_table(ss: &[Series]) -> String {
    let with: Vec<&Series> = ss.iter().filter(|s| !s.pmf.is_empty()).collect();
    if with.is_empty() {
        return String::new();
    }
    let mut rows: BTreeMap<i64, Vec<Option<f64>>> = BTreeMap::new();
    for (j, s) in with.iter().enumerate() {
        for &(k, p) in &s.pmf {
            rows.entry(k).or_insert_with(|| vec![None; with.len()])[j] = Some(p);
        }
    }
    let mut o = String::from("<details><summary>PMF table</summary><table><tr><th>k</th>");
    for s in &with {
        let _ = write!(o, "<th>{}</th>", esc(&s.label));
    }
    o.push_str("</tr>");
    for (k, v) in rows {
        let _ = write!(o, "<tr><td>{k}</td>");
        for p in v {
            let _ = write!(
                o,
                "<td>{}</td>",
                p.map_or(String::new(), |p| format!("{p:.4e}"))
            );
        }
        o.push_str("</tr>");
    }
    o.push_str("</table></details>");
    o
}

/// The proven worst case beside the distribution.
fn worst(rep: &Report) -> String {
    let Some(w) = &rep.worst_case else {
        return String::new();
    };
    let mut o = format!(
        "<h2>Worst case ({})</h2><p><b>{}</b></p>",
        if w.kind == "pair" {
            "equivalence proof"
        } else {
            "the kernel's own bound"
        },
        esc(&w.human)
    );
    if w.available {
        let _ = write!(o, "<p><code>{}</code></p>", esc(&w.formula));
    }
    for p in &w.preconditions {
        let _ = write!(o, "<p>precondition: {}</p>", esc(p));
    }
    if let Some(d) = &w.dist {
        let _ = write!(
            o,
            "<p>The bound over the Tier 3 inputs, in output ulps of y: mean {:.3}; {}; max {:.3} ({} outputs).</p>",
            d.mean_ulp,
            d.quantiles_ulp
                .iter()
                .map(|(q, x)| format!("q{q} {x:.3}"))
                .collect::<Vec<_>>()
                .join(", "),
            d.max_ulp,
            d.n
        );
    }
    if !w.checks.is_empty() {
        o.push_str("<table><tr><th>observed</th><th>outputs</th><th>largest |difference| / bound</th><th>exceeding</th></tr>");
        for c in &w.checks {
            let _ = write!(
                o,
                "<tr><td>{}</td><td>{}</td><td>{:.3e}</td><td>{}</td></tr>",
                esc(&c.source),
                c.checked,
                c.max_ratio,
                if c.exceed > 0 {
                    format!(
                        "<b style=\"color:#c62828\">&#9888; {} (bound violated)</b>",
                        c.exceed
                    )
                } else {
                    "0".into()
                }
            );
        }
        o.push_str("</table>");
    }
    o
}

pub fn render(rep: &Report, text: &str) -> String {
    let ss = series(rep);
    let title = format!(
        "ptxv errdist: {}{}",
        rep.reference.case,
        rep.candidate
            .as_ref()
            .map_or(String::new(), |c| format!(" vs {} ({})", c.case, c.variant))
    );
    format!(
        r#"<!doctype html><html><head><meta charset="utf-8"><title>{t}</title><style>
body{{font-family:system-ui,sans-serif;background:{SURFACE};color:{INK};margin:24px;max-width:1100px}}
h1{{font-size:18px}} h2{{font-size:15px;margin-top:28px}} p,li{{color:{INK2};font-size:13px}}
pre{{font-size:12px;background:#f3f2ee;padding:12px;overflow-x:auto;white-space:pre-wrap}}
table{{border-collapse:collapse;font-size:12px}} td,th{{border:1px solid {GRID};padding:2px 8px;text-align:right}}
svg{{max-width:100%;height:auto}}
</style></head><body><h1>{t}</h1>
<p>{mode} mode, {quant}; {n} outputs ({fmt}) per input. Reference for k: {refname}. Hover a mark for its value.</p>
<h2>P(bits differ)</h2>{pc}{wc}
<h2>PMF over output ulps</h2><p>Dashed: predictions; solid: observations. Log probability axis; k beyond +-24 is in the table only.</p>{pm}{tb}
<h2>Text summary</h2><pre>{txt}</pre></body></html>
"#,
        t = esc(&title),
        mode = esc(&rep.mode),
        quant = esc(&rep.quantity),
        n = rep.outputs,
        fmt = esc(&rep.output_format),
        refname = if rep.mode == "pair" {
            "the reference kernel R"
        } else {
            "the correctly rounded exact value"
        },
        pc = p_chart(&ss),
        wc = worst(rep),
        pm = pmf_chart(&ss),
        tb = pmf_table(&ss),
        txt = esc(text),
    )
}
