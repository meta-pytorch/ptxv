/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! `ptxv check --nonfinite`: NaN / Inf in the outputs of every terminal state, with the
//! instructions that produced them (ptxv_sim::nonfinite).

use ptxv_case::Case;
use ptxv_sim::explore::Verdict;
use ptxv_sim::machine::Finding;
use ptxv_sim::machine::Machine;
use ptxv_sim::nonfinite::Birth;
use ptxv_sim::nonfinite::BufScan;
use ptxv_sim::nonfinite::Fmt;
use serde::Serialize;

/// The `nonfinite` part of a check row (docs/json.md).
#[derive(Clone, Debug, Serialize)]
pub(crate) struct Report {
    /// `--nonfinite-strict`: non-finite outputs are findings even when an input holds NaN /
    /// Inf.
    pub strict: bool,
    /// Per float output buffer, the terminal state with the most non-finite elements.
    pub outputs: Vec<BufScan>,
    /// Terminal outputs (distinct) scanned, and how many hold a NaN / Inf.
    pub terminal_outputs: usize,
    pub terminal_outputs_nonfinite: usize,
    /// Float input buffers with NaN / Inf elements.
    pub inputs: Vec<BufScan>,
    /// Instructions that produced a non-finite value from operands without one of that kind,
    /// in program order.
    pub births: Vec<BirthAt>,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct BirthAt {
    /// The instruction (kernel, address, text).
    pub at: String,
    #[serde(flatten)]
    pub birth: Birth,
}

fn dtype<'a>(case: &'a Case, name: &str) -> Option<&'a str> {
    case.m
        .buffers
        .iter()
        .find(|b| b.name == name)
        .map(|b| b.dtype.as_str())
}

/// Scan the verdict's terminal outputs (one run of `m` if it kept none) and the inputs; add a
/// `nonfinite-output` finding per output buffer with NaN / Inf when every float input element
/// is finite (or `strict`), else a note.
pub(crate) fn analyze(
    case: &Case,
    m: &Machine,
    verdict: &mut Verdict,
    births: Vec<Birth>,
    strict: bool,
) -> Report {
    let l = &m.launch;
    let mut outs = verdict.outputs.clone();
    if outs.is_empty() {
        let (m2, _, ok) = ptxv_sim::explore::run_once(m.clone());
        if ok {
            outs.push(m2.outputs());
        }
    }
    // Output buffers: (name, dtype, format, byte range in the concatenated output).
    let mut at = 0usize;
    let mut layout = Vec::new();
    for &i in &l.outputs {
        let n = m.gmem[i].bytes.len();
        let name = &l.allocs[i].name;
        if let Some(d) = dtype(case, name) {
            if let Some(f) = Fmt::of_dtype(d) {
                layout.push((name.clone(), d.to_string(), f, at..at + n));
            }
        }
        at += n;
    }
    let mut worst: Vec<Option<BufScan>> = vec![None; layout.len()];
    let mut with_nf = 0;
    for o in &outs {
        let mut any = false;
        for (k, (name, d, f, r)) in layout.iter().enumerate() {
            let Some(bytes) = o.get(r.clone()) else {
                continue;
            };
            let s = ptxv_sim::nonfinite::scan(name, d, *f, bytes);
            any |= s.nonfinite() > 0;
            if worst[k]
                .as_ref()
                .is_none_or(|w| s.nonfinite() > w.nonfinite())
            {
                worst[k] = Some(s);
            }
        }
        with_nf += any as usize;
    }
    let outputs: Vec<BufScan> = worst.into_iter().flatten().collect();
    // Inputs: float buffers holding data before the launch.
    let mut inputs = Vec::new();
    for (i, a) in l.allocs.iter().enumerate() {
        let Some(b) = case.m.buffers.iter().find(|b| b.name == a.name) else {
            continue;
        };
        if l.outputs.contains(&i) && b.init != "file" {
            continue;
        }
        let Some(f) = Fmt::of_dtype(&b.dtype) else {
            continue;
        };
        let s = ptxv_sim::nonfinite::scan(&a.name, &b.dtype, f, &a.init);
        if s.nonfinite() > 0 {
            inputs.push(s);
        }
    }
    let births: Vec<BirthAt> = births
        .into_iter()
        .map(|b| BirthAt {
            at: m.k.describe(b.pc as usize),
            birth: b,
        })
        .collect();
    // The births that can explain an output buffer: Inf-producing ones for an Inf, NaN-producing
    // ones for a NaN.
    let why = |s: &BufScan| -> String {
        let fits: Vec<&BirthAt> = births
            .iter()
            .filter(|b| {
                let nan = b.birth.kind.starts_with("nan");
                (nan && s.nan > 0) || (!nan && s.inf > 0)
            })
            .collect();
        if fits.is_empty() {
            return "no arithmetic instruction produced a NaN / Inf of that kind from operands without one (a non-finite constant, input or select reached the output)".to_string();
        }
        fits.iter()
            .take(3)
            .map(|b| {
                format!(
                    "{}: {} ({} -> {}), {} time(s)",
                    b.at,
                    b.birth.kind,
                    b.birth.operands.join(", "),
                    b.birth.result,
                    b.birth.count
                )
            })
            .collect::<Vec<_>>()
            .join("; ")
    };
    let finite_in = inputs.is_empty();
    for s in outputs.iter().filter(|s| s.nonfinite() > 0) {
        let (i, v) = s.first.clone().unwrap_or_default();
        let what = format!(
            "output {} ({}): {} of {} elements NaN / Inf ({} NaN, {} Inf; first: element {i} = {v})",
            s.buffer,
            s.dtype,
            s.nonfinite(),
            s.elements,
            s.nan,
            s.inf
        );
        if finite_in || strict {
            verdict.findings.push(Finding {
                kind: "nonfinite-output".into(),
                sig: format!("nonfinite-output:{}", s.buffer),
                msg: format!(
                    "{what}{}; produced at {}",
                    if finite_in {
                        " with every float input element finite"
                    } else {
                        ", inputs hold NaN / Inf (strict)"
                    },
                    why(s)
                ),
                pcs: births.iter().map(|b| b.birth.pc).collect(),
                ..Default::default()
            });
        } else {
            verdict.notes.push(format!(
                "nonfinite: {what}; inputs already hold NaN / Inf ({}), so not a finding (--nonfinite-strict makes it one); produced at {}",
                inputs
                    .iter()
                    .map(|s| format!("{}: {}", s.buffer, s.nonfinite()))
                    .collect::<Vec<_>>()
                    .join(", "),
                why(s)
            ));
        }
    }
    Report {
        strict,
        outputs,
        terminal_outputs: outs.len(),
        terminal_outputs_nonfinite: with_nf,
        inputs,
        births,
    }
}
