/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Replays the recorded GB300 litmus results for E-POLL-ACQ and E-SMEM-CUMUL
//! (probes/poll_acq, docs/poll-acq-litmus.md) and checks the claims ASSUMPTIONS.md makes
//! from them:
//! - E-POLL-ACQ is refuted: weak reads after leaving a warp-level poll loop return stale data
//!   when the payload line is valid in L1, in every placement;
//! - DeepEP's shapes (barrier_block with its entry fence, volatile reads, the combine relay)
//!   and E-SMEM-CUMUL's relay after a real acquire never returned stale data;
//! - the positive controls saw stale data in every run (every placement and stress mode), and
//!   the negative controls never did;
//! - the totals quoted in the documents are the totals of the recorded files.

use std::collections::BTreeMap;
use std::path::PathBuf;

#[derive(Default, Clone, Copy, Debug)]
struct Tot {
    runs: u64,
    reads: u64,
    stale: u64,
    other: u64,
    timeout: u64,
}

struct Run {
    plan: String,
    cfg: String,
    family: String,
    place: String,
    stale: u64,
}

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn field(s: &str, k: &str) -> u64 {
    let pat = format!(" {k}=");
    let i = s.find(&pat).unwrap_or_else(|| panic!("no {k} in {s}")) + pat.len();
    s[i..]
        .split(|c: char| !c.is_ascii_digit())
        .next()
        .unwrap()
        .parse()
        .unwrap()
}

fn load() -> (Vec<Run>, BTreeMap<String, Tot>) {
    let dir = root().join("probes/poll_acq/results");
    let mut files: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect();
    files.sort();
    let mut runs = Vec::new();
    let mut fams: BTreeMap<String, Tot> = BTreeMap::new();
    for f in files
        .iter()
        .filter(|f| f.extension().is_some_and(|x| x == "txt"))
    {
        for line in std::fs::read_to_string(f).unwrap().lines() {
            let Some(rest) = line.strip_prefix('[') else {
                continue;
            };
            let (tag, body) = rest.split_once("] ").unwrap();
            assert!(
                body.starts_with("RESULT"),
                "{}: failed run {line}",
                f.display()
            );
            let parts: Vec<&str> = tag.split('/').collect();
            let (plan, cfg) = (parts[0], parts[1]);
            // the family is the config name without its stress suffix
            let family = match cfg.rsplit_once('-') {
                Some((f, s)) if ["none", "l2", "atom", "nvl", "l1", "payld"].contains(&s) => f,
                _ => cfg,
            };
            let reads = field(body, "reads");
            let stale = field(body, "stale");
            let other = field(body, "other");
            let timeout = field(body, "timeout");
            assert_eq!(reads, field(body, "ok") + stale + other, "{line}");
            let t = fams.entry(format!("{plan}/{family}")).or_default();
            t.runs += 1;
            t.reads += reads;
            t.stale += stale;
            t.other += other;
            t.timeout += timeout;
            runs.push(Run {
                plan: plan.into(),
                cfg: cfg.into(),
                family: family.into(),
                place: parts[2].into(),
                stale,
            });
        }
    }
    (runs, fams)
}

fn sum(fams: &BTreeMap<String, Tot>, f: impl Fn(&str) -> bool) -> Tot {
    let mut t = Tot::default();
    for (_, v) in fams.iter().filter(|(k, _)| f(k)) {
        t.runs += v.runs;
        t.reads += v.reads;
        t.stale += v.stale;
        t.other += v.other;
        t.timeout += v.timeout;
    }
    t
}

#[test]
fn poll_acq_litmus_results_support_the_assumption_statuses() {
    let (runs, fams) = load();
    assert!(!runs.is_empty());
    for (k, t) in &fams {
        assert_eq!(t.timeout, 0, "{k}: timed-out runs");
        assert_eq!(t.other, 0, "{k}: values that are neither old nor new");
        if k.starts_with("refute/") {
            // E-POLL-ACQ refuted: stale reads in every family (per placement below)
            assert!(t.stale > 0, "{k}: expected stale reads");
        } else if k.starts_with("deepep/") || k.starts_with("cumul/") {
            assert_eq!(t.stale, 0, "{k}: stale read in a shape that must be fresh");
        } else if k.starts_with("controls/ctl-neg") {
            assert_eq!(t.stale, 0, "{k}: negative control saw stale data");
        } else if k.starts_with("controls/ctl-pos") {
            assert!(t.stale > 0, "{k}");
        } else {
            panic!("unknown family {k}");
        }
    }
    // Cross-SM placements (cross-GPU, intra-GPU far / near): every refuting family saw stale
    // reads in every placement, and the positive control in every placement x stress cell. With
    // producer and consumer on the same SM (i-same) the producer's stores pass through the
    // consumer's L1 and nothing is stale: those runs are not counted as refutations or controls.
    let mut cells: BTreeMap<(String, String, String), u64> = BTreeMap::new();
    for r in runs.iter().filter(|r| r.place != "i-same") {
        if r.plan == "refute" {
            *cells
                .entry((r.plan.clone(), r.family.clone(), r.place.clone()))
                .or_default() += r.stale;
        } else if r.cfg.starts_with("ctl-pos") {
            *cells
                .entry((r.plan.clone(), r.cfg.clone(), r.place.clone()))
                .or_default() += r.stale;
        }
    }
    assert!(cells.len() > 100);
    for (k, stale) in &cells {
        assert!(*stale > 0, "{k:?}: no stale read");
    }
}

/// The totals ASSUMPTIONS.md and docs/poll-acq-litmus.md quote.
#[test]
fn poll_acq_litmus_totals_match_the_documents() {
    let (_, fams) = load();
    let deepep = sum(&fams, |k| k.starts_with("deepep/"));
    let cumul = sum(&fams, |k| k.starts_with("cumul/"));
    let refute = sum(&fams, |k| k.starts_with("refute/"));
    let pos = sum(&fams, |k| k.starts_with("controls/ctl-pos"));
    let neg = sum(&fams, |k| k.starts_with("controls/ctl-neg"));
    let got = [
        ("deepep", deepep.runs, deepep.reads, deepep.stale),
        ("cumul", cumul.runs, cumul.reads, cumul.stale),
        ("refute", refute.runs, refute.reads, refute.stale),
        ("ctl-pos", pos.runs, pos.reads, pos.stale),
        ("ctl-neg", neg.runs, neg.reads, neg.stale),
    ];
    assert_eq!(
        got, EXPECTED,
        "recorded totals changed: update the documents and EXPECTED"
    );
    let doc = std::fs::read_to_string(root().join("docs/poll-acq-litmus.md")).unwrap();
    for (name, _, reads, stale) in EXPECTED {
        let s = format!("{reads}");
        assert!(
            doc.contains(&s),
            "docs/poll-acq-litmus.md does not quote {name} reads {s}"
        );
        let s = format!("{stale}");
        assert!(
            doc.contains(&s),
            "docs/poll-acq-litmus.md does not quote {name} stale {s}"
        );
    }
}

// (family, runs, payload reads, stale reads), from probes/poll_acq/results (2026-11-04).
const EXPECTED: [(&str, u64, u64, u64); 5] = [
    ("deepep", 90, 3538947554048, 0),
    ("cumul", 180, 8274458067200, 0),
    ("refute", 264, 1223923395072, 476705632772),
    ("ctl-pos", 72, 312806190080, 128308936902),
    ("ctl-neg", 36, 122035419392, 0),
];

/// DeepEP dispatch ring-slot reuse (probes/poll_acq/ringwar.cu, ringwar.txt): the receiver's
/// bulk read (UBLKCP.S.G + mbarrier wait) followed by a relaxed head store never returned the
/// sender's later overwrite; the control that stores the head before the wait did.
#[test]
fn ringwar_results() {
    let s = std::fs::read_to_string(root().join("probes/poll_acq/ringwar.txt")).unwrap();
    let (mut dee, mut pos) = ((0u64, 0u64, 0u64), (0u64, 0u64, 0u64));
    for line in s.lines() {
        let (tag, body) = line.split_once("] ").unwrap();
        assert!(body.starts_with("RESULT"), "failed run {line}");
        assert_eq!(field(body, "timeout"), 0, "{line}");
        assert_eq!(field(body, "stale"), 0, "{line}");
        let copies = field(body, "copies");
        let early = field(body, "early");
        let t = if tag.contains("/deepep-hb0/") {
            &mut dee
        } else if tag.contains("/ctl-pos-hb1/") {
            assert!(early > 0, "{line}: control saw no early value");
            &mut pos
        } else {
            panic!("{line}")
        };
        t.0 += 1;
        t.1 += copies;
        t.2 += early;
    }
    assert_eq!(dee.2, 0, "DeepEP order returned an overwritten slot");
    assert_eq!(
        (dee, pos),
        RINGWAR,
        "update docs/poll-acq-litmus.md and RINGWAR"
    );
    let doc = std::fs::read_to_string(root().join("docs/poll-acq-litmus.md")).unwrap();
    for n in [dee.1, pos.1, pos.2] {
        assert!(
            doc.contains(&n.to_string()),
            "docs/poll-acq-litmus.md does not quote {n}"
        );
    }
}

// ((runs, copies, early) DeepEP order, (runs, copies, early) control), 2026-11-04.
const RINGWAR: ((u64, u64, u64), (u64, u64, u64)) = ((36, 21312000000, 0), (4, 710400000, 3772856));
