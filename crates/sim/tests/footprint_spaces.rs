// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

//! DPOR footprint space ids stay disjoint: a named space (`SPACE_*: u8 = N`) has a value of its
//! own, no literal `rec(N, ..)` / `rec_sync(N, ..)` / `add(N, ..)` reuses it, and every id in
//! use is documented on `Footprint` (machine.rs). Two subsystems sharing an id make each one's
//! dependencies trip the other's reduction (e.g. a collector write read as an mbarrier arrival).

use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::path::PathBuf;

fn sources() -> Vec<(String, String)> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut out = Vec::new();
    let mut stack = vec![dir];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).unwrap() {
            let p = e.unwrap().path();
            if p.is_dir() {
                stack.push(p);
            } else if p.extension().is_some_and(|x| x == "rs") {
                out.push((
                    p.display().to_string(),
                    std::fs::read_to_string(&p).unwrap(),
                ));
            }
        }
    }
    out
}

/// The leading decimal number of `s` (after optional spaces), if any.
fn num(s: &str) -> Option<u8> {
    let t = s.trim_start();
    let d: String = t.chars().take_while(|c| c.is_ascii_digit()).collect();
    if d.is_empty() || !t[d.len()..].trim_start().starts_with(',') {
        return None;
    }
    d.parse().ok()
}

#[test]
fn footprint_spaces_are_disjoint() {
    let src = sources();
    // Named spaces.
    let mut named: BTreeMap<u8, Vec<String>> = BTreeMap::new();
    for (_, s) in &src {
        for l in s.lines() {
            let l = l.trim();
            let Some(rest) = l.strip_prefix("pub const SPACE_") else {
                continue;
            };
            let (name, v) = rest.split_once(": u8 = ").unwrap();
            let v: u8 = v.trim_end_matches(';').parse().unwrap();
            named.entry(v).or_default().push(format!("SPACE_{name}"));
        }
    }
    for (v, ns) in &named {
        assert_eq!(ns.len(), 1, "footprint space {v} is named twice: {ns:?}");
    }
    // Literal ids.
    let mut literal: BTreeMap<u8, BTreeSet<String>> = BTreeMap::new();
    for (path, s) in &src {
        for pat in ["rec(", "rec_sync(", ".add("] {
            for (i, _) in s.match_indices(pat) {
                if let Some(v) = num(&s[i + pat.len()..]) {
                    literal.entry(v).or_default().insert(path.clone());
                }
            }
        }
    }
    for (v, files) in &literal {
        assert!(
            !named.contains_key(v),
            "footprint space {v} is named {:?} but also used as a literal in {files:?}",
            named[v]
        );
    }
    // Documented on `Footprint`.
    let machine = &src
        .iter()
        .find(|(p, _)| p.ends_with("machine.rs"))
        .unwrap()
        .1;
    let at = machine.find("pub struct Footprint {").unwrap();
    let doc_start = machine[..at].rfind("\n\n").unwrap();
    let doc = &machine[doc_start..at];
    for v in literal.keys().chain(named.keys()) {
        if *v == 98 {
            continue; // SPACE_CTA_ALL: slices.rs's whole-CTA key, documented at its definition
        }
        let tokens: BTreeSet<&str> = doc
            .split(|c: char| !c.is_ascii_digit())
            .filter(|t| !t.is_empty())
            .collect();
        assert!(
            tokens.contains(v.to_string().as_str()),
            "footprint space {v} is not documented on Footprint"
        );
    }
}
