/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! ASSUMPTIONS.md, the registry in `assumptions.rs`, and the `ASSUME(ID)` tags in the code
//! must agree: every assumption is documented, registered, and (for `code` assumptions)
//! tagged where the code relies on it; every tag names a registered assumption.

use std::collections::BTreeSet;
use std::path::Path;
use std::path::PathBuf;

use ptxv_sim::assumptions::REGISTRY;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    for e in std::fs::read_dir(dir).unwrap() {
        let p = e.unwrap().path();
        if p.is_dir() {
            rust_files(&p, out);
        } else if p.extension().is_some_and(|x| x == "rs") {
            out.push(p);
        }
    }
}

#[test]
fn registry_doc_and_code_agree() {
    let doc = std::fs::read_to_string(root().join("ASSUMPTIONS.md")).unwrap();
    // Headings: `### ID · kind`
    let documented: BTreeSet<(String, String)> = doc
        .lines()
        .filter_map(|l| l.strip_prefix("### "))
        .filter_map(|h| {
            h.split_once(" · ")
                .map(|(id, kind)| (id.trim().to_string(), kind.trim().to_string()))
        })
        .collect();
    let registered: BTreeSet<(String, String)> = REGISTRY
        .iter()
        .map(|a| (a.id.to_string(), a.kind.to_string()))
        .collect();
    assert_eq!(
        documented, registered,
        "ASSUMPTIONS.md headings vs crates/sim/src/assumptions.rs"
    );

    let mut files = Vec::new();
    rust_files(&root().join("crates"), &mut files);
    let mut tagged = BTreeSet::new();
    for f in &files {
        if f.ends_with("tests/assumptions.rs") || f.ends_with("src/assumptions.rs") {
            continue;
        }
        let s = std::fs::read_to_string(f).unwrap();
        for (i, _) in s.match_indices("ASSUME(") {
            let rest = &s[i + 7..];
            let ids = &rest[..rest.find(')').unwrap()];
            for id in ids.split(',') {
                let id = id.trim().to_string();
                assert!(
                    REGISTRY.iter().any(|a| a.id == id),
                    "{}: ASSUME({id}) is not in the registry",
                    f.display()
                );
                tagged.insert(id);
            }
        }
    }
    for a in REGISTRY.iter().filter(|a| a.kind == "code") {
        assert!(
            tagged.contains(a.id),
            "code assumption {} is not tagged ASSUME({}) anywhere in crates/",
            a.id,
            a.id
        );
    }
}
