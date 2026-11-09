/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Every target directory follows targets/README.md: a valid target.toml, its registry and
//! model files, IDs with the target's prefix, unique across targets and the top-level
//! ASSUMPTIONS.md.

use std::path::PathBuf;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

#[test]
fn targets_are_consistent() {
    let top = std::fs::read_to_string(root().join("ASSUMPTIONS.md")).unwrap();
    let top_ids: Vec<String> = ptxv_targets::parse_assumptions(&top)
        .into_iter()
        .map(|e| e.id)
        .collect();
    assert!(!top_ids.is_empty());
    let errs = ptxv_targets::check(&root(), &top_ids).unwrap();
    assert!(errs.is_empty(), "{}", errs.join("\n"));
    let targets = ptxv_targets::discover(&root()).unwrap();
    assert!(targets.iter().any(|t| t.spec.id == "gb300"));
}
