/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! The synchronization reduction without its random seed executions (PTXV_POR_SEED_WALKS=0),
//! so that only what the search itself learns justifies its choices. Seeding once hid a
//! soundness bug (rung 2: CuTeDSL CM1 lost its race); a separate test binary, because the
//! setting is process-wide.

use std::path::PathBuf;
use std::sync::Arc;

use ptxv_case::Case;
use ptxv_sim::build_launch;
use ptxv_sim::explore::explore;
use ptxv_sim::machine::Machine;

#[test]
fn cute_small_cm1_sass_unseeded() {
    // Set before any exploration thread exists.
    std::env::set_var("PTXV_POR_SEED_WALKS", "0");
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    // Third-party kernel, not distributed (fixtures/VENDOR_FIXTURES.toml): skip if absent.
    if ptxv_case::skip_missing_vendor_files(&root.join("fixtures"), &["r2_cute_small"]) {
        return;
    }
    let case = Case::load(root.join("fixtures/r2_cute_small")).unwrap();
    let ptx = case.variant_ptx("CM1").unwrap();
    let dir = std::env::temp_dir().join("ptxv-test-unseeded-CM1");
    std::fs::create_dir_all(&dir).unwrap();
    let (p, c) = (dir.join("k.ptx"), dir.join("k.cubin"));
    std::fs::write(&p, &ptx).unwrap();
    ptxv_sass::ptxas(&p, &c, &case.m.arch).unwrap();
    let k = ptxv_sass::load_cubin(&c, &case.m.entry).unwrap();
    let launch = build_launch(&case, &k).unwrap();
    let v = explore(Machine::new(Arc::new(k), Arc::new(launch)), 2_000_000);
    let expect = case.variant("CM1").unwrap().expect_for("sass");
    assert!(
        v.complete && v.meets(&expect),
        "got {:?} (complete: {}), want {expect:?}",
        v.classes(),
        v.complete
    );
}
