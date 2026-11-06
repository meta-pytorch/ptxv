/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! `ptxv errdist`'s worst case beside the distribution: the proven bound (a single kernel's
//! own bound, a pair's equivalence bound), its distribution, and no observed difference
//! above it (docs/errdist.md §3.1).

fn root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn errdist(path: &str, spec: &str, extra: &[&str]) -> serde_json::Value {
    // Tests run in parallel and may share a path: a counter keeps every call's JSON and
    // scratch directory distinct.
    static CALL: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let js = std::env::temp_dir().join(format!(
        "ptxv-test-errdist-wc-{}-{}-{}.json",
        path.replace('/', "_"),
        std::process::id(),
        CALL.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    let scratch = js.with_extension("d");
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_ptxv"))
        .args(["errdist", path, "--inputs", spec, "--mc", "200"])
        .args(["--sim-trials", "0"])
        .args(["--json", js.to_str().unwrap()])
        .args(["--out", scratch.to_str().unwrap()])
        .args(extra)
        .current_dir(root())
        .env("CUDA_VISIBLE_DEVICES", "")
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let j = serde_json::from_str(&std::fs::read_to_string(&js).unwrap()).unwrap();
    let _ = std::fs::remove_file(&js);
    let _ = std::fs::remove_dir_all(&scratch);
    j
}

fn checks_hold(w: &serde_json::Value) {
    let cs = w["checks"].as_array().unwrap();
    assert!(!cs.is_empty());
    for c in cs {
        assert_eq!(c["exceed"].as_u64(), Some(0), "{c}");
        assert!(c["max_ratio"].as_f64().unwrap() <= 1.0, "{c}");
        assert!(c["checked"].as_u64().unwrap() > 0);
    }
}

#[test]
fn single_kernel_own_bound() {
    // A float sum: |C - y| <= k u sum|x| with k the reduction depth's.
    let j = errdist("fixtures/r4_sum_torch", "fixtures/errdist/normal.json", &[]);
    let w = &j["worst_case"];
    assert_eq!(w["kind"], "single");
    assert_eq!(w["available"], true, "{w}");
    assert_eq!(w["form"], "sum");
    let k = w["k_u"].as_f64().unwrap();
    assert!(k > 1.0 && k < 64.0, "{k}");
    assert!(w["dist"]["max_ulp"].as_f64().unwrap() > 0.0);
    checks_hold(w);
}

#[test]
fn pair_equivalence_bound() {
    // Split-K against the row reduction: an L1 pair, |C - R| <= k u sum|x|.
    let j = errdist(
        "fixtures/c_equiv_sum/equiv.toml",
        "fixtures/errdist/normal.json",
        &["--candidate", "1"],
    );
    let w = &j["worst_case"];
    assert_eq!(w["kind"], "pair");
    assert_eq!(w["available"], true, "{w}");
    assert!(w["formula"].as_str().unwrap().starts_with("|C - R| <= "));
    checks_hold(w);
}

#[test]
fn no_bound_says_why() {
    let j = errdist(
        "fixtures/r4_softmax_torch",
        "fixtures/errdist/normal.json",
        &[],
    );
    let w = &j["worst_case"];
    assert_eq!(w["available"], false);
    assert!(w["why"].as_str().unwrap().contains("MUFU"), "{w}");
}
