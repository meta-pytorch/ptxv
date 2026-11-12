/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! E-LAUNCH-ORDER through `ptxv check` and `ptxv equiv`: the strict reading of an unknown launch
//! order (`PTXV_UNRECORDED_ORDER=concurrent`), and the certificate check's verdict cache
//! (vcache.rs) for a variant that edits the launch order.

use std::path::PathBuf;
use std::process::Command;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn text(out: &std::process::Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    )
}

/// `ptxv check fixtures/e_launch_order` (PTX front, no GB300 re-run) with
/// `PTXV_UNRECORDED_ORDER` set to `order`, or unset.
fn check(variant: &str, order: Option<&str>) -> (bool, String) {
    let dir = std::env::temp_dir().join(format!(
        "ptxv-test-launch-order-{variant}-{}-{}",
        order.unwrap_or("unset"),
        std::process::id()
    ));
    let mut c = Command::new(env!("CARGO_BIN_EXE_ptxv"));
    c.arg("check")
        .arg(root().join("fixtures/e_launch_order"))
        .args([
            "--variant",
            variant,
            "--front",
            "ptx",
            "--no-gb300",
            "--no-input-coverage",
            "--out",
        ])
        .arg(&dir)
        .env("CUDA_VISIBLE_DEVICES", "")
        .env("PTXV_NO_VCACHE", "1")
        .env_remove("PTXV_UNRECORDED_ORDER");
    if let Some(o) = order {
        c.env("PTXV_UNRECORDED_ORDER", o);
    }
    let out = c.output().unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    (out.status.success(), text(&out))
}

/// `PTXV_UNRECORDED_ORDER=concurrent`: every grid of an unknown-order case loses its
/// predecessors, so a coherent read of the previous grid's output races (PLAIN_UNKNOWN; by
/// default, and with `ordered`, E-PDL still orders the list: clean). A recorded order keeps its
/// edges (PLAIN). Any other value is refused.
#[test]
fn unrecorded_order_concurrent_drops_unknown_predecessors() {
    let (_, t) = check("PLAIN_UNKNOWN", None);
    assert!(t.contains("[clean]"), "{t}");
    let (_, t) = check("PLAIN_UNKNOWN", Some("ordered"));
    assert!(t.contains("[clean]"), "{t}");
    let (_, t) = check("PLAIN_UNKNOWN", Some("concurrent"));
    assert!(t.contains("[race]") && t.contains("RACE:"), "{t}");
    let (ok, t) = check("PLAIN", Some("concurrent"));
    assert!(ok && t.contains("[clean]"), "{t}");
    let (ok, t) = check("PLAIN_UNKNOWN", Some("parallel"));
    assert!(!ok && t.contains("expected ordered or concurrent"), "{t}");
}

/// The precondition of `ptxv equiv` / `errdist` runs on the case as loaded, without a
/// variant's launch-order edits, so the verdict `ptxv check --certify` stored for a variant
/// that edits the order must not serve it. fixtures/e_vcache_order: the case's order is
/// unknown (an S-NC violation); variant DECL declares it (clean). With DECL's verdict reused,
/// the precondition was empty (a wrong clean).
#[test]
fn launch_order_variant_verdict_does_not_serve_equiv() {
    let tag = std::process::id();
    let cache = std::env::temp_dir().join(format!("ptxv-test-vcache-order-{tag}"));
    let out_dir = std::env::temp_dir().join(format!("ptxv-test-vcache-order-out-{tag}"));
    let eq_dir = std::env::temp_dir().join(format!("ptxv-test-vcache-order-eq-{tag}"));
    let _ = std::fs::remove_dir_all(&cache);
    let out = Command::new(env!("CARGO_BIN_EXE_ptxv"))
        .arg("check")
        .arg(root().join("fixtures/e_vcache_order"))
        .args([
            "--variant",
            "DECL",
            "--front",
            "ptx",
            "--certify",
            "--no-gb300",
            "--no-input-coverage",
            "--out",
        ])
        .arg(&out_dir)
        .env("CUDA_VISIBLE_DEVICES", "")
        .env("PTXV_CACHE", &cache)
        .env_remove("PTXV_NO_VCACHE")
        .env_remove("PTXV_UNRECORDED_ORDER")
        .output()
        .unwrap();
    let t = text(&out);
    assert!(out.status.success() && t.contains("[clean]"), "{t}");
    let out = Command::new(env!("CARGO_BIN_EXE_ptxv"))
        .args([
            "equiv",
            "fixtures/e_vcache_order/equiv.toml",
            "--no-hw",
            "--out",
        ])
        .arg(&eq_dir)
        .current_dir(root())
        .env("CUDA_VISIBLE_DEVICES", "")
        .env("PTXV_QUIET", "1")
        .env("PTXV_CACHE", &cache)
        .env_remove("PTXV_NO_VCACHE")
        .env_remove("PTXV_UNRECORDED_ORDER")
        .output()
        .unwrap();
    let t = text(&out);
    assert!(out.status.success(), "{t}");
    let doc: serde_json::Value = serde_json::from_slice(
        &std::fs::read(eq_dir.join("equiv.json")).unwrap_or_else(|e| panic!("{e}: {t}")),
    )
    .unwrap();
    let pre = doc[0]["preconditions"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    assert!(
        !pre.is_empty()
            && pre
                .iter()
                .all(|p| p.as_str().unwrap_or("").contains("not clean")),
        "{doc:#}"
    );
    assert!(
        !doc.to_string()
            .contains("an earlier `ptxv check --certify`"),
        "{doc:#}"
    );
    for d in [&cache, &out_dir, &eq_dir] {
        let _ = std::fs::remove_dir_all(d);
    }
}
