/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! The certificate check's verdict cache (vcache.rs) is keyed on the case files, so it holds
//! only verdicts of the case as its directory holds it: `ptxv check --residency N` changes the
//! manifest in memory, and equiv / errdist, which check the case as they load it, must not be
//! served that verdict.

use std::path::Path;
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

/// `ptxv check <case> --variant chain --front ptx --certify` with the verdict cache in `cache`,
/// plus `extra`: its exit status and output.
fn check(case: &Path, out: &Path, cache: &Path, extra: &[&str]) -> (bool, String) {
    let r = Command::new(env!("CARGO_BIN_EXE_ptxv"))
        .arg("check")
        .arg(case)
        .args([
            "--variant",
            "chain",
            "--front",
            "ptx",
            "--certify",
            "--no-gb300",
            "--no-input-coverage",
        ])
        .args(extra)
        .arg("--out")
        .arg(out)
        .env("CUDA_VISIBLE_DEVICES", "")
        .env("PTXV_CACHE", cache)
        .env_remove("PTXV_NO_VCACHE")
        .env_remove("PTXV_RESIDENCY")
        .output()
        .unwrap();
    (r.status.success(), text(&r))
}

/// `ptxv equiv` of the chain variant against itself (certificate check, no GB300) with the
/// verdict cache in `cache`: its equiv.json.
fn equiv(case: &Path, dir: &Path, cache: &Path) -> serde_json::Value {
    let k = format!(
        "case = {:?}\nvariant = \"chain\"\nfront = \"ptx\"\ncertify = true\n",
        case.to_str().unwrap()
    );
    std::fs::create_dir_all(dir).unwrap();
    let toml = dir.join("equiv_chain.toml");
    std::fs::write(
        &toml,
        format!("max_states = 200000\n[reference]\n{k}[[candidates]]\n{k}"),
    )
    .unwrap();
    let out = dir.join("eq");
    let r = Command::new(env!("CARGO_BIN_EXE_ptxv"))
        .arg("equiv")
        .arg(&toml)
        .args(["--no-hw", "--out"])
        .arg(&out)
        .env("CUDA_VISIBLE_DEVICES", "")
        .env("PTXV_QUIET", "1")
        .env("PTXV_CACHE", cache)
        .env_remove("PTXV_NO_VCACHE")
        .env_remove("PTXV_RESIDENCY")
        .output()
        .unwrap();
    let t = text(&r);
    assert!(r.status.success(), "{t}");
    serde_json::from_slice(
        &std::fs::read(out.join("equiv.json")).unwrap_or_else(|e| panic!("{e}: {t}")),
    )
    .unwrap()
}

/// Both kernels' precondition 1 (`not clean: ...` lines; empty: clean).
fn not_clean(doc: &serde_json::Value) -> Vec<String> {
    ["pre_reference", "pre_candidate"]
        .iter()
        .flat_map(|side| doc[0][side].as_array().cloned().unwrap_or_default())
        .filter_map(|p| p.as_str().map(str::to_string))
        .filter(|p| p.starts_with("not clean"))
        .collect()
}

/// Whether the reference's precondition was served from the verdict cache (the candidate, the
/// same kernel, may be served the reference's own check, which equiv stores).
fn reference_served(doc: &serde_json::Value) -> bool {
    doc[0]["notes_reference"]
        .as_array()
        .unwrap_or_else(|| panic!("no notes_reference: {doc:#}"))
        .iter()
        .any(|n| {
            n.as_str()
                .is_some_and(|n| n.contains("an earlier `ptxv check --certify`"))
        })
}

/// fixtures/e_resid with `residency = 1` in the case: the ctaid chain (variant chain) needs every
/// CTA resident, so it deadlocks at the case's residency and is clean at `--residency 4`. A
/// check at `--residency 4` is not stored: equiv checks the case again and gets the deadlock.
/// A check of the case as it is (residency 1) is stored and serves equiv (another cache: equiv
/// stores its own check too).
#[test]
fn a_residency_override_is_not_stored() {
    let tmp = std::env::temp_dir().join(format!("ptxv-test-vcache-resid-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    let case = tmp.join("e_resid_toml1");
    std::fs::create_dir_all(&case).unwrap();
    let fx = root().join("fixtures/e_resid");
    std::fs::copy(fx.join("kernel.ptx"), case.join("kernel.ptx")).unwrap();
    let toml = std::fs::read_to_string(fx.join("case.toml")).unwrap();
    let outputs = toml.find("\noutputs = ").unwrap() + 1;
    let eol = outputs + toml[outputs..].find('\n').unwrap() + 1;
    std::fs::write(
        case.join("case.toml"),
        format!("{}residency = 1\n{}", &toml[..eol], &toml[eol..]),
    )
    .unwrap();

    let cache = tmp.join("cache4");
    let (ok, t) = check(&case, &tmp.join("c4"), &cache, &["--residency", "4"]);
    assert!(ok && t.contains("[clean]"), "--residency 4: {t}");
    let doc = equiv(&case, &tmp.join("e4"), &cache);
    assert_eq!(
        not_clean(&doc),
        [r#"not clean: ["deadlock"]"#; 2],
        "{doc:#}"
    );
    assert!(!reference_served(&doc), "{doc:#}");

    // The control: a check of the case's own residency is stored and served.
    let cache = tmp.join("cache1");
    let (ok, t) = check(&case, &tmp.join("c1"), &cache, &[]);
    assert!(!ok && t.contains("[deadlock]"), "residency 1: {t}");
    let doc = equiv(&case, &tmp.join("e1"), &cache);
    assert_eq!(
        not_clean(&doc),
        [r#"not clean: ["deadlock"]"#; 2],
        "{doc:#}"
    );
    assert!(reference_served(&doc), "{doc:#}");
    let _ = std::fs::remove_dir_all(&tmp);
}
