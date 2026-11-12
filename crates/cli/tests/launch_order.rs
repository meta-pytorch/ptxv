/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! E-LAUNCH-ORDER through `ptxv check` and `ptxv equiv`: the strict reading of an unknown launch
//! order (the default; `PTXV_UNRECORDED_ORDER=ordered` keeps E-PDL's order, under which the
//! launch-order gates are checked), equiv's precondition for a variant that edits the launch
//! order, the certificate check's verdict cache (vcache.rs) for such a variant, and the
//! variant's launch order in `ptxv num` and in the simulator runs behind `ptxv hw`.

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

/// By default (and with `PTXV_UNRECORDED_ORDER=concurrent`) every grid of an unknown-order case
/// loses its predecessors, so a coherent read of the previous grid's output races
/// (PLAIN_UNKNOWN); with `ordered`, E-PDL still orders the list: clean. A recorded order keeps
/// its edges (PLAIN), in either reading. With `ordered`, an .nc read the list orders after its
/// writer is a violation whose message names the gate (UNKNOWN). Any other value is refused.
#[test]
fn unrecorded_order_concurrent_drops_unknown_predecessors() {
    let (_, t) = check("PLAIN_UNKNOWN", None);
    assert!(t.contains("[race]") && t.contains("RACE:"), "{t}");
    let (_, t) = check("PLAIN_UNKNOWN", Some("concurrent"));
    assert!(t.contains("[race]") && t.contains("RACE:"), "{t}");
    let (_, t) = check("PLAIN_UNKNOWN", Some("ordered"));
    assert!(t.contains("[clean]"), "{t}");
    let (ok, t) = check("PLAIN", None);
    assert!(ok && t.contains("[clean]"), "{t}");
    let (ok, t) = check("PLAIN", Some("ordered"));
    assert!(ok && t.contains("[clean]"), "{t}");
    let (_, t) = check("UNKNOWN", Some("ordered"));
    assert!(
        t.contains("[violation]") && t.contains("launch order is unknown"),
        "{t}"
    );
    let (ok, t) = check("PLAIN_UNKNOWN", Some("parallel"));
    assert!(!ok && t.contains("expected ordered or concurrent"), "{t}");
}

/// The classes `ptxv check fixtures/<fx> --variant <variant>` reports (PTX front, no GB300
/// re-run, `--json`) with `PTXV_UNRECORDED_ORDER` set to `order`, or unset.
fn got(fx: &str, variant: &str, order: Option<&str>) -> Vec<String> {
    let dir = std::env::temp_dir().join(format!(
        "ptxv-test-launch-order-got-{fx}-{variant}-{}",
        std::process::id()
    ));
    let mut c = Command::new(env!("CARGO_BIN_EXE_ptxv"));
    c.arg("check")
        .arg(root().join("fixtures").join(fx))
        .args([
            "--variant",
            variant,
            "--front",
            "ptx",
            "--no-gb300",
            "--no-input-coverage",
            "--json",
            "--out",
        ])
        .arg(&dir)
        .env("CUDA_VISIBLE_DEVICES", "")
        .env("PTXV_NO_VCACHE", "1")
        .env_remove("PTXV_UNRECORDED_ORDER");
    for v in ["PTXV_HW_L1", "PTXV_L1_CORES", "PTXV_L1_HAVOC"] {
        c.env_remove(v);
    }
    if let Some(o) = order {
        c.env("PTXV_UNRECORDED_ORDER", o);
    }
    let out = c.output().unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    let doc: serde_json::Value = serde_json::from_slice(&out.stdout)
        .unwrap_or_else(|e| panic!("{fx} {variant}: {e}: {}", text(&out)));
    doc["rows"][0]["got"]
        .as_array()
        .unwrap_or_else(|| panic!("{fx} {variant}: no got: {doc:#}"))
        .iter()
        .map(|x| x.as_str().unwrap_or("").to_string())
        .collect()
}

/// The launch-order gates (`Launch::order_recorded`) under `PTXV_UNRECORDED_ORDER=ordered`, where
/// an unknown-order list keeps E-PDL's order and only loses the exclusions: S-NC's stream-order
/// exclusion (e_launch_order and e_launch_waits UNKNOWN: violation), its done-clock exemption
/// (e_ld_nc_pdl UNKNOWN_DONE: violation), its launch-clock exemption (e_ld_nc_xgpu_unord good:
/// violation) and E-L1-CORES' kernel boundary (e_launch_order_l1 UNKNOWN: stale). Under the
/// strict default such a list's grids have no predecessor to exclude by, so only this reading
/// tells a gate from its absence (source mutants lo-*-gate, nc-start-gate).
#[test]
fn ordered_reading_keeps_the_launch_order_gates() {
    for (fx, variant, want) in [
        ("e_launch_order", "UNKNOWN", ["violation"]),
        ("e_launch_waits", "UNKNOWN", ["violation"]),
        ("e_ld_nc_pdl", "UNKNOWN", ["violation"]),
        ("e_ld_nc_pdl", "UNKNOWN_DONE", ["violation"]),
        ("e_ld_nc_xgpu_unord", "good", ["violation"]),
        ("e_launch_order_l1", "UNKNOWN", ["stale"]),
    ] {
        assert_eq!(got(fx, variant, Some("ordered")), want, "{fx} {variant}");
    }
}

/// `ptxv num` and the simulator runs that give `ptxv hw` its expected bytes run the variant's
/// launch order, as `ptxv check` does. fixtures/e_launch_waits_gate deadlocks as the case lists
/// it (good: kc waits for kb, which waits for kc's flag); NOWAIT drops the wait. No GPU is used:
/// with CUDA_VISIBLE_DEVICES empty, `hw` stops at the hardware step after its simulator run.
#[test]
fn num_and_hw_expect_run_the_variants_launch_order() {
    let tmp = std::env::temp_dir().join(format!("ptxv-test-num-order-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    let fx = root().join("fixtures/e_launch_waits_gate");
    // A copy whose good variant drops the wait (PTXV_HW_EXPECT=run and the default expected
    // bytes are the good variant's run).
    let copy = tmp.join("gate_good_nowait");
    std::fs::create_dir_all(&copy).unwrap();
    std::fs::copy(fx.join("kernel.ptx"), copy.join("kernel.ptx")).unwrap();
    let toml = std::fs::read_to_string(fx.join("case.toml")).unwrap();
    let good = "[[variants]]\nid = \"good\"\n";
    assert!(toml.contains(good));
    std::fs::write(
        copy.join("case.toml"),
        toml.replacen(good, &format!("{good}waits = [[2]]\n"), 1),
    )
    .unwrap();
    let mut n = 0;
    let mut run = |case: &Path, args: &[&str], expect: Option<&str>| -> (bool, String) {
        n += 1;
        let mut c = Command::new(env!("CARGO_BIN_EXE_ptxv"));
        c.arg(args[0])
            .arg(case)
            .args(&args[1..])
            .arg("--out")
            .arg(tmp.join(format!("out{n}")))
            .env("CUDA_VISIBLE_DEVICES", "")
            .env_remove("PTXV_GPUS")
            .env_remove("PTXV_UNRECORDED_ORDER");
        match expect {
            Some(e) => c.env("PTXV_HW_EXPECT", e),
            None => c.env_remove("PTXV_HW_EXPECT"),
        };
        let o = c.output().unwrap();
        (o.status.success(), text(&o))
    };
    let (ok, t) = run(&fx, &["num", "--variant", "NOWAIT"], None);
    assert!(ok, "num NOWAIT: {t}");
    let (ok, t) = run(&fx, &["num", "--variant", "good"], None);
    assert!(!ok && t.contains("did not terminate"), "num good: {t}");
    let hw = ["hw", "--variant", "good", "--runs", "1"];
    let stuck = "does not terminate in the simulator";
    let (_, t) = run(
        &fx,
        &["hw", "--variant", "NOWAIT", "--runs", "1"],
        Some("variant"),
    );
    assert!(!t.contains(stuck), "hw NOWAIT, PTXV_HW_EXPECT=variant: {t}");
    let (_, t) = run(&fx, &hw, Some("variant"));
    assert!(
        t.contains(&format!("good {stuck}")),
        "hw good, PTXV_HW_EXPECT=variant: {t}"
    );
    let (_, t) = run(&copy, &hw, Some("run"));
    assert!(
        !t.contains(stuck),
        "hw good of the copy, PTXV_HW_EXPECT=run: {t}"
    );
    let (_, t) = run(&fx, &hw, Some("run"));
    assert!(t.contains(stuck), "hw good, PTXV_HW_EXPECT=run: {t}");
    let (_, t) = run(&copy, &hw, None);
    assert!(!t.contains("did not terminate"), "hw good of the copy: {t}");
    let (_, t) = run(&fx, &hw, None);
    assert!(t.contains("did not terminate"), "hw good: {t}");
    let _ = std::fs::remove_dir_all(&tmp);
}

/// `ptxv first-access` (the capture side's scratch suggestions) runs the variant's launch order
/// too. fixtures/e_launch_waits_gate as listed never starts kc, so kb spins on the flag and never
/// writes `b`; in NOWAIT kc sets the flag and kb writes `b`.
#[test]
fn first_access_runs_the_variants_launch_order() {
    let first = |variant: &str| -> serde_json::Value {
        let dir = std::env::temp_dir().join(format!(
            "ptxv-test-first-access-{variant}-{}",
            std::process::id()
        ));
        let out = Command::new(env!("CARGO_BIN_EXE_ptxv"))
            .arg("first-access")
            .arg(root().join("fixtures/e_launch_waits_gate"))
            .args(["--variant", variant, "--front", "ptx", "--out"])
            .arg(&dir)
            .env("CUDA_VISIBLE_DEVICES", "")
            .env_remove("PTXV_UNRECORDED_ORDER")
            .output()
            .unwrap();
        let _ = std::fs::remove_dir_all(&dir);
        assert!(out.status.success(), "{variant}: {}", text(&out));
        serde_json::from_slice(&out.stdout)
            .unwrap_or_else(|e| panic!("{variant}: {e}: {}", text(&out)))
    };
    let nowait = first("NOWAIT");
    assert_eq!(nowait["b"]["first"], "write", "NOWAIT: {nowait}");
    let good = first("good");
    assert!(good.get("b").is_none(), "good: {good}");
    assert_eq!(good["a"]["first"], "write", "good: {good}");
}

/// An equiv spec of one kernel against itself: `variant` of the case `case` (relative to the
/// repository root), PTX front, certificate check.
fn spec(dir: &Path, case: &str, variant: &str) -> PathBuf {
    std::fs::create_dir_all(dir).unwrap();
    let case = root().join(case).canonicalize().unwrap();
    let k = format!(
        "case = {:?}\nvariant = {variant:?}\nfront = \"ptx\"\ncertify = true\n",
        case.to_str().unwrap()
    );
    let p = dir.join(format!("{variant}.toml"));
    std::fs::write(
        &p,
        format!("max_states = 200000\n[reference]\n{k}[[candidates]]\n{k}"),
    )
    .unwrap();
    p
}

/// `ptxv equiv <toml> --no-hw` (the preconditions only matter here): its equiv.json. With
/// `cache`, the verdict cache in that directory; without, no verdict cache.
fn equiv(toml: &Path, out: &Path, cache: Option<&Path>) -> serde_json::Value {
    let mut c = Command::new(env!("CARGO_BIN_EXE_ptxv"));
    c.arg("equiv")
        .arg(toml)
        .args(["--no-hw", "--out"])
        .arg(out)
        .current_dir(root())
        .env("CUDA_VISIBLE_DEVICES", "")
        .env("PTXV_QUIET", "1")
        .env_remove("PTXV_UNRECORDED_ORDER");
    match cache {
        Some(d) => c.env("PTXV_CACHE", d).env_remove("PTXV_NO_VCACHE"),
        None => c.env("PTXV_NO_VCACHE", "1"),
    };
    let r = c.output().unwrap();
    let t = text(&r);
    assert!(r.status.success(), "{}: {t}", toml.display());
    serde_json::from_slice(
        &std::fs::read(out.join("equiv.json")).unwrap_or_else(|e| panic!("{e}: {t}")),
    )
    .unwrap()
}

/// Asserts that both kernels' precondition 1 in `doc` is `want`: the classes of the variant's
/// own check (`ptxv check`), or clean (`None`).
fn assert_precondition(doc: &serde_json::Value, want: Option<&[&str]>, what: &str) {
    for side in ["pre_reference", "pre_candidate"] {
        let pre: Vec<String> = doc[0][side]
            .as_array()
            .unwrap_or_else(|| panic!("{what}: no {side}: {doc:#}"))
            .iter()
            .map(|p| p.as_str().unwrap_or("").to_string())
            .collect();
        let not_clean: Vec<&String> = pre.iter().filter(|p| p.starts_with("not clean")).collect();
        match want {
            None => assert!(not_clean.is_empty(), "{what} {side}: {pre:?}"),
            Some(w) => assert_eq!(not_clean, [&format!("not clean: {w:?}")], "{what} {side}"),
        }
    }
}

/// equiv's and errdist's precondition 1 checks the variant's launch order, as `ptxv check`
/// does (`Case::for_variant`), for each kind of edit: `launch_order` (UNK, UNKNOWN), `after`
/// (UNORD) and `waits` (NOWAIT). fixtures/e_equiv_order declares its order: with the case as
/// loaded, UNK and UNORD were clean (a wrong clean). The verdict cache is off.
#[test]
fn equiv_precondition_is_the_variants_launch_order() {
    let tmp = std::env::temp_dir().join(format!("ptxv-test-equiv-order-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    let fx = root().join("fixtures/e_equiv_order");
    let rows: [(PathBuf, Option<&[&str]>); 7] = [
        (spec(&tmp, "fixtures/e_equiv_order", "good"), None),
        (fx.join("equiv_UNK.toml"), Some(&["race", "violation"])),
        (fx.join("equiv_UNORD.toml"), Some(&["race", "violation"])),
        (
            spec(&tmp.join("w"), "fixtures/e_launch_waits", "good"),
            None,
        ),
        (
            spec(&tmp.join("w"), "fixtures/e_launch_waits", "NOWAIT"),
            Some(&["race", "violation"]),
        ),
        (
            spec(&tmp.join("w"), "fixtures/e_launch_waits", "UNKNOWN"),
            Some(&["race", "violation"]),
        ),
        (
            spec(&tmp.join("w"), "fixtures/e_launch_waits", "UNORD"),
            Some(&["race"]),
        ),
    ];
    for (i, (toml, want)) in rows.iter().enumerate() {
        let doc = equiv(toml, &tmp.join(format!("out{i}")), None);
        assert_precondition(&doc, *want, &toml.display().to_string());
    }
    let _ = std::fs::remove_dir_all(&tmp);
}

/// The verdict `ptxv check --certify` stores (vcache.rs) serves equiv's precondition only for
/// a variant without launch-order edits: after a check of each kind of edit (`launch_order`:
/// e_vcache_order DECL, `after`: e_equiv_order UNORD, `waits`: e_launch_waits NOWAIT), equiv
/// checks the variant again and gets its verdict. A variant without edits (e_equiv_order good)
/// is served from the cache.
#[test]
fn launch_order_edits_keep_out_of_the_verdict_cache() {
    let tmp = std::env::temp_dir().join(format!("ptxv-test-vcache-order-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    let cache = tmp.join("cache");
    let rows: [(&str, &str, Option<&[&str]>, bool); 4] = [
        ("fixtures/e_equiv_order", "good", None, true),
        ("fixtures/e_vcache_order", "DECL", None, false),
        (
            "fixtures/e_equiv_order",
            "UNORD",
            Some(&["race", "violation"]),
            false,
        ),
        (
            "fixtures/e_launch_waits",
            "NOWAIT",
            Some(&["race", "violation"]),
            false,
        ),
    ];
    for (i, (case, variant, want, served)) in rows.iter().enumerate() {
        let out = Command::new(env!("CARGO_BIN_EXE_ptxv"))
            .arg("check")
            .arg(root().join(case))
            .args([
                "--variant",
                variant,
                "--front",
                "ptx",
                "--certify",
                "--no-gb300",
                "--no-input-coverage",
                "--out",
            ])
            .arg(tmp.join(format!("check{i}")))
            .env("CUDA_VISIBLE_DEVICES", "")
            .env("PTXV_CACHE", &cache)
            .env_remove("PTXV_NO_VCACHE")
            .env_remove("PTXV_UNRECORDED_ORDER")
            .output()
            .unwrap();
        let t = text(&out);
        assert!(out.status.success(), "{case} {variant}: {t}");
        let toml = spec(&tmp.join(format!("spec{i}")), case, variant);
        let doc = equiv(&toml, &tmp.join(format!("eq{i}")), Some(&cache));
        let what = format!("{case} {variant}");
        assert_precondition(&doc, *want, &what);
        assert_eq!(
            doc.to_string()
                .contains("an earlier `ptxv check --certify`"),
            *served,
            "{what}: {doc:#}"
        );
    }
    let _ = std::fs::remove_dir_all(&tmp);
}
