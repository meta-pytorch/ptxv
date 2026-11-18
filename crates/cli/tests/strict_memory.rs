/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::path::PathBuf;
use std::process::Command;
use std::process::Output;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

static NEXT_CASE: AtomicU64 = AtomicU64::new(0);

struct CaseDir(PathBuf);

impl Drop for CaseDir {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).expect("remove isolated strict memory case");
    }
}

fn case(body: &str, threads: u32, undefined: bool) -> CaseDir {
    let id = NEXT_CASE.fetch_add(1, Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!("ptxv-strict-memory-{}-{id}", std::process::id()));
    std::fs::create_dir_all(&path).unwrap();
    std::fs::write(
        path.join("kernel.ptx"),
        format!(".version 9.0\n.target sm_103a\n.address_size 64\n.global .align 8 .u64 x[2] = {{0, 0}};\n.visible .entry k() {{\n.reg .b64 %r<2>;\n{body}\nret;\n}}"),
    )
    .unwrap();
    let init = if undefined { "undef" } else { "zeros" };
    std::fs::write(
        path.join("case.toml"),
        format!(
            r#"name = "strict-memory"
description = "production acceptance route controls"
ptx = "kernel.ptx"
entry = "k"
grid = [1, 1, 1]
block = [{threads}, 1, 1]
outputs = []
params = []
[[buffers]]
name = "x"
dtype = "u64"
shape = [2]
init = "{init}"
scratch = {undefined}
symbol = "x"
[[variants]]
id = "good"
summary = "control"
expect = ["clean"]
"#
        ),
    )
    .unwrap();
    CaseDir(path)
}

fn run(case: &CaseDir, strict: bool, cap: usize, json: bool) -> Output {
    run_with_flags(
        case,
        if strict {
            &["--strict-memory-safety"]
        } else {
            &[]
        },
        cap,
        json,
    )
}

fn run_with_flags(case: &CaseDir, flags: &[&str], cap: usize, json: bool) -> Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_ptxv"));
    command.args([
        "check",
        case.0.to_str().unwrap(),
        "--front",
        "ptx",
        "--variant",
        "good",
        "--no-input-coverage",
        "--no-gb300",
        "--max-states",
        &cap.to_string(),
    ]);
    command.arg("--out").arg(case.0.join("check-out"));
    command.args(flags);
    if json {
        command.arg("--json");
    }
    command
        .env("CUDA_VISIBLE_DEVICES", "")
        .current_dir(&case.0)
        .output()
        .unwrap()
}

fn check(case: &CaseDir, strict: bool, cap: usize) -> (bool, serde_json::Value) {
    let output = run(case, strict, cap, true);
    let value = serde_json::from_slice(&output.stdout).unwrap_or_else(|error| {
        panic!(
            "JSON: {error}; stdout={}; stderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        )
    });
    (output.status.success(), value)
}

fn assert_headline(case: &CaseDir, strict: bool, cap: usize, status: &str, pass: bool) {
    let output = run(case, strict, cap, false);
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert_eq!(output.status.success(), pass, "{stdout}");
    let marker = if pass { "PASS " } else { "FAIL " };
    let headline = stdout
        .lines()
        .find(|line| line.starts_with(marker))
        .unwrap_or_else(|| panic!("missing {marker} headline: {stdout}"));
    if strict {
        assert!(
            headline.contains(&format!("strict-memory={status}")),
            "{headline}"
        );
        if !pass {
            assert!(
                !stdout.lines().any(|line| line.starts_with("PASS ")),
                "{stdout}"
            );
        }
    } else {
        assert!(!headline.contains("strict-memory="), "{headline}");
    }
}

#[test]
fn production_check_reports_strict_certificate_safety() {
    let case = case("st.global.u64 [x], 1;\nld.global.u64 %r0, [x];", 1, false);
    let (passed, value) = check(&case, true, 100);
    assert!(passed, "{value}");
    assert_eq!(value["command"], "check");
    let report = &value["rows"][0]["strict_memory_safety"];
    assert!(value["rows"][0].get("nonfinite").is_none());
    assert_eq!(report["status"], "safe", "{value}");
    assert_eq!(
        report["scope"],
        "raw_ptx_fixed_launch_all_payloads_legal_placements"
    );
    assert_eq!(report["states"], 4);
    assert_headline(&case, true, 100, "safe", true);
}

#[test]
fn strict_shortcut_enables_all_strict_checks() {
    let case = case("st.global.u64 [x], 1;\nld.global.u64 %r0, [x];", 1, false);
    let mut reports = Vec::new();
    for flags in [
        &["--strict"][..],
        &["--strict-memory-safety", "--nonfinite-strict"][..],
        &["--strict", "--strict-memory-safety", "--nonfinite-strict"][..],
    ] {
        let output = run_with_flags(&case, flags, 100, true);
        let value: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        assert!(output.status.success(), "{value}");
        let row = &value["rows"][0];
        assert_eq!(row["strict_memory_safety"]["status"], "safe", "{value}");
        assert_eq!(row["nonfinite"]["strict"], true, "{value}");
        reports.push((
            row["strict_memory_safety"].clone(),
            row["nonfinite"].clone(),
        ));
    }
    assert!(reports.windows(2).all(|pair| pair[0] == pair[1]));
}

#[test]
fn strict_shortcut_cannot_pass_unsafe_refused_or_capped_checks() {
    for (body, cap, status) in [
        ("ld.global.u64 %r0, [x+16];", 100, "unsafe"),
        ("shl.b64 %r0, 1, 2;", 100, "inconclusive"),
        ("st.global.u64 [x], 1;", 1, "inconclusive"),
    ] {
        let case = case(body, 1, false);
        let output = run_with_flags(&case, &["--strict"], cap, true);
        let value: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        assert!(!output.status.success(), "{value}");
        assert_eq!(value["pass"], false, "{value}");
        assert_eq!(
            value["rows"][0]["strict_memory_safety"]["status"], status,
            "{value}"
        );
    }
}

#[test]
fn strict_shortcut_rejects_options_that_bypass_memory_safety() {
    let case = case("ld.global.u64 %r0, [x];", 1, true);
    for flags in [
        &["--strict", "--coverage-only"][..],
        &["--strict", "--residency", "auto"][..],
        &["--strict", "--tmem-cotenant"][..],
    ] {
        let output = run_with_flags(&case, flags, 100, true);
        assert_eq!(output.status.code(), Some(2));
        let stderr = String::from_utf8(output.stderr).unwrap();
        assert!(stderr.contains("cannot be used with"), "{stderr}");
        assert!(output.stdout.is_empty());
    }
}

#[test]
fn production_check_does_not_accept_dead_reads_same_value_races_or_oob() {
    for (body, threads, undefined) in [
        ("ld.global.u64 %r0, [x];", 1, true),
        ("st.global.u64 [x], 0;", 2, false),
        ("ld.global.u64 %r0, [x+16];", 1, false),
    ] {
        let case = case(body, threads, undefined);
        let (passed, value) = check(&case, true, 100);
        assert!(!passed, "{value}");
        assert_eq!(
            value["rows"][0]["strict_memory_safety"]["status"], "unsafe",
            "{value}"
        );
        assert_headline(&case, true, 100, "unsafe", false);
    }
}

#[test]
fn production_cap_and_ordinary_clean_cannot_promote_to_strict_safe() {
    let bounded = case("ld.global.u64 %r0, [x];", 1, false);
    let (passed, value) = check(&bounded, true, 1);
    assert!(!passed, "{value}");
    assert_eq!(
        value["rows"][0]["strict_memory_safety"]["status"],
        "inconclusive"
    );
    assert_eq!(
        value["rows"][0]["strict_memory_safety"]["reason"]["kind"],
        "state_cap"
    );
    assert_headline(&bounded, true, 1, "inconclusive", false);
    let unsupported = case("shl.b64 %r0, 1, 2;", 1, false);
    let (passed, ordinary) = check(&unsupported, false, 100);
    assert!(passed, "{ordinary}");
    assert_eq!(ordinary["rows"][0]["got"], serde_json::json!(["clean"]));
    assert!(ordinary["rows"][0].get("strict_memory_safety").is_none());
    assert_headline(&unsupported, false, 100, "", true);
    let (passed, strict) = check(&unsupported, true, 100);
    assert!(!passed, "{strict}");
    assert_eq!(
        strict["rows"][0]["strict_memory_safety"]["status"],
        "inconclusive"
    );
    assert_eq!(
        strict["rows"][0]["strict_memory_safety"]["reason"]["kind"],
        "source_refused"
    );
    assert_headline(&unsupported, true, 100, "inconclusive", false);
}

#[test]
fn production_source_register_control_and_finite_loop_are_checked() {
    let case = case("", 1, true);
    let path = case.0.join("kernel.ptx");
    let source = std::fs::read_to_string(&path).unwrap().replace(
        ".reg .b64 %r<2>;",
        ".reg .b32 %r<2>;\n.reg .b64 %rd<2>;\n.reg .pred %p<1>;\n\
         mov.u32 %r0, 0;\nL: cvt.u64.u32 %rd1, %r0;\n\
         mov.u64 %rd0, x;\nadd.u64 %rd0, %rd0, %rd1;\n\
         st.global.u8 [%rd0], 1;\nadd.u32 %r0, %r0, 1;\n\
         setp.lt.u32 %p0, %r0, 16;\n@%p0 bra L;",
    );
    std::fs::write(&path, &source).unwrap();
    let (passed, value) = check(&case, true, 512);
    assert!(passed, "{value}");
    assert_eq!(value["rows"][0]["strict_memory_safety"]["status"], "safe");
    std::fs::write(&path, source.replace("%r0, 16", "%r0, 20")).unwrap();
    let (passed, value) = check(&case, true, 512);
    assert!(!passed, "{value}");
    assert_ne!(value["rows"][0]["strict_memory_safety"]["status"], "safe");
}

#[test]
fn production_source_thread_ids_cover_every_actor() {
    let case = case("", 2, false);
    let path = case.0.join("kernel.ptx");
    let source = std::fs::read_to_string(&path).unwrap().replace(
        ".reg .b64 %r<2>;",
        ".reg .b32 %r<2>;\n.reg .b64 %rd<2>;\n\
         mov.u32 %r0, %tid.x;\nmul.wide.u32 %rd1, %r0, 8;\n\
         mov.u64 %rd0, x;\nadd.u64 %rd0, %rd0, %rd1;\n\
         st.global.u64 [%rd0], 1;",
    );
    std::fs::write(&path, &source).unwrap();
    let (passed, value) = check(&case, true, 512);
    assert!(passed, "{value}");
    assert_eq!(value["rows"][0]["strict_memory_safety"]["status"], "safe");
    std::fs::write(&path, source.replace("%tid.x", "0")).unwrap();
    let (passed, value) = check(&case, true, 512);
    assert!(!passed, "{value}");
    assert_eq!(value["rows"][0]["strict_memory_safety"]["status"], "unsafe");
}

#[test]
fn production_raw_alignment_and_object_extent_are_preserved() {
    let case = case("st.global.u64 [x], 1;", 1, false);
    let source_path = case.0.join("kernel.ptx");
    let source = std::fs::read_to_string(&source_path).unwrap();
    std::fs::write(&source_path, source.replace(".align 8", ".align 1")).unwrap();
    let (passed, value) = check(&case, true, 100);
    assert!(!passed, "{value}");
    assert_ne!(value["rows"][0]["strict_memory_safety"]["status"], "safe");
    std::fs::write(&source_path, source).unwrap();
    let manifest_path = case.0.join("case.toml");
    let manifest = std::fs::read_to_string(&manifest_path).unwrap();
    std::fs::write(
        &manifest_path,
        manifest.replace("shape = [2]", "shape = [4]"),
    )
    .unwrap();
    let (passed, value) = check(&case, true, 100);
    assert!(!passed, "{value}");
    assert_eq!(
        value["rows"][0]["strict_memory_safety"]["status"],
        "inconclusive"
    );
}
