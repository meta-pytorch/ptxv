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
const STATE_CAP: usize = 8192;

struct CaseDir(PathBuf);

impl Drop for CaseDir {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).expect("remove strict memory composition case");
    }
}

fn case(body: &str, grid: [u32; 3], block: [u32; 3], init: &str, outputs: &[&str]) -> CaseDir {
    let id = NEXT_CASE.fetch_add(1, Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!(
        "ptxv-strict-memory-composition-{}-{id}",
        std::process::id()
    ));
    std::fs::create_dir_all(&path).unwrap();
    std::fs::write(
        path.join("kernel.ptx"),
        format!(
            ".version 9.0\n.target sm_103a\n.address_size 64\n\
             .global .align 8 .u64 x[2] = {{0, 0}};\n\
             .global .align 8 .u64 y[2] = {{0, 0}};\n\
             .visible .entry k() {{\n\
             .reg .b32 %r<2>;\n.reg .b64 %rd<2>;\n.reg .pred %p<1>;\n\
             {body}\n}}"
        ),
    )
    .unwrap();
    std::fs::write(
        path.join("case.toml"),
        format!(
            r#"name = "strict-memory-composition"
description = "raw source, metadata, actor and observer composition controls"
ptx = "kernel.ptx"
entry = "k"
grid = {grid:?}
block = {block:?}
outputs = {outputs:?}
params = []
[[buffers]]
name = "x"
dtype = "u64"
shape = [2]
init = "{init}"
symbol = "x"
[[buffers]]
name = "y"
dtype = "u64"
shape = [2]
init = "{init}"
symbol = "y"
[[variants]]
id = "control"
summary = "strict composition control"
expect = ["clean"]
"#
        ),
    )
    .unwrap();
    CaseDir(path)
}

fn run(case: &CaseDir, json: bool, run_once: bool, additional: &[&str]) -> Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_ptxv"));
    command.args([
        "check",
        case.0.to_str().unwrap(),
        "--front",
        "ptx",
        "--variant",
        "control",
        "--strict-memory-safety",
        "--no-input-coverage",
        "--no-gb300",
        "--max-states",
        &STATE_CAP.to_string(),
    ]);
    command.args(additional);
    command.arg("--out").arg(case.0.join("out"));
    if json {
        command.arg("--json");
    }
    if run_once {
        command.env("PTXV_RUN_ONCE", "1");
    } else {
        command.env_remove("PTXV_RUN_ONCE");
    }
    command
        .env("CUDA_VISIBLE_DEVICES", "")
        .current_dir(&case.0)
        .output()
        .unwrap()
}

fn assert_report(case: &CaseDir, status: &str, reason: Option<&str>) {
    assert_report_mode(case, status, reason, false);
}

fn assert_report_mode(case: &CaseDir, status: &str, reason: Option<&str>, run_once: bool) {
    let output = run(case, true, run_once, &[]);
    let value: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap_or_else(|error| {
        panic!(
            "JSON: {error}; stdout={}; stderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        )
    });
    let report = &value["rows"][0]["strict_memory_safety"];
    assert_eq!(report["status"], status, "{value}");
    assert_eq!(
        report["scope"], "raw_ptx_fixed_launch_all_payloads_legal_placements",
        "{value}"
    );
    assert_eq!(output.status.success(), status == "safe", "{value}");
    if let Some(reason) = reason {
        assert_eq!(report["reason"]["kind"], reason, "{value}");
    }
    let text = run(case, false, run_once, &[]);
    let stdout = String::from_utf8(text.stdout).unwrap();
    assert_eq!(text.status.success(), status == "safe", "{stdout}");
    let marker = if status == "safe" { "PASS " } else { "FAIL " };
    let headline = stdout
        .lines()
        .find(|line| line.starts_with(marker))
        .unwrap_or_else(|| panic!("missing {marker} headline: {stdout}"));
    assert!(
        headline.contains(&format!("strict-memory={status}")),
        "{headline}"
    );
    if status != "safe" {
        assert!(
            !stdout.lines().any(|line| line.starts_with("PASS ")),
            "{stdout}"
        );
    }
}

fn assert_refused_before_safe(case: &CaseDir) {
    let output = run(case, true, false, &[]);
    assert!(!output.status.success());
    let value: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    if let Some(error) = value["error"].as_str() {
        assert!(
            error.contains("ptxas") && error.contains("failed:"),
            "{value}"
        );
    } else {
        assert_eq!(
            value["rows"][0]["strict_memory_safety"]["status"], "inconclusive",
            "{value}"
        );
        assert_eq!(
            value["rows"][0]["strict_memory_safety"]["reason"]["kind"], "source_refused",
            "{value}"
        );
        assert_eq!(value["rows"][0]["pass"], false, "{value}");
    }
    assert!(!String::from_utf8_lossy(&output.stdout).contains("\"status\":\"safe\""));
    let text = run(case, false, false, &[]);
    assert!(!text.status.success());
    let stdout = String::from_utf8(text.stdout).unwrap();
    assert!(
        !stdout.lines().any(|line| line.starts_with("PASS ")),
        "{stdout}"
    );
    assert!(!stdout.contains("strict-memory=safe"), "{stdout}");
}

#[test]
fn non_ptx_nop_cannot_report_safe() {
    let rejected = case("nop; ret;", [1, 1, 1], [1, 1, 1], "zeros", &[]);
    let result = run(&rejected, true, false, &[]);
    assert!(
        !result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stdout)
    );
    assert!(!String::from_utf8_lossy(&result.stdout).contains("\"status\":\"safe\""));
    assert!(!String::from_utf8_lossy(&result.stdout).contains("PASS"));
    let accepted = case("ret;", [1, 1, 1], [1, 1, 1], "zeros", &[]);
    assert_report(&accepted, "safe", None);
}

#[test]
fn bare_subtraction_cannot_report_safe() {
    let rejected = case(
        "mov.u64 %rd0,x; add.u64 %rd0,%rd0,8; ld.global.u32 %r0,[%rd0-4]; ret;",
        [1, 1, 1],
        [1, 1, 1],
        "zeros",
        &[],
    );
    let result = run(&rejected, true, false, &[]);
    assert!(
        !result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stdout)
    );
    assert!(!String::from_utf8_lossy(&result.stdout).contains("\"status\":\"safe\""));
    assert!(!String::from_utf8_lossy(&result.stdout).contains("PASS"));
    let accepted = case(
        "mov.u64 %rd0,x; ld.global.u32 %r0,[%rd0+4]; ret;",
        [1, 1, 1],
        [1, 1, 1],
        "zeros",
        &[],
    );
    assert_report(&accepted, "safe", None);
}

#[test]
fn register_index_address_cannot_report_safe() {
    let rejected = case(
        "mov.u64 %rd0,4; ld.global.u32 %r0,[x+%rd0]; ret;",
        [1, 1, 1],
        [1, 1, 1],
        "zeros",
        &[],
    );
    let result = run(&rejected, true, false, &[]);
    assert!(
        !result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stdout)
    );
    assert!(!String::from_utf8_lossy(&result.stdout).contains("\"status\":\"safe\""));
    assert!(!String::from_utf8_lossy(&result.stdout).contains("PASS"));
    let accepted = case(
        "mov.u64 %rd0,x; add.u64 %rd0,%rd0,4; ld.global.u32 %r0,[%rd0]; ret;",
        [1, 1, 1],
        [1, 1, 1],
        "zeros",
        &[],
    );
    assert_report(&accepted, "safe", None);
}

#[test]
fn incompatible_header_cannot_report_safe() {
    let rejected = case("ret;", [1, 1, 1], [1, 1, 1], "zeros", &[]);
    let path = rejected.0.join("kernel.ptx");
    let source = std::fs::read_to_string(&path)
        .unwrap()
        .replace(".version 9.0", ".version 8.0");
    std::fs::write(path, source).unwrap();
    assert_refused_before_safe(&rejected);
    let accepted = case("ret;", [1, 1, 1], [1, 1, 1], "zeros", &[]);
    assert_report(&accepted, "safe", None);
}

#[test]
fn global_label_shadow_cannot_report_safe() {
    let rejected = case(
        "x: ld.global.u32 %r0,[x]; ret;",
        [1, 1, 1],
        [1, 1, 1],
        "zeros",
        &[],
    );
    let result = run(&rejected, true, false, &[]);
    assert!(
        !result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stdout)
    );
    assert!(!String::from_utf8_lossy(&result.stdout).contains("\"status\":\"safe\""));
    assert!(!String::from_utf8_lossy(&result.stdout).contains("PASS"));
    let accepted = case(
        "label: ld.global.u32 %r0,[x]; ret;",
        [1, 1, 1],
        [1, 1, 1],
        "zeros",
        &[],
    );
    assert_report(&accepted, "safe", None);
}

#[test]
fn scalar_shadow_of_builtin_cannot_report_safe() {
    let rejected = case(
        "mov.u32 %r0,%tid.x; st.global.u32 [x],%r0; ret;",
        [1, 1, 1],
        [1, 1, 1],
        "zeros",
        &[],
    );
    let source = rejected.0.join("kernel.ptx");
    let raw = std::fs::read_to_string(&source)
        .unwrap()
        .replace(".reg .b32 %r<2>;", ".reg .b32 %r<2>; .reg .b32 %tid;");
    std::fs::write(source, raw).unwrap();
    assert_refused_before_safe(&rejected);
    let accepted = case(
        "mov.u32 %r0,%tid.x; st.global.u32 [x],%r0; ret;",
        [1, 1, 1],
        [1, 1, 1],
        "zeros",
        &[],
    );
    assert_report(&accepted, "safe", None);
}

#[test]
fn reserved_global_symbol_cannot_report_safe() {
    let rejected = case("ret;", [1, 1, 1], [1, 1, 1], "zeros", &[]);
    let source = rejected.0.join("kernel.ptx");
    let raw = std::fs::read_to_string(&source)
        .unwrap()
        .replace(".u64 x[2]", ".u64 WARP_SZ[2]");
    std::fs::write(source, raw).unwrap();
    assert_refused_before_safe(&rejected);
    let accepted = case("ret;", [1, 1, 1], [1, 1, 1], "zeros", &[]);
    assert_report(&accepted, "safe", None);
}
