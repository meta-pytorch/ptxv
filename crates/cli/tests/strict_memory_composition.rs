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

#[test]
fn partial_store_cannot_erase_neighboring_payload_dependencies() {
    let body = "st.global.u8 [x], 0;\n\
                ld.global.u64 %rd0, [x];\n\
                setp.ne.u64 %p0, %rd0, 0;\n\
                @%p0 ld.global.u64 %rd1, [x+16];\nret;";
    let partial = case(body, [1, 1, 1], [1, 1, 1], "zeros", &[]);
    assert_report(&partial, "inconclusive", Some("input_dependent"));
    let complete = case(
        &body.replace("st.global.u8", "st.global.u64"),
        [1, 1, 1],
        [1, 1, 1],
        "zeros",
        &[],
    );
    assert_report(&complete, "safe", None);
}

#[test]
fn stored_and_reloaded_payload_retains_input_dependency() {
    let body = "ld.global.u64 %rd0, [x];\n\
                st.global.u64 [y], %rd0;\n\
                ld.global.u64 %rd1, [y];\n\
                setp.ne.u64 %p0, %rd1, 0;\n\
                @%p0 ld.global.u64 %rd0, [x+16];\nret;";
    let dependent = case(body, [1, 1, 1], [1, 1, 1], "zeros", &[]);
    assert_report(&dependent, "inconclusive", Some("input_dependent"));
    let constant = case(
        &format!("st.global.u64 [x], 0;\n{body}"),
        [1, 1, 1],
        [1, 1, 1],
        "zeros",
        &[],
    );
    assert_report(&constant, "safe", None);
}

#[test]
fn last_z_dimension_actor_cannot_be_omitted() {
    let body = "mov.u32 %r0, %ctaid.z;\n\
                setp.eq.u32 %p0, %r0, 1;\n@!%p0 ret;\n\
                mov.u32 %r1, %tid.z;\n\
                setp.eq.u32 %p0, %r1, 1;\n@!%p0 ret;\n\
                ld.global.u64 %rd0, [x+16];\nret;";
    let last = case(body, [1, 1, 2], [1, 1, 2], "zeros", &[]);
    assert_report(&last, "unsafe", Some("bad_state"));
    let bounded = case(
        &body.replace("[x+16]", "[x]"),
        [1, 1, 2],
        [1, 1, 2],
        "zeros",
        &[],
    );
    assert_report(&bounded, "safe", None);
}

#[test]
fn terminal_observer_checks_outputs_without_instruction_reads() {
    let undefined = case("ret;", [1, 1, 1], [1, 1, 1], "undef", &["x"]);
    assert_report(&undefined, "unsafe", Some("bad_state"));
    let initialized = case(
        "st.global.u64 [x], 0;\nst.global.u64 [x+8], 0;\nret;",
        [1, 1, 1],
        [1, 1, 1],
        "undef",
        &["x"],
    );
    assert_report(&initialized, "safe", None);
}

#[test]
fn false_guarded_final_exit_is_not_termination() {
    let skipped = case(
        "setp.eq.u32 %p0, 0, 1;\n@%p0 ret;",
        [1, 1, 1],
        [1, 1, 1],
        "zeros",
        &[],
    );
    assert_report(&skipped, "inconclusive", Some("execution_fault"));
    assert_report_mode(&skipped, "inconclusive", Some("execution_fault"), true);
    let taken = case(
        "setp.eq.u32 %p0, 0, 0;\n@%p0 ret;",
        [1, 1, 1],
        [1, 1, 1],
        "zeros",
        &[],
    );
    assert_report(&taken, "safe", None);
}

#[test]
fn coverage_only_cannot_override_required_strict_safety() {
    let uninitialized = case(
        "ld.global.u64 %rd0, [x];\nret;",
        [1, 1, 1],
        [1, 1, 1],
        "undef",
        &[],
    );
    let output = run(&uninitialized, true, false, &["--coverage-only"]);
    assert_eq!(output.status.code(), Some(2));
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("cannot be used with"), "{stderr}");
    assert!(output.stdout.is_empty());
    assert!(!stderr.lines().any(|line| line.starts_with("PASS ")));
}
