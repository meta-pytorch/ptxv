/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! `ptxv check --nonfinite` on GB300-captured fixtures (no GPU): which outputs are NaN / Inf,
//! the instruction that produced them, inputs that already hold NaN / Inf, and errdist's
//! P(output NaN / Inf) on the same kernel.

use std::path::Path;
use std::path::PathBuf;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// `ptxv check CASE --certify --front sass --json` with `flag`: the first row.
fn check(case: &Path, flag: &str) -> serde_json::Value {
    let dir = std::env::temp_dir().join(format!(
        "ptxv-test-nonfinite-{}-{}",
        case.file_name().unwrap().to_string_lossy(),
        std::process::id()
    ));
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_ptxv"))
        .args([
            "check",
            case.to_str().unwrap(),
            "--certify",
            "--front",
            "sass",
        ])
        .args(["--json", "--out", dir.to_str().unwrap(), flag])
        .current_dir(root())
        .env("PTXV_QUIET", "1")
        .env("CUDA_VISIBLE_DEVICES", "")
        .output()
        .unwrap();
    let doc: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap_or_else(|e| {
        panic!(
            "{e}: {}\n{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        )
    });
    doc["rows"][0].clone()
}

fn kinds(row: &serde_json::Value) -> Vec<String> {
    row["verdict"]["findings"]
        .as_array()
        .unwrap()
        .iter()
        .map(|f| f["kind"].as_str().unwrap().to_string())
        .collect()
}

fn output<'a>(row: &'a serde_json::Value, name: &str) -> &'a serde_json::Value {
    row["nonfinite"]["outputs"]
        .as_array()
        .unwrap()
        .iter()
        .find(|o| o["buffer"] == name)
        .unwrap()
}

/// A fully masked softmax row: max = -inf, x - max = -inf - -inf = NaN, so 128 NaN outputs from
/// finite inputs, born at the FADD (the -inf of the mask is a select, not a birth).
#[test]
fn masked_softmax_row_is_nan() {
    let row = check(
        &root().join("fixtures/c_nonfinite_masked_softmax"),
        "--nonfinite",
    );
    assert!(
        kinds(&row).contains(&"nonfinite-output".to_string()),
        "{row}"
    );
    let o = output(&row, "out");
    assert_eq!((o["nan"].as_u64(), o["inf"].as_u64()), (Some(128), Some(0)));
    let b = &row["nonfinite"]["births"][0];
    assert_eq!(b["kind"], "nan from inf", "{b}");
    assert!(b["at"].as_str().unwrap().contains("FADD"), "{b}");
    assert_eq!(b["operands"][0], "-inf");
    // The simulator agrees with the GB300 capture bitwise, NaN row included.
    assert!(
        row["expect_report"][0]
            .as_str()
            .unwrap()
            .contains("bitwise equal"),
        "{row}"
    );
}

/// exp in f32 stored as f16: 114 of 1024 outputs overflow to +inf in the conversion.
#[test]
fn exp_overflow_in_f16_conversion() {
    let row = check(&root().join("fixtures/c_nonfinite_exp_f16"), "--nonfinite");
    assert!(
        kinds(&row).contains(&"nonfinite-output".to_string()),
        "{row}"
    );
    let o = output(&row, "out");
    assert_eq!((o["nan"].as_u64(), o["inf"].as_u64()), (Some(0), Some(114)));
    let b = &row["nonfinite"]["births"][0];
    assert_eq!(b["kind"], "inf from finite", "{b}");
    assert!(b["at"].as_str().unwrap().contains("F2FP.F16.F32"), "{b}");
}

/// GB300 semantics (each output bitwise equal to the capture): cvt.rn.satfinite to e4m3 / e5m2
/// saturates (no NaN / Inf), MUFU.EX2 overflows to +inf (23 outputs), and max(inf - inf, 0) = 0:
/// the NaN born at the FADD never reaches an output, and is not blamed for out2's Inf.
#[test]
fn probe_satfinite_ex2_and_max() {
    let row = check(&root().join("fixtures/c_nonfinite_probe"), "--nonfinite");
    for (b, n) in [("out0", 0), ("out1", 0), ("out2", 23), ("out3", 0)] {
        let o = output(&row, b);
        assert_eq!(
            o["nan"].as_u64().unwrap() + o["inf"].as_u64().unwrap(),
            n,
            "{b}: {o}"
        );
    }
    let f = row["verdict"]["findings"]
        .as_array()
        .unwrap()
        .iter()
        .find(|f| f["kind"] == "nonfinite-output")
        .unwrap();
    let msg = f["msg"].as_str().unwrap();
    assert!(
        msg.contains("output out2") && msg.contains("MUFU.EX2"),
        "{msg}"
    );
    assert!(!msg.contains("nan from inf"), "{msg}");
    let births = row["nonfinite"]["births"].as_array().unwrap();
    assert!(births.iter().any(|b| b["kind"] == "nan from inf"), "{row}");
    for e in row["expect_report"].as_array().unwrap() {
        assert!(e.as_str().unwrap().contains("bitwise equal"), "{e}");
    }
}

/// No NaN / Inf: no finding, and the scan says so.
#[test]
fn clean_kernel_has_no_finding() {
    let row = check(&root().join("fixtures/c_silu"), "--nonfinite");
    assert!(
        !kinds(&row).contains(&"nonfinite-output".to_string()),
        "{row}"
    );
    assert_eq!(output(&row, "out")["inf"], 0);
    assert_eq!(row["nonfinite"]["terminal_outputs_nonfinite"], 0);
}

/// An input that already holds an Inf: a note, not a finding; --nonfinite-strict: a finding.
#[test]
fn nonfinite_input_is_a_note_unless_strict() {
    let src = root().join("fixtures/c_nonfinite_exp_f16");
    let dst = std::env::temp_dir().join(format!("ptxv-nonfinite-input-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dst);
    std::fs::create_dir_all(dst.join("data")).unwrap();
    std::fs::create_dir_all(dst.join("expected")).unwrap();
    for f in ["case.toml", "kernel.ptx", "jit.cubin", "expected/out.bin"] {
        std::fs::copy(src.join(f), dst.join(f)).unwrap();
    }
    let mut x = std::fs::read(src.join("data/x.bin")).unwrap();
    x[0..2].copy_from_slice(&0x7c00u16.to_le_bytes());
    std::fs::write(dst.join("data/x.bin"), x).unwrap();
    let row = check(&dst, "--nonfinite");
    assert!(
        !kinds(&row).contains(&"nonfinite-output".to_string()),
        "{row}"
    );
    assert_eq!(row["nonfinite"]["inputs"][0]["inf"], 1, "{row}");
    assert!(
        row["verdict"]["notes"]
            .as_array()
            .unwrap()
            .iter()
            .any(|n| n.as_str().unwrap().starts_with("nonfinite:")),
        "{row}"
    );
    let row = check(&dst, "--nonfinite-strict");
    assert!(
        kinds(&row).contains(&"nonfinite-output".to_string()),
        "{row}"
    );
    let _ = std::fs::remove_dir_all(&dst);
}

/// errdist's P(output NaN / Inf) on the f16 exp kernel, x ~ N(6, 4): e^x reaches 65520 (the
/// f16 overflow point) for x > ln 65520 = 11.09, P = 1 - Phi(1.273) = 0.1015.
#[test]
fn errdist_p_nonfinite_matches_the_overflow_probability() {
    let dir = std::env::temp_dir().join(format!("ptxv-test-nonfinite-ed-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let js = dir.join("r.json");
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_ptxv"))
        .args(["errdist", "fixtures/c_nonfinite_exp_f16"])
        .args(["--inputs", "fixtures/errdist/exp_overflow.json"])
        .args([
            "--out",
            dir.to_str().unwrap(),
            "--json",
            js.to_str().unwrap(),
        ])
        .current_dir(root())
        .env("PTXV_QUIET", "1")
        .env("PTXV_XCHECK_SECONDS", "0")
        .env("PTXV_NL_SECONDS", "0")
        .env("CUDA_VISIBLE_DEVICES", "")
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let r: serde_json::Value = serde_json::from_slice(&std::fs::read(&js).unwrap()).unwrap();
    let p = &r["p_nonfinite"]["element"];
    let (lo, hi) = (p[1].as_f64().unwrap(), p[2].as_f64().unwrap());
    assert!(lo - 0.003 < 0.1015 && 0.1015 < hi + 0.003, "{p}");
}
