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

#[test]
fn packed_f32_births_report_the_actual_high_and_low_lanes() {
    let dir = std::env::temp_dir().join(format!("ptxv-packed-nonfinite-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("kernel.ptx"),
        r#"
.version 8.8
.target sm_103a
.address_size 64
.visible .entry k(.param .u64 input, .param .u64 output) {
    .reg .b64 %p<2>, %v<3>;
    .reg .b32 %r;
    ld.param.u64 %p0, [input];
    ld.param.u64 %p1, [output];
    ld.global.b64 %v0, [%p0];
    mov.b32 %r, 0f40000000;
    mov.b64 %v1, {%r, %r};
    mul.rn.f32x2 %v2, %v0, %v1;
    st.global.b64 [%p1], %v2;
    ret;
}
"#,
    )
    .unwrap();
    std::fs::write(
        dir.join("case.toml"),
        r#"
name = "packed-nonfinite"
description = "packed nonfinite attribution"
ptx = "kernel.ptx"
entry = "k"
grid = [1,1,1]
block = [1,1,1]
outputs = ["out"]
params = [{kind="ptr", buffer="x"}, {kind="ptr", buffer="out"}]
[[buffers]]
name = "x"
dtype = "f32"
shape = [2]
init = "file"
file = "input.bin"
[[buffers]]
name = "out"
dtype = "f32"
shape = [2]
init = "zeros"
[[variants]]
id = "good"
expect = ["nonfinite-output"]
"#,
    )
    .unwrap();
    for (input, want) in [
        ([0u32, 0x7f7fffff], "0e0, inf"),
        ([0x7f7fffff, 0], "inf, 0e0"),
        ([0x3f800000, 0x3f800000], ""),
    ] {
        std::fs::write(
            dir.join("input.bin"),
            input
                .into_iter()
                .flat_map(u32::to_le_bytes)
                .collect::<Vec<_>>(),
        )
        .unwrap();
        for front in ["ptx", "sass"] {
            let result = std::process::Command::new(env!("CARGO_BIN_EXE_ptxv"))
                .args([
                    "check",
                    dir.to_str().unwrap(),
                    "--front",
                    front,
                    "--variant",
                    "good",
                    "--nonfinite",
                    "--json",
                    "--hw-profile",
                    "none",
                    "--no-gb300",
                    "--out",
                    dir.join("result").to_str().unwrap(),
                ])
                .env("CUDA_VISIBLE_DEVICES", "")
                .output()
                .unwrap();
            let doc: serde_json::Value = serde_json::from_slice(&result.stdout)
                .unwrap_or_else(|e| panic!("{e}: {}", String::from_utf8_lossy(&result.stderr)));
            let births = doc["rows"][0]["nonfinite"]["births"].as_array().unwrap();
            if want.is_empty() {
                assert!(births.is_empty(), "{front}: {births:?}");
            } else {
                assert_eq!(births.len(), 1, "{front}: {births:?}");
                assert_eq!(births[0]["result"], want, "{front}");
            }
        }
    }
    std::fs::remove_dir_all(dir).unwrap();
}

/// MISS 67 (fixtures/r1h_nvls_inf, expected/ constructed): a multicast case's outputs are
/// compared with the one-ulp tolerance of N-MULTIMEM-ROUND16. The simulated (max, inf) against
/// the expected (inf, NaN) are adjacent encodings, and were reported within one ulp; a
/// non-finite value on either side passes only if both sides are the same infinity or both NaN,
/// so both outputs are a mismatch, on both fronts.
#[test]
fn one_ulp_tolerance_never_crosses_an_infinity() {
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_ptxv"))
        .args(["check", "fixtures/r1h_nvls_inf", "--variant", "good"])
        .current_dir(root())
        .env("CUDA_VISIBLE_DEVICES", "")
        .output()
        .unwrap();
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "{text}");
    for o in ["out0", "out1"] {
        assert_eq!(
            text.matches(&format!("{o}: MISMATCH vs the captured GPU output"))
                .count(),
            2,
            "{text}"
        );
    }
    assert!(!text.contains("within one ulp"), "{text}");
}
