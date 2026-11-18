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

/// `ptxv errdist CASE --inputs SPEC ARGS --json` without the analytic tiers' time budget: the
/// output, the JSON report (written before a refusal) and whether errdist succeeded.
fn errdist_json(
    tag: &str,
    case: &str,
    spec: &str,
    args: &[&str],
) -> (String, serde_json::Value, bool) {
    let dir =
        std::env::temp_dir().join(format!("ptxv-test-nonfinite-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let js = dir.join("r.json");
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_ptxv"))
        .args(["errdist", case, "--inputs", spec])
        .args(args)
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
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let r: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&js).unwrap_or_else(|e| panic!("{e}: {text}")))
            .unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    (text, r, out.status.success())
}

/// MISS 64: the same kernel at x ~ N(1000, 1): every output and its exact value overflow, so
/// no output is compared and every input is out of the domain. Tier 3 is refused and errdist
/// reports no headline. Until 2026-11-11 Tier 3 was the headline, P(differ) 0 over [0, 1].
#[test]
fn errdist_every_input_out_of_domain_is_refused() {
    let (text, r, ok) = errdist_json(
        "ood",
        "fixtures/c_nonfinite_exp_f16",
        "fixtures/errdist/exp_overflow_all.json",
        &["--mc", "300"],
    );
    assert!(!ok, "{text}");
    assert!(
        text.contains("Tier 3 compared no output") && text.contains("refusing to report"),
        "{text}"
    );
    assert_eq!(r["out_of_domain"]["mass"], 1.0, "{}", r["out_of_domain"]);
    assert!(r["best"].is_null(), "{}", r["best"]);
    let t3 = r["estimates"]
        .as_array()
        .unwrap()
        .iter()
        .find(|e| e["tier"] == 3)
        .unwrap();
    assert_eq!(t3["applicable"], false, "{t3}");
}

/// The estimates of a report, by tier.
fn tier(r: &serde_json::Value, t: u64) -> Vec<&serde_json::Value> {
    r["estimates"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|e| e["tier"] == t)
        .collect()
}

/// [p, lo, hi] of an estimate's P(differ).
fn p_of(e: &serde_json::Value) -> [f64; 3] {
    let p = &e["p_differ"];
    [0, 1, 2].map(|i| p[i].as_f64().unwrap())
}

/// MISS 66 (N-ERRDIST-NONFINITE): an intermediate overflow at 0.5% of the elements
/// (fixtures/errdist/nonfinite/ovf, y = RN(RN(x x) 2^-64), x ~ spec_mix.json): the kernel stores
/// +Inf where the exact value is finite, so P(differ) is 0.005. Tier 3 and the elementwise law
/// compare those outputs; a linearized tier, which models rounding errors only, is refused, and
/// its validation does not agree. Until 2026-11-11 the outputs were left out: Tier 3 headlined
/// P(differ) 0 [0, 1.45e-5] and the elementwise law a conditional P.
#[test]
fn errdist_nonfinite_outputs_that_differ_are_counted() {
    let (text, r, ok) = errdist_json(
        "ovf",
        "fixtures/errdist/nonfinite/ovf",
        "fixtures/errdist/nonfinite/spec_mix.json",
        &["--mc", "3000", "--sim-trials", "50"],
    );
    assert!(ok, "{text}");
    let t3 = tier(&r, 3);
    let [p, lo, hi] = p_of(t3[0]);
    assert!(
        lo <= 0.005 && 0.005 <= hi && p > 0.003,
        "{}",
        t3[0]["p_differ"]
    );
    assert!(
        text.contains("are NaN / Inf in Tier 3 and compared in P(differ)"),
        "{text}"
    );
    // The elementwise law (f32 quadrature) counts the overflow's mass itself.
    let el: Vec<_> = r["estimates"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|e| e["method"].as_str().unwrap().starts_with("elementwise"))
        .collect();
    assert_eq!(el.len(), 1, "{text}");
    let [pe, _, _] = p_of(el[0]);
    assert!((pe - 0.005).abs() < 2.5e-4, "{}", el[0]["p_differ"]);
    // A linearized tier is refused, and its simulator validation disagrees.
    let lin: Vec<_> = r["estimates"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|e| e["method"].as_str().unwrap().starts_with("linearized"))
        .collect();
    assert!(!lin.is_empty(), "{text}");
    for e in &lin {
        assert_eq!(e["applicable"], false, "{e}");
        assert!(
            e["applicability"].as_array().unwrap().iter().any(|l| l
                .as_str()
                .unwrap()
                .contains("NaN / Inf outputs that differ")),
            "{e}"
        );
    }
    assert!(
        r["validation"]
            .as_array()
            .unwrap()
            .iter()
            .any(|v| v["source"] == "sim"
                && v["against"] == "tier 2"
                && v["agree"] == false
                && v["notes"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|n| n.as_str().unwrap().contains("which tier 2 does not model"))),
        "{}",
        r["validation"]
    );
    // The headline covers the overflow.
    let best = &r["estimates"][r["best"].as_u64().unwrap() as usize];
    let [_, blo, bhi] = p_of(best);
    assert!(blo <= 0.005 && 0.005 <= bhi, "{best}");
    assert!(
        r["assumptions"]
            .as_array()
            .unwrap()
            .iter()
            .any(|a| a == "N-ERRDIST-NONFINITE")
    );
}

/// MISS 66: y = RN(x x) at the same inputs (fixtures/errdist/nonfinite/ovf_sq): where x^2
/// overflows, RN of the exact value is +Inf too, so the +Inf output equals it. Tier 3 compares
/// it, and P(differ) is 0 (the product of two f32 values is correctly rounded), with nothing
/// out of the domain. Until 2026-11-11 those outputs were left out of the domain.
#[test]
fn errdist_nonfinite_outputs_equal_to_rn_exact_match() {
    let (text, r, ok) = errdist_json(
        "ovfsq",
        "fixtures/errdist/nonfinite/ovf_sq",
        "fixtures/errdist/nonfinite/spec_mix.json",
        &["--mc", "3000"],
    );
    assert!(ok, "{text}");
    let t3 = tier(&r, 3);
    let [p, _, hi] = p_of(t3[0]);
    assert!(p == 0.0 && hi < 2e-3, "{}", t3[0]["p_differ"]);
    assert!(text.contains("compared in P(differ) (0 differ"), "{text}");
    assert_eq!(r["out_of_domain"]["mass"], 0.0, "{}", r["out_of_domain"]);
}

/// The out-of-domain causes of a report.
fn causes(r: &serde_json::Value) -> Vec<String> {
    r["out_of_domain"]["by_cause"]
        .as_object()
        .unwrap()
        .keys()
        .cloned()
        .collect()
}

/// The applicability lines of an estimate that contain `pat`.
fn lines<'a>(e: &'a serde_json::Value, pat: &str) -> Vec<&'a str> {
    e["applicability"]
        .as_array()
        .unwrap()
        .iter()
        .map(|l| l.as_str().unwrap())
        .filter(|l| l.contains(pat))
        .collect()
}

const UNIDENT: &str = "no exact value: a slice without an identified real function";

/// MISS 88: a slice whose guard the identification probes take both ways
/// (fixtures/errdist/nonfinite/unident): no output has an exact value, though every output is
/// finite. The out-of-domain cause names the unidentified slice and P(output NaN / Inf) is 0;
/// with no output compared Tier 3 is refused (MISS 64). Until 2026-11-11 the cause read
/// "non-finite output (NaN / Inf)" and Tier 3 headlined P(differ) 0 over [0, 1].
#[test]
fn errdist_unidentified_slice_is_not_nonfinite() {
    let (text, r, ok) = errdist_json(
        "unident",
        "fixtures/errdist/nonfinite/unident",
        "fixtures/errdist/normal.json",
        &["--mc", "300", "--sim-trials", "0"],
    );
    assert!(!ok, "{text}");
    assert_eq!(causes(&r), [UNIDENT], "{text}");
    assert_eq!(r["p_nonfinite"]["element"][0], 0.0, "{}", r["p_nonfinite"]);
    assert!(r["best"].is_null(), "{}", r["best"]);
}

/// MISS 88: the same slice under a maximum (fixtures/errdist/nonfinite/unident_max,
/// y = max(a', v)). The slice's exact value is unknown, and a maximum must not pass on v as the
/// exact value because the other child is NaN: no output has an exact value, and Tier 3 is
/// refused. Until 2026-11-11 the exact value was v, and errdist compared max(a', v) against it.
#[test]
fn errdist_unidentified_slice_is_not_passed_over_by_a_maximum() {
    let (text, r, ok) = errdist_json(
        "unidentmax",
        "fixtures/errdist/nonfinite/unident_max",
        "fixtures/errdist/normal.json",
        &["--mc", "300", "--sim-trials", "0"],
    );
    assert!(!ok, "{text}");
    assert_eq!(causes(&r), [UNIDENT], "{text}");
    assert_eq!(r["out_of_domain"]["mass"], 1.0, "{}", r["out_of_domain"]);
    assert!(r["best"].is_null(), "{}", r["best"]);
}

/// MISS 66: outputs left out of P(differ) widen its interval. In
/// fixtures/errdist/nonfinite/unident_one only output 0 is above the unidentified slice: 1 of 64
/// outputs per input is left out, and Tier 3's P(differ) over the other outputs (about 0.24) is
/// widened by the share's bound q, within 10% of P, so Tier 3 stands. Until 2026-11-11 the
/// interval covered the compared outputs only. Since MISS 139 q is the input-level bound at
/// any count, which cannot exclude a rare input that leaves out more than its one output: at
/// 300 inputs q is about 3.8e-2, above 10% of P, and 1500 inputs bring it under.
#[test]
fn errdist_outputs_left_out_widen_the_interval() {
    let (text, r, ok) = errdist_json(
        "unidentone",
        "fixtures/errdist/nonfinite/unident_one",
        "fixtures/errdist/normal.json",
        &["--mc", "1500", "--sim-trials", "0"],
    );
    assert!(ok, "{text}");
    assert_eq!(causes(&r), [UNIDENT], "{text}");
    let t3 = tier(&r, 3)[0];
    assert_ne!(t3["applicable"], false, "{t3}");
    let l = lines(t3, "outputs left out of P(differ)");
    assert!(l.len() == 1 && l[0].ends_with(" ok"), "{t3}");
    // The share left out widens the interval by q a side.
    let [p, lo, hi] = p_of(t3);
    assert!(p > 0.15 && hi - lo > 0.04, "{}", t3["p_differ"]);
    let best = &r["estimates"][r["best"].as_u64().unwrap() as usize];
    assert_eq!(best["tier"], 3, "{best}");
}

/// MISS 66: a share left out that is more than 10% of P(differ) refuses the estimates. The
/// f16 exp kernel at x ~ N(0, 1) w.p. 0.999 and N(1000, 1) w.p. 0.001
/// (fixtures/errdist/exp_mix_rare.json): e^x has no determined exact value on 0.1% of the
/// outputs, and P(differ) over the others is about 4e-5. Tier 3 and the exact elementwise law
/// are refused, and errdist reports no headline. Until 2026-11-11 Tier 3 and the elementwise
/// law reported P(differ) over the other outputs.
#[test]
fn errdist_a_large_share_left_out_is_refused() {
    let (text, r, ok) = errdist_json(
        "exprare",
        "fixtures/c_nonfinite_exp_f16",
        "fixtures/errdist/exp_mix_rare.json",
        &["--mc", "300"],
    );
    assert!(!ok, "{text}");
    assert_eq!(
        causes(&r),
        ["non-finite exact value (NaN / Inf in the exact arithmetic)"],
        "{text}"
    );
    let t3 = tier(&r, 3)[0];
    let l = lines(t3, "outputs left out of P(differ)");
    assert!(l.len() == 1 && l[0].ends_with("FAILS"), "{t3}");
    let t0 = tier(&r, 0);
    assert!(
        t0.iter()
            .any(|e| e["method"].as_str().unwrap().starts_with("elementwise")
                && e["applicable"] == false
                && lines(e, "has no determined outcome")
                    .iter()
                    .any(|l| l.ends_with("FAILS"))),
        "{t0:?}"
    );
    assert!(r["best"].is_null(), "{}", r["best"]);
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
