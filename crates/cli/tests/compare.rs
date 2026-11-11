/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! `ptxv compare` on already-captured fixture pairs (no GPU): the report's verdicts, level,
//! bounds, GPU comparison and blockers.

fn compare(r: &str, c: &str) -> (String, serde_json::Value, String) {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let dir = std::env::temp_dir().join(format!(
        "ptxv-test-compare-{}-{}-{}",
        r.replace('/', "_"),
        c.replace('/', "_"),
        std::process::id()
    ));
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_ptxv"))
        .args(["compare", r, c, "--out", dir.to_str().unwrap()])
        .current_dir(&root)
        .env("PTXV_QUIET", "1")
        .env("CUDA_VISIBLE_DEVICES", "")
        .output()
        .unwrap();
    let text = String::from_utf8_lossy(&out.stdout).to_string();
    assert!(
        out.status.success(),
        "{text}\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let j: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("report.json")).unwrap()).unwrap();
    let md = std::fs::read_to_string(dir.join("report.md")).unwrap();
    (text, j, md)
}

/// torch.cat vs a Triton cat: both certified clean, sim = GPU, L0 proven, the captured GPU
/// outputs bitwise equal, nothing blocking.
#[test]
fn compare_cat_is_l0() {
    let (text, j, md) = compare("fixtures/c_cat", "fixtures/c_triton_cat");
    assert!(text.contains("PROVEN bitwise identical (L0)"), "{text}");
    for s in ["reference", "candidate"] {
        let side = &j[s]["side"];
        assert_eq!(side["lifted"], true, "{j}");
        assert_eq!(
            side["check"]["verdict"],
            serde_json::json!(["clean"]),
            "{j}"
        );
        assert_eq!(side["check"]["certified"], true, "{j}");
        assert!(
            side["sim_vs_gpu"][0]
                .as_str()
                .unwrap()
                .contains("bitwise equal"),
            "{j}"
        );
    }
    let e = &j["equivalence"];
    assert_eq!(e["level"], "L0", "{j}");
    assert_eq!(e["proven"], true, "{j}");
    assert_eq!(e["per_output_bound"]["with_bound"], 512, "{j}");
    assert_eq!(e["per_output_bound"]["max_u"], 0.0, "{j}");
    assert_eq!(j["gpu"]["bitwise_equal"], 512, "{j}");
    assert_eq!(j["blockers"], serde_json::json!([]), "{j}");
    assert!(md.contains("Nothing: every step is proven."), "{md}");
}

/// torch.cumsum vs Triton tl.cumsum: L1 with per-output bounds from 0 to 15u; the GPU's
/// difference on the captured input stays within the bound there.
#[test]
fn compare_cumsum_is_bounded() {
    let (text, j, _) = compare("fixtures/c_cumsum", "fixtures/c_triton_cumsum");
    assert!(text.contains("PROVEN L1"), "{text}");
    let b = &j["equivalence"]["per_output_bound"];
    assert_eq!(b["with_bound"], 512, "{j}");
    assert!(b["max_u"].as_f64().unwrap() <= 15.001, "{j}");
    assert_eq!(b["min_u"], 0.0, "{j}");
    let f = j["gpu"]["max_fraction_of_bound"].as_f64().unwrap();
    assert!(f > 0.0 && f <= 1.0, "{j}");
}

/// tanh-GELU, torch (EX2/RCP tanhf, data-dependent guards) vs Triton (MUFU.TANH): no normal
/// form covers it, so elementwise mode proves it: every output is one input's function, both
/// swept over all 2^32 f32 inputs (the committed sweep is reused), the function identified as
/// gelu_tanh, nothing blocking, and the GPU outputs still compared.
#[test]
fn compare_gelu_is_elementwise() {
    let (text, j, _) = compare("fixtures/c_gelu_tanh", "fixtures/c_triton_gelu_tanh");
    assert!(
        text.contains("PROVEN elementwise for every f32 input (all 2^32 values)"),
        "{text}"
    );
    let e = &j["equivalence"]["elementwise"];
    assert_eq!(e["level"], "EW", "{e}");
    assert_eq!(e["inputs"], 1u64 << 32, "{e}");
    assert_eq!(e["true_fn"], "gelu_tanh", "{e}");
    assert!(e["max_abs"].as_f64().unwrap() < 1e-5, "{e}");
    assert!(
        e["reference_vs_true"]["max_abs"].as_f64().unwrap() < 1e-6,
        "{e}"
    );
    assert!(
        e["candidate_vs_true"]["max_abs"].as_f64().unwrap() < 1e-5,
        "{e}"
    );
    assert!(text.contains("data-dependent guards"), "{text}");
    assert_eq!(j["blockers"], serde_json::json!([]), "{j}");
    assert_eq!(j["gpu"]["outputs"], 1024, "{j}");
    assert!(j["gpu"]["max_abs_diff"].as_f64().unwrap() > 0.0, "{j}");
}

/// torch.compile vs eager (docs/compile-vs-eager.md), a fused pair: Inductor's persistent
/// residual-add + RMSNorm kernel against the 7-kernel eager sequence it replaces (add, pow,
/// mean, add, rsqrt, mul, mul), on the same captured inputs. Both certified clean, sim = GPU,
/// L1 proven in the norm form; the GPU outputs happen to be bitwise equal.
#[test]
fn compare_inductor_add_rmsnorm_vs_eager_is_l1() {
    let (text, j, md) = compare(
        "fixtures/c_cve_addnorm_eager",
        "fixtures/c_cve_addnorm_inductor",
    );
    assert!(text.contains("PROVEN L1"), "{text}");
    assert_eq!(
        j["reference"]["side"]["kernels"].as_array().unwrap().len(),
        7,
        "{j}"
    );
    for s in ["reference", "candidate"] {
        assert_eq!(j[s]["side"]["check"]["certified"], true, "{j}");
    }
    let e = &j["equivalence"];
    assert_eq!(e["proven"], true, "{j}");
    assert_eq!(e["per_output_bound"]["with_bound"], 1024, "{j}");
    assert_eq!(j["gpu"]["outputs"], 1024, "{j}");
    assert_eq!(j["blockers"], serde_json::json!([]), "{j}");
    assert!(md.contains("Nothing: every step is proven."), "{md}");
}

/// Inductor's fused `silu(a) * b` (one Triton kernel) against eager's silu kernel + mul kernel:
/// every output is rn(f(a) * b) on both sides, so elementwise mode sweeps f over all 2^32 f32
/// values (the committed sweep is reused) and bounds |C - R| for every a and b.
#[test]
#[ignore = "heavy: ptxv compare subprocess, three modes; 247 s"]
fn compare_inductor_silu_mul_vs_eager_is_elementwise_product() {
    let (text, j, _) = compare(
        "fixtures/c_cve_silumul_eager",
        "fixtures/c_cve_silumul_inductor",
    );
    assert!(text.contains("every output is rn(f(x) * y)"), "{text}");
    let e = &j["equivalence"]["elementwise"];
    assert_eq!(e["product"], true, "{e}");
    assert_eq!(e["inputs"], 1u64 << 32, "{e}");
    assert_eq!(e["true_fn"], "silu", "{e}");
    assert!(e["max_abs"].as_f64().unwrap() < 2e-6, "{e}");
    assert_eq!(j["equivalence"]["proven"], true, "{j}");
    assert_eq!(j["blockers"], serde_json::json!([]), "{j}");
    assert_eq!(j["gpu"]["outputs"], 2048, "{j}");
}

/// bf16 residual add (docs/compile-vs-eager.md): eager rounds x + a to bf16 before adding b,
/// Inductor adds in f32 and rounds once. Both outputs are bf16 roundings of the same exact
/// polynomial (eager's inner rounding bounded as error); the per-input bound is the distance
/// between the bf16 roundings of the two kernels' intervals, and the GPU's difference reaches
/// it (one bf16 ulp).
#[test]
fn compare_bf16_residual_add_is_bounded() {
    let (text, j, _) = compare(
        "fixtures/c_cve_add_bf16_eager",
        "fixtures/c_cve_add_bf16_inductor",
    );
    assert!(
        text.contains("PROVEN L2 for every input: both outputs are bf16 roundings"),
        "{text}"
    );
    assert!(
        text.contains("the reference rounds to bf16 before the output"),
        "{text}"
    );
    let e = &j["equivalence"];
    assert_eq!(e["proven"], true, "{j}");
    assert_eq!(e["bound_form"], serde_json::json!(["round"]), "{j}");
    let st = e["round16"]["captured_steps"].as_array().unwrap();
    assert_eq!(
        st.iter().map(|x| x.as_u64().unwrap()).sum::<u64>(),
        8192,
        "{j}"
    );
    let f = j["gpu"]["max_fraction_of_bound"].as_f64().unwrap();
    assert!(f > 0.5 && f <= 1.0, "{j}");
    assert_eq!(j["blockers"], serde_json::json!([]), "{j}");
}

/// bf16 transpose copy: no floating-point operation; the outputs are the input elements
/// moved byte for byte, L0 for every input.
#[test]
fn compare_bf16_clone_is_l0() {
    let (text, j, _) = compare(
        "fixtures/c_cve_clone_bf16_eager",
        "fixtures/c_cve_clone_bf16_inductor",
    );
    assert!(
        text.contains("PROVEN bitwise identical (L0) for every input"),
        "{text}"
    );
    assert_eq!(j["gpu"]["bitwise_equal"], 8192, "{j}");
    assert_eq!(j["blockers"], serde_json::json!([]), "{j}");
}

/// bf16 silu * mul: eager rounds silu(x) to bf16 before the mul kernel, Inductor fuses in f32.
/// Every output is bf16(rn(f(x) * y)); both kernels are evaluated at every (x, y) pair of bf16
/// values: at most 1 bf16 ulp apart where f and the outputs are normal.
#[test]
#[ignore = "heavy: 39 s"]
fn compare_bf16_silu_mul_is_exhaustive() {
    let (text, j, _) = compare(
        "fixtures/c_cve_silumul_bf16_eager",
        "fixtures/c_cve_silumul_bf16_inductor",
    );
    assert!(
        text.contains("PROVEN elementwise: every output is bf16(rn(f(x) * y))"),
        "{text}"
    );
    let p = &j["equivalence"]["elementwise"]["product16"];
    assert_eq!(p["pairs"], 2u64 << 32, "{j}");
    assert_eq!(p["max_ulps_normal"], 1, "{j}");
    assert_eq!(j["equivalence"]["elementwise"]["true_fn"], "silu", "{j}");
}
