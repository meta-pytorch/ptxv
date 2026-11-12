/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! errdist's guards against reporting from bad data (docs/errdist.md §1): NaN / Inf outputs
//! are out-of-domain mass, never dropped silently; a numerics DAG that disagrees with the
//! simulator is re-extracted in segment mode, and refused if that disagrees too; a run whose
//! control, addresses or guards depend on input data where its DAG does not model them is
//! refused.

use std::path::PathBuf;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn run(name: &str, path: &str, spec: &str) -> (String, serde_json::Value) {
    run_args(name, path, spec, &["--mc", "200"])
}

/// `run` with other arguments than `--mc 200` (none: the default, growing Tier 3).
fn run_args(name: &str, path: &str, spec: &str, extra: &[&str]) -> (String, serde_json::Value) {
    let (text, j, ok) = run_any(name, path, spec, extra);
    assert!(ok, "{text}");
    (text, j)
}

/// `run_args` without the success check: the output, the JSON report (written before a
/// refusal) and whether errdist succeeded.
fn run_any(
    name: &str,
    path: &str,
    spec: &str,
    extra: &[&str],
) -> (String, serde_json::Value, bool) {
    let js = std::env::temp_dir().join(format!(
        "ptxv-test-guard-{name}-{}.json",
        std::process::id()
    ));
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_ptxv"))
        .args(["errdist", path, "--inputs", spec])
        .args(extra)
        .args(["--json", js.to_str().unwrap()])
        .current_dir(root())
        .env("CUDA_VISIBLE_DEVICES", "")
        .output()
        .unwrap();
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let j = serde_json::from_str(
        &std::fs::read_to_string(&js).unwrap_or_else(|e| panic!("{e}: {text}")),
    )
    .unwrap();
    let _ = std::fs::remove_file(&js);
    (text, j, out.status.success())
}

/// Run errdist where it must refuse: its output.
fn run_refused(path: &str, spec: &str) -> String {
    run_refused_args(path, spec, &["--mc", "200"])
}

/// `run_refused` with other arguments than `--mc 200`.
fn run_refused_args(path: &str, spec: &str, extra: &[&str]) -> String {
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_ptxv"))
        .args(["errdist", path, "--inputs", spec])
        .args(extra)
        .current_dir(root())
        .env("CUDA_VISIBLE_DEVICES", "")
        .output()
        .unwrap();
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(!out.status.success(), "{text}");
    text
}

/// Row sums of N(3e37, 1e36) f32 inputs overflow: every output is Inf. Tier 3 compares
/// nothing, with the default input count does not grow, and is refused: no headline
/// (MISS 64; until 2026-11-11 Tier 3 headlined P(differ) 0 over [0, 1]).
#[test]
fn nonfinite_outputs_are_out_of_domain() {
    let spec =
        std::env::temp_dir().join(format!("ptxv-test-guard-huge-{}.json", std::process::id()));
    std::fs::write(
        &spec,
        r#"{"version": 1, "seed": 1, "name": "huge", "default": {"dist": {"kind": "normal", "mean": 3e37, "std": 1e36}}}"#,
    )
    .unwrap();
    let (text, j, ok) = run_any(
        "inf",
        "fixtures/r4_sum_torch",
        spec.to_str().unwrap(),
        &["--mc", "200"],
    );
    // With the default input count Tier 3 does not grow at P 0 where it compared nothing
    // (MISS 120's growth): more inputs cannot narrow [0, 1].
    let (_, jd, okd) = run_any("infd", "fixtures/r4_sum_torch", spec.to_str().unwrap(), &[]);
    let _ = std::fs::remove_file(&spec);
    assert!(!ok && !okd, "{text}");
    assert!(
        text.contains("Tier 3 compared no output") && text.contains("refusing to report"),
        "{text}"
    );
    for r in [&j, &jd] {
        assert!(r["best"].is_null(), "{}", r["best"]);
    }
    let t3 = jd["estimates"]
        .as_array()
        .unwrap()
        .iter()
        .find(|e| e["tier"] == 3)
        .unwrap();
    assert_eq!(t3["n"], 300, "{t3}");
    assert_eq!(t3["applicable"], false, "{t3}");
    let ood = &j["out_of_domain"];
    assert!(ood["mass"].as_f64().unwrap() > 0.99, "{ood}");
    assert!(
        ood["by_cause"]["non-finite output (NaN / Inf)"]
            .as_f64()
            .unwrap()
            > 0.99,
        "{ood}"
    );
    assert!(text.contains("TIER 3 COMPARED NO OUTPUTS"), "{text}");
    // The simulator's outputs are Inf too: the validation says so and does not agree.
    let v = j["validation"]
        .as_array()
        .unwrap()
        .iter()
        .find(|v| v["source"] == "sim")
        .unwrap();
    assert_eq!(v["agree"], false, "{v}");
}

/// Kahan compensation stays input-dependent even when captured as zero. The extracted DAG
/// must reproduce sampled simulator outputs without needing to repair a lost dependency.
#[test]
fn kahan_compensation_keeps_input_history() {
    let (text, j) = run(
        "kahan",
        "fixtures/c_sum_kahan",
        "fixtures/errdist/normal.json",
    );
    assert!(!text.contains("retrying in segment mode"), "{text}");
    for v in j["validation"].as_array().unwrap() {
        assert_eq!(v["dag_mismatch"], 0, "{v}");
    }
}

/// MISS 53: an EXIT under a data-dependent guard whose predicate has no slice
/// (fixtures/errdist/miss53/exit, `bfind(bits(x)) == 30` on half of the threads). On the
/// captured input those threads exit, so their outputs had neither a node nor taint: errdist
/// skipped them, reported over the other half, and its validation compared only those. Every
/// input that stores there was evaluated as if it exited. The run is refused.
#[test]
fn data_dependent_exit_is_refused() {
    let text = run_refused(
        "fixtures/errdist/miss53/exit",
        "fixtures/errdist/uniform01.json",
    );
    assert!(
        text.contains("guard of") && text.contains("EXIT") && text.contains("refusing to report"),
        "{text}"
    );
}

/// MISS 53: a store address chosen by a select on input data (fixtures/errdist/miss53/scatter):
/// the DAG holds the captured address's values only. Refused.
#[test]
fn data_dependent_address_is_refused() {
    let text = run_refused(
        "fixtures/errdist/miss53/scatter",
        "fixtures/errdist/uniform01.json",
    );
    assert!(
        text.contains("an address / control operand") && text.contains("refusing to report"),
        "{text}"
    );
}

/// A guarded store whose guard holds for every thread of the captured run
/// (fixtures/errdist/miss53/sel): the default extraction tracks every output and keeps the guard
/// as a data-dependent precondition; segment mode models the store as a selection
/// (N-ERRDIST-SEL), so errdist reports, and its DAG agrees with the simulator on sampled inputs
/// where the guard fails as well.
#[test]
fn guarded_store_is_a_selection_not_a_refusal() {
    let (text, j) = run(
        "m53sel",
        "fixtures/errdist/miss53/sel",
        "fixtures/errdist/normal.json",
    );
    assert!(text.contains("N-ERRDIST-SEL"), "{text}");
    assert!(!text.contains("precondition: data-dependent"), "{text}");
    assert!(j["best"].is_number(), "{}", j["best"]);
    let v = j["validation"]
        .as_array()
        .unwrap()
        .iter()
        .find(|v| v["source"] == "sim")
        .unwrap();
    assert_eq!(v["dag_mismatch"], 0, "{v}");
}

/// MISS 65: control decided by other lanes through a warp ballot (fixtures/errdist/m65vote).
/// Each lane's ballot result carried only its own predicate's taint, so errdist reported the
/// captured path's DAG for every input:
/// - ballot: lanes 0..15 ballot x >= 3, lanes 16..31 branch on the ballot (rn(3 t) or t);
/// - lane0 (the independent review's rE_vote): lane 0's x[0] >= 3 guards the other lanes'
///   stores; the kernel's P(bits differ) is 0.2550, errdist reported 0.2415 [0.2410, 0.2419]
///   from the path where no store runs.
///
/// Both are refused.
#[test]
fn ballot_decided_control_is_refused() {
    for (case, spec) in [("ballot", "ballot.json"), ("lane0", "x10_s2.json")] {
        let text = run_refused(
            &format!("fixtures/errdist/m65vote/{case}"),
            &format!("fixtures/errdist/m65vote/{spec}"),
        );
        assert!(
            text.contains("depends on input data") && text.contains("refusing to report"),
            "{case}: {text}"
        );
    }
}

/// MISS 53 in a pair (fixtures/errdist/miss53/pair.toml): the reference (sel) is clean in
/// segment mode, the candidate (exit) is not; the pair is refused on the candidate.
#[test]
fn data_dependent_candidate_is_refused() {
    let text = run_refused(
        "fixtures/errdist/miss53/pair.toml",
        "fixtures/errdist/normal.json",
    );
    assert!(
        text.contains("m53_exit") && text.contains("EXIT") && text.contains("refusing to report"),
        "{text}"
    );
}

/// MISS 56: a store under a data-dependent guard, false on the captured input, whose address
/// depends on input data (fixtures/errdist/miss56/selscat: thread 0 stores RN(3 t_0) to
/// y[(bits(x[0]) >> 19) & 15] where x[0] >= 3). Segment mode modeled it as a selection at the
/// captured address y[0] (N-ERRDIST-SEL), the address check covering only the threads whose
/// guard held: with x[0] in [3, 4) on 1% of the inputs, two outputs were wrong on each such
/// input and errdist reported. The address is a data-dependent precondition: refused.
#[test]
fn selection_at_an_input_address_is_refused() {
    let text = run_refused(
        "fixtures/errdist/miss56/selscat",
        "fixtures/errdist/miss56/x1.json",
    );
    assert!(
        text.contains("an address / control operand")
            && text.contains("STG")
            && text.contains("refusing to report"),
        "{text}"
    );
}

/// MISS 56, the other side: a store under an input-independent guard
/// (fixtures/errdist/miss56/tid, `@!P0 STG [y + 4 k_i]` with P0 = tid != 0) whose address
/// depends on input data for the threads whose guard is false. They store on no input, and
/// thread 0's address is y[0] on every input: errdist reports, and its DAG agrees with the
/// simulator.
#[test]
fn input_independent_guard_keeps_its_addresses() {
    let (text, j) = run(
        "m56tid",
        "fixtures/errdist/miss56/tid",
        "fixtures/errdist/miss56/x1.json",
    );
    assert!(!text.contains("precondition: data-dependent"), "{text}");
    assert!(j["best"].is_number(), "{}", j["best"]);
    let v = j["validation"]
        .as_array()
        .unwrap()
        .iter()
        .find(|v| v["source"] == "sim")
        .unwrap();
    assert_eq!(v["dag_mismatch"], 0, "{v}");
}

/// MISS 57: the slice of `a' = x[0] >= -3 ? RN(3 a) : a` (fixtures/errdist/miss57/guard, a
/// predicated FMUL) is identified as 3 a on probes that all take the guard (the captured x = 0,
/// perturbed by N(0, 1)). With x uniform on [-5, -4] every input takes the other outcome:
/// errdist used 3 a as the exact reference there and reported P(bits differ) = 1 (the kernel's
/// is about 0.29). The inputs whose slice decisions no probe took have no exact reference:
/// refused.
#[test]
fn slice_off_its_probed_guard_outcome_is_refused() {
    let text = run_refused(
        "fixtures/errdist/miss57/guard",
        "fixtures/errdist/miss57/xneg.json",
    );
    assert!(
        text.contains("200 of 200 Tier 3 inputs")
            && text.contains("no other output has an exact reference")
            && text.contains("identification's probes never took")
            && text.contains("refusing to report"),
        "{text}"
    );
}

/// MISS 57 with a select (fixtures/errdist/miss57/sel: an `FSEL` on x[0] >= -3) and under a
/// maximum (fixtures/errdist/miss57/max: y = max(a', v), whose exact pick would pass over an
/// unknown child): refused on the same inputs.
#[test]
fn select_or_maximum_off_its_probed_outcome_is_refused() {
    for case in ["fixtures/errdist/miss57/sel", "fixtures/errdist/miss57/max"] {
        let text = run_refused(case, "fixtures/errdist/miss57/xneg.json");
        assert!(
            text.contains("identification's probes never took") && text.contains("refusing"),
            "{case}: {text}"
        );
    }
}

/// MISS 57, the other side: on N(0, 1) inputs (x[0] < -3 on about 0.13% of them) errdist
/// reports, names the slice's decision and the patterns its probes took, and its DAG agrees
/// with the simulator.
#[test]
fn slice_on_its_probed_guard_outcome_reports() {
    let (text, j) = run(
        "m57guard",
        "fixtures/errdist/miss57/guard",
        "fixtures/errdist/normal.json",
    );
    assert!(
        text.contains("1 data-dependent decision(s)")
            && text.contains("outcome patterns probed: 1 of 2"),
        "{text}"
    );
    let b = j["best"].as_u64().expect("a headline") as usize;
    let p = j["estimates"][b]["p_differ"][0].as_f64().unwrap();
    assert!((0.3..0.4).contains(&p), "{p}");
    let v = j["validation"]
        .as_array()
        .unwrap()
        .iter()
        .find(|v| v["source"] == "sim")
        .unwrap();
    assert_eq!(v["dag_mismatch"], 0, "{v}");
}

/// MISS 57, the round-4 review's repro (fixtures/errdist/miss57/ffma): a guarded FFMA,
/// `@!P0 FFMA a, a, v, w` with P0 = (x[0] >= -3), inside the slice that single-path auto mode
/// identifies as a on the captured side. With x[0] in [-5, -4) on 30% of the inputs (xn30)
/// errdist reported 0.483 [0.474, 0.491] where the kernel's is 0.276 (exact rationals); now
/// refused.
#[test]
fn guarded_ffma_off_its_probed_outcome_is_refused() {
    let text = run_refused(
        "fixtures/errdist/miss57/ffma",
        "fixtures/errdist/miss57/xn30.json",
    );
    assert!(
        text.contains("identification's probes never took") && text.contains("refusing"),
        "{text}"
    );
}

/// MISS 57 in an elementwise kernel (fixtures/errdist/miss57/elem: the guard on the thread's
/// own x[i]; x[i] in [-5, -4] w.p. 0.002): the elementwise law leaves the mass outside the
/// probed outcome out, measured (2e-3), and widens its P(differ) interval by it; Tier 3 leaves
/// those inputs out and widens by their bound.
#[test]
fn elementwise_law_leaves_out_the_unknown_mass() {
    let (text, j) = run_args(
        "m57elem",
        "fixtures/errdist/miss57/elem",
        "fixtures/errdist/miss57/xrare.json",
        &["--mc", "1000"],
    );
    let e = j["estimates"]
        .as_array()
        .unwrap()
        .iter()
        .find(|e| e["method"].as_str().unwrap().starts_with("elementwise"))
        .unwrap_or_else(|| panic!("no elementwise estimate: {text}"));
    let line = e["applicability"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|a| a.as_str())
        .find(|a| a.contains("MISS 57"))
        .unwrap_or_else(|| panic!("{e}"));
    assert!(
        line.starts_with("2.000e-3 of the law") && line.ends_with(" ok"),
        "{line}"
    );
    let p = &e["p_differ"];
    let (lo, hi) = (p[1].as_f64().unwrap(), p[2].as_f64().unwrap());
    assert!(hi - lo >= 2.0 * 2e-3, "{p}");
    assert!(
        text.contains("took another pattern, where the exact value is unknown"),
        "{text}"
    );
}

/// MISS 57, the analytic tiers: they use the identified function on every input, so they
/// take the bound of the share outside the probed outcomes even when no Tier 3 input fell
/// there (x N(0, 0.5^2): x[0] < -3 at 6 sigma), as they take the branch-flip bound.
#[test]
fn analytic_tiers_take_the_unknown_bound_at_zero_inputs_off() {
    let (text, j) = run_args(
        "m57max0",
        "fixtures/errdist/miss57/max",
        "fixtures/errdist/miss57/xnarrow.json",
        &[],
    );
    assert!(text.contains(": 0 of "), "{text}");
    let t2: Vec<&serde_json::Value> = j["estimates"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|e| e["tier"] == 2)
        .collect();
    assert!(!t2.is_empty(), "{text}");
    for e in t2 {
        assert!(
            e["applicability"].as_array().unwrap().iter().any(|a| a
                .as_str()
                .unwrap()
                .starts_with("exact value unknown on a share")),
            "{e}"
        );
    }
}

/// MISS 58 follow-up (MISS 42's rule for Tier 3): a validation run that contradicts Tier 3's
/// P(differ) refuses Tier 3, and with nothing else standing the run has no headline and fails.
/// fixtures/errdist/miss58/rare: y[i] = RN(RN(x[i] s[0]) + v[i]), exact where s[0] = 1. Under
/// dense.json (x = 0.1, v = -0.3, s[0] = 1 w.p. 0.2, else U[2.999, 3.001]) RN(x s[0]) - 0.3
/// cancels where s[0] is not 1 and all 64 outputs differ together: P(differ) about 0.8. Tier 3's
/// 3 inputs (seed 129) all draw s[0] = 1, P(differ) 0, and since MISS 120 its interval counts
/// inputs, [0, 0.71]; the simulator's 1000 inputs put P(differ) at least 0.77 (99.9%), with the
/// DAG matching the simulator bit for bit. The analytic tiers (0.67 to 0.80) fail their
/// cross-check against Tier 3's 0. Before the follow-up: DISAGREE printed and the headline
/// P(bits differ) = 0, rc 0. Under rare.json (5 Tier 3 inputs, all with s[0] = 1; P about 0.047)
/// Tier 3's [0, 0.52] holds the run's P since MISS 120 (it said [0, 1.1e-2] before) and the
/// elementwise law, 4.67e-2, is the headline.
#[test]
fn tier3_contradicted_by_a_validation_is_no_headline() {
    let js = std::env::temp_dir().join(format!("ptxv-test-guard-t3r-{}.json", std::process::id()));
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_ptxv"))
        .args([
            "errdist",
            "fixtures/errdist/miss58/rare",
            "--inputs",
            "fixtures/errdist/miss58/dense.json",
            "--mc",
            "3",
            "--sim-trials",
            "1000",
            "--json",
            js.to_str().unwrap(),
        ])
        .current_dir(root())
        .env("CUDA_VISIBLE_DEVICES", "")
        .output()
        .unwrap();
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(!out.status.success(), "{text}");
    assert!(
        text.contains("summary: no estimate passed its checks (tier 3 refused: sim validation (1000 inputs): Tier 3's interval [0.000e0, 7.076e-1] against the observed inputs' exact bounds")
            && text.contains("refusing to report from it"),
        "{text}"
    );
    let j: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&js).unwrap()).unwrap();
    let _ = std::fs::remove_file(&js);
    assert!(j["best"].is_null(), "{}", j["best"]);
    let t3 = j["estimates"]
        .as_array()
        .unwrap()
        .iter()
        .find(|e| e["tier"] == 3)
        .unwrap();
    assert_eq!(t3["applicable"], false, "{t3}");
    let (text, j) = run_args(
        "t3rare",
        "fixtures/errdist/miss58/rare",
        "fixtures/errdist/miss58/rare.json",
        &["--mc", "5", "--sim-trials", "1000"],
    );
    let t3 = j["estimates"]
        .as_array()
        .unwrap()
        .iter()
        .find(|e| e["tier"] == 3)
        .unwrap();
    assert_ne!(t3["applicable"], false, "{text}");
    assert!(t3["p_differ"][2].as_f64().unwrap() > 0.5, "{text}");
    let b = j["best"].as_u64().expect("a headline") as usize;
    let p = j["estimates"][b]["p_differ"][0].as_f64().unwrap();
    assert!(p > 0.04 && p < 0.055, "{text}");
}

/// The store word of a selection (MISS 56 class; fixtures/errdist/storeword): thread 0 clears
/// a shared flag under x[0] >= 3, false on the captured input, and other threads read the flag
/// after a barrier as an address (addr) or a guard (guard, all). The thread that did not store
/// ran no taint write, so the word read as input-independent: with x[0] in [3, 4) on 10% of
/// the inputs errdist reported 2.415e-1 for `all` where the kernel's P(bits differ) is 0.2553
/// [0.2531, 0.2575] (exact rationals), and its DAG disagreed with the simulator on 4-8 outputs
/// over 400 inputs. The word now depends on input data on every input: the address and the
/// guards that read it are data-dependent preconditions, refused.
#[test]
fn selection_store_word_read_by_another_thread_is_refused() {
    for case in ["addr", "guard", "all"] {
        for spec in ["x10", "x1"] {
            let text = run_refused(
                &format!("fixtures/errdist/storeword/{case}"),
                &format!("fixtures/errdist/storeword/{spec}.json"),
            );
            assert!(
                text.contains("depends on input data") && text.contains("refusing to report"),
                "{case} {spec}: {text}"
            );
        }
    }
}

/// fp32 atomic adds flush subnormal operands and results (`red/atom.add.f32`; MISS 58 class;
/// fixtures/errdist/atomftz/sum: y[j] = x[j] + x[j + 32] from 0, x scaled by 2^-130 on 1% of the
/// inputs). The DAG added without the flush: P(bits differ) 0 [0, 3.8e-4] with rc 0, and 400
/// simulator inputs refused on 96 mismatching outputs. Now the DAG matches the simulator, and
/// Tier 3's interval holds the share of flushed inputs (GB300 differs from the correctly
/// rounded sum on 128 of 12800 outputs, 1.00e-2).
#[test]
fn fp32_atomic_adds_flush_subnormal_operands() {
    let (text, j) = run_args(
        "atomftz",
        "fixtures/errdist/atomftz/sum",
        "fixtures/errdist/atomftz/sub1.json",
        &["--sim-trials", "400"],
    );
    let v = j["validation"]
        .as_array()
        .unwrap()
        .iter()
        .find(|v| v["source"] == "sim" && v["against"] == "tier 3")
        .unwrap_or_else(|| panic!("{text}"));
    assert_eq!(v["dag_mismatch"], 0, "{v}");
    let b = j["best"].as_u64().expect("a headline") as usize;
    let p = &j["estimates"][b]["p_differ"];
    let (lo, hi) = (p[1].as_f64().unwrap(), p[2].as_f64().unwrap());
    assert!(lo <= 1e-2 && 1e-2 <= hi, "{p}: {text}");
}

/// The captured-output check at atomic outputs (MISS 58 class): until 2026-11-11 a kernel with
/// any atomic add skipped it. fixtures/errdist/atomftz/capgood carries GB300's output at the
/// captured input (`ptxv probe`), which matches the DAG: reported. capoff has y[5] one ulp off,
/// which no order of two adds from zero gives: refused.
#[test]
fn captured_atomic_outputs_outside_any_order_are_refused() {
    let text = run_refused(
        "fixtures/errdist/atomftz/capoff",
        "fixtures/errdist/normal.json",
    );
    assert!(
        text.contains("1 of 32 outputs of the numerics DAG differ from the captured GPU outputs"),
        "{text}"
    );
    let (text, j) = run(
        "capgood",
        "fixtures/errdist/atomftz/capgood",
        "fixtures/errdist/normal.json",
    );
    assert!(j["best"].is_number(), "{text}");
}

/// MISS 84 follow-up, point (b): another order of the adds excuses a captured GPU output only
/// where the run's happens-before leaves some two adds unordered. fixtures/errdist/atomord:
/// three adds per output from three warps, y[5] of the captured output one ulp off the DAG,
/// inside the reordering bound. Unordered (no barrier), another order explains it; with a
/// barrier before each warp's add the order is the DAG's on every execution, and the run is
/// refused. Before, both were excused. The exact captured output passes under the barriers.
#[test]
fn barrier_ordered_atomic_adds_have_one_order() {
    let (text, j) = run(
        "atomord-free",
        "fixtures/errdist/atomord/free",
        "fixtures/errdist/normal.json",
    );
    assert!(j["best"].is_number(), "{text}");
    let text = run_refused(
        "fixtures/errdist/atomord/bar",
        "fixtures/errdist/normal.json",
    );
    assert!(
        text.contains("1 of 32 outputs of the numerics DAG differ from the captured GPU outputs"),
        "{text}"
    );
    let (text, j) = run(
        "atomord-barexact",
        "fixtures/errdist/atomord/barexact",
        "fixtures/errdist/normal.json",
    );
    assert!(j["best"].is_number(), "{text}");
}

/// Value clamps inside an instruction slice (MISS 57 class; fixtures/errdist/clamp): a
/// saturating multiply (`FMUL.SAT`) and a minimum against a constant (`FMNMX.FTZ`, no `Max`
/// node) are piecewise without a predicate operand. Every probe took the unclamped piece and
/// the slice was fitted as x^2 / 32: with x in [6, 8] w.p. 0.3 errdist reported 0.331 where the
/// kernel's P(bits differ) is 0.0312 (exact rationals). The clamp's piece is now a decision:
/// refused. On N(0, 1) inputs (no clamp) the decisions are named and errdist reports.
#[test]
fn clamps_inside_a_slice_are_decisions() {
    for case in ["sat", "min"] {
        let text = run_refused_args(
            &format!("fixtures/errdist/clamp/{case}"),
            "fixtures/errdist/clamp/big30.json",
            &[],
        );
        assert!(
            text.contains("identification's probes never took") && text.contains("refusing"),
            "{case}: {text}"
        );
    }
    let (text, j) = run(
        "clampn",
        "fixtures/errdist/clamp/sat",
        "fixtures/errdist/normal.json",
    );
    assert!(
        text.contains("data-dependent decision(s)") && text.contains("outcome patterns probed"),
        "{text}"
    );
    assert!(j["best"].is_number(), "{text}");
}

/// MISS 57 at 0 Tier 3 inputs outside the probed outcomes (fixtures/errdist/k0: y = 2x where
/// x[0] >= -3, else RN(RN(3x) w); x[0] in [-5, -4) w.p. 0.002; the kernel's P(bits differ)
/// 5.2e-4). On seed 2 no Tier 3 input fell outside, and Tier 3 reported 0 [0, 1.9e-4]; it now
/// widens by the share's 95% bound as the analytic tiers do. Each seed refuses (inputs outside)
/// or reports an interval that holds 5.2e-4. At 300 inputs (the default count, fixed: at P 0 the
/// default run grows to its budget, where every seed takes inputs outside and refuses).
#[test]
fn tier3_takes_the_unknown_bound_at_zero_inputs_off() {
    let mut reported = 0;
    for seed in 1..=6 {
        let spec = format!("fixtures/errdist/k0/s{seed}.json");
        let js = std::env::temp_dir().join(format!(
            "ptxv-test-guard-k0-{seed}-{}.json",
            std::process::id()
        ));
        let out = std::process::Command::new(env!("CARGO_BIN_EXE_ptxv"))
            .args(["errdist", "fixtures/errdist/k0/guard", "--inputs", &spec])
            .args(["--mc", "300", "--json", js.to_str().unwrap()])
            .current_dir(root())
            .env("CUDA_VISIBLE_DEVICES", "")
            .output()
            .unwrap();
        let text = format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        if !out.status.success() {
            assert!(
                text.contains("identification's probes never took"),
                "{text}"
            );
            continue;
        }
        let j: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&js).unwrap()).unwrap();
        let _ = std::fs::remove_file(&js);
        let b = j["best"].as_u64().expect("a headline") as usize;
        let hi = j["estimates"][b]["p_differ"][2].as_f64().unwrap();
        assert!(hi >= 5.2e-4, "seed {seed}: {hi}: {text}");
        reported += 1;
    }
    assert!(reported > 0, "every seed refused");
}

/// `--pre-round` keeps the refusals where it changes nothing (no output is an MMA chain's):
/// fixtures/errdist/miss58/rare, whose Tier 3 (P(differ) 0) the simulator's 1000 inputs
/// contradict, was reported with --pre-round (P(bits differ) 0, rc 0), every refusal skipped.
/// Now refused, as without it. Since MISS 120 the contradiction needs dense.json (3 Tier 3
/// inputs, [0, 0.71], against a run at about 0.8; see
/// `tier3_contradicted_by_a_validation_is_no_headline`).
#[test]
fn pre_round_keeps_the_refusals_where_it_changes_nothing() {
    let text = run_refused_args(
        "fixtures/errdist/miss58/rare",
        "fixtures/errdist/miss58/dense.json",
        &["--mc", "3", "--sim-trials", "1000", "--pre-round"],
    );
    assert!(
        text.contains("no estimate passed its checks") && text.contains("refusing to report"),
        "{text}"
    );
}

/// MISS 99 (integer steps inside a slice, the MISS 85 class; fixtures/errdist/trunc): an F2I's
/// integer, an exponent field and a mantissa with a fixed exponent cut from a float were no
/// decisions. Every probe took one step (trunc(x / 1024) = 0, one binade of x * 2^-20 + 1.5),
/// and the slice was fitted as that step's function: with 30% of the inputs on another step
/// errdist reported P(bits differ) 0.300, 0.300 and 0.545 (Tier 1, rc 0), where the step
/// kernel's is 0 (exact; GB300 0 of 12800 outputs). Each is now refused; on N(0, 1) inputs
/// (the probes' step) the step kernel still reports P 0.
#[test]
fn integer_steps_inside_a_slice_are_decisions() {
    for (case, spec) in [("step", "trunc30"), ("exp", "big20"), ("mant", "big20")] {
        let text = run_refused(
            &format!("fixtures/errdist/trunc/{case}"),
            &format!("fixtures/errdist/trunc/{spec}.json"),
        );
        assert!(
            text.contains("identification's probes never took") && text.contains("refusing"),
            "{case}: {text}"
        );
    }
    let (text, j) = run(
        "truncn",
        "fixtures/errdist/trunc/step",
        "fixtures/errdist/normal.json",
    );
    let b = j["best"].as_u64().expect("a headline") as usize;
    assert_eq!(
        j["estimates"][b]["p_differ"][0].as_f64(),
        Some(0.0),
        "{text}"
    );
}

/// MISS 99 follow-up (integer steps made by FP rounding; fixtures/errdist/trunc/fpstep.ptx, the
/// review's probes): RM(t + 1.5 * 2^23) - 1.5 * 2^23 (FADD.RM), the same in RN, and FFMA.RM (an
/// exp range reduction's form) give floor(t) or round(t), t = x * 2^-10. Every probe took step
/// 0 (x ~ N(0, 1)) and the slice was fitted as RN(0 + v): with 30% of the inputs on step 2 (or
/// 3) errdist reported P(bits differ) 0.300 (Tier 1, rc 0) where the kernels' is 0 (exact;
/// GB300 0 of 12800 outputs). The rounding took one value on the probes while its operand took
/// many, so its bits are decisions: each is refused, and on N(0, 1) inputs (the probes' step)
/// each still reports Tier 1 P 0.
#[test]
fn fp_rounding_steps_inside_a_slice_are_decisions() {
    for case in ["floor", "round", "ffrm"] {
        let text = run_refused(
            &format!("fixtures/errdist/trunc/{case}"),
            "fixtures/errdist/trunc/trunc30.json",
        );
        assert!(
            text.contains("identification's probes never took") && text.contains("refusing"),
            "{case}: {text}"
        );
        let (text, j) = run(
            &format!("fpstep-{case}"),
            &format!("fixtures/errdist/trunc/{case}"),
            "fixtures/errdist/normal.json",
        );
        assert!(
            text.contains("rounding(s) acting as a step"),
            "{case}: {text}"
        );
        let b = j["best"].as_u64().expect("a headline") as usize;
        assert_eq!(
            j["estimates"][b]["tier"].as_u64(),
            Some(1),
            "{case}: {text}"
        );
        assert_eq!(
            j["estimates"][b]["p_differ"][0].as_f64(),
            Some(0.0),
            "{case}: {text}"
        );
    }
}

/// MISS 87 follow-up: `--pre-round` marked every MMA chain output as taken before its rounding,
/// an f32 D that is its chain's last step included, and then skipped the worst-case check and
/// Tier 3's validation refusal for the whole run. fixtures/e_tc_handback (f32 D): with and
/// without --pre-round the estimates are the same and both worst-case checks (Tier 3's DAG
/// evaluation and the simulator's) stand.
#[test]
#[ignore = "heavy: two errdist runs of a tcgen05 GEMM, about 45 s"]
fn pre_round_keeps_the_checks_of_an_output_it_leaves_alone() {
    let mut texts = vec![];
    for (name, extra) in [("tc-off", &[][..]), ("tc-on", &["--pre-round"][..])] {
        let mut args = vec!["--mc", "5", "--sim-trials", "1", "--threads", "8"];
        args.extend_from_slice(extra);
        let (text, j) = run_args(
            name,
            "fixtures/e_tc_handback",
            "fixtures/errdist/normal.json",
            &args,
        );
        let checks = j["worst_case"]["checks"]
            .as_array()
            .cloned()
            .unwrap_or_default();
        let srcs: Vec<&str> = checks.iter().filter_map(|c| c["source"].as_str()).collect();
        assert!(
            srcs.iter().any(|s| s.starts_with("tier 3")) && srcs.contains(&"sim"),
            "{name}: {srcs:?}: {text}"
        );
        assert!(
            checks
                .iter()
                .all(|c| c["checked"].as_u64().unwrap_or(0) > 0),
            "{name}: {text}"
        );
        let est: Vec<String> = j["estimates"]
            .as_array()
            .unwrap()
            .iter()
            .map(|e| format!("{} {} {}", e["tier"], e["p_differ"], e["applicable"]))
            .collect();
        texts.push(est);
    }
    assert_eq!(texts[0], texts[1]);
}

/// A pair's elementwise law counts P(differ) where the exact value is unknown (MISS 85
/// follow-up): it compares C with R bit for bit and needs no exact value, as Tier 3 counts it.
/// With the value pieces 7.5e-8 of c_equiv_elementwise/gelu_bf16's codes on uniform inputs take
/// a piece the identification's probes never took; the law left them out and widened by them,
/// which refused its exact Tier 0 (P 0), and the headline fell to a Tier 2 estimate,
/// 4.4e-6 [4.39e-6, 4.40e-6], that the enumeration contradicts. Tier 0 stands again: P 0.
#[test]
fn pair_elementwise_law_counts_codes_with_an_unknown_exact_value() {
    let (text, j) = run_args(
        "gelupair",
        "fixtures/c_equiv_elementwise/gelu_bf16.toml",
        "fixtures/errdist/uniform.json",
        &["--mc", "200", "--candidate", "0"],
    );
    let b = j["best"].as_u64().expect("a headline") as usize;
    let e = &j["estimates"][b];
    assert_eq!(e["tier"], 0, "{text}");
    assert_eq!(e["p_differ"][2].as_f64(), Some(0.0), "{text}");
    let notes = e["error_bar"]["notes"].to_string();
    assert!(notes.contains("counted in P(differ)"), "{notes}");
}
