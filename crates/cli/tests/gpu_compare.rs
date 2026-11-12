/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! `rows[].gpu_compare`: the decision each `expect_report` line states, per comparison with the
//! captured GPU output (`bitwise`, `within_ulp`, `mismatch`, `no_capture`), next to the
//! unchanged lines. fixtures/r1h_nvls (an f32 multimem.ld_reduce: no calibrated rounding, so
//! N-MULTIMEM-ROUND16's one-ulp tolerance does not apply) and fixtures/r1h_nvls_bf16 (bf16 lanes
//! the calibrated rule rounds) copied with captured outputs written from their own simulated
//! output; the review probes fixtures/r1h_nvls_ulp and r1h_nvls_nan (MISS 110).

use std::path::Path;
use std::path::PathBuf;

/// `ptxv check <case> --variant good --front ptx --json` (PTXV_DUMP_OUTPUTS=1 into `out`): the
/// row and the files ptxv wrote under `out`.
fn check(case: &Path, out: &Path) -> serde_json::Value {
    check_front(case, out, "ptx")
}

/// `check` on `front`.
fn check_front(case: &Path, out: &Path, front: &str) -> serde_json::Value {
    let o = std::process::Command::new(env!("CARGO_BIN_EXE_ptxv"))
        .arg("check")
        .arg(case)
        .args([
            "--variant",
            "good",
            "--front",
            front,
            "--json",
            "--no-input-coverage",
            "--out",
        ])
        .arg(out)
        .env("PTXV_DUMP_OUTPUTS", "1")
        .env("CUDA_VISIBLE_DEVICES", "")
        .output()
        .unwrap();
    let doc: serde_json::Value = serde_json::from_slice(&o.stdout).unwrap_or_else(|e| {
        panic!("{e}: {}", String::from_utf8_lossy(&o.stderr));
    });
    doc["rows"][0].clone()
}

/// (part, result) of each `gpu_compare` entry.
fn results(row: &serde_json::Value) -> Vec<(String, String)> {
    row["gpu_compare"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| {
            assert_eq!(c["output"], 0, "{c}");
            (
                c["part"].as_str().unwrap().to_string(),
                c["result"].as_str().unwrap().to_string(),
            )
        })
        .collect()
}

fn pair(a: &str, b: &str) -> Vec<(String, String)> {
    vec![
        ("out0".to_string(), a.to_string()),
        ("out1".to_string(), b.to_string()),
    ]
}

fn find(dir: &Path, name: &str) -> Option<PathBuf> {
    for e in std::fs::read_dir(dir).ok()?.flatten() {
        let p = e.path();
        if p.is_dir() {
            if let Some(f) = find(&p, name) {
                return Some(f);
            }
        } else if p.file_name().is_some_and(|n| n == name) {
            return Some(p);
        }
    }
    None
}

#[test]
fn gpu_compare_states_each_comparison() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/r1h_nvls");
    let tmp = std::env::temp_dir().join(format!("ptxv-gpu-compare-{}", std::process::id()));
    let case = tmp.join("case");
    let out = tmp.join("out");
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(case.join("expected")).unwrap();
    for f in ["case.toml", "kernel.ptx"] {
        std::fs::copy(root.join(f), case.join(f)).unwrap();
    }
    // No captured output: one `no_capture` entry per output buffer, no line.
    let row = check(&case, &out);
    assert_eq!(results(&row), pair("no_capture", "no_capture"));
    assert_eq!(row["expect_report"], serde_json::json!([]));
    let sim = std::fs::read(find(&out, "good.ptx.output0.bin").expect("dumped output")).unwrap();
    assert_eq!(sim.len(), 32, "out0 and out1, 4 f32 each");
    let (out0, out1) = sim.split_at(16);
    let toml = std::fs::read_to_string(case.join("case.toml")).unwrap();
    let toml = toml
        .replace(
            "name = \"out0\"\n",
            "name = \"out0\"\nexpect = \"expected/out0.bin\"\n",
        )
        .replace(
            "name = \"out1\"\n",
            "name = \"out1\"\nexpect = \"expected/out1.bin\"\n",
        );
    std::fs::write(case.join("case.toml"), toml).unwrap();
    // out0's second element one unit in the last place off, out1 as simulated: an f32 sum
    // holds no calibrated f16 / bf16 lane, so a mismatch (MISS 110; within one ulp before).
    let mut near = out0.to_vec();
    let w = u32::from_le_bytes(near[4..8].try_into().unwrap()) ^ 1;
    near[4..8].copy_from_slice(&w.to_le_bytes());
    std::fs::write(case.join("expected/out0.bin"), &near).unwrap();
    std::fs::write(case.join("expected/out1.bin"), out1).unwrap();
    let row = check(&case, &out);
    assert_eq!(results(&row), pair("mismatch", "bitwise"));
    let lines: Vec<&str> = row["expect_report"]
        .as_array()
        .unwrap()
        .iter()
        .map(|l| l.as_str().unwrap())
        .collect();
    assert!(
        lines[0].starts_with("output 0: out0: MISMATCH vs the captured GPU output: 1 of 16 bytes"),
        "{lines:?}"
    );
    assert!(
        lines[1].starts_with("output 0: out1: bitwise equal to the captured GPU output (16 bytes"),
        "{lines:?}"
    );
    // out0 as simulated, out1 far off: bitwise and mismatch.
    let mut far = out1.to_vec();
    far[3] ^= 0x40;
    std::fs::write(case.join("expected/out0.bin"), out0).unwrap();
    std::fs::write(case.join("expected/out1.bin"), &far).unwrap();
    let row = check(&case, &out);
    assert_eq!(results(&row), pair("bitwise", "mismatch"));
    assert!(
        row["expect_report"][1]
            .as_str()
            .unwrap()
            .contains("MISMATCH"),
        "{}",
        row["expect_report"]
    );
    let _ = std::fs::remove_dir_all(&tmp);
}

/// The output buffers of fixtures/r1h_nvls_bf16, in `outputs` order (4 bf16 each).
const NVLS16: [&str; 8] = [
    "outa0", "outb0", "outc0", "outm0", "outa1", "outb1", "outc1", "outm1",
];

/// MISS 110 (N-MULTIMEM-ROUND16 scope): fixtures/r1h_nvls_bf16 with captured outputs written
/// from its own simulated output, one bf16 element one encoding off at a time, on both fronts.
/// Within one ulp only in lanes 0 and 2 (the sums the calibrated rule rounds) of outa (stored as
/// reduced) and outm (broadcast by multimem.st, on the other GPU too); a mismatch in an exact lane
/// (1), in outb (the pair doubled in its register: rounded data is no longer the lane) and in outc
/// (the input pair exchanged in over the stored lane).
#[test]
fn only_calibrated_bf16_lanes_are_within_one_ulp() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/r1h_nvls_bf16");
    for front in ["ptx", "sass"] {
        let tmp =
            std::env::temp_dir().join(format!("ptxv-gpu-compare16-{front}-{}", std::process::id()));
        let case = tmp.join("case");
        let out = tmp.join("out");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(case.join("expected")).unwrap();
        std::fs::create_dir_all(case.join("data")).unwrap();
        for f in ["case.toml", "kernel.ptx", "data/in0.bin", "data/in1.bin"] {
            std::fs::copy(root.join(f), case.join(f)).unwrap();
        }
        let row = check_front(&case, &out, front);
        assert_eq!(
            row["got"],
            serde_json::json!(["clean"]),
            "{front}: {}",
            row["got"]
        );
        let sim = std::fs::read(find(&out, &format!("good.{front}.output0.bin")).unwrap()).unwrap();
        assert_eq!(sim.len(), 64, "{front}: 8 outputs of 4 bf16");
        let mut toml = std::fs::read_to_string(case.join("case.toml")).unwrap();
        for b in NVLS16 {
            toml = toml.replace(
                &format!("name = \"{b}\"\n"),
                &format!("name = \"{b}\"\nexpect = \"expected/{b}.bin\"\n"),
            );
        }
        std::fs::write(case.join("case.toml"), toml).unwrap();
        // (buffer, lane, result of that buffer with the element one encoding up)
        for (b, lane, want) in [
            ("outa0", 0, "within_ulp"),
            ("outa1", 2, "within_ulp"),
            ("outm0", 2, "within_ulp"),
            ("outm1", 0, "within_ulp"),
            ("outa0", 1, "mismatch"),
            ("outm1", 3, "mismatch"),
            ("outb0", 0, "mismatch"),
            ("outb1", 2, "mismatch"),
            ("outc0", 0, "mismatch"),
            ("outc1", 2, "mismatch"),
        ] {
            let bi = NVLS16.iter().position(|&x| x == b).unwrap();
            for (i, name) in NVLS16.iter().enumerate() {
                let mut v = sim[8 * i..8 * i + 8].to_vec();
                if i == bi {
                    let e = u16::from_le_bytes([v[2 * lane], v[2 * lane + 1]]) + 1;
                    v[2 * lane..2 * lane + 2].copy_from_slice(&e.to_le_bytes());
                }
                std::fs::write(case.join(format!("expected/{name}.bin")), v).unwrap();
            }
            let row = check_front(&case, &out, front);
            let got = results(&row);
            for (part, r) in &got {
                let w = if part == b { want } else { "bitwise" };
                assert_eq!(r, w, "{front}: {b} lane {lane}: {part}: {got:?}");
            }
            assert_eq!(got.len(), 8, "{front}: {got:?}");
        }
        let _ = std::fs::remove_dir_all(&tmp);
    }
}

/// MISS 110: the review probes, each with constructed captured outputs one f32 encoding off
/// (r1h_nvls_ulp, an exact f32 sum) or NaN against a NaN with another payload (r1h_nvls_nan):
/// both outputs are a mismatch on both fronts (both were within one ulp).
#[test]
fn review_probes_are_mismatches() {
    for probe in ["r1h_nvls_ulp", "r1h_nvls_nan"] {
        let case = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures")
            .join(probe);
        for front in ["ptx", "sass"] {
            let out = std::env::temp_dir().join(format!(
                "ptxv-gpu-compare-{probe}-{front}-{}",
                std::process::id()
            ));
            let _ = std::fs::remove_dir_all(&out);
            let row = check_front(&case, &out, front);
            assert_eq!(
                results(&row),
                pair("mismatch", "mismatch"),
                "{probe} {front}: {}",
                row["expect_report"]
            );
            let _ = std::fs::remove_dir_all(&out);
        }
    }
}
