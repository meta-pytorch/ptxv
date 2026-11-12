/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! `rows[].gpu_compare`: the decision each `expect_report` line states, per comparison with the
//! captured GPU output (`bitwise`, `within_ulp`, `mismatch`, `no_capture`), next to the
//! unchanged lines. fixtures/r1h_nvls (multimem.ld_reduce, so N-MULTIMEM-ROUND16's one-ulp
//! tolerance applies) copied with captured outputs written from its own simulated output.

use std::path::Path;
use std::path::PathBuf;

/// `ptxv check <case> --variant good --front ptx --json` (PTXV_DUMP_OUTPUTS=1 into `out`): the
/// row and the files ptxv wrote under `out`.
fn check(case: &Path, out: &Path) -> serde_json::Value {
    let o = std::process::Command::new(env!("CARGO_BIN_EXE_ptxv"))
        .arg("check")
        .arg(case)
        .args([
            "--variant",
            "good",
            "--front",
            "ptx",
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
    // out0's second element one unit in the last place off, out1 as simulated.
    let mut near = out0.to_vec();
    let w = u32::from_le_bytes(near[4..8].try_into().unwrap()) ^ 1;
    near[4..8].copy_from_slice(&w.to_le_bytes());
    std::fs::write(case.join("expected/out0.bin"), &near).unwrap();
    std::fs::write(case.join("expected/out1.bin"), out1).unwrap();
    let row = check(&case, &out);
    assert_eq!(results(&row), pair("within_ulp", "bitwise"));
    let lines: Vec<&str> = row["expect_report"]
        .as_array()
        .unwrap()
        .iter()
        .map(|l| l.as_str().unwrap())
        .collect();
    // The lines are as before; the within-one-ulp one also says "the rest bitwise equal".
    assert!(
        lines[0].starts_with("output 0: out0: within one ulp of the captured GPU output: 1 of 4"),
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
