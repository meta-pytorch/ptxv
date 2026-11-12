/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! The hardware runner's GPU path (crates/hw `run_with`: a hardware stream per plan stream,
//! events recorded after launches and waited for by later ones) on a GB300, which no other test
//! runs. GPU-gated: `PTXV_TEST_GPU=<index>` names the GPU; without it, or when `ptxv hw` finds
//! that GPU busy (a compute process of another program on it), the test is skipped.

use std::path::PathBuf;
use std::process::Command;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// `ptxv hw fixtures/<fx> --variant <variant> --runs <runs> --delay <delay>` on GPU `gpu`: the
/// number of runs whose output differs from the simulator's ordered output, or None when the
/// GPU is busy.
fn differing(gpu: &str, fx: &str, variant: &str, delay: &str, runs: usize) -> Option<usize> {
    let out = std::env::temp_dir().join(format!(
        "ptxv-test-hw-runner-{fx}-{variant}-{}",
        std::process::id()
    ));
    let r = Command::new(env!("CARGO_BIN_EXE_ptxv"))
        .arg("hw")
        .arg(root().join("fixtures").join(fx))
        .args([
            "--variant",
            variant,
            "--runs",
            &runs.to_string(),
            "--delay",
            delay,
            "--out",
        ])
        .arg(&out)
        .env("PTXV_GPUS", gpu)
        .env("PTXV_NO_VCACHE", "1")
        .env_remove("CUDA_VISIBLE_DEVICES")
        .env_remove("PTXV_HW_EXPECT")
        .output()
        .unwrap();
    let _ = std::fs::remove_dir_all(&out);
    let t = format!(
        "{}{}",
        String::from_utf8_lossy(&r.stdout),
        String::from_utf8_lossy(&r.stderr)
    );
    if t.contains("no free GPU") {
        return None;
    }
    // "hw <variant> N/M runs differ ..." or "hw <variant> matches ... (M runs, ...)"
    let line = t
        .lines()
        .find(|l| l.starts_with(&format!("hw {variant} ")))
        .unwrap_or_else(|| panic!("{fx} {variant}: no result line: {t}"));
    if line.contains("runs differ") {
        let n = line[format!("hw {variant} ").len()..]
            .split('/')
            .next()
            .and_then(|x| x.trim().parse().ok())
            .unwrap_or_else(|| panic!("{fx} {variant}: {line}"));
        Some(n)
    } else {
        assert!(line.contains("matches"), "{fx} {variant}: {line}");
        Some(0)
    }
}

/// With a 2 ms delay before the writer's store (`--delay`), a reader the case orders after the
/// writer through an event (e_launch_order good: another stream; e_launch_waits good: `waits`)
/// or the stream order (e_runner_after CHAIN) reads the written bytes in every run, and one the
/// case leaves unordered with it (e_launch_order UNORD, e_launch_waits NOWAIT, e_runner_after
/// good) reads them before the write in most runs (200 of 200 when measured). An event recorded
/// where it does not cover the writer (on another stream) shows up as the first kind differing.
#[test]
fn gpu_runs_keep_the_case_order() {
    let Ok(gpu) = std::env::var("PTXV_TEST_GPU") else {
        eprintln!("skipped: set PTXV_TEST_GPU=<index of an idle GPU>");
        return;
    };
    let runs = 200;
    for (fx, variant, delay, ordered) in [
        ("e_launch_order", "good", "33:2000000@0", true),
        ("e_launch_waits", "good", "48:2000000@0", true),
        ("e_runner_after", "CHAIN", "24:2000000@0", true),
        ("e_launch_order", "UNORD", "33:2000000@0", false),
        ("e_launch_waits", "NOWAIT", "48:2000000@0", false),
        ("e_runner_after", "good", "24:2000000@0", false),
    ] {
        let Some(n) = differing(&gpu, fx, variant, delay, runs) else {
            eprintln!("skipped: GPU {gpu} is busy");
            return;
        };
        if ordered {
            assert_eq!(n, 0, "{fx} {variant}: {n}/{runs} runs differ");
        } else {
            assert!(n > runs / 2, "{fx} {variant}: only {n}/{runs} runs differ");
        }
    }
}
