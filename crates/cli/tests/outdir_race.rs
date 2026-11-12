/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Concurrent runs of one case without `--out`. A run used to materialize its variants in
//! `$PTXV_ROOT/out/<case>`, shared by every run of the case: a variant's cubin is copied there,
//! patched in place and read back, so another run's copy in between was checked as this run's
//! variant. r4_sum_torch's `nosb` (the warp reduction's SHFL loses its write scoreboard:
//! `violation`) came out `clean` that way in 3 of 119 concurrent pairs of an A/B. Every run now
//! materializes in a directory of its own under `$PTXV_ROOT/out/<case>`, removed at its end.

use std::path::Path;
use std::path::PathBuf;
use std::process::Child;
use std::process::Command;
use std::process::Stdio;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/r4_sum_torch")
}

fn scratch(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("ptxv-outdir-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// `ptxv check r4_sum_torch --variant V --front sass --json` without `--out`, PTXV_ROOT = `root`.
fn spawn(root: &Path, variant: &str) -> Child {
    Command::new(env!("CARGO_BIN_EXE_ptxv"))
        .args(["check", fixture().to_str().unwrap(), "--variant", variant])
        .args(["--front", "sass", "--json", "--no-input-coverage"])
        .env("PTXV_ROOT", root)
        .env("PTXV_THREADS", "1")
        .env("PTXV_NO_VCACHE", "1")
        .env("CUDA_VISIBLE_DEVICES", "")
        .current_dir(root)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap()
}

/// (variant, classes) of every row of a finished check, or the run's error (no rows).
fn rows(c: Child) -> Result<Vec<(String, Vec<String>)>, String> {
    let out = c.wait_with_output().unwrap();
    let err = || String::from_utf8_lossy(&out.stderr).trim().to_string();
    let j: serde_json::Value = serde_json::from_slice(&out.stdout).map_err(|_| err())?;
    let rs = j["rows"].as_array().ok_or_else(err)?;
    Ok(rs
        .iter()
        .map(|r| {
            let got = r["got"].as_array().unwrap().iter();
            (
                r["variant"].as_str().unwrap().to_string(),
                got.map(|x| x.as_str().unwrap().to_string()).collect(),
            )
        })
        .collect())
}

fn expected(variant: &str) -> Vec<String> {
    vec![
        if variant == "nosb" {
            "violation"
        } else {
            "clean"
        }
        .to_string(),
    ]
}

/// Four runs of the whole case at once per round (as the two binaries of an A/B start them),
/// PTXV_OUTDIR_ROUNDS rounds (default 4): every row keeps its verdict, and every run removes
/// its directory (only the case's `.case` claim is left).
#[test]
fn concurrent_runs_without_out_keep_their_verdicts() {
    let rounds: usize = std::env::var("PTXV_OUTDIR_ROUNDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);
    let d = scratch("concurrent");
    let (mut wrong, mut failed, mut n) = (Vec::new(), Vec::new(), 0);
    for round in 0..rounds {
        let runs: Vec<Child> = (0..4).map(|_| spawn(&d, "all")).collect();
        for (i, c) in runs.into_iter().enumerate() {
            n += 1;
            match rows(c) {
                Ok(rs) if rs.len() == 2 => {
                    for (v, got) in rs {
                        if got != expected(&v) {
                            wrong.push(format!("round {round} run {i}: {v} {got:?}"));
                        }
                    }
                }
                Ok(rs) => failed.push(format!("round {round} run {i}: rows {rs:?}")),
                Err(e) => failed.push(format!("round {round} run {i}: {e}")),
            }
        }
    }
    let mut left: Vec<String> = std::fs::read_dir(d.join("out/r4_sum_torch"))
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    left.sort();
    let _ = std::fs::remove_dir_all(&d);
    assert!(
        wrong.is_empty() && failed.is_empty(),
        "of {n} runs: {} rows with a wrong verdict {wrong:?}; {} runs failed {failed:?}",
        wrong.len(),
        failed.len()
    );
    assert_eq!(left, [".case"], "run directories left behind");
}

/// What another run of the case does in the shared directory, all the time: it puts the
/// captured (unpatched) cubin at `out/r4_sum_torch/nosb.cubin` (here: write and rename, so
/// every read sees a whole file). A run that read that path checked the unpatched kernel and
/// reported `clean`; this run's own directory keeps `violation`.
#[test]
fn another_runs_copy_in_the_shared_directory_is_not_read() {
    let d = scratch("writer");
    let shared = d.join("out/r4_sum_torch");
    std::fs::create_dir_all(&shared).unwrap();
    let unpatched = std::fs::read(fixture().join("k0.cubin")).unwrap();
    let stop = Arc::new(AtomicBool::new(false));
    let writer = {
        let (stop, shared) = (stop.clone(), shared.clone());
        std::thread::spawn(move || {
            let mut n = 0u64;
            while !stop.load(Ordering::Relaxed) {
                let t = shared.join(format!(".copy{}", n % 2));
                std::fs::write(&t, &unpatched).unwrap();
                std::fs::rename(&t, shared.join("nosb.cubin")).unwrap();
                n += 1;
                std::thread::sleep(std::time::Duration::from_millis(2));
            }
            n
        })
    };
    let got = rows(spawn(&d, "nosb")).unwrap_or_else(|e| panic!("{e}"));
    stop.store(true, Ordering::Relaxed);
    let copies = writer.join().unwrap();
    let _ = std::fs::remove_dir_all(&d);
    assert!(copies > 0);
    assert_eq!(
        got,
        [("nosb".to_string(), expected("nosb"))],
        "{copies} copies written during the run"
    );
}
