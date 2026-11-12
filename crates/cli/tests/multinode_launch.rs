/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Launches of a multi-node job (PTXV_DIST_DIR, PTXV_RANK, PTXV_RANKS) where one rank stops
//! with an error: it refuses an argument before connecting (--time-budget, which only that
//! rank was given; an argument it cannot read), or fails after the ranks connected (a case it
//! cannot load). The others used to wait for it: ten minutes at the rendezvous for a lower
//! rank, forever for a higher one or after connecting. The stopping rank now leaves its reason
//! in the rendezvous directory (`refused.<rank>`, dist::refuse) and the others stop with it.
//! And `check` refuses, in multi-node mode, what lets each rank decide by its own clock which
//! mutated inputs to check.

use std::path::PathBuf;
use std::process::Child;
use std::process::Command;
use std::process::Stdio;
use std::time::Duration;
use std::time::Instant;

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/r4_sum_torch")
}

fn dist_dir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("ptxv-mn-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// Rank `rank` of `ranks`: `ptxv <args>`.
fn rank(dist: &PathBuf, rank: usize, ranks: usize, args: &[&str]) -> Child {
    Command::new(env!("CARGO_BIN_EXE_ptxv"))
        .args(args)
        .env("PTXV_DIST_DIR", dist)
        .env("PTXV_RANK", rank.to_string())
        .env("PTXV_RANKS", ranks.to_string())
        .env_remove("SLURM_PROCID")
        .env_remove("SLURM_NTASKS")
        .env_remove("PTXV_TIME_BUDGET")
        .env("PTXV_THREADS", "2")
        .env("CUDA_VISIBLE_DEVICES", "")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap()
}

/// (exit code, stderr, seconds) of a rank, killed after `limit` (code None).
fn finish(mut c: Child, t0: Instant, limit: Duration) -> (Option<i32>, String, f64) {
    loop {
        if let Some(st) = c.try_wait().unwrap() {
            let out = c.wait_with_output().unwrap();
            return (
                st.code(),
                String::from_utf8_lossy(&out.stderr).into_owned(),
                t0.elapsed().as_secs_f64(),
            );
        }
        if t0.elapsed() > limit {
            let _ = c.kill();
            let out = c.wait_with_output().unwrap();
            return (
                None,
                String::from_utf8_lossy(&out.stderr).into_owned(),
                t0.elapsed().as_secs_f64(),
            );
        }
        std::thread::sleep(Duration::from_millis(50));
    }
}

/// `check r4_sum_torch --variant good --front sass` and `extra`.
fn check_args<'a>(extra: &[&'a str]) -> Vec<&'a str> {
    static FIXTURE: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    let f = FIXTURE.get_or_init(|| fixture().to_str().unwrap().to_string());
    let mut a = vec!["check", f.as_str(), "--variant", "good", "--front", "sass"];
    a.extend_from_slice(extra);
    a
}

/// One rank refuses before connecting, the other has nothing to refuse: whichever rank refuses
/// (the lower one, which the other polls for, or the higher one, which the other accepts), the
/// other stops within seconds with exit status 1 and the refusing rank's reason.
#[test]
fn a_rank_that_refuses_stops_the_others() {
    let limit = Duration::from_secs(60);
    for (refusing, args, reason) in [
        (
            0,
            &["--time-budget", "0"][..],
            "not supported in multi-node mode",
        ),
        (
            1,
            &["--time-budget", "0"][..],
            "not supported in multi-node mode",
        ),
        (1, &["--no-such-flag"][..], "--no-such-flag"),
    ] {
        let d = dist_dir(&format!("refuse{refusing}"));
        let t0 = Instant::now();
        let ranks: Vec<Child> = (0..2)
            .map(|r| {
                let extra = if r == refusing { args } else { &[][..] };
                rank(&d, r, 2, &check_args(extra))
            })
            .collect();
        let done: Vec<_> = ranks.into_iter().map(|c| finish(c, t0, limit)).collect();
        let _ = std::fs::remove_dir_all(&d);
        let (code, err, secs) = &done[1 - refusing];
        assert_eq!(
            *code,
            Some(1),
            "rank {} after {secs:.0} s (rank {refusing} refused {args:?}): {err}",
            1 - refusing
        );
        assert!(
            err.contains(&format!("rank {refusing} stopped")) && err.contains(reason),
            "{err}"
        );
        assert!(*secs < 30.0, "stopped only after {secs:.0} s");
        assert_ne!(done[refusing].0, Some(0), "{}", done[refusing].1);
    }
}

/// A rank that fails after the ranks connected (here: its case directory does not exist on
/// its node) stops the others, which were in their first search waiting for its messages.
#[test]
fn a_rank_that_fails_after_connecting_stops_the_others() {
    let d = dist_dir("late");
    let t0 = Instant::now();
    let r0 = rank(&d, 0, 2, &check_args(&["--max-states", "200000"]));
    let r1 = rank(
        &d,
        1,
        2,
        &[
            "check",
            "/nonexistent/ptxv-case",
            "--variant",
            "good",
            "--front",
            "sass",
        ],
    );
    let limit = Duration::from_secs(180);
    let (c0, e0, s0) = finish(r0, t0, limit);
    let (c1, e1, _) = finish(r1, t0, limit);
    let _ = std::fs::remove_dir_all(&d);
    assert_ne!(c1, Some(0), "{e1}");
    assert_eq!(c0, Some(1), "rank 0 after {s0:.0} s: {e0}");
    assert!(e0.contains("rank 1 stopped"), "{e0}");
}

/// Input mutation and the concolic modes are refused in multi-node mode, before any rank
/// connects; with one rank (not multi-node) they run.
#[test]
fn multi_node_refuses_clock_dependent_mutation() {
    for extra in [
        &["--mutate-inputs", "1"][..],
        &["--mutate-input", "x[0]=1"][..],
        &["--concolic"][..],
        &["--exhaustive-concolic"][..],
    ] {
        let d = dist_dir("mutate");
        let t0 = Instant::now();
        let (code, err, secs) = finish(
            rank(&d, 0, 2, &check_args(extra)),
            t0,
            Duration::from_secs(60),
        );
        let _ = std::fs::remove_dir_all(&d);
        assert_eq!(code, Some(1), "{extra:?} after {secs:.0} s: {err}");
        assert!(
            err.contains(&format!("{} not supported in multi-node mode", extra[0])),
            "{err}"
        );
    }
    let d = dist_dir("one");
    let t0 = Instant::now();
    let (code, err, _) = finish(
        rank(&d, 0, 1, &check_args(&["--mutate-input", "x[0]=1"])),
        t0,
        Duration::from_secs(120),
    );
    let _ = std::fs::remove_dir_all(&d);
    assert_eq!(code, Some(0), "{err}");
}
