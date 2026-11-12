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
    rows_of(&c.wait_with_output().unwrap())
}

/// (variant, classes) of every row of a check's output, or the run's error (no rows).
fn rows_of(out: &std::process::Output) -> Result<Vec<(String, Vec<String>)>, String> {
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

/// This host as run directory names carry it (main.rs `run_host`): host name, boot (16 hex
/// digits of the boot id), pid namespace unless the initial one; and the same token with
/// another boot (this machine before a reboot, or another machine of the same name).
fn host_tokens() -> (String, String) {
    let name: String = std::fs::read_to_string("/proc/sys/kernel/hostname")
        .unwrap()
        .trim()
        .chars()
        .map(|c| match c {
            'a'..='z' | 'A'..='Z' | '0'..='9' | '.' | '-' | '_' => c,
            _ => '_',
        })
        .collect();
    let boot: String = std::fs::read_to_string("/proc/sys/kernel/random/boot_id")
        .unwrap()
        .chars()
        .filter(char::is_ascii_hexdigit)
        .take(16)
        .collect();
    let other: String = boot
        .chars()
        .map(|c| if c == '0' { '1' } else { '0' })
        .collect();
    let ns = std::fs::read_link("/proc/self/ns/pid").unwrap();
    let ns = ns.to_string_lossy().into_owned();
    let ino = ns.trim_start_matches("pid:[").trim_end_matches(']');
    let pidns = if ino == "4026531836" {
        String::new()
    } else {
        format!(".pidns{ino}")
    };
    (
        format!("{name}.boot{boot}{pidns}"),
        format!("{name}.boot{other}{pidns}"),
    )
}

/// An exclusive flock on `path` (created), held while the returned file is open: what a live
/// run holds on its directory's `.lock`.
fn hold_lock(path: &Path) -> std::fs::File {
    let f = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(path)
        .unwrap();
    f.try_lock().unwrap();
    f
}

/// (pid, start time in clock ticks since boot) of this process as /proc names it.
fn proc_self() -> (u32, u64) {
    let pid: u32 = std::fs::read_link("/proc/self")
        .unwrap()
        .to_str()
        .unwrap()
        .parse()
        .unwrap();
    let stat = std::fs::read_to_string(format!("/proc/{pid}/stat")).unwrap();
    let (_, rest) = stat.rsplit_once(')').unwrap();
    (
        pid,
        rest.split_whitespace().nth(19).unwrap().parse().unwrap(),
    )
}

fn names(d: &Path) -> Vec<String> {
    let mut v: Vec<String> = std::fs::read_dir(d)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    v.sort();
    v
}

/// A run killed by a signal (here SIGKILL; `timeout` sends SIGTERM) skips the removal of its
/// run directory. The next run of the case on the host removes it: the kernel released the
/// killed run's lock (`<run dir>/.lock`, flock) with its process. Only directories of this host
/// token (host name, boot, pid namespace) and user whose lock no process holds are removed,
/// whatever /proc says of the pid in their name: those of live runs (the lock held, here by
/// this test), of runs still making their directory (no lock file yet), of other hosts or
/// boots, of older binaries (`run-<pid>-<ns>`) and kept ones (`.kept`) stay.
#[test]
fn killed_runs_directories_are_removed_by_the_next_run() {
    let d = scratch("killed");
    let base = d.join("out/r4_sum_torch");
    std::fs::create_dir_all(&base).unwrap();
    let (host, other_boot) = host_tokens();
    let (me, start) = proc_self();
    let dead = {
        let mut c = Command::new("true").spawn().unwrap();
        let pid = c.id();
        c.wait().unwrap();
        pid
    };
    enum Lock {
        Free,
        Held,
        Missing,
    }
    let mut held = Vec::new();
    let mut keep = vec![".case".to_string()];
    for (name, lock, stays) in [
        // A dead run: nobody holds its lock, though its pid is live (this test's).
        (format!("run-{host}-{me}-{start}-1"), Lock::Free, false),
        // A live run, though its pid is gone: its lock is held (by this test).
        (format!("run-{host}-{dead}-12345-2"), Lock::Held, true),
        // A run making its directory: no lock file yet.
        (format!("run-{host}-{dead}-12345-3"), Lock::Missing, true),
        // Another host's run (a shared file system): not this host's locks to judge.
        (
            format!("run-elsewhere.example.boot0123456789abcdef-{dead}-12345-4"),
            Lock::Free,
            true,
        ),
        // This host name in another boot: before a reboot, or another machine of the name.
        (format!("run-{other_boot}-{dead}-12345-5"), Lock::Free, true),
        // An older binary's name says nothing of the host.
        (format!("run-{dead}-6"), Lock::Free, true),
        // Kept.
        (format!("run-{host}-{dead}-12345-7"), Lock::Free, true),
    ] {
        let dir = base.join(&name);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("k0.cubin"), b"x").unwrap();
        match lock {
            Lock::Free => std::fs::write(dir.join(".lock"), b"").unwrap(),
            Lock::Held => held.push(hold_lock(&dir.join(".lock"))),
            Lock::Missing => {}
        }
        if name.ends_with("-7") {
            std::fs::write(dir.join(".kept"), b"").unwrap();
        }
        if stays {
            keep.push(name);
        }
    }
    keep.sort();
    let before = names(&base);
    // Kill a run as soon as its directory holds its lock file (again if it finished first).
    let killed = (0..10)
        .find_map(|_| {
            let mut c = spawn(&d, "all");
            let t0 = std::time::Instant::now();
            loop {
                let new: Vec<String> = names(&base)
                    .into_iter()
                    .filter(|n| !before.contains(n) && n != ".case")
                    .collect();
                if let Some(n) = new
                    .into_iter()
                    .find(|n| base.join(n).join(".lock").exists())
                {
                    c.kill().unwrap();
                    c.wait().unwrap();
                    return base.join(&n).exists().then_some(n);
                }
                if c.try_wait().unwrap().is_some() || t0.elapsed().as_secs() > 60 {
                    let _ = c.kill();
                    let _ = c.wait();
                    return None;
                }
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        })
        .expect("no run was killed while its directory was there");
    let got = rows(spawn(&d, "all"));
    let left = names(&base);
    drop(held);
    let _ = std::fs::remove_dir_all(&d);
    let got = got.unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(got.len(), 2, "{got:?}");
    for (v, classes) in got {
        assert_eq!(classes, expected(&v), "{v}");
    }
    assert!(
        killed.starts_with(&format!("run-{host}-")),
        "the run's directory {killed} does not carry the host token {host}"
    );
    assert!(
        !left.contains(&killed),
        "the killed run's {killed} is left: {left:?}"
    );
    assert_eq!(left, keep);
}

/// A run removes a dead run's directory with `.lock` last, so what it cannot remove keeps
/// `.lock`, and the next run removes the rest. `remove_dir_all` stops at the first entry it
/// cannot remove, which may come after `.lock` (on NFS, a removal that a racing reaper's
/// reading skipped): the directory stayed without `.lock`, and no run reaped it. Here a
/// subdirectory without write permission stands for what a run fails to remove, in 12 dead
/// directories whose `.lock` is made first in half of them and last in the others, beside
/// subdirectories of different names (directories list their entries in creation order, the
/// reverse, or by a hash of the name). Skipped where permissions do not hold (root).
#[test]
fn what_a_reaper_cannot_remove_keeps_the_lock_file() {
    use std::os::unix::fs::PermissionsExt;
    let d = scratch("partial");
    let base = d.join("out/r4_sum_torch");
    std::fs::create_dir_all(&base).unwrap();
    let mode = |p: &Path, m: u32| std::fs::set_permissions(p, std::fs::Permissions::from_mode(m));
    let probe = d.join("probe");
    std::fs::create_dir_all(&probe).unwrap();
    mode(&probe, 0o500).unwrap();
    let enforced = std::fs::write(probe.join("x"), b"x").is_err();
    mode(&probe, 0o700).unwrap();
    if !enforced {
        let _ = std::fs::remove_dir_all(&d);
        eprintln!("skipped: a directory without write permission can be written to here");
        return;
    }
    let (host, _) = host_tokens();
    let dead = {
        let mut c = Command::new("true").spawn().unwrap();
        let pid = c.id();
        c.wait().unwrap();
        pid
    };
    let mut planted = Vec::new();
    for i in 0..12 {
        let dir = base.join(format!("run-{host}-{dead}-12345-{i}"));
        let sub = dir.join(format!("sub{i}"));
        std::fs::create_dir_all(&dir).unwrap();
        if i % 2 == 0 {
            std::fs::write(dir.join(".lock"), b"").unwrap();
        }
        std::fs::create_dir_all(&sub).unwrap();
        std::fs::write(sub.join("f"), b"x").unwrap();
        std::fs::write(dir.join("k0.cubin"), b"x").unwrap();
        if i % 2 == 1 {
            std::fs::write(dir.join(".lock"), b"").unwrap();
        }
        mode(&sub, 0o500).unwrap();
        planted.push((dir, sub, format!("sub{i}")));
    }
    // A run reaps: it can remove k0.cubin only.
    let first = rows(spawn(&d, "good"));
    let after_first: Vec<(PathBuf, Option<Vec<String>>)> = planted
        .iter()
        .map(|(dir, _, _)| (dir.clone(), dir.exists().then(|| names(dir))))
        .collect();
    // With the subdirectories writable again, the next run removes the rest.
    for (_, sub, _) in &planted {
        let _ = mode(sub, 0o700);
    }
    let second = rows(spawn(&d, "good"));
    let left: Vec<String> = planted
        .iter()
        .filter(|(dir, _, _)| dir.exists())
        .map(|(dir, _, _)| dir.display().to_string())
        .collect();
    let _ = std::fs::remove_dir_all(&d);
    let good = vec![("good".to_string(), expected("good"))];
    assert_eq!(first.unwrap_or_else(|e| panic!("the first run: {e}")), good);
    assert_eq!(
        second.unwrap_or_else(|e| panic!("the second run: {e}")),
        good
    );
    for ((dir, got), (_, _, sub)) in after_first.iter().zip(&planted) {
        assert_eq!(
            got.as_deref(),
            Some(&[".lock".to_string(), sub.clone()][..]),
            "{}: what the first run left of it",
            dir.display()
        );
    }
    assert!(
        left.is_empty(),
        "dead directories the second run left: {left:?}"
    );
}

/// A run stops emptying a level of a dead directory after a round that removes nothing. With
/// its rounds retried at every level, what a run could not remove k levels down cost up to 8^k
/// listings: here a dead directory whose 9th level has no write permission, 8^9 listings at the
/// bottom, many minutes of every run of the case. The run must end within 60 s (it takes a
/// second or two) and leave the directory with `.lock` and its first level (k0.cubin goes).
/// Skipped where permissions do not hold (root).
#[test]
fn a_reaper_stops_at_what_it_cannot_remove_deep_down() {
    use std::os::unix::fs::PermissionsExt;
    let d = scratch("deep");
    let base = d.join("out/r4_sum_torch");
    std::fs::create_dir_all(&base).unwrap();
    let mode = |p: &Path, m: u32| std::fs::set_permissions(p, std::fs::Permissions::from_mode(m));
    let probe = d.join("probe");
    std::fs::create_dir_all(&probe).unwrap();
    mode(&probe, 0o500).unwrap();
    let enforced = std::fs::write(probe.join("x"), b"x").is_err();
    mode(&probe, 0o700).unwrap();
    if !enforced {
        let _ = std::fs::remove_dir_all(&d);
        eprintln!("skipped: a directory without write permission can be written to here");
        return;
    }
    let (host, _) = host_tokens();
    let dead = {
        let mut c = Command::new("true").spawn().unwrap();
        let pid = c.id();
        c.wait().unwrap();
        pid
    };
    let dir = base.join(format!("run-{host}-{dead}-12345-0"));
    let deep = (0..9).fold(dir.clone(), |p, level| p.join(format!("l{level}")));
    std::fs::create_dir_all(&deep).unwrap();
    std::fs::write(deep.join("f"), b"x").unwrap();
    std::fs::write(dir.join("k0.cubin"), b"x").unwrap();
    std::fs::write(dir.join(".lock"), b"").unwrap();
    mode(&deep, 0o500).unwrap();
    let t0 = std::time::Instant::now();
    let mut run = spawn(&d, "good");
    let ended = loop {
        if run.try_wait().unwrap().is_some() {
            break true;
        }
        if t0.elapsed().as_secs() >= 60 {
            let _ = run.kill();
            break false;
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    };
    let secs = t0.elapsed().as_secs_f64();
    let got = rows(run);
    let left = dir.exists().then(|| names(&dir));
    let bottom = deep.join("f").exists();
    let _ = mode(&deep, 0o700);
    let _ = std::fs::remove_dir_all(&d);
    assert!(
        ended,
        "the run that reaps a dead directory it cannot empty 9 levels down was still running \
         after {secs:.0} s"
    );
    let good = vec![("good".to_string(), expected("good"))];
    assert_eq!(got.unwrap_or_else(|e| panic!("the run: {e}")), good);
    assert_eq!(
        left.as_deref(),
        Some(&[".lock".to_string(), "l0".to_string()][..]),
        "what the run left of {}",
        dir.display()
    );
    assert!(bottom, "the file the run cannot remove is gone");
}

/// A subdirectory of a dead directory that another run's reaper removes between this run's
/// listing of the directory and its reading of the subdirectory (read_dir: NotFound) counts as
/// gone, as an entry whose removal finds it gone does: the round made progress, the next one
/// lists the directory empty, and the run removes it. The run ended the level there, leaving
/// the dead directory with `.lock` for the next run. The shim (tests/shim_flock.c, LD_PRELOAD)
/// removes the empty subdirectory `gone` when the run opens it; skipped where the shim cannot
/// be built (no C compiler).
#[test]
fn a_subdirectory_removed_under_a_reaper_counts_as_gone() {
    let d = scratch("vanish");
    let so = match flock_shim(&d) {
        Ok(so) => so,
        Err(why) => {
            let _ = std::fs::remove_dir_all(&d);
            eprintln!("skipped: the flock shim cannot be built ({why})");
            return;
        }
    };
    let (host, _) = host_tokens();
    let dead = {
        let mut c = Command::new("true").spawn().unwrap();
        let pid = c.id();
        c.wait().unwrap();
        pid
    };
    let dir = d.join(format!("out/r4_sum_torch/run-{host}-{dead}-12345-0"));
    std::fs::create_dir_all(dir.join("gone")).unwrap();
    std::fs::write(dir.join(".lock"), b"").unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_ptxv"))
        .args(["check", fixture().to_str().unwrap(), "--variant", "good"])
        .args(["--front", "sass", "--json", "--no-input-coverage"])
        .env("PTXV_ROOT", &d)
        .env("PTXV_THREADS", "1")
        .env("PTXV_NO_VCACHE", "1")
        .env("CUDA_VISIBLE_DEVICES", "")
        .env("LD_PRELOAD", &so)
        .env("SHIM_VANISH_DIR", "gone")
        .current_dir(&d)
        .output()
        .unwrap();
    let err = String::from_utf8_lossy(&out.stderr).to_string();
    let left = dir.exists().then(|| names(&dir));
    let _ = std::fs::remove_dir_all(&d);
    assert!(
        err.contains("shim: opendir("),
        "the shim never removed `gone` under the run: {err}"
    );
    let good = vec![("good".to_string(), expected("good"))];
    assert_eq!(
        rows_of(&out).unwrap_or_else(|e| panic!("the run: {e}")),
        good
    );
    assert_eq!(left, None, "what the run left of {}", dir.display());
}

/// `ptxv check r4_sum_torch --variant all` in its own process group, PTXV_ROOT = `root`, run
/// through `prefix` (e.g. `unshare ...`) when it is not empty.
fn spawn_group(root: &Path, prefix: &[String]) -> Child {
    use std::os::unix::process::CommandExt;
    let bin = env!("CARGO_BIN_EXE_ptxv").to_string();
    let fx = fixture();
    let args = [
        "check",
        fx.to_str().unwrap(),
        "--variant",
        "all",
        "--front",
        "sass",
        "--json",
        "--no-input-coverage",
    ];
    let mut cmd = match prefix.split_first() {
        Some((p, rest)) => {
            let mut c = Command::new(p);
            c.args(rest).arg(&bin);
            c
        }
        None => Command::new(&bin),
    };
    cmd.args(args)
        .env("PTXV_ROOT", root)
        .env("PTXV_THREADS", "1")
        .env("PTXV_NO_VCACHE", "1")
        .env("CUDA_VISIBLE_DEVICES", "")
        .current_dir(root)
        .process_group(0)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap()
}

/// `kill -SIG -- -<pgid>`.
fn signal_group(pgid: u32, sig: &str) {
    let _ = Command::new("kill")
        .args([sig, "--", &format!("-{pgid}")])
        .status();
}

/// A stopped process group, continued and then killed if the test ends before `release`.
struct Stopped(Option<u32>);

impl Stopped {
    fn release(&mut self) {
        if let Some(g) = self.0.take() {
            signal_group(g, "-CONT");
        }
    }
}

impl Drop for Stopped {
    fn drop(&mut self) {
        if let Some(g) = self.0.take() {
            signal_group(g, "-CONT");
            signal_group(g, "-KILL");
        }
    }
}

/// The prefix that runs a command in a new user and time namespace whose boot clock is 100000 s
/// ahead (`unshare --user --map-root-user --time --boottime 100000`), or why there is none.
fn time_namespace() -> Result<Vec<String>, String> {
    let pre: Vec<String> = [
        "unshare",
        "--user",
        "--map-root-user",
        "--time",
        "--boottime",
        "100000",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    match Command::new(&pre[0]).args(&pre[1..]).arg("true").output() {
        Ok(o) if o.status.success() => Ok(pre),
        Ok(o) => Err(String::from_utf8_lossy(&o.stderr).trim().to_string()),
        Err(e) => Err(e.to_string()),
    }
}

/// A run of the case in another time namespace of the host does not remove a live run's
/// directory. /proc shows a process's start time shifted by the reader's boot clock offset, so
/// a reaper that judged liveness by the start time in /proc took the live run for a reused pid
/// and removed its directory: the live run then failed (nvdisasm: cannot open its good.cubin).
/// Both ways round: the live run outside the namespace and the other run in it, and the
/// reverse. Skipped where unprivileged user (and time) namespaces are not available.
#[test]
// The stopped live run's child leaves the retry loop in a tuple; `rows(live)` waits for it.
#[allow(clippy::zombie_processes)]
fn a_live_run_in_another_time_namespace_keeps_its_directory() {
    let tns = match time_namespace() {
        Ok(p) => p,
        Err(why) => {
            eprintln!("skipped: no unprivileged user and time namespace here ({why})");
            return;
        }
    };
    for (live_in_ns, tag) in [(false, "tns-other-inside"), (true, "tns-live-inside")] {
        let (live_pre, other_pre) = if live_in_ns {
            (tns.clone(), Vec::new())
        } else {
            (Vec::new(), tns.clone())
        };
        let d = scratch(tag);
        let base = d.join("out/r4_sum_torch");
        // Stop the live run (its whole process group) once its directory holds good.cubin;
        // again if it finished first.
        let (live, mut stopped, dir) = (0..10)
            .find_map(|_| {
                let mut c = spawn_group(&d, &live_pre);
                let g = c.id();
                let t0 = std::time::Instant::now();
                loop {
                    let found = std::fs::read_dir(&base).ok().and_then(|es| {
                        es.flatten()
                            .map(|e| e.path())
                            .find(|p| p.join("good.cubin").exists())
                    });
                    if let Some(dir) = found {
                        signal_group(g, "-STOP");
                        if dir.join("good.cubin").exists() {
                            return Some((c, Stopped(Some(g)), dir));
                        }
                        signal_group(g, "-CONT");
                        signal_group(g, "-KILL");
                        let _ = c.wait();
                        return None;
                    }
                    if t0.elapsed().as_secs() > 60 {
                        signal_group(g, "-KILL");
                        let _ = c.wait();
                        return None;
                    }
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
            })
            .expect("the live run was never stopped while its directory held good.cubin");
        let other = rows(spawn_group(&d, &other_pre));
        let after: Vec<String> = std::fs::read_dir(&dir)
            .map(|es| {
                let mut v: Vec<String> = es
                    .flatten()
                    .map(|e| e.file_name().to_string_lossy().into_owned())
                    .collect();
                v.sort();
                v
            })
            .unwrap_or_default();
        stopped.release();
        let got = rows(live);
        let _ = std::fs::remove_dir_all(&d);
        let other = other.unwrap_or_else(|e| panic!("{tag}: the other run: {e}"));
        assert_eq!(other.len(), 2, "{tag}: the other run: {other:?}");
        assert!(
            after.iter().any(|n| n == "good.cubin"),
            "{tag}: the live run's directory {} lost its files: {after:?}",
            dir.display()
        );
        let got = got.unwrap_or_else(|e| panic!("{tag}: the live run: {e}"));
        assert_eq!(got.len(), 2, "{tag}: the live run: {got:?}");
        for (v, classes) in got {
            assert_eq!(classes, expected(&v), "{tag}: {v}");
        }
    }
}

/// The flock shim (tests/shim_flock.c) built for LD_PRELOAD into `dir`, or why it cannot be.
fn flock_shim(dir: &Path) -> Result<PathBuf, String> {
    let so = dir.join("shim_flock.so");
    let src = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/shim_flock.c");
    let out = Command::new("cc")
        .args(["-shared", "-fPIC", "-O1", "-o"])
        .arg(&so)
        .arg(&src)
        .arg("-ldl")
        .output();
    match out {
        Ok(o) if o.status.success() => Ok(so),
        Ok(o) => Err(String::from_utf8_lossy(&o.stderr).trim().to_string()),
        Err(e) => Err(e.to_string()),
    }
}

/// A run's `.lock` appears only once it is locked (taken on `.lock.new`, then renamed), so
/// another run that reaps while the run is still taking its lock finds no `.lock` and leaves
/// the directory alone (no `.lock`: live). Created as `.lock` and locked after, the file read
/// as a dead run's lock in between, and the other run removed the live run's new directory.
/// The shim (tests/shim_flock.c, LD_PRELOAD) delays the live run's flock on its own lock file
/// by 10 s; skipped where the shim cannot be built (no C compiler).
#[test]
fn a_run_taking_its_lock_keeps_its_directory() {
    reap_during_shim_delay("lockwin", "SHIM_FLOCK_DELAY_MS", "shim: flock(");
}

/// A rename of `.lock.new` to `.lock` reported as failed after it happened (on NFS, a reply
/// lost after the server renamed) leaves `.lock` there, and the run keeps its lock all the
/// same, so another run that reaps meanwhile finds `.lock` held and leaves the directory alone.
/// The run released the lock on a failed rename before, and the other run removed the live
/// run's directory. The shim does the rename, reports ENOENT and delays the unlink of
/// `.lock.new` that follows by 10 s; skipped where it cannot be built.
#[test]
fn a_run_whose_lock_rename_is_reported_failed_keeps_its_directory() {
    reap_during_shim_delay("renamelost", "SHIM_RENAME_LOST_MS", "shim: rename(");
}

/// A live run of the case with the shim's fault `shim_var` (a 10 s delay), and another run of
/// the case, which reaps first, once the live run's stderr says `marker` (the delay began): the
/// live run's directory must keep its lock file, and both runs their verdicts.
fn reap_during_shim_delay(tag: &str, shim_var: &str, marker: &'static str) {
    let d = scratch(tag);
    let so = match flock_shim(&d) {
        Ok(so) => so,
        Err(why) => {
            let _ = std::fs::remove_dir_all(&d);
            eprintln!("skipped: the flock shim cannot be built ({why})");
            return;
        }
    };
    let base = d.join("out/r4_sum_torch");
    let mut live = Command::new(env!("CARGO_BIN_EXE_ptxv"))
        .args(["check", fixture().to_str().unwrap(), "--variant", "good"])
        .args(["--front", "sass", "--json", "--no-input-coverage"])
        .env("PTXV_ROOT", &d)
        .env("PTXV_THREADS", "1")
        .env("PTXV_NO_VCACHE", "1")
        .env("CUDA_VISIBLE_DEVICES", "")
        .env("LD_PRELOAD", &so)
        .env(shim_var, "10000")
        .current_dir(&d)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    // The shim says on stderr when the live run's delay starts.
    let stderr = live.stderr.take().unwrap();
    let (tx, rx) = std::sync::mpsc::channel();
    let reader = std::thread::spawn(move || {
        use std::io::BufRead;
        let mut all = String::new();
        for line in std::io::BufReader::new(stderr)
            .lines()
            .map_while(Result::ok)
        {
            if line.starts_with(marker) {
                let _ = tx.send(());
            }
            all.push_str(&line);
            all.push('\n');
        }
        all
    });
    let delayed = rx.recv_timeout(std::time::Duration::from_secs(120)).is_ok();
    let dirs = run_dirs(&base);
    let contents: Vec<Vec<String>> = dirs.iter().map(|n| names(&base.join(n))).collect();
    // Another run of the case, which reaps first, while the live run is in the shim's delay.
    let other = rows(spawn(&d, "good"));
    // The live run's directory is there with its lock file, under either name (on NFS a
    // removed open file is renamed to .nfs* and keeps the directory).
    let kept = dirs.iter().all(|n| {
        std::fs::read_dir(base.join(n)).is_ok_and(|es| {
            es.flatten()
                .any(|e| e.file_name() == ".lock" || e.file_name() == ".lock.new")
        })
    });
    let got = rows(live);
    let err = reader.join().unwrap();
    let _ = std::fs::remove_dir_all(&d);
    assert!(
        delayed,
        "the shim never delayed the live run's lock ({marker}): {err}"
    );
    assert_eq!(
        dirs.len(),
        1,
        "run directories while the live run waits: {dirs:?}"
    );
    assert!(
        kept,
        "the other run removed the live run's directory {dirs:?} (it held {contents:?})"
    );
    let good = vec![("good".to_string(), expected("good"))];
    assert_eq!(other.unwrap_or_else(|e| panic!("the other run: {e}")), good);
    assert_eq!(
        got.unwrap_or_else(|e| panic!("the live run: {e} {err}")),
        good
    );
}

fn copy_dir(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for e in std::fs::read_dir(from).unwrap() {
        let e = e.unwrap();
        if e.file_type().unwrap().is_dir() {
            copy_dir(&e.path(), &to.join(e.file_name()));
        } else {
            std::fs::copy(e.path(), to.join(e.file_name())).unwrap();
        }
    }
}

/// `ptxv check --certify` stores each verdict in the verdict cache (vcache.rs) under a key of
/// the case's files. Run from inside the case directory without `--out` or PTXV_ROOT, the
/// run's own directory `out/<case>/run-...` is in the case and was part of the key: a name of
/// its own per run, so no later run (equiv's precondition, errdist) found a stored verdict.
/// Two runs now store under the same keys.
#[test]
fn runs_inside_the_case_share_the_verdict_cache() {
    let d = scratch("vcache");
    copy_dir(&fixture(), &d.join("case"));
    let cache = d.join("cache");
    let mut failed = Vec::new();
    for _ in 0..2 {
        let out = Command::new(env!("CARGO_BIN_EXE_ptxv"))
            .args(["check", ".", "--certify", "--front", "sass", "--json"])
            .arg("--no-input-coverage")
            .env_remove("PTXV_ROOT")
            .env_remove("PTXV_NO_VCACHE")
            .env("PTXV_CACHE", &cache)
            .env("PTXV_THREADS", "1")
            .env("CUDA_VISIBLE_DEVICES", "")
            .current_dir(d.join("case"))
            .output()
            .unwrap();
        if !out.status.success() {
            failed.push(String::from_utf8_lossy(&out.stderr).into_owned());
        }
    }
    let entries = names(&cache.join("check"));
    let _ = std::fs::remove_dir_all(&d);
    assert!(failed.is_empty(), "{failed:?}");
    // One entry per (variant, front): good and nosb, SASS front.
    assert_eq!(entries.len(), 2, "{entries:?}");
}

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Run directories under `pair` (compare's and equiv's default directories hold only them).
fn run_dirs(pair: &Path) -> Vec<String> {
    names(pair)
        .into_iter()
        .filter(|n| n.starts_with("run-"))
        .collect()
}

/// `ptxv compare` without `--out` wrote into one directory per pair, `$TMPDIR/ptxv-compare-
/// <ref>-<cand>`, shared by concurrent runs of the pair: each run's report.json was the last
/// writer's, and `report_dir` named the same directory for both. Each run now has its own
/// (printed, `report_dir`), kept when it ends; the next run of the pair removes the finished
/// ones.
#[test]
fn concurrent_compares_without_out_keep_their_own_reports() {
    let d = scratch("compare");
    let spawn = |label: &str| {
        Command::new(env!("CARGO_BIN_EXE_ptxv"))
            .args([
                "compare",
                "fixtures/c_cat",
                "fixtures/c_triton_cat",
                "--json",
            ])
            .args(["--ref-label", label])
            .current_dir(repo())
            .env("TMPDIR", &d)
            .env("PTXV_QUIET", "1")
            .env("CUDA_VISIBLE_DEVICES", "")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap()
    };
    let wait = |label: &str, c: Child| -> Result<PathBuf, String> {
        let out = c.wait_with_output().unwrap();
        let err = String::from_utf8_lossy(&out.stderr).into_owned();
        let doc: serde_json::Value = serde_json::from_slice(&out.stdout).map_err(|_| err)?;
        if doc["reference"]["label"] != label || doc["equivalence"]["level"] != "L0" {
            return Err(format!("{label}: {doc}"));
        }
        Ok(PathBuf::from(
            doc["report_dir"].as_str().unwrap_or_default(),
        ))
    };
    let labels = ["first run", "second run"];
    let runs: Vec<Child> = labels.iter().map(|l| spawn(l)).collect();
    let dirs: Vec<Result<PathBuf, String>> =
        labels.iter().zip(runs).map(|(l, c)| wait(l, c)).collect();
    // What each run's report directory holds once both ended.
    let after: Vec<String> = dirs
        .iter()
        .map(|r| match r {
            Ok(dir) => std::fs::read_to_string(dir.join("report.json"))
                .ok()
                .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok())
                .map_or("no report".into(), |j| {
                    j["reference"]["label"].as_str().unwrap_or("").to_string()
                }),
            Err(e) => e.clone(),
        })
        .collect();
    let pair = d.join("ptxv-compare-c_cat-c_triton_cat");
    let before_next = run_dirs(&pair);
    let next = wait("next run", spawn("next run"));
    let after_next = run_dirs(&pair);
    let _ = std::fs::remove_dir_all(&d);
    assert_eq!(after, labels, "reports {dirs:?}");
    assert_ne!(dirs[0], dirs[1], "one report directory for two runs");
    assert_eq!(before_next.len(), 2, "{before_next:?}");
    let next = next.unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(
        after_next,
        [next.file_name().unwrap().to_string_lossy().into_owned()],
        "the finished runs' directories are removed by the next run"
    );
}

/// The same for `ptxv equiv` without `--out` (`$TMPDIR/ptxv-equiv-<dir>-<stem>`): concurrent
/// runs of one file each get their own directory and summary.
#[test]
fn concurrent_equivs_without_out_keep_their_own_summaries() {
    let d = scratch("equiv");
    let spawn = || {
        Command::new(env!("CARGO_BIN_EXE_ptxv"))
            .args(["equiv", "fixtures/c_equiv_cat/equiv.toml", "--no-hw"])
            .current_dir(repo())
            .env("TMPDIR", &d)
            .env("PTXV_QUIET", "1")
            .env("CUDA_VISIBLE_DEVICES", "")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap()
    };
    let runs: Vec<Child> = (0..2).map(|_| spawn()).collect();
    let mut summaries = Vec::new();
    let mut failed = Vec::new();
    for c in runs {
        let out = c.wait_with_output().unwrap();
        let text = String::from_utf8_lossy(&out.stdout).into_owned();
        let summary = text
            .lines()
            .find_map(|l| l.strip_prefix("JSON summary: "))
            .map(PathBuf::from);
        match summary {
            Some(s) if out.status.success() && text.contains("expected L0: ok") => {
                summaries.push(s)
            }
            _ => failed.push(format!("{text}{}", String::from_utf8_lossy(&out.stderr))),
        }
    }
    let readable = summaries
        .iter()
        .filter(|s| {
            std::fs::read_to_string(s)
                .ok()
                .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok())
                .is_some_and(|j| j[0]["strongest"] == "L0")
        })
        .count();
    let _ = std::fs::remove_dir_all(&d);
    assert!(failed.is_empty(), "{failed:?}");
    assert_ne!(summaries[0], summaries[1], "one directory for two runs");
    assert_eq!(readable, 2, "{summaries:?}");
}
