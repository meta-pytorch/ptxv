/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Memory budget of an exploration: the explorers stop, incomplete, when the process gets near
//! its memory limit, instead of aborting on a failed allocation (`memory allocation of N bytes
//! failed`, SIGABRT, no verdict). Limits: `PTXV_MAX_RSS` (bytes, or with a K / M / G / T
//! suffix) on the resident set; else 90% of the cgroup's `memory.max` (systemd-run
//! MemoryMax, Slurm) on the resident set, and 90% of `RLIMIT_AS` (`ulimit -v`) on the
//! virtual size. `PTXV_MAX_RSS=0` turns the budget off.

use std::sync::OnceLock;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use std::time::Instant;

#[derive(Debug, Default)]
struct Limits {
    rss: Option<u64>,
    vsize: Option<u64>,
}

/// `"96G"` / `"512M"` / `"1048576"` -> bytes.
fn parse_size(s: &str) -> Option<u64> {
    let s = s.trim();
    let (n, mul) = match s.chars().last()?.to_ascii_uppercase() {
        'K' => (&s[..s.len() - 1], 1u64 << 10),
        'M' => (&s[..s.len() - 1], 1 << 20),
        'G' => (&s[..s.len() - 1], 1 << 30),
        'T' => (&s[..s.len() - 1], 1 << 40),
        _ => (s, 1),
    };
    let v: f64 = n.trim().parse().ok()?;
    (v >= 0.0).then_some((v * mul as f64) as u64)
}

/// The cgroup v2 `memory.max` of this process (None: unlimited or not found).
fn cgroup_max() -> Option<u64> {
    let cg = std::fs::read_to_string("/proc/self/cgroup").ok()?;
    let path = cg.lines().find_map(|l| l.strip_prefix("0::"))?;
    // The process's own cgroup, then its ancestors (a scope inside a limited slice).
    let mut p = std::path::PathBuf::from(format!("/sys/fs/cgroup{}", path.trim()));
    let mut best: Option<u64> = None;
    loop {
        if let Ok(s) = std::fs::read_to_string(p.join("memory.max")) {
            if let Ok(v) = s.trim().parse::<u64>() {
                best = Some(best.map_or(v, |b| b.min(v)));
            }
        }
        if !p.pop() || p.as_os_str() == "/sys/fs" {
            break;
        }
    }
    best
}

/// `RLIMIT_AS` soft limit from /proc/self/limits (None: unlimited).
fn rlimit_as() -> Option<u64> {
    let s = std::fs::read_to_string("/proc/self/limits").ok()?;
    let l = s.lines().find(|l| l.starts_with("Max address space"))?;
    l["Max address space".len()..]
        .split_whitespace()
        .next()?
        .parse()
        .ok()
}

fn limits() -> &'static Limits {
    static L: OnceLock<Limits> = OnceLock::new();
    L.get_or_init(|| {
        if let Ok(s) = std::env::var("PTXV_MAX_RSS") {
            return match parse_size(&s) {
                Some(0) => Limits::default(),
                Some(v) => Limits {
                    rss: Some(v),
                    vsize: rlimit_as().map(|v| v / 10 * 9),
                },
                None => {
                    eprintln!("PTXV_MAX_RSS={s}: not a size; no memory budget");
                    Limits::default()
                }
            };
        }
        Limits {
            rss: cgroup_max().map(|v| v / 10 * 9),
            vsize: rlimit_as().map(|v| v / 10 * 9),
        }
    })
}

/// (virtual size, resident set) of this process in bytes, from /proc/self/status.
fn usage() -> (u64, u64) {
    let s = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    let kb = |key: &str| -> u64 {
        s.lines()
            .find(|l| l.starts_with(key))
            .and_then(|l| l.split_whitespace().nth(1)?.parse::<u64>().ok())
            .unwrap_or(0)
            * 1024
    };
    (kb("VmSize:"), kb("VmRSS:"))
}

static HIT: AtomicBool = AtomicBool::new(false);
static LAST_MS: AtomicU64 = AtomicU64::new(0);
static WHY: OnceLock<String> = OnceLock::new();

/// Has the process reached its memory budget? Cheap: reads /proc at most every 100 ms; once
/// reached it stays reached.
pub fn exceeded() -> bool {
    if HIT.load(Ordering::Relaxed) {
        return true;
    }
    let l = limits();
    if l.rss.is_none() && l.vsize.is_none() {
        return false;
    }
    static T0: OnceLock<Instant> = OnceLock::new();
    let now = T0.get_or_init(Instant::now).elapsed().as_millis() as u64 + 100;
    let last = LAST_MS.load(Ordering::Relaxed);
    if now < last + 100
        || LAST_MS
            .compare_exchange(last, now, Ordering::Relaxed, Ordering::Relaxed)
            .is_err()
    {
        return false;
    }
    let (vs, rss) = usage();
    let g = |b: u64| b as f64 / (1u64 << 30) as f64;
    let why = match (l.rss, l.vsize) {
        (Some(m), _) if rss >= m => {
            format!("resident set {:.1} GiB >= budget {:.1} GiB", g(rss), g(m))
        }
        (_, Some(m)) if vs >= m => format!(
            "virtual size {:.1} GiB >= 90% of the address-space limit ({:.1} GiB)",
            g(vs),
            g(m)
        ),
        _ => return false,
    };
    let _ = WHY.set(why);
    HIT.store(true, Ordering::Relaxed);
    true
}

/// Time budget of the current exploration (`set_deadline`): milliseconds since `T_BASE`, 0 for
/// none.
static DEADLINE_MS: AtomicU64 = AtomicU64::new(0);
static DEADLINE_HIT: AtomicBool = AtomicBool::new(false);
static T_BASE: OnceLock<Instant> = OnceLock::new();

/// Sets (or clears) a wall-clock deadline for the explorations that follow (the total budget
/// of `--mutate-inputs`): they stop, incomplete, when it passes. Clearing it also clears the
/// note.
pub fn set_deadline(d: Option<Instant>) {
    let base = *T_BASE.get_or_init(Instant::now);
    DEADLINE_HIT.store(false, Ordering::Relaxed);
    DEADLINE_MS.store(
        d.map_or(0, |d| {
            d.saturating_duration_since(base).as_millis() as u64 + 1
        }),
        Ordering::Relaxed,
    );
}

/// Has the deadline set by `set_deadline` passed? Once it has, it stays passed (until the
/// next `set_deadline`).
pub fn past_deadline() -> bool {
    let d = DEADLINE_MS.load(Ordering::Relaxed);
    if d == 0 {
        return false;
    }
    if DEADLINE_HIT.load(Ordering::Relaxed) {
        return true;
    }
    let now = T_BASE.get_or_init(Instant::now).elapsed().as_millis() as u64 + 1;
    if now >= d {
        DEADLINE_HIT.store(true, Ordering::Relaxed);
        return true;
    }
    false
}

/// The note an incomplete verdict carries when the deadline stopped it.
pub fn deadline_note() -> Option<String> {
    DEADLINE_HIT.load(Ordering::Relaxed).then(|| {
        "exploration stopped at the time budget (--mutate-budget-seconds): incomplete".to_string()
    })
}

/// The resident set of this process in bytes.
pub fn rss() -> u64 {
    usage().1
}

/// The resident-set limit the budget uses (`PTXV_MAX_RSS`, else 90% of the cgroup limit).
pub fn rss_limit() -> Option<u64> {
    limits().rss
}

/// The note an incomplete verdict carries when the budget stopped it.
pub fn note() -> Option<String> {
    HIT.load(Ordering::Relaxed).then(|| {
        format!(
            "exploration stopped at the memory budget ({}; PTXV_MAX_RSS sets it): incomplete",
            WHY.get().map_or("", |s| s.as_str())
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes() {
        assert_eq!(parse_size("96G"), Some(96 << 30));
        assert_eq!(parse_size("512m"), Some(512 << 20));
        assert_eq!(parse_size("1000"), Some(1000));
        assert_eq!(parse_size("0"), Some(0));
        assert_eq!(parse_size("x"), None);
    }
}
