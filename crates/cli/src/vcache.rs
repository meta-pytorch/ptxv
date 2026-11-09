/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Execution-check verdict cache: `ptxv check --certify` stores the level-1 verdict of each
//! (variant, front-end), and the precondition of `errdist` / `equiv` (equiv.rs `prepare`,
//! the same `wellsync::check_certified` on the same machine) reads it instead of exploring the
//! kernel again.
//!
//! The key covers everything the verdict is a function of: every file under the case
//! directory (recursively: `data/` holds the inputs that steer control flow), the variant,
//! the front-end, the ptxv binary (path, size, modification time) and every `PTXV_*`
//! environment variable except the ones that only change scheduling of the search threads,
//! output and caching. A stored verdict serves any request when it is complete (the same
//! verdict at every state cap), and an incomplete one serves requests for at most as many
//! states as it was computed with (it explored at least as far as the request would).
//! Files: `$PTXV_CACHE/check/<key>.json` (default `~/.cache/ptxv`). `PTXV_NO_VCACHE=1`
//! disables both reading and writing.

use std::path::Path;
use std::path::PathBuf;

use ptxv_sim::explore::Verdict;

/// Variables that do not change a verdict's classes: thread counts, progress and debug
/// output, caches, errdist budgets, out dirs.
const IGNORED: &[&str] = &[
    "PTXV_THREADS",
    "PTXV_SLICE_THREADS",
    "PTXV_QUIET",
    "PTXV_STATS",
    "PTXV_CACHE",
    "PTXV_NO_VCACHE",
    "PTXV_LIFT_REPORT",
    "PTXV_RUN_ONCE_PROGRESS",
    "PTXV_XCHECK_SECONDS",
    "PTXV_NL_SECONDS",
    "PTXV_ERRDIST_THREADS",
    "PTXV_ERRDIST_MEM",
    "PTXV_ERRDIST_DEBUG",
    "PTXV_GPUS",
    "PTXV_ROOT",
    "PTXV_PYTHON",
    "PTXV_CUDA_HOME",
    "PTXV_BIN",
    "PTXV_TRACE_DUMP",
    "PTXV_DUMP_OUTPUTS",
    "PTXV_WS_DEBUG",
    "PTXV_POR_DEBUG",
    "PTXV_RING_DEBUG",
    "PTXV_LA_DEBUG",
    "PTXV_LA_STATS",
    "PTXV_LA_TRACE",
    "PTXV_NO_INPUT_COVERAGE",
];

const VERSION: u32 = 1;

fn on() -> bool {
    std::env::var_os("PTXV_NO_VCACHE").is_none()
}

struct Fnv(u64);

impl Fnv {
    fn eat(&mut self, b: &[u8]) {
        for &x in b {
            self.0 ^= x as u64;
            self.0 = self.0.wrapping_mul(0x100000001b3);
        }
        // Length separator: ("ab", "c") and ("a", "bc") differ.
        for x in (b.len() as u64).to_le_bytes() {
            self.0 ^= x as u64;
            self.0 = self.0.wrapping_mul(0x100000001b3);
        }
    }
}

fn eat_dir(h: &mut Fnv, base: &Path, d: &Path) -> std::io::Result<()> {
    let mut es: Vec<PathBuf> = std::fs::read_dir(d)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .collect();
    es.sort();
    for p in es {
        let rel = p.strip_prefix(base).unwrap_or(&p);
        if p.is_dir() {
            h.eat(b"d");
            h.eat(rel.as_os_str().as_encoded_bytes());
            eat_dir(h, base, &p)?;
        } else if p.is_file() {
            h.eat(b"f");
            h.eat(rel.as_os_str().as_encoded_bytes());
            h.eat(&std::fs::read(&p)?);
        }
    }
    Ok(())
}

/// The cache key of a level-1 check of `variant` / `front` of the case in `case_dir`, in
/// `mode` ("certify"), or None when it cannot be formed (then nothing is cached).
pub(crate) fn key(case_dir: &Path, variant: &str, front: &str, mode: &str) -> Option<String> {
    if !on() {
        return None;
    }
    let mut h = Fnv(0xcbf29ce484222325);
    h.eat(&VERSION.to_le_bytes());
    h.eat(env!("CARGO_PKG_VERSION").as_bytes());
    let exe = std::env::current_exe().ok()?;
    let md = std::fs::metadata(&exe).ok()?;
    h.eat(exe.as_os_str().as_encoded_bytes());
    h.eat(&md.len().to_le_bytes());
    let t = md
        .modified()
        .ok()?
        .duration_since(std::time::UNIX_EPOCH)
        .ok()?;
    h.eat(&t.as_nanos().to_le_bytes());
    h.eat(variant.as_bytes());
    h.eat(front.as_bytes());
    h.eat(mode.as_bytes());
    let mut env: Vec<(String, String)> = std::env::vars()
        .filter(|(k, _)| k.starts_with("PTXV_") && !IGNORED.contains(&k.as_str()))
        .collect();
    env.sort();
    for (k, v) in env {
        h.eat(k.as_bytes());
        h.eat(v.as_bytes());
    }
    let dir = case_dir.canonicalize().ok()?;
    eat_dir(&mut h, &dir, &dir).ok()?;
    Some(format!("{:016x}", h.0))
}

fn base() -> PathBuf {
    std::env::var_os("PTXV_CACHE")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(std::env::var_os("HOME").unwrap_or_default()).join(".cache/ptxv")
        })
}

fn path(base: &Path, key: &str) -> PathBuf {
    base.join("check").join(format!("{key}.json"))
}

#[derive(serde::Serialize, serde::Deserialize)]
struct Entry {
    version: u32,
    max_states: usize,
    verdict: Verdict,
}

/// Store a verdict computed with `max_states` (best effort: errors are ignored).
pub(crate) fn store(key: &str, max_states: usize, v: &Verdict) {
    store_in(&base(), key, max_states, v)
}

fn store_in(base: &Path, key: &str, max_states: usize, v: &Verdict) {
    let p = path(base, key);
    let e = Entry {
        version: VERSION,
        max_states,
        verdict: v.clone(),
    };
    let Ok(b) = serde_json::to_vec(&e) else {
        return;
    };
    if let Some(d) = p.parent() {
        let _ = std::fs::create_dir_all(d);
    }
    // Write then rename: concurrent readers see the old file or the new one.
    let tmp = p.with_extension(format!("tmp{}", std::process::id()));
    if std::fs::write(&tmp, b).is_ok() {
        let _ = std::fs::rename(&tmp, &p);
    }
}

/// A stored verdict usable for a request of `max_states` states.
pub(crate) fn load(key: &str, max_states: usize) -> Option<Verdict> {
    load_in(&base(), key, max_states)
}

fn load_in(base: &Path, key: &str, max_states: usize) -> Option<Verdict> {
    let b = std::fs::read(path(base, key)).ok()?;
    let e: Entry = serde_json::from_slice(&b).ok()?;
    if e.version != VERSION {
        return None;
    }
    (e.verdict.complete || e.max_states >= max_states).then_some(e.verdict)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn verdict(complete: bool) -> Verdict {
        Verdict {
            states: 7,
            transitions: 0,
            complete,
            terminal_states: 1,
            distinct_outputs: 1,
            findings: vec![],
            assumptions: vec!["E-WELLSYNC".into()],
            notes: vec![],
            seconds: 0.0,
            outputs: vec![],
        }
    }

    /// The key changes with any file of the case (including `data/`), the variant and the
    /// front-end; a stored incomplete verdict serves only requests it explored as far as.
    #[test]
    fn key_covers_the_case_and_caps_are_respected() {
        let d = std::env::temp_dir().join(format!("ptxv-vcache-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(d.join("case/data")).unwrap();
        std::fs::write(d.join("case/case.toml"), "name = \"x\"").unwrap();
        std::fs::write(d.join("case/data/a.bin"), [1u8, 2, 3]).unwrap();
        let k0 = key(&d.join("case"), "good", "sass", "certify").unwrap();
        assert_eq!(k0, key(&d.join("case"), "good", "sass", "certify").unwrap());
        assert_ne!(k0, key(&d.join("case"), "B1", "sass", "certify").unwrap());
        assert_ne!(k0, key(&d.join("case"), "good", "ptx", "certify").unwrap());
        std::fs::write(d.join("case/data/a.bin"), [1u8, 2, 4]).unwrap();
        let k1 = key(&d.join("case"), "good", "sass", "certify").unwrap();
        assert_ne!(k0, k1, "a data file change must change the key");
        let cache = d.join("cache");
        store_in(&cache, &k1, 1000, &verdict(false));
        assert!(load_in(&cache, &k1, 1000).is_some());
        assert!(load_in(&cache, &k1, 500).is_some());
        assert!(load_in(&cache, &k1, 2000).is_none());
        store_in(&cache, &k1, 1000, &verdict(true));
        assert!(load_in(&cache, &k1, 2000).is_some());
        assert!(load_in(&cache, &k0, 10).is_none());
        let _ = std::fs::remove_dir_all(&d);
    }
}
