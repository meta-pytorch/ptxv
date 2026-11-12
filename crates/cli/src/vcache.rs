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
//! directory (recursively: `data/` holds the inputs that steer control flow; ptxv's own output
//! directories there are not input, `is_output_dir`), the variant,
//! the front-end, the ptxv binary (path, size, modification time) and every `PTXV_*`
//! environment variable except the ones that only change scheduling of the search threads,
//! output and caching. A stored verdict serves any request when it is complete (the same
//! verdict at every state cap), and an incomplete one serves requests for at most as many
//! states as it was computed with (it explored at least as far as the request would).
//! Files: `$PTXV_CACHE/check/<key>.json` (default `~/.cache/ptxv`). `PTXV_NO_VCACHE=1`
//! disables both reading and writing.
//!
//! The key is of the files, so only a check of the case as its directory holds it is stored
//! (`as_on_disk`): `ptxv check --residency N` changes the manifest in memory, and its verdict
//! would otherwise serve equiv / errdist, which check the case with its own residency.

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

const VERSION: u32 = 2;

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

/// Whether directory `p` under the case directory holds ptxv's own output, not case input: a
/// default output directory (it holds the `.case` claim, main.rs `default_out`) or a run
/// directory in one (`out/<case>/run-...`, main.rs `run_dir`; multi-node runs make no claim).
/// A run from inside the case directory without `--out` or PTXV_ROOT writes there, and every
/// run's directory has a name of its own, so the key never matched another run's.
fn is_output_dir(p: &Path) -> bool {
    let run = p
        .file_name()
        .and_then(|n| n.to_str())
        .is_some_and(crate::is_run_dir_name);
    let in_out = p
        .parent()
        .and_then(Path::parent)
        .and_then(Path::file_name)
        .is_some_and(|n| n == "out");
    p.join(".case").is_file() || run && in_out
}

fn eat_dir(h: &mut Fnv, base: &Path, d: &Path) -> std::io::Result<()> {
    let mut es: Vec<PathBuf> = std::fs::read_dir(d)?
        .map(|e| e.map(|e| e.path()))
        .collect::<std::io::Result<_>>()?;
    es.sort();
    for p in es {
        let rel = p.strip_prefix(base).unwrap_or(&p);
        // Directories count by the files in them (their paths name the directories): one that
        // holds only output, such as `out/`, adds nothing.
        if p.is_dir() {
            if !is_output_dir(&p) {
                eat_dir(h, base, &p)?;
            }
        } else if p.is_file() {
            h.eat(b"f");
            h.eat(rel.as_os_str().as_encoded_bytes());
            h.eat(&std::fs::read(&p)?);
        }
    }
    Ok(())
}

/// Whether `case` is the case its directory holds (`Case::load`, the manifest compared field by
/// field), so that a verdict of it may be stored under the key of the directory's files. A
/// manifest changed in memory (`ptxv check --residency N`) is not.
pub(crate) fn as_on_disk(case: &ptxv_case::Case) -> bool {
    ptxv_case::Case::load(&case.dir).is_ok_and(|d| format!("{:?}", d.m) == format!("{:?}", case.m))
}

/// The cache key of a level-1 check of `variant` / `front` of the case in `case_dir`, in
/// `mode` ("certify"), or None when it cannot be formed or PTXV_NO_VCACHE is set (then nothing
/// is cached).
pub(crate) fn key(case_dir: &Path, variant: &str, front: &str, mode: &str) -> Option<String> {
    if !on() {
        return None;
    }
    key_of(case_dir, variant, front, mode)
}

/// `key` whatever PTXV_NO_VCACHE says (the tests read the key with it set too).
fn key_of(case_dir: &Path, variant: &str, front: &str, mode: &str) -> Option<String> {
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
    let case = ptxv_case::Case::load(&dir).ok()?;
    let mut dependencies = Vec::new();
    if !case.m.ptx.is_empty() {
        dependencies.push(dir.join(&case.m.ptx));
    }
    for file in case
        .m
        .cubin
        .iter()
        .chain(case.m.launches.iter().filter_map(|l| l.cubin.as_ref()))
        .chain(
            case.m
                .variants
                .iter()
                .flat_map(|v| v.patch.iter())
                .filter_map(|p| p.cubin.as_ref()),
        )
        .chain(
            case.m
                .buffers
                .iter()
                .flat_map(|b| [b.file.as_ref(), b.cubin.as_ref()].into_iter().flatten()),
        )
    {
        dependencies.push(dir.join(file));
    }
    let mut dependencies = dependencies
        .into_iter()
        .map(|p| p.canonicalize())
        .collect::<std::io::Result<Vec<_>>>()
        .ok()?;
    dependencies.sort();
    dependencies.dedup();
    for path in dependencies {
        h.eat(path.as_os_str().as_encoded_bytes());
        h.eat(&std::fs::read(path).ok()?);
    }
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

    /// Only the case as its directory holds it is stored: a manifest changed in memory
    /// (`ptxv check --residency N`) is not, nor a case whose directory holds another.
    #[test]
    fn only_the_case_on_disk_is_stored() {
        let fx = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/e_resid");
        let mut c = ptxv_case::Case::load(&fx).unwrap();
        assert!(as_on_disk(&c));
        assert!(as_on_disk(&c.for_variant("chain").unwrap()));
        c.m.residency = Some(4);
        assert!(!as_on_disk(&c));
        let mut other = ptxv_case::Case::load(&fx).unwrap();
        other.dir = fx.join("../e_equiv_order");
        assert!(!as_on_disk(&other));
        other.dir = fx.join("no_such_case");
        assert!(!as_on_disk(&other));
    }

    /// The key changes with any file of the case (including `data/`), the variant and the
    /// front-end; a stored incomplete verdict serves only requests it explored as far as.
    #[test]
    fn key_covers_the_case_and_caps_are_respected() {
        let d = std::env::temp_dir().join(format!("ptxv-vcache-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(d.join("case/data")).unwrap();
        std::fs::write(
            d.join("case/case.toml"),
            r#"
name = "x"
description = "cache dependencies"
ptx = "../kernel.ptx"
cubin = "../kernel.cubin"
entry = "k"
grid = [1,1,1]
block = [1,1,1]
outputs = ["out"]
params = []
[[buffers]]
name = "out"
dtype = "u8"
shape = [3]
init = "file"
file = "../input.bin"
[[variants]]
id = "good"
expect = ["clean"]
"#,
        )
        .unwrap();
        for file in ["kernel.ptx", "kernel.cubin", "input.bin"] {
            std::fs::write(d.join(file), [1u8, 2, 3]).unwrap();
        }
        std::fs::write(d.join("case/data/a.bin"), [1u8, 2, 3]).unwrap();
        let k0 = key_of(&d.join("case"), "good", "sass", "certify").unwrap();
        assert_eq!(
            k0,
            key_of(&d.join("case"), "good", "sass", "certify").unwrap()
        );
        assert_ne!(
            k0,
            key_of(&d.join("case"), "B1", "sass", "certify").unwrap()
        );
        assert_ne!(
            k0,
            key_of(&d.join("case"), "good", "ptx", "certify").unwrap()
        );
        std::fs::write(d.join("case/data/a.bin"), [1u8, 2, 4]).unwrap();
        let k1 = key_of(&d.join("case"), "good", "sass", "certify").unwrap();
        assert_ne!(k0, k1, "a data file change must change the key");
        let mut previous = k1.clone();
        for file in ["kernel.ptx", "kernel.cubin", "input.bin"] {
            std::fs::write(d.join(file), [1u8, 2, 4]).unwrap();
            let changed = key_of(&d.join("case"), "good", "sass", "certify").unwrap();
            assert_ne!(previous, changed, "external {file} must change the key");
            previous = changed;
        }
        std::fs::remove_file(d.join("input.bin")).unwrap();
        assert!(key_of(&d.join("case"), "good", "sass", "certify").is_none());
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

    /// A run from inside the case directory without `--out` or PTXV_ROOT materializes in
    /// `out/<case>/run-<host>-<pid>-<start>-<ns>`, a name of its own per run: such directories
    /// (and the default output directory that holds them) are not part of the key, so a later
    /// run finds the verdict. Case files elsewhere still are, whatever their name.
    #[test]
    fn run_directories_in_the_case_are_not_part_of_the_key() {
        let root = std::env::temp_dir().join(format!("ptxv-vcache-out-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let d = root.join("case");
        std::fs::create_dir_all(d.join("data")).unwrap();
        std::fs::write(
            d.join("case.toml"),
            r#"
name = "x"
description = "out dirs"
ptx = "../kernel.ptx"
cubin = "../kernel.cubin"
entry = "k"
grid = [1,1,1]
block = [1,1,1]
outputs = ["out"]
params = []
[[buffers]]
name = "out"
dtype = "u8"
shape = [3]
init = "file"
file = "../input.bin"
[[variants]]
id = "good"
expect = ["clean"]
"#,
        )
        .unwrap();
        for file in ["kernel.cubin", "input.bin"] {
            std::fs::write(root.join(file), [1u8, 2, 3]).unwrap();
        }
        std::fs::write(root.join("kernel.ptx"), [1u8, 2, 3]).unwrap();
        std::fs::write(d.join("data/a.bin"), [1u8]).unwrap();
        let k = || key_of(&d, "good", "sass", "certify").unwrap();
        let k0 = k();
        let put = |rel: &str| {
            let p = d.join(rel);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(p, [7u8]).unwrap();
        };
        // A run's directory, with its claim; an older binary's; a multi-node run's (no claim).
        put("out/x/.case");
        put("out/x/run-host.example-123-4567-1794443314609297873/good.cubin");
        put("out/x/run-123-1794443314609297873/good.cubin");
        put("out/y/run-host.example-9-8-7/rank0/good.cubin");
        // A kept run directory (probe, calib-mma, sb-sweep) and a stray file in the claim.
        put("out/x/run-host.example-5-6-7/.kept");
        put("out/x/notes.txt");
        assert_eq!(k0, k(), "ptxv output directories must not change the key");
        // Case input that only looks like a run directory: still part of the key.
        put("data/run-1-2/b.bin");
        let k1 = k();
        assert_ne!(
            k0, k1,
            "a data directory named like a run directory is input"
        );
        put("out/z/c.bin");
        assert_ne!(k1, k(), "an `out` directory without a claim is input");
        let _ = std::fs::remove_dir_all(&root);
    }
}
