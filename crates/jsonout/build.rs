/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Build stamp: the ptxv commit this binary is built from, as `PTXV_REV` for `env!`.
//!
//! In order: the `PTXV_REV` environment variable; `git rev-parse HEAD` of the workspace when the
//! workspace is the root of its own git work tree ("<12 hex>+dirty" when crates/, Cargo.toml or
//! Cargo.lock have uncommitted changes, as tools/build.sh stamps it); a `REV` file at the
//! workspace root (a source copy without .git, e.g. a remote build snapshot); else empty
//! (unknown, reported as null).

use std::path::Path;
use std::path::PathBuf;
use std::process::Command;

fn git(root: &Path, args: &[&str]) -> Option<String> {
    let o = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(args)
        .output()
        .ok()?;
    o.status
        .success()
        .then(|| String::from_utf8_lossy(&o.stdout).trim().to_string())
}

fn main() {
    let root = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap())
        .join("../..")
        .canonicalize()
        .unwrap();
    println!("cargo:rerun-if-env-changed=PTXV_REV");
    // Only existing paths: cargo reruns a script on every build for a missing one. Sources whose
    // edits make the tree dirty (tests and fixtures do not change the binary).
    watch(&root.join("REV"));
    for f in ["Cargo.toml", "Cargo.lock"] {
        watch(&root.join(f));
    }
    if let Ok(rd) = std::fs::read_dir(root.join("crates")) {
        for e in rd.flatten() {
            for sub in ["src", "Cargo.toml", "build.rs"] {
                watch(&e.path().join(sub));
            }
        }
    }
    let rev = std::env::var("PTXV_REV")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.trim().to_string())
        .or_else(|| from_git(&root))
        .or_else(|| {
            std::fs::read_to_string(root.join("REV"))
                .ok()
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
        })
        .unwrap_or_default();
    println!("cargo:rustc-env=PTXV_REV={rev}");
    let ver = std::env::var("CARGO_PKG_VERSION").unwrap_or_default();
    let long = if rev.is_empty() {
        format!("{ver} (rev unknown)")
    } else {
        format!("{ver} (rev {rev})")
    };
    println!("cargo:rustc-env=PTXV_VERSION={long}");
}

fn watch(p: &Path) {
    if p.exists() {
        println!("cargo:rerun-if-changed={}", p.display());
    }
}

/// HEAD of the workspace's own git work tree (not of an enclosing repository).
fn from_git(root: &Path) -> Option<String> {
    let top = PathBuf::from(git(root, &["rev-parse", "--show-toplevel"])?);
    if top.canonicalize().ok()? != root {
        return None;
    }
    // Rerun when HEAD moves: HEAD itself, the branch ref it names, and packed refs.
    let path = |name: &str| {
        git(
            root,
            &["rev-parse", "--path-format=absolute", "--git-path", name],
        )
    };
    if let Some(p) = path("HEAD") {
        watch(Path::new(&p));
    }
    if let Some(r) = git(root, &["symbolic-ref", "-q", "HEAD"]) {
        if let Some(p) = path(&r) {
            watch(Path::new(&p));
        }
    }
    if let Some(p) = path("packed-refs") {
        watch(Path::new(&p));
    }
    let rev = git(root, &["rev-parse", "--short=12", "HEAD"])?;
    let dirty = git(
        root,
        &[
            "status",
            "--porcelain",
            "--untracked-files=no",
            "--",
            "crates",
            "Cargo.toml",
            "Cargo.lock",
        ],
    )
    .is_some_and(|s| !s.is_empty());
    Some(if dirty { format!("{rev}+dirty") } else { rev })
}
