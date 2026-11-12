/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! `ptxv check --slices` on a multi-launch case keeps launch order: fixtures/r1e_pdl (a
//! producer and a PDL consumer on one stream) is clean, as without `--slices`. Checking one
//! cluster at a time across launches reported the consumer's reads of the producer's output
//! as cross-cluster races.

use std::path::PathBuf;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

#[test]
fn slices_keep_launch_order() {
    let dir = std::env::temp_dir().join(format!("ptxv-test-slices-{}", std::process::id()));
    for extra in [&["--slices"][..], &["--slices", "--certify"][..]] {
        let out = std::process::Command::new(env!("CARGO_BIN_EXE_ptxv"))
            .arg("check")
            .arg(root().join("fixtures/r1e_pdl"))
            .args([
                "--variant",
                "good",
                "--no-gb300",
                "--no-input-coverage",
                "--out",
            ])
            .arg(&dir)
            .args(extra)
            .env("CUDA_VISIBLE_DEVICES", "")
            .output()
            .unwrap();
        let text = format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        assert!(out.status.success(), "{extra:?}: {text}");
        assert!(!text.contains("RACE:"), "{extra:?}: {text}");
        assert!(text.contains("--groups auto --slices"), "{extra:?}: {text}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}
