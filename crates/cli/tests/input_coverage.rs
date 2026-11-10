/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! `ptxv check` input coverage output: a kernel whose control depends on input data that
//! reaches it through TMEM (fixtures/e_cov_tccp: tcgen05.cp, then tcgen05.ld into a guard)
//! is never printed as input-independent, on either front-end.

use std::path::Path;
use std::path::PathBuf;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

#[test]
fn tmem_dependent_control_is_not_called_input_independent() {
    let case = root().join("fixtures/e_cov_tccp");
    for front in ["ptx", "sass"] {
        let dir = std::env::temp_dir().join(format!(
            "ptxv-test-input-coverage-{front}-{}",
            std::process::id()
        ));
        let out = std::process::Command::new(env!("CARGO_BIN_EXE_ptxv"))
            .args([
                "check",
                case.to_str().unwrap(),
                "--variant",
                "good",
                "--front",
                front,
            ])
            .args(["--out", dir.to_str().unwrap()])
            .current_dir(root())
            .env("CUDA_VISIBLE_DEVICES", "")
            .output()
            .unwrap();
        let text = String::from_utf8_lossy(&out.stdout);
        assert!(
            out.status.success() || text.contains("PASS"),
            "{front}: {text}"
        );
        assert!(!text.contains("input-independent:"), "{front}: {text}");
        assert!(text.contains("input-dependent control:"), "{front}: {text}");
    }
}
