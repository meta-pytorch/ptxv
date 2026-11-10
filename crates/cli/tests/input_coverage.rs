/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! `ptxv check` input coverage output: a kernel whose control depends on input data that
//! reaches it through TMEM (fixtures/e_cov_tccp: tcgen05.cp, then tcgen05.ld into a guard),
//! through cp.async (fixtures/e_cov_cpasync) or through a parameter-space lookup table
//! (fixtures/e_cov_ldconst) is never printed as input-independent, on either front-end.

use std::path::Path;
use std::path::PathBuf;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn dependent_control_is_not_called_input_independent(fixture: &str, variant: &str) -> Vec<String> {
    let mut texts = Vec::new();
    let case = root().join("fixtures").join(fixture);
    for front in ["ptx", "sass"] {
        let dir = std::env::temp_dir().join(format!(
            "ptxv-test-input-coverage-{fixture}-{variant}-{front}-{}",
            std::process::id()
        ));
        let out = std::process::Command::new(env!("CARGO_BIN_EXE_ptxv"))
            .args([
                "check",
                case.to_str().unwrap(),
                "--variant",
                variant,
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
            "{fixture} {front}: {text}"
        );
        assert!(
            !text.contains("input-independent:"),
            "{fixture} {front}: {text}"
        );
        assert!(
            text.contains("input-dependent control:"),
            "{fixture} {front}: {text}"
        );
        texts.push(text.to_string());
    }
    texts
}

#[test]
fn tmem_dependent_control_is_not_called_input_independent() {
    dependent_control_is_not_called_input_independent("e_cov_tccp", "good");
}

#[test]
fn cp_async_dependent_control_is_not_called_input_independent() {
    dependent_control_is_not_called_input_independent("e_cov_cpasync", "good");
}

/// A comparison of a parameter-space table entry indexed by input data is an input-dependent
/// comparison on both front-ends (before 2026-11-08 the PTX `ld.param` through a register
/// address left the entry clean: no comparison).
#[test]
fn table_lookup_comparison_is_a_site() {
    for text in dependent_control_is_not_called_input_independent("e_cov_ldconst", "good") {
        assert!(
            text.contains("input coverage: 1 input-dependent comparison(s)"),
            "{text}"
        );
    }
}
