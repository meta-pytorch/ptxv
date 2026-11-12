/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Every source mutant of tools/mutate/srcmut.py finds its text exactly once in this tree
//! (`srcmut.py --check-anchors`, which builds and runs nothing). A change to code a mutant
//! anchors on re-points the mutant's find text in the same commit, or this test fails and names
//! the mutant. A mutant whose find text is gone can no longer be applied, and srcmut reports it
//! as an error only when the mutant is run.

use std::path::PathBuf;
use std::process::Command;
use std::process::Output;

/// The positive control's mutant (an entry of srcmut.py's own) and its file.
const CONTROL: &str = "srcmut-none-chosen-ok";
const CONTROL_FILE: &str = "tools/mutate/srcmut.py";

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// `$PTXV_PYTHON -I tools/mutate/srcmut.py ARGS` (default `python3`) in the repository root.
fn srcmut(args: &[&str]) -> Output {
    let root = repo();
    let srcmut = root.join("tools/mutate/srcmut.py");
    let python = std::env::var("PTXV_PYTHON").unwrap_or_else(|_| "python3".to_string());
    Command::new(&python)
        .arg("-I")
        .arg(&srcmut)
        .args(args)
        .current_dir(&root)
        .output()
        .unwrap_or_else(|e| panic!("{python} {}: {e}", srcmut.display()))
}

/// `srcmut.py --check-anchors` exits 0. Pass or fail is its exit status; its output only names
/// the broken mutants. Two controls come first. Negative: the check of a mutant that does not
/// exist, and the check with an --only that names nothing, must exit 2 (srcmut's --only error),
/// so that an interpreter that exits the same whatever it is given (PTXV_PYTHON=/bin/true, say)
/// or an --only path that checks anchors instead cannot pass. Any error is not enough: in a
/// tree with a mutant of srcmut.py applied, that mutant's own anchor is gone, so a check of
/// every mutant exits 1. Positive: the check of one mutant (CONTROL) against a scratch tree that
/// holds its file (`--root`) exits 0 with the file as it is and 1 with a broken anchor planted
/// (the file twice, so that the find text occurs twice; the file emptied; no file), so that a
/// check that reports nothing, or that reads another tree, cannot pass.
#[test]
fn every_srcmut_mutant_finds_its_text_once() {
    for only in ["no-such-mutant", ""] {
        let none = srcmut(&["--check-anchors", "--only", only]);
        assert_eq!(
            none.status.code(),
            Some(2),
            "srcmut.py --check-anchors --only {only:?} exited with {}, not with the --only \
             error (2), so the check below could pass without checking anything:\n{}{}",
            none.status,
            String::from_utf8_lossy(&none.stdout),
            String::from_utf8_lossy(&none.stderr)
        );
    }
    let text = std::fs::read_to_string(repo().join(CONTROL_FILE)).unwrap();
    let scratch = std::env::temp_dir().join(format!("ptxv-srcmut-anchors-{}", std::process::id()));
    let file = scratch.join(CONTROL_FILE);
    std::fs::create_dir_all(file.parent().unwrap()).unwrap();
    for (planted, want, what) in [
        (Some(text.clone()), 0, "the file as it is"),
        (Some(text.repeat(2)), 1, "the file twice"),
        (Some(String::new()), 1, "the file emptied"),
        (None, 1, "no file"),
    ] {
        match planted {
            Some(t) => std::fs::write(&file, t).unwrap(),
            None => std::fs::remove_file(&file).unwrap(),
        }
        let root = scratch.to_str().unwrap();
        let out = srcmut(&["--check-anchors", "--only", CONTROL, "--root", root]);
        let why = if want == 0 {
            format!("{CONTROL} must stay a mutant of {CONTROL_FILE}")
        } else {
            "the check passed a broken anchor".to_string()
        };
        assert_eq!(
            out.status.code(),
            Some(want),
            "positive control, {what}: srcmut.py --check-anchors --only {CONTROL} --root {root} \
             exited with {}, not {want} ({why}):\n{}{}",
            out.status,
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
    }
    let _ = std::fs::remove_dir_all(&scratch);
    let out = srcmut(&["--check-anchors"]);
    assert!(
        out.status.success(),
        "srcmut.py --check-anchors exited with {}; re-point each mutant below in the commit \
         that changed its code:\n{}{}",
        out.status,
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
}
